"""
Streamlit FAST dashboard for Jira -> LLM Explorer (optimized for large datasets)

Key optimizations:
- DEFAULT_MAX_DOCS limits the number of loaded documents (default 500).
- Smart sampling: selects most recently-updated issues first if timestamps exist.
- Embedding cache on disk (.cache/embeddings_<hash>.npz) so embeddings are computed once.
- Incremental batch embeddings with Streamlit progress bar.
- Uses float32 embeddings and sklearn NearestNeighbors with n_jobs=-1.
- Graceful fallbacks on memory errors.

Place at repo root and run:
    streamlit run streamlit_app_fast.py
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# optional OpenAI usage
try:
    import openai
except Exception:
    openai = None

# ---------------------- CONFIG ----------------------
DATA_PROCESSED = Path("data/processed")
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH = 64
DEFAULT_MAX_DOCS = 500  # default sample size to keep memory low
TOP_K_DEFAULT = 5
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------- HELPERS ----------------------
def file_list_from_processed(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.glob("*.jsonl")])


def read_jsonl_limited(path: Path, limit: int) -> List[Dict[str, Any]]:
    """Read at most `limit` objects from a JSONL file (streaming)."""
    docs = []
    with path.open(encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            docs.append(obj)
            if len(docs) >= limit:
                break
    return docs


def load_corpus_limited(processed_dir: Path, max_docs: int, project_filter: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load up to max_docs documents in smart order:
    - Prefer most recently updated documents (if 'updated' field exists)
    - Otherwise, read files sequentially and stop when sample reached
    """
    docs = []
    files = file_list_from_processed(processed_dir)
    if not files:
        return []
    # stream and collect but keep only up to some larger pool then sort by updated
    pool = []
    pool_limit = max_docs * 4  # collect a pool larger than needed to choose recent
    for f in files:
        with f.open(encoding="utf8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if project_filter and obj.get("project") != project_filter:
                    continue
                # try parse updated timestamp; keep as string for sorting fallback
                updated = obj.get("updated") or obj.get("created") or ""
                pool.append((updated, obj))
                if len(pool) >= pool_limit:
                    break
        if len(pool) >= pool_limit:
            break

    # sort pool by updated desc if we have timestamps (strings ISO-like sort works usually)
    try:
        pool.sort(key=lambda x: x[0] or "", reverse=True)
    except Exception:
        pass

    # take top max_docs
    docs = [item[1] for item in pool[:max_docs]]
    # fallback: if pool empty, try simpler sequential load
    if not docs:
        for f in files:
            docs.extend(read_jsonl_limited(f, max_docs - len(docs)))
            if len(docs) >= max_docs:
                break
    return docs[:max_docs]


def texts_from_doc(doc: Dict[str, Any]) -> str:
    parts = []
    if doc.get("title"):
        parts.append(doc["title"])
    if doc.get("text"):
        parts.append(doc["text"])
    for c in (doc.get("comments") or [])[:3]:
        if c.get("body"):
            parts.append(c.get("body"))
    return "\n\n".join(parts).strip() or " "


def compute_hash_for_docs(docs: List[Dict[str, Any]], model_name: str, max_docs: int) -> str:
    """Stable hash based on doc ids + model_name + max_docs to version the embedding cache."""
    m = hashlib.sha1()
    m.update(model_name.encode("utf8"))
    m.update(str(max_docs).encode("utf8"))
    # include doc ids and their updated timestamps for cache invalidation
    for d in docs:
        mid = d.get("id", "") + "|" + str(d.get("updated", "")) + "|"
        m.update(mid.encode("utf8"))
    return m.hexdigest()


# ---------------------- EMBEDDING & CACHE ----------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = EMBED_MODEL_NAME):
    return SentenceTransformer(model_name)


def load_or_build_embeddings(docs: List[Dict[str, Any]], model_name: str, force_rebuild: bool, progress_callback=None) -> Tuple[np.ndarray, List[int]]:
    """
    Load embeddings from disk cache if available, else compute and write to cache.
    Returns embeddings (N x D) and index list mapping (0..N-1).
    """
    if not docs:
        return np.zeros((0, 384), dtype=np.float32), []

    cache_key = compute_hash_for_docs(docs, model_name, len(docs))
    cache_file = CACHE_DIR / f"embeddings_{cache_key}.npz"
    ids_file = CACHE_DIR / f"ids_{cache_key}.json"

    if cache_file.exists() and not force_rebuild:
        try:
            with np.load(cache_file) as data:
                emb = data["embeddings"].astype(np.float32)
            with ids_file.open(encoding="utf8") as f:
                idxs = json.load(f)
            return emb, idxs
        except Exception:
            # if loading fails, rebuild
            pass

    embedder = get_embedder(model_name)
    texts = [texts_from_doc(d) for d in docs]
    texts = [t if t else " " for t in texts]

    # incremental batch encoding with progress callback
    batches = []
    total = len(texts)
    emb_dim = embedder.get_sentence_embedding_dimension()
    try:
        for i in range(0, total, EMBED_BATCH):
            batch_texts = texts[i: i + EMBED_BATCH]
            emb_batch = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            # ensure float32
            if emb_batch.dtype != np.float32:
                emb_batch = emb_batch.astype(np.float32)
            batches.append(emb_batch)
            if progress_callback:
                progress_callback(min(i + EMBED_BATCH, total), total)
        if batches:
            embeddings = np.vstack(batches).astype(np.float32)
        else:
            embeddings = np.zeros((0, emb_dim), dtype=np.float32)
    except MemoryError as me:
        st.error("MemoryError during embedding. Try reducing max docs or use fewer CPU cores.")
        raise me

    # write cache atomically
    try:
        np.savez_compressed(cache_file, embeddings=embeddings)
        with ids_file.open("w", encoding="utf8") as f:
            json.dump(list(range(len(docs))), f)
    except Exception:
        # if cache write fails, ignore but return embeddings
        pass

    return embeddings, list(range(len(docs)))


# ---------------------- RETRIEVER ----------------------
def build_retriever(embeddings: np.ndarray):
    if embeddings.shape[0] == 0:
        return None
    # sklearn NearestNeighbors with cosine metric uses distances in [0,2]; n_jobs=-1 uses all CPUs
    try:
        nn = NearestNeighbors(n_neighbors=min(50, embeddings.shape[0]), metric="cosine", n_jobs=-1)
        nn.fit(embeddings)
        return nn
    except Exception as e:
        st.warning(f"Failed to build retriever: {e}")
        return None


def retrieve(knn, embeddings, q_emb: np.ndarray, top_k: int):
    if knn is None or embeddings.shape[0] == 0:
        return []
    dists, idxs = knn.kneighbors(q_emb.reshape(1, -1), n_neighbors=min(top_k, embeddings.shape[0]))
    sims = (1.0 - dists[0]).tolist()
    indices = idxs[0].tolist()
    return list(zip(indices, sims))


# ---------------------- RAG / ANSWER ----------------------
def call_openai_rag(question: str, contexts: List[str], max_tokens: int = 512):
    if openai is None:
        return "OpenAI library not installed or import failed."
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set; set it to enable LLM RAG answers."
    openai.api_key = OPENAI_API_KEY
    system = "You are an assistant that answers technical questions referencing provided Jira issue contexts. If answer is not in context, say you don't know and provide steps to find it."
    context_block = "\n\n---\n\n".join(contexts[:5])
    prompt = f"Context:\n{context_block}\n\nQuestion:\n{question}\n\nAnswer concisely and cite context indices."
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"


def extractive_answer(question: str, contexts: List[str]) -> str:
    if not contexts:
        return "No relevant context found."
    joined = "\n\n---\n\n".join(contexts[:5])
    return f"Top contexts:\n\n{joined}\n\n(Extractive fallback — no LLM.)"


# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="Jira LLM Explorer (FAST)", layout="wide")
st.title("⚡ Jira → LLM Explorer (FAST Mode)")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    max_docs = st.slider("Max documents to load", min_value=50, max_value=5000, value=DEFAULT_MAX_DOCS, step=50,
                         help="Reduce this to lower memory and speed up startup. Default=500")
    project_filter = st.selectbox("Project (filter)", options=["All"] + sorted(list({d.get("project") for d in load_corpus_limited(DATA_PROCESSED, 50, None)})), index=0)
    model_name = st.text_input("Embedding model", value=EMBED_MODEL_NAME)
    top_k = st.slider("Top K results", 1, 20, TOP_K_DEFAULT)
    force_rebuild = st.button("Recompute embeddings (force)", key="force_rebuild")
    st.markdown("---")
    st.markdown("Tips:\n- Lower `Max documents` to improve speed.\n- Use `Recompute embeddings` after changing model or sample size.")

# Load limited corpus
with st.spinner("Loading documents (sampleing)..."):
    selected_proj = None if project_filter == "All" else project_filter
    docs = load_corpus_limited(DATA_PROCESSED, max_docs, selected_proj)

st.success(f"Loaded {len(docs)} documents (max requested {max_docs})")

if not docs:
    st.warning("No processed documents found. Run transform step first.")
    st.stop()

# Embedding & retriever (with progress)
embeddings = None
idx_map = None
retriever = None

progress_text = st.empty()
progress_bar = st.progress(0)

def progress_cb(done, total):
    try:
        progress_bar.progress(int(done / total * 100))
        progress_text.text(f"Embedding progress: {done}/{total}")
    except Exception:
        pass

try:
    embeddings, idx_map = load_or_build_embeddings(docs, model_name, force_rebuild, progress_callback=progress_cb)
    retriever = build_retriever(embeddings)
    progress_bar.empty()
    progress_text.empty()
except MemoryError:
    st.error("Not enough memory to build embeddings. Lower max_docs or run on a machine with more RAM.")
    st.stop()

# Main layout
left, right = st.columns([3, 2])

with left:
    st.subheader("Search / Ask")
    query = st.text_area("Enter search query or question:", height=120)
    if st.button("Search / Ask"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            t0 = time.time()
            embedder = get_embedder(model_name)
            q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype(np.float32)
            hits = retrieve(retriever, embeddings, q_emb, top_k)
            contexts = []
            results = []
            for idx, sim in hits:
                d = docs[idx]
                ctx = texts_from_doc(d)
                contexts.append(ctx)
                results.append((d, sim))
            elapsed = time.time() - t0
            st.write(f"Retrieved {len(results)} results in {elapsed:.2f}s")

            if OPENAI_API_KEY and openai is not None:
                with st.spinner("Calling OpenAI (RAG)..."):
                    ans = call_openai_rag(query, contexts)
                    st.markdown("### LLM Answer (OpenAI)")
                    st.write(ans)
            else:
                st.markdown("### Extractive Answer (Fallback)")
                st.write(extractive_answer(query, contexts))

            st.markdown("### Retrieved Items")
            for i, (doc, sim) in enumerate(results):
                with st.expander(f"[{i+1}] {doc.get('id')} — {doc.get('title')[:120]} (sim={sim:.3f})", expanded=(i==0)):
                    st.write("**Project:**", doc.get("project"))
                    st.write("**Status:**", doc.get("status"))
                    st.write("**Labels:**", doc.get("labels"))
                    st.write("**Created / Updated:**", doc.get("created"), "/", doc.get("updated"))
                    st.write("**Snippet:**")
                    st.write(texts_from_doc(doc)[:1500])
                    st.write("**Full JSON**")
                    st.code(json.dumps(doc, indent=2, ensure_ascii=False))

with right:
    st.subheader("Browse & Export")
    id_list = [f"{d.get('id')} — {d.get('title')[:60]}" for d in docs]
    sel = st.selectbox("Select issue", options=id_list)
    if sel:
        sel_id = sel.split(" — ")[0]
        sel_doc = next((d for d in docs if d.get("id") == sel_id), None)
        if sel_doc:
            st.markdown(f"### {sel_doc.get('id')}: {sel_doc.get('title')}")
            st.write("**Project:**", sel_doc.get("project"))
            st.write("**Status:**", sel_doc.get("status"))
            st.write("**Priority:**", sel_doc.get("priority"))
            st.write("**Labels:**", sel_doc.get("labels"))
            st.write("**Created / Updated:**", sel_doc.get("created"), "/", sel_doc.get("updated"))
            st.markdown("**Description**")
            st.write(sel_doc.get("text"))
            if sel_doc.get("comments"):
                st.markdown("**Comments (latest 5)**")
                for c in sel_doc.get("comments")[:5]:
                    st.write(f"- `{c.get('author')}` {c.get('created')}: {c.get('body')[:600]}")

    st.markdown("---")
    st.subheader("Quick stats")
    st.write(f"- Documents loaded: **{len(docs)}**")
    lens = [len(d.get("text","")) for d in docs]
    if lens:
        st.write(f"- Avg text length: **{int(sum(lens)/len(lens))}** chars")
        st.write(f"- Max text length: **{max(lens)}** chars")
        st.write(f"- Min text length: **{min(lens)}** chars")

    st.markdown("---")
    if st.button("Export loaded docs to JSONL"):
        outp = Path("dash_exports")
        outp.mkdir(exist_ok=True)
        filename = outp / f"export_{(selected_proj or 'ALL')}_{int(time.time())}.jsonl"
        with open(filename, "w", encoding="utf8") as fo:
            for d in docs:
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
        st.success(f"Exported {len(docs)} docs to {filename}")

st.markdown("---")
st.caption("Tip: reduce `Max documents` to speed up. Use `Recompute embeddings` if you change model or sample size.")
