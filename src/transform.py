import json
from pathlib import Path
from typing import Set
from .config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from .utils import strip_html, scrub_pii, normalize_timestamp, stable_hash

Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

def process_file(infile: str, outfile: str, min_text_len: int = 20):
    seen: Set[str] = set()
    written = 0
    with open(infile, "r", encoding="utf8") as fin, open(outfile, "a", encoding="utf8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Compose canonical text
            text = obj.get("text") or ""
            text = scrub_pii(text)
            text = strip_html(text)
            if len(text) < min_text_len:
                continue

            # Normalize timestamps
            obj["created"] = normalize_timestamp(obj.get("created"))
            obj["updated"] = normalize_timestamp(obj.get("updated"))

            # dedupe
            h = stable_hash((obj.get("title") or "") + "||" + text)
            if h in seen:
                continue
            seen.add(h)

            # derived skeleton (empty now); optionally you can call an LLM here
            obj.setdefault("derived", {})

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    return written

def run_all(in_dir: str = DATA_RAW_DIR, out_dir: str = DATA_PROCESSED_DIR):
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in sorted(in_path.glob("*.jsonl")):
        target = out_path / f.name
        wrote = process_file(str(f), str(target))
        print(f"Processed {f.name} -> {target.name}: {wrote} records")
        total += wrote
    print(f"Total processed: {total}")

if __name__ == "__main__":
    run_all()
