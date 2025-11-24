import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Dict, Any, List
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from aiohttp import ClientResponseError
import time

from .config import DATA_RAW_DIR, STATE_DIR, USER_AGENT, MAX_RESULTS, CONCURRENT_REQUESTS
from .utils import strip_html

BASE = "https://issues.apache.org/jira"

HEADERS = {"Accept": "application/json", "User-Agent": USER_AGENT}

Path(DATA_RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(STATE_DIR).mkdir(parents=True, exist_ok=True)

class RateLimitError(Exception):
    pass

@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(6),
       retry=retry_if_exception_type((aiohttp.ClientError, RateLimitError, ClientResponseError)))
async def fetch(session: aiohttp.ClientSession, url: str, params: Dict[str, Any] = None):
    async with session.get(url, params=params, headers=HEADERS) as resp:
        if resp.status == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = int(retry_after)
                except Exception:
                    delay = 10
                await asyncio.sleep(delay)
            raise RateLimitError("HTTP 429")
        if 500 <= resp.status < 600:
            raise ClientResponseError(status=resp.status, request_info=resp.request_info, history=resp.history)
        return await resp.json()

def load_state(project_key: str) -> Dict[str, Any]:
    p = Path(STATE_DIR) / f"{project_key}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf8"))
    return {"startAt": 0, "fetched": 0, "total": None}

def save_state(project_key: str, state: Dict[str, Any]):
    p = Path(STATE_DIR) / f"{project_key}.json"
    p.write_text(json.dumps(state, indent=2), encoding="utf8")

def transform_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    fields = issue.get("fields", {})
    # description and summary
    summary = fields.get("summary")
    desc = fields.get("description")
    text = "\n\n".join(filter(None, [summary, desc]))
    text_plain = strip_html(text)

    comments = []
    for c in fields.get("comment", {}).get("comments", []):
        comments.append({
            "author": c.get("author", {}).get("displayName"),
            "created": c.get("created"),
            "body": strip_html(c.get("body")) if c.get("body") else None
        })

    return {
        "id": issue.get("key"),
        "project": fields.get("project", {}).get("key"),
        "title": summary,
        "status": fields.get("status", {}).get("name"),
        "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
        "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None,
        "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
        "labels": fields.get("labels", []),
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "text": text_plain,
        "comments": comments,
        "raw": issue
    }

async def scrape_project(project_key: str, out_path: str):
    state = load_state(project_key)
    start_at = state.get("startAt", 0)
    total = state.get("total")
    session_timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(timeout=session_timeout, connector=connector) as session:
        while True:
            params = {
                "jql": f"project={project_key}",
                "startAt": start_at,
                "maxResults": MAX_RESULTS,
                "fields": "summary,description,comment,labels,assignee,reporter,status,priority,created,updated,project"
            }
            url = f"{BASE}/rest/api/2/search"
            try:
                data = await fetch(session, url, params=params)
            except Exception as e:
                print(f"[{project_key}] fetch error: {e}")
                raise

            issues = data.get("issues", [])
            if total is None:
                total = data.get("total", None)

            if not issues:
                print(f"[{project_key}] no more issues.")
                break

            with open(out_path, "a", encoding="utf8") as f:
                for issue in issues:
                    try:
                        obj = transform_issue(issue)
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    except Exception as e:
                        dlq_dir = Path("dlq")
                        dlq_dir.mkdir(exist_ok=True)
                        (dlq_dir / f"{project_key}_dlq.jsonl").write_text(json.dumps(issue), encoding="utf8")
                        print(f"[{project_key}] malformed issue saved to DLQ: {e}")

            fetched = start_at + len(issues)
            state.update({"startAt": fetched, "fetched": fetched, "total": total})
            save_state(project_key, state)

            if total and fetched >= total:
                print(f"[{project_key}] finished. fetched={fetched} total={total}")
                break
            start_at = fetched
            await asyncio.sleep(0.2)

async def run(projects: List[str]):
    tasks = []
    for p in projects:
        out = Path(DATA_RAW_DIR) / f"{p}.jsonl"
        tasks.append(scrape_project(p, str(out)))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    import argparse
    from .config import JIRA_PROJECTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", default=JIRA_PROJECTS)
    args = parser.parse_args()
    asyncio.run(run(args.projects))
