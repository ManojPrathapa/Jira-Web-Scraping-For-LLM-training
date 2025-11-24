"""Utility helpers: html cleaning, dedupe, timestamp normalization."""
import re
from bs4 import BeautifulSoup
import hashlib
from dateutil import parser
from typing import Optional

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def strip_html(text: Optional[str]) -> str:
    if not text:
        return ""

    # Sometimes Jira description is plain text or JSON – no need for BS
    if "<" not in text and ">" not in text:
        return text.strip()

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(" ").strip()
    except Exception:
        return text.strip()

def scrub_pii(text: str) -> str:
    # basic email removal; you can extend to other PII
    if not text:
        return text
    return EMAIL_RE.sub("[REDACTED_EMAIL]", text)

def normalize_timestamp(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    try:
        dt = parser.parse(ts)
        return dt.isoformat()
    except Exception:
        return ts

def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf8")).hexdigest()
