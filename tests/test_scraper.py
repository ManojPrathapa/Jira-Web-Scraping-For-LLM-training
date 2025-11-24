import asyncio
import json
from unittest.mock import AsyncMock, patch
from src import jira_scraper

def test_transform_issue_minimal():
    # minimal shape
    issue = {
        "key": "HADOOP-1",
        "fields": {"summary": "s", "description": "d", "project": {"key": "HADOOP"}}
    }
    out = jira_scraper.transform_issue(issue)
    assert out["id"] == "HADOOP-1"
    assert out["project"] == "HADOOP"
