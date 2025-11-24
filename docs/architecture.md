# Architecture overview

- **Scraper (src/jira_scraper.py)**: Async, resumable scraper using Jira REST API. Writes raw JSONL per project.
- **Transform (src/transform.py)**: Cleans HTML, scrubs PII, deduplicates and normalizes timestamps to create `data/processed/*.jsonl`.
- **State directory**: per-project checkpoint files `state/{PROJECT}.json`.
- **DLQ**: malformed items saved to `dlq/` for manual inspection.
- **Demo**: `notebooks/quick_demo.py` for quick EDA.
- **CI**: GitHub Actions run lint and tests.

Design reasoning, tradeoffs, and future improvements are described in README.
