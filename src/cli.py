"""Simple CLI to run scraper / transformer."""
import argparse
from . import jira_scraper, transform
from .config import JIRA_PROJECTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scrape", "transform"], required=True)
    parser.add_argument("--projects", nargs="+", default=JIRA_PROJECTS)
    args = parser.parse_args()

    if args.mode == "scrape":
        import asyncio
        asyncio.run(jira_scraper.run(args.projects))
    elif args.mode == "transform":
        transform.run_all()

if __name__ == "__main__":
    main()
