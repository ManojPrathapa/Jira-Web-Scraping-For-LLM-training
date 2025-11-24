"""Configuration and toggles for the pipeline."""
from typing import List

USE_MOCK_DATA = False  # Set True to use existing mock data in repo
JIRA_PROJECTS: List[str] = ["HADOOP", "SPARK", "HIVE"]

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
STATE_DIR = "state"

# Scraper tuning
MAX_RESULTS = 200
CONCURRENT_REQUESTS = 8

# User agent
USER_AGENT = "Manoj-Jira-Scraper/1.0 (+https://github.com/ManojPrathapa)"
