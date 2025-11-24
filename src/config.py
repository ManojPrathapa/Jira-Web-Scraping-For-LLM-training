import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # JIRA CONFIGURATION
    JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "https://your-domain.atlassian.net")
    JIRA_USERNAME = os.getenv("JIRA_USERNAME", "user@example.com")
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "your-api-token")
    JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "KAN")
    
    # VECTOR DB
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME = "jira_issues_v2"
    
    # AI & ML
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") # Optional: For real LLM generation
    
    # FLAGS
    USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "True").lower() == "true"
    MOCK_COUNT =  2000  # Generated 2000 issues for a dense dashboard