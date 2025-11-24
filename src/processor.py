from sentence_transformers import SentenceTransformer
from src.models import JiraIssue, EnrichedIssue
from src.config import Config
import time
from typing import List

class AIProcessor:
    def __init__(self):
        print(f"Loading AI Model: {Config.EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        print("Model loaded successfully.")

    def enrich_issue(self, issue: JiraIssue) -> EnrichedIssue:
        start_time = time.time()
        
        # We combine fields to create a rich context for the AI
        # This makes the search smarter (it knows about comments and summaries)
        context_text = f"""
        Title: {issue.summary}
        Description: {issue.description}
        Assignee: {issue.assignee}
        Labels: {', '.join(issue.labels)}
        Latest Comments: {' '.join([c.body for c in issue.comments[:3]])}
        """
        
        # Generate Vector (The heavy lifting)
        embedding = self.model.encode(context_text).tolist()
        
        duration = (time.time() - start_time) * 1000
        
        return EnrichedIssue(
            **issue.model_dump(),
            embedding=embedding,
            token_count=len(context_text.split()),
            processing_time_ms=duration
        )

    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()