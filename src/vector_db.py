from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import Config
from src.models import EnrichedIssue
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        print("📂 Using Local Vector Storage (No Docker)")
        self.client = QdrantClient(path="./qdrant_data_local") 
        
        self.collection_name = Config.COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if it doesn't exist with correct vector config."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating collection {self.collection_name}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Matching all-MiniLM-L6-v2
                    distance=models.Distance.COSINE
                )
            )
        else:
            logger.info(f"Connected to collection {self.collection_name}")

    def upsert_batch(self, issues: List[EnrichedIssue]):
        """Uploads data points + vectors to Qdrant."""
        if not issues:
            return
        
        points = [
            models.PointStruct(
                id=abs(hash(issue.key)),  # Deterministic integer ID from string key
                vector=issue.embedding,
                payload={
                    "key": issue.key,
                    "summary": issue.summary,
                    "description": issue.description,
                    "status": issue.status,
                    "assignee": issue.assignee,
                    "labels": issue.labels,
                    "url": issue.url,
                    "created_at": issue.created_at.isoformat()
                }
            )
            for issue in issues
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(issues)} vectors to Qdrant.")

    def search(self, query_vector: List[float], limit: int = 5):
        """Semantic Search implementation."""
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )