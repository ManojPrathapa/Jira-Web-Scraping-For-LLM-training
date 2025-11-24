from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any
from datetime import datetime

class JiraComment(BaseModel):
    author: str
    body: str
    created: datetime

class JiraIssue(BaseModel):
    """
    The Raw Data Contract. 
    Represents the clean structure we expect from the source.
    """
    key: str
    summary: str
    description: Optional[str] = "No description provided."
    status: str
    priority: str
    assignee: Optional[str] = "Unassigned"
    created_at: datetime
    labels: List[str] = Field(default_factory=list)
    comments: List[JiraComment] = Field(default_factory=list)
    url: str

    @field_validator('description')
    def clean_description(cls, v):
        if not v:
            return "No description provided."
        return v.strip()

class EnrichedIssue(JiraIssue):
    """
    The AI-Ready Data Contract.
    Contains the vector embedding and token count.
    """
    embedding: List[float]
    token_count: int
    processing_time_ms: float