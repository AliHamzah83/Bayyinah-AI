"""General domain models used throughout the application."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BaseEntity:
    """Common base for domain entities."""

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class DocumentRecord(BaseEntity):
    """Represents an uploaded document in the system."""

    id: Optional[str] = None
    filename: Optional[str] = None
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    mime_type: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "uploaded"


@dataclass
class ChunkRecord(BaseEntity):
    """Represents a processed chunk for retrieval."""

    id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: int = 0
    parent_chunk_id: Optional[str] = None
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_id: Optional[str] = None


@dataclass
class QueryRequest:
    """Simple request model used by the API layer."""

    question: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5
    use_hybrid_search: bool = True
