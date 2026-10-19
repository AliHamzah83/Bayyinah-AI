"""Document indexing dependency wiring for the FastAPI API."""

from __future__ import annotations

from app.api.dependencies.retrieval import get_retrieval_service
from src.indexing import DocumentIndexingService


def get_indexing_service() -> DocumentIndexingService:
    """Construct the indexing service with the shared retrieval infrastructure."""
    return DocumentIndexingService(retrieval_service=get_retrieval_service())