"""Retrieval dependency wiring for the FastAPI API."""

from __future__ import annotations

from src.core.settings import get_settings
from src.embeddings import EMBEDDING_DIMENSIONS, EmbeddingService
from src.storage.qdrant_store import QdrantVectorStore, RetrievalService


def get_retrieval_service(*, collection_name: str | None = None) -> RetrievalService:
    """Construct the retrieval service using the project’s shared Qdrant abstraction."""
    settings = get_settings()
    vector_store = QdrantVectorStore(
        collection_name=collection_name or settings.qdrant_collection_name,
        vector_size=EMBEDDING_DIMENSIONS,
        timeout=settings.qdrant_timeout,
    )
    embedding_service = EmbeddingService(expected_dimensions=EMBEDDING_DIMENSIONS)
    return RetrievalService(embedding_service=embedding_service, vector_store=vector_store)
