"""Answer-generation dependency wiring."""

from __future__ import annotations

from app.api.dependencies.retrieval import get_retrieval_service
from src.answering import AnswerGenerationService
from src.core.settings import get_settings

ANSWER_COLLECTION_NAME = "arabic_documents_e5_small"


def get_answering_service() -> AnswerGenerationService:
    """Construct the answering service with configured retrieval and LLM settings."""
    settings = get_settings()
    return AnswerGenerationService(
        retrieval_service=get_retrieval_service(collection_name=ANSWER_COLLECTION_NAME),
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )