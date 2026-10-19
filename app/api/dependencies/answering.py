"""Answer-generation dependency wiring."""

from __future__ import annotations

from app.api.dependencies.retrieval import get_retrieval_service
from src.answering import AnswerGenerationService
from src.core.settings import get_settings


def get_answering_service() -> AnswerGenerationService:
    """Construct the answering service with configured retrieval and LLM settings."""
    settings = get_settings()
    return AnswerGenerationService(
        retrieval_service=get_retrieval_service(),
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )