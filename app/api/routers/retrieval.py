"""Retrieval endpoints for vector search in Arabic documents."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.dependencies.answering import get_answering_service
from app.api.dependencies.retrieval import get_retrieval_service
from src.answering import AnswerGenerationService, AnswerResult
from src.storage.qdrant_store import RetrievalHit, RetrievalService

router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval"])


class RetrievalQuery(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/answer", response_model=AnswerResult)
async def answer_question(
    payload: RetrievalQuery,
    service: AnswerGenerationService = Depends(get_answering_service),
) -> AnswerResult:
    """Retrieve supporting passages and generate an answer with citations."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        return await run_in_threadpool(
            service.answer,
            payload.query,
            top_k=payload.top_k,
        )
    except Exception as exc:  # pragma: no cover - external LLM/vector services
        raise HTTPException(status_code=503, detail="Answer generation service unavailable") from exc


@router.post("/search")
def search_documents(
    payload: RetrievalQuery,
    service: RetrievalService = Depends(get_retrieval_service),
) -> dict[str, object]:
    """Embed the query, retrieve the top-k vectors, and return citation metadata."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        hits = service.search(payload.query, top_k=payload.top_k)
    except Exception as exc:  # pragma: no cover - guard for unavailable vector service
        raise HTTPException(status_code=503, detail=f"Retrieval service unavailable: {exc}") from exc

    return {
        "query": payload.query,
        "top_k": payload.top_k,
        "results": [_serialize_hit(hit) for hit in hits],
    }


def _serialize_hit(hit: RetrievalHit) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "text": hit.text,
        "score": hit.score,
        "page_number": hit.page_number,
        "source": {
            "filename": hit.source_filename,
            "page_number": hit.page_number,
            "text": hit.source_text,
            "metadata": hit.source_metadata,
        },
        "metadata": hit.metadata,
    }
