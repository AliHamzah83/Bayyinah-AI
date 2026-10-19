"""Entry point for the Arabic RAG API."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routers.indexing import router as indexing_router
from app.api.routers.retrieval import router as retrieval_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Arabic RAG API", version="0.1.0")
app.include_router(indexing_router)
app.include_router(retrieval_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
