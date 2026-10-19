"""API tests for the retrieval layer."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.api.dependencies.retrieval as retrieval_dependencies
from app.api.dependencies.retrieval import get_retrieval_service
from app.main import app
from src.storage.qdrant_store import RetrievalHit


class FakeRetrievalService:
    def search(self, query: str, *, top_k: int = 5):
        return [
            RetrievalHit(
                chunk_id="0" * 64,
                document_id="a" * 64,
                text="الذكاء الاصطناعي وتطبيقاته في التعليم",
                score=0.92,
                page_number=1,
                source_filename="demo.pdf",
                source_text="الذكاء الاصطناعي وتطبيقاته في التعليم",
                source_metadata={"page_count": 1, "sha256": "b" * 64},
                metadata={"page_count": 1, "sha256": "b" * 64},
            )
        ]


app.dependency_overrides = {}
app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()


client = TestClient(app)


def test_retrieval_dependency_uses_configured_collection(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        retrieval_dependencies,
        "get_settings",
        lambda: SimpleNamespace(
            qdrant_collection_name="arabic_documents_e5_small",
            qdrant_timeout=30,
        ),
    )
    monkeypatch.setattr(
        retrieval_dependencies,
        "QdrantVectorStore",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(retrieval_dependencies, "EmbeddingService", lambda **_kwargs: object())

    retrieval_dependencies.get_retrieval_service()

    assert captured["collection_name"] == "arabic_documents_e5_small"


def test_retrieval_endpoint_returns_query_hits_with_source_references():
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "ما هو الذكاء الاصطناعي؟", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "ما هو الذكاء الاصطناعي؟"
    assert payload["top_k"] == 5
    assert len(payload["results"]) == 1
    assert payload["results"][0]["text"] == "الذكاء الاصطناعي وتطبيقاته في التعليم"
    assert payload["results"][0]["source"]["filename"] == "demo.pdf"
    assert payload["results"][0]["source"]["page_number"] == 1
