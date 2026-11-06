"""End-to-end API tests for document upload, indexing, and retrieval."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pymupdf
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.api.routers import indexing as indexing_router
from app.api.dependencies.indexing import get_indexing_service
from app.api.dependencies.retrieval import get_retrieval_service
from app.api.dependencies.answering import ANSWER_COLLECTION_NAME, get_answering_service
from app.main import app
from src.answering import AnswerGenerationService
from src.embeddings import EmbeddingService
from src.indexing import DocumentIndexingService
from src.storage.qdrant_store import QdrantVectorStore, RetrievalService


class FixedEncoder:
    def encode(self, sentences, **_kwargs):
        return [[1.0, 0.0, 0.0] for _ in sentences]


class FixedChatCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(content="The document describes an indexing workflow.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FixedOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FixedChatCompletions())


def make_pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Arabic document indexing integration test")
    content = pdf.tobytes()
    pdf.close()
    return content


class EmptyRetrievalService:
    def search(self, query, *, top_k=5):
        return []


def test_upload_indexes_document_and_makes_it_searchable(tmp_path, monkeypatch):
    vector_store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        collection_name="test_document_upload",
        vector_size=3,
    )
    retrieval_service = RetrievalService(
        embedding_service=EmbeddingService(expected_dimensions=3, encoder=FixedEncoder()),
        vector_store=vector_store,
    )
    indexing_service = DocumentIndexingService(
        retrieval_service=retrieval_service,
        output_dir=tmp_path / "processed",
    )
    llm_client = FixedOpenAIClient()
    answering_service = AnswerGenerationService(
        retrieval_service=retrieval_service,
        model="test-model",
        client=llm_client,
    )
    monkeypatch.setitem(app.dependency_overrides, get_retrieval_service, lambda: retrieval_service)
    monkeypatch.setitem(app.dependency_overrides, get_indexing_service, lambda: indexing_service)
    monkeypatch.setitem(app.dependency_overrides, get_answering_service, lambda: answering_service)

    with TestClient(app) as client:
        upload_response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("integration.pdf", make_pdf(), "application/pdf")},
        )
        search_response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "find the integration document", "top_k": 5},
        )
        answer_response = client.post(
            "/api/v1/retrieval/answer",
            json={"query": "What does the document describe?", "top_k": 5},
        )

    assert upload_response.status_code == 201
    indexed = upload_response.json()
    assert indexed["status"] == "indexed"
    assert indexed["metadata"]["source_filename"] == "integration.pdf"
    assert indexed["statistics"]["page_count"] == 1
    assert indexed["statistics"]["chunk_count"] == 1
    assert indexed["statistics"]["indexed_vectors"] == 1
    assert indexed["statistics"]["embedding_model"] == "intfloat/multilingual-e5-small"

    stored_metadata = tmp_path / "processed" / f"{indexed['document_id']}.json"
    assert (
        json.loads(stored_metadata.read_text(encoding="utf-8"))["document_id"]
        == indexed["document_id"]
    )
    assert search_response.status_code == 200
    hits = search_response.json()["results"]
    assert hits[0]["document_id"] == indexed["document_id"]
    assert hits[0]["source"]["filename"] == "integration.pdf"

    assert answer_response.status_code == 200
    answer = answer_response.json()
    assert answer["model"] == "test-model"
    assert "indexing workflow" in answer["answer"]
    assert "- [1] integration.pdf, page 1" in answer["answer"]
    assert answer["sources"][0]["document_id"] == indexed["document_id"]
    assert answer["sources"][0]["citation_id"] == "[1]"
    assert llm_client.chat.completions.request["model"] == "test-model"
    prompt = llm_client.chat.completions.request["messages"][1]["content"]
    assert "Arabic document indexing integration test" in prompt


def test_answering_uses_configured_ollama_model_and_endpoint(monkeypatch):
    settings = SimpleNamespace(
        ollama_base_url="http://ollama-host:11434/v1",
        ollama_model="qwen2.5:3b",
    )
    monkeypatch.setattr("src.answering.get_settings", lambda: settings)
    created_clients = []
    llm_client = FixedOpenAIClient()

    def make_client(**kwargs):
        created_clients.append(kwargs)
        return llm_client

    monkeypatch.setattr("src.answering.OpenAI", make_client)
    hit = SimpleNamespace(
        document_id="document-1",
        source_filename="source.pdf",
        page_number=1,
        text="معلومة من المستند",
        score=0.9,
    )

    class RetrievalWithContext:
        def search(self, _query, *, top_k=5):
            return [hit]

    service = AnswerGenerationService(retrieval_service=RetrievalWithContext())
    result = service.answer("ما المعلومة؟")

    assert result.model == "qwen2.5:3b"
    assert created_clients == [
        {"api_key": "ollama", "base_url": "http://ollama-host:11434/v1"}
    ]
    assert llm_client.chat.completions.request["model"] == "qwen2.5:3b"


def test_answering_uses_e5_small_collection(monkeypatch):
    retrieval_service = EmptyRetrievalService()
    requested_collections = []

    def make_retrieval_service(*, collection_name=None):
        requested_collections.append(collection_name)
        return retrieval_service

    monkeypatch.setattr(
        "app.api.dependencies.answering.get_retrieval_service",
        make_retrieval_service,
    )

    service = get_answering_service()

    assert requested_collections == [ANSWER_COLLECTION_NAME]
    assert ANSWER_COLLECTION_NAME == "arabic_documents_e5_small"
    assert service.retrieval_service is retrieval_service


def test_answer_without_matching_context_does_not_call_llm(monkeypatch):
    llm_client = FixedOpenAIClient()
    answering_service = AnswerGenerationService(
        retrieval_service=EmptyRetrievalService(),
        model="test-model",
        client=llm_client,
    )
    monkeypatch.setitem(app.dependency_overrides, get_answering_service, lambda: answering_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/retrieval/answer",
            json={"query": "ما هي المعلومات المتاحة؟", "top_k": 5},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == (
        "### الإجابة\n"
        "لم أجد معلومات كافية في الوثائق المتاحة للإجابة عن هذا السؤال.\n\n"
        "### المصادر\n"
        "لا توجد مصادر متاحة."
    )
    assert result["sources"] == []
    assert llm_client.chat.completions.request is None


def test_upload_rejects_file_over_configured_size_limit(monkeypatch):
    class IndexingServiceMustNotRun:
        def index_pdf(self, _file_path):
            raise AssertionError("indexing must not start for an oversized upload")

    monkeypatch.setattr(
        indexing_router,
        "get_settings",
        lambda: SimpleNamespace(max_pdf_size_bytes=4),
    )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_indexing_service,
        lambda: IndexingServiceMustNotRun(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("large.pdf", b"12345", "application/pdf")},
        )

    assert response.status_code == 413
    assert "maximum file size" in response.json()["detail"]
