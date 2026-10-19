"""Integration-style tests for Qdrant-backed Arabic retrieval."""

from __future__ import annotations

from qdrant_client import QdrantClient

from src.chunking import TextChunk
from src.embeddings import EmbeddingService
from src.storage.qdrant_store import QdrantVectorStore, RetrievalService


class FakeEncoder:
    def __init__(self, vectors=None):
        self.calls = []
        self.vectors = vectors or {
            "الذكاء الاصطناعي": [1.0, 0.0, 0.0],
            "الذكاء الاصطناعي وتطبيقاته في التعليم": [1.0, 0.0, 0.0],
            "الاستثمار": [0.0, 1.0, 0.0],
            "الاستثمار يتطلب تحليل المخاطر": [0.0, 1.0, 0.0],
            "الأمن": [0.0, 0.0, 1.0],
            "الأمن السيبراني يحمي الأنظمة": [0.0, 0.0, 1.0],
        }

    def encode(self, sentences, **_kwargs):
        self.calls.append(sentences)
        return [self.vectors.get(sentence, [0.0, 0.0, 1.0]) for sentence in sentences]


def make_chunk(index: int, text: str, *, document_id: str = "a" * 64) -> TextChunk:
    return TextChunk(
        chunk_id=f"{index:064x}",
        document_id=document_id,
        chunk_index=index,
        page_number=1,
        strategy="recursive",
        text=text,
        source_text=text,
        estimated_token_count=4,
        char_count=len(text),
        source_start=0,
        source_end=len(text),
        source_filename="source.pdf",
        page_language="arabic",
        extraction_method="native",
        source_metadata={"page_count": 1, "sha256": "b" * 64},
    )


def test_qdrant_vector_store_indexes_and_searches_similarity():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(
        client=client,
        collection_name="test_arabic_vectors",
        vector_size=3,
    )
    chunks = [
        make_chunk(0, "الذكاء الاصطناعي يغير العالم"),
        make_chunk(1, "الاستثمار المالي يتطلب تحليل السوق"),
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.index_chunks(chunks, vectors)
    hits = store.search([1.0, 0.0, 0.0], limit=1)

    assert hits
    assert hits[0].chunk_id == chunks[0].chunk_id
    assert hits[0].text == chunks[0].text
    assert hits[0].source_filename == "source.pdf"


def test_retrieval_service_returns_top_k_chunks_with_source_metadata():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(
        client=client,
        collection_name="test_arabic_retrieval",
        vector_size=3,
    )
    encoder = EmbeddingService(expected_dimensions=3, encoder=FakeEncoder())
    service = RetrievalService(embedding_service=encoder, vector_store=store)

    chunks = [
        make_chunk(0, "الذكاء الاصطناعي وتطبيقاته في التعليم"),
        make_chunk(1, "الاستثمار يتطلب تحليل المخاطر"),
        make_chunk(2, "الأمن السيبراني يحمي الأنظمة"),
    ]
    service.index_chunks(chunks)

    hits = service.search("الذكاء الاصطناعي", top_k=2)

    assert len(hits) == 2
    assert hits[0].chunk_id == chunks[0].chunk_id
    assert hits[0].source_filename == "source.pdf"
    assert hits[0].metadata["page_count"] == 1
    assert hits[0].source_metadata["sha256"] == "b" * 64


def test_retrieval_service_embeds_and_persists_incremental_batches():
    class RecordingStore:
        def __init__(self):
            self.batch_sizes = []

        def index_chunks(self, chunks, vectors=None):
            self.batch_sizes.append((len(chunks), len(vectors or [])))

    chunks = [make_chunk(index, f"معلومة {index}") for index in range(5)]
    store = RecordingStore()
    encoder = FakeEncoder()
    service = RetrievalService(
        embedding_service=EmbeddingService(
            expected_dimensions=3,
            batch_size=2,
            encoder=encoder,
        ),
        vector_store=store,
    )

    summary = service.index_chunks(chunks)

    assert store.batch_sizes == [(2, 2), (2, 2), (1, 1)]
    assert [len(batch) for batch in encoder.calls] == [2, 2, 1]
    assert summary.performance.chunk_count == 5
    assert summary.model_name == "intfloat/multilingual-e5-small"
