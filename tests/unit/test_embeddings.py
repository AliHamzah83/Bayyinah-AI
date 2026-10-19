"""Offline tests for the embedding service contract."""

import math
import sys
from types import ModuleType

import pytest

from src.chunking import TextChunk
from src.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingError,
    EmbeddingService,
    compare_storage_requirements,
    generate_embedding_id,
)


class FakeEncoder:
    def __init__(self, vector=None):
        self.vector = vector or [3.0, 4.0, 0.0]
        self.calls = []

    def encode(self, sentences, **_kwargs):
        self.calls.append(sentences)
        return [self.vector for _ in sentences]


def make_chunk(index=0):
    chunk_id = f"{index + 1:064x}"
    return TextChunk(
        chunk_id=chunk_id,
        document_id="a" * 64,
        chunk_index=index,
        page_number=index + 1,
        strategy="recursive",
        text=f"نص عربي للاختبار {index}",
        source_text=f"نص عربي للاختبار {index}",
        estimated_token_count=4,
        char_count=20,
        source_start=0,
        source_end=20,
        source_filename="source.pdf",
        page_language="arabic",
        extraction_method="native",
        source_metadata={"page_count": 2, "sha256": "b" * 64},
    )


def test_embedding_service_normalizes_vectors_and_preserves_metadata():
    encoder = FakeEncoder()
    service = EmbeddingService(
        model_name="test-model",
        batch_size=1,
        expected_dimensions=3,
        encoder=encoder,
    )

    result = service.embed_chunks([make_chunk(0), make_chunk(1)])

    assert len(encoder.calls) == 2
    assert result.model_name == "test-model"
    assert result.dimensions == 3
    assert result.bytes_per_vector == 12
    assert result.estimated_storage_bytes == 24
    assert result.performance.chunk_count == 2
    assert result.performance.generation_seconds >= 0
    assert result.performance.total_seconds >= result.performance.generation_seconds
    assert result.performance.chunks_per_second > 0
    for embedded, chunk in zip(result.embeddings, [make_chunk(0), make_chunk(1)], strict=True):
        assert embedded.chunk_id == chunk.chunk_id
        assert embedded.document_id == chunk.document_id
        assert embedded.page_number == chunk.page_number
        assert embedded.source["filename"] == "source.pdf"
        assert embedded.source["source_text"] == chunk.source_text
        assert embedded.source["metadata"] == chunk.source_metadata
        assert embedded.text == chunk.text
        assert math.isclose(math.sqrt(sum(value * value for value in embedded.vector)), 1.0)


def test_default_adapter_loads_sentence_transformer_model(monkeypatch):
    calls = {}
    sentence_transformers = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name, **options):
            calls["model"] = model_name
            calls["options"] = options

        def encode(self, sentences, **options):
            calls["sentences"] = sentences
            calls["encode"] = options
            return [[3.0, 4.0, 0.0] for _ in sentences]

    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    service = EmbeddingService(expected_dimensions=3)

    result = service.embed_chunks([make_chunk()])

    assert calls["model"] == EMBEDDING_MODEL
    assert calls["sentences"] == ["passage: نص عربي للاختبار 0"]
    assert calls["encode"]["normalize_embeddings"] is True
    assert len(result.embeddings) == 1


def test_embedding_service_validates_dimension_and_finite_values():
    wrong_dimension = EmbeddingService(
        expected_dimensions=3,
        encoder=FakeEncoder(vector=[1.0, 2.0]),
    )
    with pytest.raises(EmbeddingError, match="Expected 3 vector dimensions"):
        wrong_dimension.embed_chunks([make_chunk()])

    non_finite = EmbeddingService(
        expected_dimensions=3,
        encoder=FakeEncoder(vector=[1.0, float("nan"), 2.0]),
    )
    with pytest.raises(EmbeddingError, match="non-finite"):
        non_finite.embed_chunks([make_chunk()])


def test_embedding_service_rejects_empty_chunks_and_zero_vectors():
    empty_chunk = make_chunk().model_copy(update={"text": "  "})
    service = EmbeddingService(expected_dimensions=3, encoder=FakeEncoder())
    with pytest.raises(EmbeddingError, match="empty chunk"):
        service.embed_chunks([empty_chunk])

    zero_encoder = FakeEncoder(vector=[0.0, 0.0, 0.0])
    zero_service = EmbeddingService(expected_dimensions=3, encoder=zero_encoder)
    with pytest.raises(EmbeddingError, match="zero-magnitude"):
        zero_service.embed_chunks([make_chunk()])


def test_empty_batch_does_not_load_encoder():
    encoder = FakeEncoder()
    service = EmbeddingService(expected_dimensions=3, encoder=encoder)

    result = service.embed_chunks([])

    assert result.embeddings == []
    assert result.performance.chunk_count == 0
    assert result.estimated_storage_bytes == 0
    assert encoder.calls == []


def test_storage_comparison_reports_float32_and_float16_payloads():
    estimates = compare_storage_requirements(100_000)
    by_dimensions = {estimate.dimensions: estimate for estimate in estimates}
    e5_small = by_dimensions[EMBEDDING_DIMENSIONS]

    assert e5_small.float32_bytes_per_vector == 1536
    assert e5_small.float32_total_bytes == 153_600_000
    assert e5_small.float16_bytes_per_vector == 768
    assert e5_small.float16_total_bytes == 76_800_000
    assert by_dimensions[384].float32_bytes_per_vector == 1536


def test_embedding_ids_are_stable_and_model_specific():
    chunk_id = "c" * 64

    assert generate_embedding_id(chunk_id) == generate_embedding_id(chunk_id)
    assert generate_embedding_id(chunk_id) != generate_embedding_id(chunk_id, "other-model")
    assert EMBEDDING_MODEL == "intfloat/multilingual-e5-small"