"""Reusable dense-embedding service for Arabic document chunks."""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import threading
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.chunking import TextChunk


EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "
logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation or validation fails."""


class Encoder(Protocol):
    """Minimal interface implemented by SentenceTransformer and test encoders."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


class _SentenceTransformerEncoder:
    """Sentence Transformers adapter for dense text embeddings."""

    def __init__(self, model_name: str, device: str | None) -> None:
        logger.info(
            "stage=embedding_model_initialization "
            "step=sentence_transformers_import status=started"
        )
        sentence_transformers = importlib.import_module("sentence_transformers")
        logger.info(
            "stage=embedding_model_initialization "
            "step=sentence_transformers_import status=completed"
        )
        options = {} if device is None else {"device": device}
        load_started = perf_counter()
        logger.info(
            "stage=embedding_model_initialization step=model_load "
            "status=started model=%s device=%s",
            model_name,
            device or "auto",
        )
        try:
            self._model = sentence_transformers.SentenceTransformer(model_name, **options)
        except Exception as exc:
            logger.exception(
                "stage=embedding_model_initialization step=model_load status=failed "
                "model=%s exception_type=%s exception_message=%s duration_seconds=%.3f",
                model_name,
                type(exc).__name__,
                exc,
                perf_counter() - load_started,
            )
            raise
        logger.info(
            "stage=embedding_model_initialization step=model_load status=completed "
            "model=%s duration_seconds=%.3f",
            model_name,
            perf_counter() - load_started,
        )

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        prefix = (
            EMBEDDING_QUERY_PREFIX
            if kwargs.get("input_type") == "query"
            else EMBEDDING_PASSAGE_PREFIX
        )
        return self._model.encode(
            [f"{prefix}{sentence}" for sentence in sentences],
            batch_size=kwargs["batch_size"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class EmbeddedChunk(BaseModel):
    """Dense vector paired with the chunk identity and citation metadata."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    chunk_id: str
    document_id: str
    page_number: int = Field(ge=1)
    source: dict[str, Any]
    text: str
    model_name: str
    dimensions: int = Field(ge=1)
    vector: list[float]


class EmbeddingPerformance(BaseModel):
    """Wall-clock measurements for one embedding batch."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    chunk_count: int = Field(ge=0)
    model_load_seconds: float = Field(ge=0)
    generation_seconds: float = Field(ge=0)
    total_seconds: float = Field(ge=0)
    chunks_per_second: float = Field(ge=0)


class EmbeddingBatch(BaseModel):
    """Generated vectors plus dimension, storage, and performance details."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    dimensions: int = Field(ge=1)
    precision: Literal["float32"] = "float32"
    bytes_per_vector: int = Field(ge=1)
    estimated_storage_bytes: int = Field(ge=0)
    embeddings: list[EmbeddedChunk]
    performance: EmbeddingPerformance


class VectorStorageEstimate(BaseModel):
    """Dense-vector payload sizes, excluding database/index overhead."""

    model_config = ConfigDict(extra="forbid")

    dimensions: int = Field(ge=1)
    float32_bytes_per_vector: int = Field(ge=1)
    float32_total_bytes: int = Field(ge=0)
    float16_bytes_per_vector: int = Field(ge=1)
    float16_total_bytes: int = Field(ge=0)


def compare_storage_requirements(
    vector_count: int,
    *,
    dimensions: Sequence[int] = (384, 768, EMBEDDING_DIMENSIONS, 1536),
) -> list[VectorStorageEstimate]:
    """Compare float32/float16 dense-vector payload sizes for common dimensions."""
    if vector_count < 0:
        raise ValueError("vector_count must be non-negative")
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("all dimensions must be positive")
    return [
        VectorStorageEstimate(
            dimensions=dimension,
            float32_bytes_per_vector=dimension * 4,
            float32_total_bytes=vector_count * dimension * 4,
            float16_bytes_per_vector=dimension * 2,
            float16_total_bytes=vector_count * dimension * 2,
        )
        for dimension in dimensions
    ]


def _batches(items: Sequence[TextChunk], batch_size: int) -> Iterator[Sequence[TextChunk]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _as_vectors(encoded: Any) -> list[list[float]]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    try:
        vectors = [[float(value) for value in vector] for vector in encoded]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("Encoder returned an invalid dense-vector batch") from exc
    return vectors


def _normalize_vector(vector: list[float], expected_dimensions: int) -> list[float]:
    if len(vector) != expected_dimensions:
        raise EmbeddingError(
            f"Expected {expected_dimensions} vector dimensions, received {len(vector)}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingError("Encoder returned a non-finite vector value")
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise EmbeddingError("Encoder returned a zero-magnitude vector")
    return [value / magnitude for value in vector]


class EmbeddingService:
    """Generate normalized dense vectors without coupling to storage."""

    def __init__(
        self,
        *,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        expected_dimensions: int = EMBEDDING_DIMENSIONS,
        encoder: Encoder | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if expected_dimensions < 1:
            raise ValueError("expected_dimensions must be at least 1")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.expected_dimensions = expected_dimensions
        self._encoder = encoder
        self._encoder_ready = encoder is not None
        self._load_lock = threading.Lock()

    def _get_encoder(self) -> tuple[Encoder, float]:
        if self._encoder_ready:
            assert self._encoder is not None
            return self._encoder, 0.0
        with self._load_lock:
            if self._encoder_ready:
                assert self._encoder is not None
                return self._encoder, 0.0
            load_started = perf_counter()
            logger.info(
                "stage=embedding_model_initialization status=started model=%s",
                self.model_name,
            )
            try:
                self._encoder = _SentenceTransformerEncoder(self.model_name, self.device)
            except Exception as exc:
                raise EmbeddingError(
                    f"Unable to load {self.model_name}; install sentence-transformers "
                    "and ensure the model is available locally or from Hugging Face"
                ) from exc
            self._encoder_ready = True
            logger.info(
                "stage=embedding_model_initialization status=completed model=%s",
                self.model_name,
            )
            return self._encoder, perf_counter() - load_started

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed arbitrary strings, normalizing each vector to unit length."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Cannot generate an embedding for an empty text value")

        encoder, _ = self._get_encoder()
        logger.info("stage=embedding_generation status=started inputs=%s", len(texts))
        try:
            encoded = encoder.encode(
                list(texts),
                batch_size=self.batch_size,
                input_type="query",
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError("Dense embedding generation failed") from exc

        vectors = _as_vectors(encoded)
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Encoder returned {len(vectors)} vectors for {len(texts)} text inputs"
            )
        normalized_vectors = [
            _normalize_vector(vector, self.expected_dimensions) for vector in vectors
        ]
        logger.info("stage=embedding_generation status=completed inputs=%s", len(texts))
        return normalized_vectors

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> EmbeddingBatch:
        """Embed chunks in bounded batches and preserve their citation metadata."""
        if not chunks:
            return self._empty_batch()
        if any(not chunk.text.strip() for chunk in chunks):
            raise EmbeddingError("Cannot generate an embedding for an empty chunk")

        total_started = perf_counter()
        encoder, model_load_seconds = self._get_encoder()
        generation_started = perf_counter()
        embedded: list[EmbeddedChunk] = []
        batch_count = (len(chunks) + self.batch_size - 1) // self.batch_size
        for batch_number, chunk_batch in enumerate(_batches(chunks, self.batch_size), start=1):
            logger.info(
                "stage=embedding_generation status=started batch=%s/%s chunks=%s",
                batch_number,
                batch_count,
                len(chunk_batch),
            )
            try:
                encoded = encoder.encode(
                    [chunk.text for chunk in chunk_batch],
                    batch_size=self.batch_size,
                    input_type="passage",
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise EmbeddingError("Dense embedding generation failed") from exc
            vectors = _as_vectors(encoded)
            if len(vectors) != len(chunk_batch):
                raise EmbeddingError(
                    f"Encoder returned {len(vectors)} vectors for {len(chunk_batch)} chunks"
                )
            for chunk, vector in zip(chunk_batch, vectors, strict=True):
                normalized_vector = _normalize_vector(vector, self.expected_dimensions)
                source = {
                    "filename": chunk.source_filename,
                    "source_text": chunk.source_text,
                    "source_start": chunk.source_start,
                    "source_end": chunk.source_end,
                    "page_language": chunk.page_language,
                    "extraction_method": chunk.extraction_method,
                    "metadata": chunk.source_metadata,
                }
                embedded.append(
                    EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        page_number=chunk.page_number,
                        source=source,
                        text=chunk.text,
                        model_name=self.model_name,
                        dimensions=self.expected_dimensions,
                        vector=normalized_vector,
                    )
                )
            logger.info(
                "stage=embedding_generation status=completed batch=%s/%s chunks=%s",
                batch_number,
                batch_count,
                len(chunk_batch),
            )
        finished = perf_counter()
        generation_seconds = finished - generation_started
        total_seconds = finished - total_started
        bytes_per_vector = self.expected_dimensions * 4
        return EmbeddingBatch(
            model_name=self.model_name,
            dimensions=self.expected_dimensions,
            bytes_per_vector=bytes_per_vector,
            estimated_storage_bytes=bytes_per_vector * len(embedded),
            embeddings=embedded,
            performance=EmbeddingPerformance(
                chunk_count=len(embedded),
                model_load_seconds=model_load_seconds,
                generation_seconds=generation_seconds,
                total_seconds=total_seconds,
                chunks_per_second=(
                    len(embedded) / generation_seconds if generation_seconds > 0 else 0.0
                ),
            ),
        )

    def _empty_batch(self) -> EmbeddingBatch:
        return EmbeddingBatch(
            model_name=self.model_name,
            dimensions=self.expected_dimensions,
            bytes_per_vector=self.expected_dimensions * 4,
            estimated_storage_bytes=0,
            embeddings=[],
            performance=EmbeddingPerformance(
                chunk_count=0,
                model_load_seconds=0.0,
                generation_seconds=0.0,
                total_seconds=0.0,
                chunks_per_second=0.0,
            ),
        )


def generate_embedding_id(chunk_id: str, model_name: str = EMBEDDING_MODEL) -> str:
    """Create a stable model-specific ID for a dense-vector record."""
    return hashlib.sha256(f"{model_name}:{chunk_id}".encode("utf-8")).hexdigest()
