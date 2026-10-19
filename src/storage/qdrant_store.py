"""Qdrant-backed vector storage and retrieval for Arabic chunks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.chunking import TextChunk
from src.core.exceptions import VectorStoreConnectionError
from src.core.memory import measure_memory
from src.core.settings import get_settings
from src.embeddings import EmbeddingBatch, EmbeddingPerformance, EmbeddingService

logger = logging.getLogger(__name__)


def _point_id(chunk_id: str) -> int:
    """Map a stable chunk hash to a valid Qdrant point identifier."""
    return int(chunk_id, 16) % ((2**63) - 1)


class RetrievalHit(BaseModel):
    """A single vector-search hit with source provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    score: float
    page_number: int = Field(ge=1)
    source_filename: str
    source_text: str
    source_metadata: dict[str, Any]
    metadata: dict[str, Any]


class IndexingSummary(BaseModel):
    """Aggregate embedding statistics for chunks persisted incrementally."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    dimensions: int = Field(ge=1)
    performance: EmbeddingPerformance


class QdrantVectorStore:
    """Persist and query dense embeddings in a Qdrant collection."""

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        collection_name: str = "arabic_documents",
        vector_size: int = 1024,
        distance: str = "cosine",
        timeout: int | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.timeout = timeout if timeout is not None else settings.qdrant_timeout
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=self.timeout,
        )
        self._ensure_collection()

    def _distance(self) -> qmodels.Distance:
        if self.distance.lower() == "cosine":
            return qmodels.Distance.COSINE
        if self.distance.lower() == "dot":
            return qmodels.Distance.DOT
        if self.distance.lower() == "euclid":
            return qmodels.Distance.EUCLID
        raise ValueError(f"Unsupported Qdrant distance: {self.distance}")

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection_name)
            return
        except Exception:
            pass

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=self._distance(),
                ),
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive layer for local service setup
            raise VectorStoreConnectionError(
                f"Unable to initialize Qdrant collection '{self.collection_name}'"
            ) from exc

    def index_chunks(
        self,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if not chunks:
            return
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("vectors length must match chunks length")

        with measure_memory("qdrant_indexing", points=len(chunks)):
            points = []
            for index, chunk in enumerate(chunks):
                vector = list(vectors[index]) if vectors is not None else [0.0] * self.vector_size
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "source_text": chunk.source_text,
                    "source_filename": chunk.source_filename,
                    "page_number": chunk.page_number,
                    "page_language": chunk.page_language,
                    "extraction_method": chunk.extraction_method,
                    "chunk_index": chunk.chunk_index,
                    "strategy": chunk.strategy,
                    "metadata": chunk.source_metadata,
                    "source_metadata": chunk.source_metadata,
                }
                points.append(
                    qmodels.PointStruct(
                        id=_point_id(chunk.chunk_id),
                        vector=vector,
                        payload=payload,
                    )
                )
            if points:
                logger.info(
                    "stage=qdrant_indexing status=started collection=%s points=%s",
                    self.collection_name,
                    len(points),
                )
                self.client.upsert(collection_name=self.collection_name, points=points)
                logger.info(
                    "stage=qdrant_indexing status=completed collection=%s points=%s",
                    self.collection_name,
                    len(points),
                )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievalHit]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        logger.info(
            "stage=retrieval step=qdrant_search status=started collection=%s limit=%s",
            self.collection_name,
            limit,
        )
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        logger.info(
            "stage=retrieval step=qdrant_search status=completed collection=%s hits=%s",
            self.collection_name,
            len(hits),
        )

        results: list[RetrievalHit] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                RetrievalHit(
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    document_id=str(payload.get("document_id", "")),
                    text=str(payload.get("text", "")),
                    score=float(hit.score),
                    page_number=int(payload.get("page_number", 1)),
                    source_filename=str(payload.get("source_filename", "unknown.pdf")),
                    source_text=str(payload.get("source_text", "")),
                    source_metadata=dict(payload.get("source_metadata", {})),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return results


class RetrievalService:
    """Embed queries and retrieve the most relevant indexed chunks."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index_chunks(self, chunks: Sequence[TextChunk]) -> IndexingSummary:
        """Embed and persist bounded batches so document-sized vectors are never retained."""
        if not chunks:
            return IndexingSummary(
                model_name=self.embedding_service.model_name,
                dimensions=self.embedding_service.expected_dimensions,
                performance=EmbeddingPerformance(
                    chunk_count=0,
                    model_load_seconds=0.0,
                    generation_seconds=0.0,
                    total_seconds=0.0,
                    chunks_per_second=0.0,
                ),
            )

        total_started = perf_counter()
        chunks_per_batch = self.embedding_service.batch_size
        count = 0
        model_load_seconds = 0.0
        generation_seconds = 0.0
        for start in range(0, len(chunks), chunks_per_batch):
            chunk_batch = chunks[start : start + chunks_per_batch]
            with measure_memory(
                "index_batch",
                batch_number=start // chunks_per_batch + 1,
                chunks=len(chunk_batch),
            ):
                batch: EmbeddingBatch = self.embedding_service.embed_chunks(chunk_batch)
                model_load_seconds += batch.performance.model_load_seconds
                generation_seconds += batch.performance.generation_seconds
                self.vector_store.index_chunks(
                    chunk_batch,
                    vectors=[item.vector for item in batch.embeddings],
                )
                count += len(batch.embeddings)
                del batch

        total_seconds = perf_counter() - total_started
        return IndexingSummary(
            model_name=self.embedding_service.model_name,
            dimensions=self.embedding_service.expected_dimensions,
            performance=EmbeddingPerformance(
                chunk_count=count,
                model_load_seconds=model_load_seconds,
                generation_seconds=generation_seconds,
                total_seconds=total_seconds,
                chunks_per_second=count / generation_seconds if generation_seconds > 0 else 0.0,
            ),
        )

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        logger.info("stage=retrieval status=started top_k=%s", top_k)
        query_vector = self.embedding_service.embed_texts([query])[0]
        hits = self.vector_store.search(query_vector, limit=top_k)
        logger.info("stage=retrieval status=completed hits=%s", len(hits))
        return hits
