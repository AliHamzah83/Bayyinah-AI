"""End-to-end PDF indexing orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.chunking import ChunkStatistics, chunk_document
from src.embeddings import EmbeddingPerformance
from src.core.memory import measure_memory
from src.ingestion.models import DocumentMetadata
from src.ingestion.pipeline import ingest_pdf
from src.storage.qdrant_store import RetrievalService

logger = logging.getLogger(__name__)


class NoIndexableContentError(ValueError):
    """Raised when PDF extraction produces no chunks to search."""


class IndexingStatistics(BaseModel):
    """Counts and model details for one completed indexing operation."""

    model_config = ConfigDict(extra="forbid")

    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    indexed_vectors: int = Field(ge=0)
    embedding_model: str
    embedding_dimensions: int = Field(ge=1)
    embedding_performance: EmbeddingPerformance
    chunking: ChunkStatistics


class DocumentIndexingResult(BaseModel):
    """Persisted document metadata and indexing statistics returned by the API."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["indexed"] = "indexed"
    document_id: str
    metadata: DocumentMetadata
    statistics: IndexingStatistics


class DocumentIndexingService:
    """Run extraction, chunking, embedding, and vector persistence for a PDF."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        output_dir: str | Path = Path("data/processed"),
    ) -> None:
        self.retrieval_service = retrieval_service
        self.output_dir = Path(output_dir)

    def index_pdf(self, file_path: str | Path) -> DocumentIndexingResult:
        logger.info("stage=ingestion status=started")
        document = ingest_pdf(file_path, self.output_dir)
        logger.info(
            "stage=ingestion status=completed document_id=%s pages=%s",
            document.document_id,
            document.metadata.page_count,
        )

        logger.info("stage=chunking status=started document_id=%s", document.document_id)
        with measure_memory("chunking", pages=document.metadata.page_count):
            chunking_result = chunk_document(document)
        if not chunking_result.chunks:
            raise NoIndexableContentError("PDF contains no extractable text to index")
        logger.info(
            "stage=chunking status=completed document_id=%s chunks=%s",
            document.document_id,
            len(chunking_result.chunks),
        )

        embedding_batch = self.retrieval_service.index_chunks(chunking_result.chunks)
        logger.info(
            "stage=embedding_and_indexing status=completed document_id=%s vectors=%s",
            document.document_id,
            embedding_batch.performance.chunk_count,
        )
        return DocumentIndexingResult(
            document_id=document.document_id,
            metadata=document.metadata,
            statistics=IndexingStatistics(
                page_count=document.metadata.page_count,
                chunk_count=len(chunking_result.chunks),
                indexed_vectors=embedding_batch.performance.chunk_count,
                embedding_model=embedding_batch.model_name,
                embedding_dimensions=embedding_batch.dimensions,
                embedding_performance=embedding_batch.performance,
                chunking=chunking_result.statistics,
            ),
        )