"""Structured records emitted by document ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DocumentLanguage = Literal["arabic", "english", "mixed", "unknown"]
ExtractionStatus = Literal["complete", "partial", "no_text"]
ExtractionMethod = Literal["native", "ocr", "none"]


class PageContent(BaseModel):
    """Extracted page text in source and normalized forms."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    raw_text: str
    normalized_text: str
    language: DocumentLanguage
    extraction_method: ExtractionMethod
    character_count: int = Field(ge=0)
    has_text: bool


class DocumentMetadata(BaseModel):
    """Stable source and extraction metadata for a PDF."""

    model_config = ConfigDict(extra="forbid")

    source_filename: str
    file_size_bytes: int = Field(ge=0)
    mime_type: Literal["application/pdf"] = "application/pdf"
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_count: int = Field(ge=1)
    extracted_at: datetime
    pdf_metadata: dict[str, str] = Field(default_factory=dict)
    language_counts: dict[str, int] = Field(default_factory=dict)
    extraction_status: ExtractionStatus


class IngestedDocument(BaseModel):
    """Versioned, JSON-serializable result of processing one PDF."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    metadata: DocumentMetadata
    pages: list[PageContent]