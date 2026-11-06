"""PDF validation, text extraction, and source metadata collection."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from src.ingestion.arabic_text import detect_language, normalize_arabic_text
from src.ingestion.models import DocumentMetadata, IngestedDocument, PageContent


class PDFIngestionError(ValueError):
    """Raised when a PDF cannot be safely read or contains no usable text."""


def load_pdf(file_path: str | Path) -> IngestedDocument:
    """Extract page-level text and metadata from a PDF without altering source text."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise PDFIngestionError(f"PDF file does not exist or is not a regular file: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFIngestionError("Unsupported file type; expected a .pdf file")

    try:
        file_bytes = path.read_bytes()
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise PDFIngestionError("Encrypted PDFs are not supported")
            if pdf.page_count < 1:
                raise PDFIngestionError("PDF contains no pages")

            pages = []
            for page_number, page in enumerate(pdf, start=1):
                raw_text = page.get_text("text", sort=True)
                normalized_text = normalize_arabic_text(raw_text)
                pages.append(
                    PageContent(
                        page_number=page_number,
                        raw_text=raw_text,
                        normalized_text=normalized_text,
                        language=detect_language(raw_text),
                        extraction_method="native" if raw_text.strip() else "none",
                        character_count=len(raw_text),
                        has_text=bool(normalized_text),
                    )
                )
            pdf_metadata = {
                str(key): str(value)
                for key, value in (pdf.metadata or {}).items()
                if value not in (None, "")
            }
    except PDFIngestionError:
        raise
    except (OSError, pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
        raise PDFIngestionError(f"Unable to read PDF '{path.name}': {exc}") from exc

    language_counts: dict[str, int] = {}
    for page in pages:
        language_counts[page.language] = language_counts.get(page.language, 0) + 1

    pages_with_text = sum(page.has_text for page in pages)
    extraction_status = (
        "no_text"
        if pages_with_text == 0
        else "complete"
        if pages_with_text == len(pages)
        else "partial"
    )
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    return IngestedDocument(
        document_id=file_hash,
        metadata=DocumentMetadata(
            source_filename=path.name,
            file_size_bytes=len(file_bytes),
            sha256=file_hash,
            page_count=len(pages),
            extracted_at=datetime.now(UTC),
            pdf_metadata=pdf_metadata,
            language_counts=language_counts,
            extraction_status=extraction_status,
        ),
        pages=pages,
    )