"""PDF validation, text extraction, and source metadata collection."""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from src.core.settings import get_settings
from src.ingestion.arabic_text import detect_language, normalize_arabic_text
from src.ingestion.models import DocumentMetadata, IngestedDocument, PageContent

_OCR_DPI = 200
_HASH_CHUNK_SIZE = 1024 * 1024


class PDFIngestionError(ValueError):
    """Raised when a PDF cannot be safely read or contains no usable text."""


def load_pdf(
    file_path: str | Path,
    *,
    uploaded_at: datetime | None = None,
    normalize_teh_marbuta: bool = False,
) -> IngestedDocument:
    """Extract native text, OCR image-only pages, and collect source metadata."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise PDFIngestionError(f"PDF file does not exist or is not a regular file: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFIngestionError("Unsupported file type; expected a .pdf file")

    settings = get_settings()
    file_size = path.stat().st_size
    if file_size > settings.max_pdf_size_bytes:
        maximum_mb = settings.max_pdf_size_bytes / (1024 * 1024)
        raise PDFIngestionError(f"PDF exceeds the maximum file size of {maximum_mb:.0f} MB")

    try:
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted:
                raise PDFIngestionError("Encrypted PDFs are not supported")
            if pdf.page_count < 1:
                raise PDFIngestionError("PDF contains no pages")
            if pdf.page_count > settings.max_pdf_pages:
                raise PDFIngestionError(
                    f"PDF exceeds the maximum page count of {settings.max_pdf_pages}"
                )

            pages = []
            for page_number, page in enumerate(pdf, start=1):
                raw_text = page.get_text("text", sort=True)
                extraction_method = "native"
                if not raw_text.strip():
                    if page.get_images(full=True):
                        try:
                            pixmap = page.get_pixmap(dpi=_OCR_DPI, alpha=False)
                            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                            raw_text = pytesseract.image_to_string(image, lang="ara+eng")
                            extraction_method = "ocr"
                            del image, pixmap
                        except pytesseract.TesseractNotFoundError as exc:
                            raise PDFIngestionError(
                                "Scanned PDF pages require Tesseract OCR with Arabic language data"
                            ) from exc
                        except pytesseract.TesseractError as exc:
                            raise PDFIngestionError(
                                "OCR failed; verify Tesseract and its ara/eng language data"
                            ) from exc
                    else:
                        extraction_method = "none"
                normalized_text = normalize_arabic_text(
                    raw_text,
                    normalize_teh_marbuta=normalize_teh_marbuta,
                )
                pages.append(
                    PageContent(
                        page_number=page_number,
                        raw_text=raw_text,
                        normalized_text=normalized_text,
                        language=detect_language(raw_text),
                        extraction_method=extraction_method,
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

    extracted_at = datetime.now(UTC)
    if uploaded_at is None:
        upload_timestamp = extracted_at
    elif uploaded_at.tzinfo is None:
        upload_timestamp = uploaded_at.replace(tzinfo=UTC)
    else:
        upload_timestamp = uploaded_at.astimezone(UTC)

    language_counts: dict[str, int] = {}
    for page in pages:
        language_counts[page.language] = language_counts.get(page.language, 0) + 1

    pages_with_text = sum(page.has_text for page in pages)
    extraction_status = (
        "no_text"
        if pages_with_text == 0
        else "complete" if pages_with_text == len(pages) else "partial"
    )
    file_digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while content := source_file.read(_HASH_CHUNK_SIZE):
            file_digest.update(content)
    file_hash = file_digest.hexdigest()

    return IngestedDocument(
        document_id=file_hash,
        metadata=DocumentMetadata(
            source_filename=path.name,
            file_size_bytes=file_size,
            sha256=file_hash,
            page_count=len(pages),
            uploaded_at=upload_timestamp,
            extracted_at=extracted_at,
            pdf_metadata=pdf_metadata,
            language_counts=language_counts,
            extraction_status=extraction_status,
        ),
        pages=pages,
    )
