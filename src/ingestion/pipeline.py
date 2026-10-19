"""Ingestion orchestration and structured JSON persistence."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from src.ingestion.models import IngestedDocument
from src.ingestion.pdf_loader import load_pdf


def save_ingested_document(
    document: IngestedDocument,
    output_dir: str | Path = Path("data/processed"),
) -> Path:
    """Persist an ingestion result as UTF-8 JSON using an atomic file replacement."""
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{document.document_id}.json"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{document.document_id}.", suffix=".tmp", dir=destination_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(document.model_dump_json(indent=2))
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def ingest_pdf(
    file_path: str | Path,
    output_dir: str | Path = Path("data/processed"),
    *,
    uploaded_at: datetime | None = None,
    normalize_teh_marbuta: bool = False,
) -> IngestedDocument:
    """Load a PDF and persist its structured extraction result as UTF-8 JSON."""
    document = load_pdf(
        file_path,
        uploaded_at=uploaded_at,
        normalize_teh_marbuta=normalize_teh_marbuta,
    )
    save_ingested_document(document, output_dir)
    return document