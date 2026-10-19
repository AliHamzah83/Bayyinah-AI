"""Document upload and indexing endpoint."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies.indexing import get_indexing_service
from src.core.settings import get_settings
from src.indexing import DocumentIndexingResult, DocumentIndexingService, NoIndexableContentError
from src.ingestion.pdf_loader import PDFIngestionError

UPLOAD_CHUNK_SIZE = 1024 * 1024
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=DocumentIndexingResult, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentIndexingService = Depends(get_indexing_service),
) -> DocumentIndexingResult:
    """Accept a PDF, persist its metadata, and make its chunks searchable."""
    filename = Path((file.filename or "").replace("\\", "/")).name
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A PDF filename is required")
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")

    max_file_size = get_settings().max_pdf_size_bytes
    try:
        with tempfile.TemporaryDirectory(prefix="arabic-rag-upload-") as temporary_dir:
            temporary_path = Path(temporary_dir) / filename
            logger.info("stage=pdf_upload status=started filename=%s", filename)
            uploaded_size = 0
            with temporary_path.open("wb") as destination:
                while content := await file.read(UPLOAD_CHUNK_SIZE):
                    uploaded_size += len(content)
                    if uploaded_size > max_file_size:
                        maximum_mb = max_file_size / (1024 * 1024)
                        raise HTTPException(
                            status_code=413,
                            detail=f"PDF exceeds the maximum file size of {maximum_mb:.0f} MB",
                        )
                    destination.write(content)
            logger.info("stage=pdf_upload status=completed filename=%s", filename)
            return await run_in_threadpool(service.index_pdf, temporary_path)
    except HTTPException:
        raise
    except PDFIngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NoIndexableContentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - guard for unavailable model/vector services
        raise HTTPException(status_code=503, detail="Document indexing service unavailable") from exc
    finally:
        await file.close()