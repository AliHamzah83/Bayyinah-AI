"""Behavior tests for Arabic PDF ingestion and structured output."""

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace

import pymupdf
import pytest
from PIL import Image

from src.ingestion.arabic_text import detect_language, normalize_arabic_text
from src.ingestion.pdf_loader import PDFIngestionError, load_pdf
from src.ingestion.pipeline import ingest_pdf


def create_pdf(path, page_texts):
    document = pymupdf.open()
    for page_text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), page_text, fontname="helv")
    document.save(path)
    document.close()


def test_arabic_normalization_preserves_mixed_text():
    source = "إِنْجِلِيزِي English ـ و آية"

    assert normalize_arabic_text(source) == "انجليزي English و اية"
    assert detect_language(source) == "mixed"
    assert detect_language("مستند عربي") == "arabic"
    assert detect_language("English document") == "english"
    assert detect_language("123 !!!") == "unknown"


def test_arabic_normalization_handles_variants_and_unwanted_characters():
    source = "آ أ إ ٱ ى ی ک ـ ة\x00  ب،"

    assert normalize_arabic_text(source) == "ا ا ا ا ي ي ك ة ب،"
    assert normalize_arabic_text(source, normalize_teh_marbuta=True) == "ا ا ا ا ي ي ك ه ب،"


def test_arabic_normalization_collapses_whitespace():
    assert normalize_arabic_text("ا  ب\nج\tد") == "ا ب ج د"


def test_load_pdf_extracts_arabic_and_metadata(tmp_path):
    pdf_path = tmp_path / "arabic-source.pdf"
    create_pdf(pdf_path, ["Arabic English 123"])

    document = load_pdf(pdf_path)

    assert document.metadata.source_filename == pdf_path.name
    assert document.metadata.page_count == 1
    assert document.metadata.extraction_status == "complete"
    assert len(document.metadata.sha256) == 64
    assert document.pages[0].raw_text.strip() == "Arabic English 123"
    assert document.pages[0].language == "english"
    assert document.pages[0].extraction_method == "native"
    assert document.metadata.uploaded_at.utcoffset() == timedelta(0)


def test_load_pdf_preserves_supplied_upload_timestamp(tmp_path):
    pdf_path = tmp_path / "timestamped.pdf"
    create_pdf(pdf_path, ["Timestamped document"])
    uploaded_at = datetime(2025, 2, 3, 12, 30, tzinfo=UTC)

    document = load_pdf(pdf_path, uploaded_at=uploaded_at)

    assert document.metadata.uploaded_at == uploaded_at


def test_load_pdf_uses_ocr_for_image_only_pages(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    image = Image.new("RGB", (32, 32), color="white")
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_image(page.rect, stream=image_bytes.getvalue())
    pdf.save(pdf_path)
    pdf.close()

    def fake_ocr(*_args, **_kwargs):
        return "مستند عربي"

    monkeypatch.setattr("src.ingestion.pdf_loader.pytesseract.image_to_string", fake_ocr)
    document = load_pdf(pdf_path)

    assert document.pages[0].raw_text == "مستند عربي"
    assert document.pages[0].normalized_text == "مستند عربي"
    assert document.pages[0].extraction_method == "ocr"


def test_ingest_pdf_writes_utf8_json(tmp_path):
    pdf_path = tmp_path / "mixed-source.pdf"
    output_dir = tmp_path / "processed"
    create_pdf(pdf_path, ["Arabic English mixed document"])

    document = ingest_pdf(pdf_path, output_dir)
    output_path = output_dir / f"{document.document_id}.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["document_id"] == document.document_id
    assert payload["pages"][0]["raw_text"].strip() == "Arabic English mixed document"


def test_load_pdf_marks_pages_without_text_as_partial(tmp_path):
    pdf_path = tmp_path / "partially-readable.pdf"
    create_pdf(pdf_path, ["Readable text", ""])

    document = load_pdf(pdf_path)

    assert document.metadata.extraction_status == "partial"
    assert [page.has_text for page in document.pages] == [True, False]
    assert [page.extraction_method for page in document.pages] == ["native", "none"]


def test_load_pdf_rejects_unsupported_file_type(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(PDFIngestionError, match="expected a .pdf"):
        load_pdf(text_file)


def test_load_pdf_rejects_file_over_configured_size_limit(tmp_path, monkeypatch):
    pdf_path = tmp_path / "too-large.pdf"
    create_pdf(pdf_path, ["Text"])
    monkeypatch.setattr(
        "src.ingestion.pdf_loader.get_settings",
        lambda: SimpleNamespace(max_pdf_size_bytes=1, max_pdf_pages=500),
    )

    with pytest.raises(PDFIngestionError, match="maximum file size"):
        load_pdf(pdf_path)


def test_load_pdf_rejects_page_count_over_configured_limit(tmp_path, monkeypatch):
    pdf_path = tmp_path / "too-many-pages.pdf"
    create_pdf(pdf_path, ["First page", "Second page"])
    monkeypatch.setattr(
        "src.ingestion.pdf_loader.get_settings",
        lambda: SimpleNamespace(max_pdf_size_bytes=1024 * 1024, max_pdf_pages=1),
    )

    with pytest.raises(PDFIngestionError, match="maximum page count"):
        load_pdf(pdf_path)