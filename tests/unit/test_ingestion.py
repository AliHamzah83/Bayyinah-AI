"""Behavior tests for Arabic PDF ingestion and structured output."""

import json

import pymupdf
import pytest

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


def test_load_pdf_rejects_unsupported_file_type(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(PDFIngestionError, match="expected a .pdf"):
        load_pdf(text_file)