"""Tests for Arabic-aware document chunking and quality statistics."""

from datetime import UTC, datetime
import hashlib

import pytest

from src.chunking import chunk_document, compare_chunking_strategies
from src.ingestion.arabic_text import normalize_arabic_text
from src.ingestion.models import DocumentMetadata, IngestedDocument, PageContent


def make_document(*pages: str) -> IngestedDocument:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    content = "\n".join(pages).encode("utf-8")
    document_id = hashlib.sha256(content).hexdigest()
    return IngestedDocument(
        document_id=document_id,
        metadata=DocumentMetadata(
            source_filename="arabic-source.pdf",
            file_size_bytes=len(content),
            sha256=document_id,
            page_count=len(pages),
            uploaded_at=timestamp,
            extracted_at=timestamp,
            extraction_status="complete",
        ),
        pages=[
            PageContent(
                page_number=page_number,
                raw_text=page_text,
                normalized_text=normalize_arabic_text(page_text),
                language="arabic",
                extraction_method="native",
                character_count=len(page_text),
                has_text=bool(page_text.strip()),
            )
            for page_number, page_text in enumerate(pages, start=1)
        ],
    )


def test_compare_strategies_preserves_provenance_and_limits():
    page_one = " ".join(f"معلومة{index}" for index in range(25))
    page_two = " ".join(f"تفصيل{index}" for index in range(25))
    document = make_document(page_one, page_two)

    results = compare_chunking_strategies(document, max_tokens=8, overlap_tokens=2)

    assert set(results) == {"fixed_size", "recursive", "semantic"}
    for result in results.values():
        assert result.document_id == document.document_id
        assert result.statistics.chunk_count == len(result.chunks)
        assert result.chunks
        assert len({chunk.chunk_id for chunk in result.chunks}) == len(result.chunks)
        assert all(chunk.document_id == document.document_id for chunk in result.chunks)
        assert all(chunk.source_filename == "arabic-source.pdf" for chunk in result.chunks)
        assert all(chunk.estimated_token_count <= 8 for chunk in result.chunks)
        assert {chunk.page_number for chunk in result.chunks} == {1, 2}
        assert all(chunk.source_metadata["page_count"] == 2 for chunk in result.chunks)


def test_recursive_strategy_prefers_sentence_boundaries():
    text = (
        "اللغة العربية جميلة ومفيدة وتساعد القارئ على فهم المعلومات بوضوح. "
        "الاقتصاد العالمي يتغير بسرعة كبيرة بسبب التجارة والاستثمار والتقنية الحديثة."
    )
    document = make_document(text)

    fixed = chunk_document(document, strategy="fixed_size", max_tokens=12, overlap_tokens=0)
    recursive = chunk_document(document, strategy="recursive", max_tokens=12, overlap_tokens=0)

    assert fixed.chunks[0].text != recursive.chunks[0].text
    assert recursive.chunks[0].text.endswith("بوضوح.")


def test_semantic_strategy_splits_at_low_cohesion_boundary():
    first_topic = "الطب الصحة علاج دواء مستشفى طبيب مريض صحة علاج دواء."
    second_topic = (
        "الاقتصاد سوق تداول استثمار شركة مال أسهم تجارة بنك أرباح "
        "تمويل سوق تداول استثمار شركة مال أسهم تجارة بنك أرباح."
    )
    document = make_document(f"{first_topic} {second_topic}")

    recursive = chunk_document(
        document,
        strategy="recursive",
        max_tokens=24,
        overlap_tokens=0,
    )
    semantic = chunk_document(
        document,
        strategy="semantic",
        max_tokens=24,
        overlap_tokens=0,
        minimum_semantic_tokens=6,
    )

    assert len(semantic.chunks) == 2
    assert semantic.chunks[0].text.endswith("دواء.")
    assert not recursive.chunks[0].text.endswith("دواء.")


def test_overlap_statistics_report_actual_source_spans():
    document = make_document(" ".join(f"كلمة{index}" for index in range(10)))

    result = chunk_document(
        document,
        strategy="fixed_size",
        max_tokens=4,
        overlap_tokens=1,
    )

    assert result.statistics.chunk_count == 3
    assert result.statistics.overlapping_pairs == 2
    assert result.statistics.total_overlap_characters == 10
    assert result.statistics.average_overlap_characters == 5
    assert 0 < result.statistics.overlap_ratio < 1


def test_empty_pages_return_zero_statistics():
    document = make_document("", "   ")

    result = chunk_document(document)

    assert result.chunks == []
    assert result.statistics.chunk_count == 0
    assert result.statistics.average_chunk_length_chars == 0
    assert result.statistics.overlap_ratio == 0


def test_rejects_invalid_chunk_parameters():
    document = make_document("نص عربي")

    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_document(document, max_tokens=8, overlap_tokens=8)