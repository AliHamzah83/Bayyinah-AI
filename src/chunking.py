"""Arabic-aware chunk generation for ingested page-level documents."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.arabic_text import normalize_arabic_text
from src.ingestion.models import IngestedDocument


ChunkStrategy = Literal["fixed_size", "recursive", "semantic"]
_WORD = re.compile(r"\S+")
_TERM = re.compile(r"[\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
_SENTENCE_END = re.compile(r"[.!?؟؛][\"'»”)]*$")


class TextChunk(BaseModel):
    """A normalized chunk with exact source spans and document provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    document_id: str
    chunk_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    strategy: ChunkStrategy
    text: str
    source_text: str
    estimated_token_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    source_filename: str
    page_language: str
    extraction_method: str
    source_metadata: dict[str, Any]


class ChunkStatistics(BaseModel):
    """Size and overlap diagnostics for one chunking strategy."""

    model_config = ConfigDict(extra="forbid")

    strategy: ChunkStrategy
    chunk_count: int = Field(ge=0)
    average_chunk_length_chars: float = Field(ge=0)
    average_estimated_token_count: float = Field(ge=0)
    overlapping_pairs: int = Field(ge=0)
    total_overlap_characters: int = Field(ge=0)
    average_overlap_characters: float = Field(ge=0)
    overlap_ratio: float = Field(ge=0, le=1)


class ChunkingResult(BaseModel):
    """Chunks and quality statistics for one document and strategy."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    strategy: ChunkStrategy
    chunks: list[TextChunk]
    statistics: ChunkStatistics


def _word_spans(text: str) -> list[re.Match[str]]:
    return list(_WORD.finditer(text))


def _boundary_strength(text: str, words: list[re.Match[str]], boundary: int) -> int:
    gap = text[words[boundary - 1].end() : words[boundary].start()]
    if "\n\n" in gap or "\r\n\r\n" in gap:
        return 3
    if "\n" in gap:
        return 2
    if _SENTENCE_END.search(words[boundary - 1].group()):
        return 1
    return 0


def _recursive_end(
    text: str,
    words: list[re.Match[str]],
    start: int,
    hard_end: int,
) -> int:
    lower_bound = start + max(1, (hard_end - start) // 2)
    boundaries = [
        (strength, boundary)
        for boundary in range(lower_bound, hard_end)
        if (strength := _boundary_strength(text, words, boundary)) > 0
    ]
    return max(boundaries)[1] if boundaries else hard_end


def _fixed_spans(
    words: list[re.Match[str]], max_tokens: int, overlap_tokens: int
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    step = max_tokens - overlap_tokens
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        spans.append((words[start].start(), words[end - 1].end()))
        if end == len(words):
            break
        start += step
    return spans


def _recursive_spans(
    text: str,
    words: list[re.Match[str]],
    max_tokens: int,
    overlap_tokens: int,
    *,
    semantic: bool = False,
    similarity_threshold: float = 0.08,
    minimum_semantic_tokens: int = 32,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(words):
        hard_end = min(start + max_tokens, len(words))
        end = hard_end
        if hard_end < len(words):
            if semantic:
                for boundary in range(
                    start + minimum_semantic_tokens,
                    hard_end,
                ):
                    if _boundary_strength(text, words, boundary) == 0:
                        continue
                    next_sentence_end = boundary + 1
                    while (
                        next_sentence_end < hard_end
                        and _boundary_strength(text, words, next_sentence_end) == 0
                    ):
                        next_sentence_end += 1
                    current_terms = {
                        term.group().casefold()
                        for word in words[start:boundary]
                        for term in _TERM.finditer(word.group())
                    }
                    next_terms = {
                        term.group().casefold()
                        for word in words[boundary:next_sentence_end]
                        for term in _TERM.finditer(word.group())
                    }
                    union = current_terms | next_terms
                    similarity = len(current_terms & next_terms) / len(union) if union else 1.0
                    if similarity < similarity_threshold:
                        end = boundary
                        break
            if end == hard_end:
                end = _recursive_end(text, words, start, hard_end)

        spans.append((words[start].start(), words[end - 1].end()))
        if end == len(words):
            break
        start = max(start + 1, end - overlap_tokens)
    return spans


def _statistics(
    strategy: ChunkStrategy,
    chunks: list[TextChunk],
) -> ChunkStatistics:
    chunk_count = len(chunks)
    source_length = sum(chunk.source_end - chunk.source_start for chunk in chunks)
    overlaps = []
    previous_by_page: dict[int, TextChunk] = {}
    for chunk in chunks:
        previous = previous_by_page.get(chunk.page_number)
        if previous is not None:
            overlap = max(0, previous.source_end - chunk.source_start)
            if overlap:
                overlaps.append(overlap)
        previous_by_page[chunk.page_number] = chunk

    total_overlap = sum(overlaps)
    return ChunkStatistics(
        strategy=strategy,
        chunk_count=chunk_count,
        average_chunk_length_chars=(
            sum(chunk.char_count for chunk in chunks) / chunk_count if chunk_count else 0.0
        ),
        average_estimated_token_count=(
            sum(chunk.estimated_token_count for chunk in chunks) / chunk_count
            if chunk_count
            else 0.0
        ),
        overlapping_pairs=len(overlaps),
        total_overlap_characters=total_overlap,
        average_overlap_characters=total_overlap / len(overlaps) if overlaps else 0.0,
        overlap_ratio=total_overlap / source_length if source_length else 0.0,
    )


def chunk_document(
    document: IngestedDocument,
    *,
    strategy: ChunkStrategy = "recursive",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    semantic_similarity_threshold: float = 0.08,
    minimum_semantic_tokens: int = 32,
) -> ChunkingResult:
    """Chunk a document without crossing page boundaries.

    Token limits use whitespace-delimited units as a conservative, model-independent
    estimate. Supply a lower limit if exact BGE-M3 tokenizer counts are required later.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between 0 and max_tokens - 1")
    if not 0 <= semantic_similarity_threshold <= 1:
        raise ValueError("semantic_similarity_threshold must be between 0 and 1")
    if minimum_semantic_tokens < 1:
        raise ValueError("minimum_semantic_tokens must be at least 1")

    source_metadata = document.metadata.model_dump(mode="json")
    chunks: list[TextChunk] = []
    for page in document.pages:
        source_text = page.raw_text or page.normalized_text
        words = _word_spans(source_text)
        if not words:
            continue
        if strategy == "fixed_size":
            spans = _fixed_spans(words, max_tokens, overlap_tokens)
        else:
            spans = _recursive_spans(
                source_text,
                words,
                max_tokens,
                overlap_tokens,
                semantic=strategy == "semantic",
                similarity_threshold=semantic_similarity_threshold,
                minimum_semantic_tokens=minimum_semantic_tokens,
            )

        for start, end in spans:
            source_chunk = source_text[start:end]
            normalized_chunk = normalize_arabic_text(source_chunk)
            if not normalized_chunk:
                continue
            chunk_index = len(chunks)
            chunk_id = hashlib.sha256(
                f"{document.document_id}:{strategy}:{page.page_number}:{start}:{end}:"
                f"{normalized_chunk}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                    strategy=strategy,
                    text=normalized_chunk,
                    source_text=source_chunk,
                    estimated_token_count=len(_word_spans(normalized_chunk)),
                    char_count=len(normalized_chunk),
                    source_start=start,
                    source_end=end,
                    source_filename=document.metadata.source_filename,
                    page_language=page.language,
                    extraction_method=page.extraction_method,
                    source_metadata=source_metadata,
                )
            )

    return ChunkingResult(
        document_id=document.document_id,
        strategy=strategy,
        chunks=chunks,
        statistics=_statistics(strategy, chunks),
    )


def compare_chunking_strategies(
    document: IngestedDocument,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    semantic_similarity_threshold: float = 0.08,
) -> dict[ChunkStrategy, ChunkingResult]:
    """Run all embedding-free chunking strategies for direct quality comparison."""
    return {
        strategy: chunk_document(
            document,
            strategy=strategy,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            semantic_similarity_threshold=semantic_similarity_threshold,
        )
        for strategy in ("fixed_size", "recursive", "semantic")
    }