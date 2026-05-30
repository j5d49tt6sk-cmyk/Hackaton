from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExtractedSection:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_number: int | None = None
    sheet_name: str | None = None
    heading: str | None = None


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    document_id: int
    document_text_id: int | None
    chunk_index: int
    expert: str | None = None
    topic: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    document_id: int | None
    content: str
    source: str | None
    file_name: str | None
    expert: str | None
    topic: str | None
    chunk_index: int | None
    metadata: dict[str, Any]
    similarity: float
    page_number: int | None = None
    sheet_name: str | None = None
    heading: str | None = None


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    sources: list[str]
    confidence: str
    decision_trail: str | None = None
