from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    source: str
    file_name: str
    expert: str | None
    topic: str | None
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    content: str
    source: str | None
    file_name: str | None
    expert: str | None
    topic: str | None
    chunk_index: int | None
    metadata: dict[str, Any]
    similarity: float


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    sources: list[str]
    confidence: str
    decision_trail: str | None = None

