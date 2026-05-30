from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from company_brain.chunking import split_text
from company_brain.loaders import extract_sections
from company_brain.metadata import infer_expert, infer_topic
from company_brain.models import RetrievedChunk


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{3,}")


class LocalKnowledgeStore:
    def __init__(self, root: Path = Path("local_knowledge")) -> None:
        self._root = root
        self._chunks_path = root / "chunks.jsonl"
        self._uploads_dir = root / "uploads"
        self._root.mkdir(exist_ok=True)
        self._uploads_dir.mkdir(exist_ok=True)

    def ingest_file(
        self,
        source_path: Path,
        expert: str | None = None,
        topic: str | None = None,
        chunk_size: int = 1200,
        overlap: int = 180,
    ) -> int:
        stored_path = self._uploads_dir / source_path.name
        stored_path.write_bytes(source_path.read_bytes())

        inferred_expert = infer_expert(source_path, expert)
        inferred_topic = infer_topic(source_path, topic)
        next_id = self._next_chunk_id()
        rows: list[dict[str, Any]] = []

        for section_index, section in enumerate(extract_sections(source_path)):
            heading_prefix = f"{section.heading}\n\n" if section.heading else ""
            for content in split_text(
                f"{heading_prefix}{section.content}",
                chunk_size=chunk_size,
                overlap=overlap,
            ):
                rows.append(
                    {
                        "id": next_id,
                        "document_id": None,
                        "content": content,
                        "source": str(stored_path),
                        "file_name": source_path.name,
                        "expert": inferred_expert,
                        "topic": inferred_topic,
                        "chunk_index": len(rows),
                        "metadata": {
                            **section.metadata,
                            "section_index": section_index,
                            "local_backend": True,
                        },
                        "similarity": 0.0,
                        "page_number": section.page_number,
                        "sheet_name": section.sheet_name,
                        "heading": section.heading,
                    }
                )
                next_id += 1

        if rows:
            with self._chunks_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        return len(rows)

    def retrieve(
        self,
        question: str,
        expert: str | None = None,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        query_tokens = _tokens(question)
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._iter_rows():
            if expert and row.get("expert") != expert:
                continue
            score = _score(query_tokens, _tokens(row.get("content", "")))
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _to_retrieved_chunk({**row, "similarity": min(score, 1.0)})
            for score, row in scored[:top_k]
        ]

    def _iter_rows(self) -> list[dict[str, Any]]:
        if not self._chunks_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self._chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _next_chunk_id(self) -> int:
        ids = [int(row.get("id", 0)) for row in self._iter_rows()]
        return max(ids, default=0) + 1


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def _score(query_tokens: set[str], content_tokens: set[str]) -> float:
    overlap = query_tokens & content_tokens
    if not overlap:
        return 0.0
    return len(overlap) / math.sqrt(len(query_tokens) * max(len(content_tokens), 1))


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    allowed = RetrievedChunk.__dataclass_fields__.keys()
    values = {key: value for key, value in row.items() if key in allowed}
    return RetrievedChunk(**values)
