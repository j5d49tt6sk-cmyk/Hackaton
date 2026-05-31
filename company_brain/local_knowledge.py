from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from company_brain.access_control import data_case_access_level, infer_document_access
from company_brain.chunking import split_text
from company_brain.loaders import extract_sections
from company_brain.metadata import infer_expert, infer_topic
from company_brain.models import RetrievedChunk


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{3,}")

QUERY_EXPANSIONS = {
    "micar": {
        "markets",
        "crypto",
        "assets",
        "regulation",
        "digital",
        "cryptoassets",
    },
    "fatca": {"tax", "reporting", "withholding", "compliance"},
    "mifid": {"product", "governance", "financial", "instruments"},
    "sfdr": {"sustainability", "disclosure", "esg"},
}


class LocalKnowledgeStore:
    def __init__(self, root: Path = Path("local_knowledge")) -> None:
        self._root = root
        self._chunks_path = root / "chunks.jsonl"
        self._uploads_dir = root / "uploads"
        self._embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self._embedding_client = OllamaEmbeddingClient(
            model=self._embedding_model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        self._root.mkdir(exist_ok=True)
        self._uploads_dir.mkdir(exist_ok=True)

    def ingest_file(
        self,
        source_path: Path,
        expert: str | None = None,
        topic: str | None = None,
        chunk_size: int = 1200,
        overlap: int = 180,
        access_level: int | None = None,
        access_tag: str | None = None,
        collaborators: list[dict[str, object]] | None = None,
    ) -> int:
        stored_path = self._uploads_dir / source_path.name
        stored_path.write_bytes(source_path.read_bytes())
        self._remove_existing_file(source_path.name)

        inferred_expert = infer_expert(source_path, expert)
        inferred_topic = infer_topic(source_path, topic)
        inferred_access_level, inferred_access_tag = infer_document_access(source_path)
        document_access_level = access_level or inferred_access_level
        document_access_tag = access_tag or inferred_access_tag
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
                            "access_level": document_access_level,
                            "access_tag": document_access_tag,
                            "collaborators": collaborators or [],
                            "embedding_model": self._embedding_model,
                        },
                        "similarity": 0.0,
                        "page_number": section.page_number,
                        "sheet_name": section.sheet_name,
                        "heading": section.heading,
                    }
                )
                next_id += 1

        self._add_embeddings(rows)
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
        requester_access_level: int = 1,
        include_inaccessible: bool = False,
    ) -> list[RetrievedChunk]:
        rows = self._iter_rows()
        try:
            self._ensure_embeddings(rows)
            query_embedding = self._embedding_client.embed_text(question)
            return self._retrieve_by_embeddings(
                rows,
                query_embedding,
                expert=expert,
                top_k=top_k,
                requester_access_level=requester_access_level,
                include_inaccessible=include_inaccessible,
            )
        except LocalEmbeddingError:
            return self._retrieve_by_keywords(
                question,
                rows,
                expert=expert,
                top_k=top_k,
                requester_access_level=requester_access_level,
                include_inaccessible=include_inaccessible,
            )

    def _retrieve_by_embeddings(
        self,
        rows: list[dict[str, Any]],
        query_embedding: list[float],
        expert: str | None,
        top_k: int,
        requester_access_level: int,
        include_inaccessible: bool,
    ) -> list[RetrievedChunk]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            if expert and row.get("expert") != expert:
                continue
            if not include_inaccessible and not _is_accessible(row, requester_access_level):
                continue
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                continue
            score = _cosine_similarity(query_embedding, embedding)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _to_retrieved_chunk({**row, "similarity": min(score, 1.0)})
            for score, row in scored[:top_k]
        ]

    def _retrieve_by_keywords(
        self,
        question: str,
        rows: list[dict[str, Any]],
        expert: str | None,
        top_k: int,
        requester_access_level: int,
        include_inaccessible: bool,
    ) -> list[RetrievedChunk]:
        query_tokens = _expand_query_tokens(_tokens(question))
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            if expert and row.get("expert") != expert:
                continue
            if not include_inaccessible and not _is_accessible(row, requester_access_level):
                continue
            score = _score(query_tokens, _tokens(row.get("content", "")))
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _to_retrieved_chunk({**row, "similarity": min(score, 1.0)})
            for score, row in scored[:top_k]
        ]

    def _add_embeddings(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        try:
            embeddings = self._embedding_client.embed_texts(
                [_embedding_text(row) for row in rows]
            )
        except LocalEmbeddingError:
            return
        for row, embedding in zip(rows, embeddings):
            row["embedding"] = embedding
            row.setdefault("metadata", {})["embedding_model"] = self._embedding_model

    def _ensure_embeddings(self, rows: list[dict[str, Any]]) -> None:
        missing_rows = [
            row
            for row in rows
            if not isinstance(row.get("embedding"), list)
            or row.get("metadata", {}).get("embedding_model") != self._embedding_model
        ]
        if not missing_rows:
            return

        self._add_embeddings(missing_rows)
        if any("embedding" in row for row in missing_rows):
            self._write_rows(rows)

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

    def _remove_existing_file(self, file_name: str) -> None:
        if not self._chunks_path.exists():
            return
        rows = [
            row
            for row in self._iter_rows()
            if row.get("file_name") != file_name
        ]
        self._write_rows(rows)

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        with self._chunks_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")


class LocalEmbeddingError(RuntimeError):
    pass


class OllamaEmbeddingClient:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
        batch_size: int = 16,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._batch_size = batch_size

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            embeddings.extend(self._embed_batch(texts[start : start + self._batch_size]))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps(
            {
                "model": self._model,
                "input": texts,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise LocalEmbeddingError("Ollama embeddings are not reachable.") from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise LocalEmbeddingError("Ollama returned no embeddings.")
        return embeddings


def _tokens(text: str) -> set[str]:
    tokens = {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}
    return tokens | {_compact(token) for token in tokens if _compact(token)}


def _expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(QUERY_EXPANSIONS.get(token, set()))
    return expanded


def _score(query_tokens: set[str], content_tokens: set[str]) -> float:
    overlap = query_tokens & content_tokens
    if not overlap:
        return 0.0
    return len(overlap) / math.sqrt(len(query_tokens) * max(len(content_tokens), 1))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dimensions = min(len(left), len(right))
    if dimensions == 0:
        return 0.0
    dot_product = sum(left[index] * right[index] for index in range(dimensions))
    left_norm = math.sqrt(sum(value * value for value in left[:dimensions]))
    right_norm = math.sqrt(sum(value * value for value in right[:dimensions]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _embedding_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("topic") or ""),
        str(row.get("file_name") or ""),
        str(row.get("heading") or ""),
        str(row.get("content") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


def _is_accessible(row: dict[str, Any], requester_access_level: int) -> bool:
    metadata = row.get("metadata") or {}
    document_access_level = (
        data_case_access_level(row.get("file_name") or row.get("source") or "")
        or int(metadata.get("access_level") or 1)
    )
    return document_access_level < 99 and document_access_level <= requester_access_level


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    allowed = RetrievedChunk.__dataclass_fields__.keys()
    values = {key: value for key, value in row.items() if key in allowed}
    return RetrievedChunk(**values)
