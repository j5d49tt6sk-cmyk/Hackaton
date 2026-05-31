from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

from supabase import Client, create_client

from company_brain.access_control import EmployeeAccount, employee_from_row
from company_brain.models import DocumentChunk, RetrievedChunk


logger = logging.getLogger(__name__)


class SupabaseDocumentStore:
    def __init__(self, url: str, service_role_key: str, bucket: str) -> None:
        self._client: Client = create_client(url, service_role_key)
        self._bucket = bucket

    def upload_file(self, path: Path, document_id: str) -> str:
        storage_path = f"documents/{document_id}/{path.name}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as handle:
            try:
                self._client.storage.from_(self._bucket).upload(
                    storage_path,
                    handle,
                    {"content-type": content_type, "upsert": "true"},
                )
            except Exception as exc:
                if "Bucket not found" not in str(exc):
                    raise
                self._client.storage.create_bucket(
                    self._bucket,
                    options={"public": False},
                )
                handle.seek(0)
                self._client.storage.from_(self._bucket).upload(
                    storage_path,
                    handle,
                    {"content-type": content_type, "upsert": "true"},
                )
        return storage_path

    def find_existing_document_id(self, path: Path) -> str | None:
        result = (
            self._client.table("documents")
            .select("id")
            .eq("file_name", path.name)
            .eq("file_size", path.stat().st_size)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        return str(result.data[0]["id"])

    def delete_document(self, document_id: str) -> None:
        result = (
            self._client.table("documents")
            .select("storage_path")
            .eq("id", document_id)
            .limit(1)
            .execute()
        )
        storage_path = result.data[0].get("storage_path") if result.data else None
        self._client.table("documents").delete().eq("id", document_id).execute()
        if storage_path:
            try:
                self._client.storage.from_(self._bucket).remove([storage_path])
            except Exception:
                logger.warning("Could not remove stale storage object %s", storage_path)

    def create_document(
        self,
        path: Path,
        expert: str | None,
        topic: str | None,
        access_level: int,
        access_tag: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        row = {
            "file_name": path.name,
            "original_file_name": path.name,
            "storage_bucket": self._bucket,
            "mime_type": mime_type,
            "file_size": path.stat().st_size,
            "source": str(path),
            "expert": expert,
            "topic": topic,
            "access_level": access_level,
            "access_tag": access_tag,
            "metadata": metadata or {},
            "status": "created",
        }
        result = self._client.table("documents").insert(row).execute()
        return str(result.data[0]["id"])

    def update_document(
        self,
        document_id: str,
        values: dict[str, Any],
    ) -> None:
        self._client.table("documents").update(values).eq("id", document_id).execute()

    def insert_document_text(
        self,
        document_id: str,
        raw_text: str,
        cleaned_text: str,
        access_level: int,
        access_tag: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        row = {
            "document_id": document_id,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "access_level": access_level,
            "access_tag": access_tag,
            "metadata": metadata or {},
        }
        result = self._client.table("document_texts").insert(row).execute()
        return str(result.data[0]["id"])

    def insert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float] | None],
        access_level: int,
        access_tag: str,
        collaborators: list[dict[str, object]] | None = None,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")
        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append(
                {
                    "document_id": chunk.document_id,
                    "document_text_id": chunk.document_text_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "embedding": embedding,
                    "page_number": chunk.page_number,
                    "sheet_name": chunk.sheet_name,
                    "heading": chunk.heading,
                    "expert": chunk.expert,
                    "topic": chunk.topic,
                    "access_level": access_level,
                    "access_tag": access_tag,
                    "metadata": {
                        **chunk.metadata,
                        "access_level": access_level,
                        "access_tag": access_tag,
                        "collaborators": collaborators or [],
                    },
                }
            )

        if not rows:
            return 0

        result = self._client.table("document_chunks").insert(rows).execute()
        inserted = len(result.data or rows)
        logger.info("Inserted %s chunks into Supabase", inserted)
        return inserted

    def match_documents(
        self,
        query_embedding: list[float],
        top_k: int,
        expert: str | None = None,
        similarity_threshold: float = 0.0,
        requester_user_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if not requester_user_id:
            logger.info("[AI Search] authenticated user: no")
            return []
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "match_expert": expert,
            "similarity_threshold": similarity_threshold,
            "requester_user_id": requester_user_id,
        }
        result = self._client.rpc("match_documents", params).execute()
        logger.info("[AI Search] authenticated user: yes user_id=%s", requester_user_id)
        logger.info(
            "[AI Search] chunks returned after DB access filter: %s",
            len(result.data or []),
        )
        return [_to_retrieved_chunk(row) for row in result.data or []]

    def keyword_search_documents(
        self,
        query: str,
        top_k: int,
        expert: str | None = None,
        requester_user_id: str | None = None,
        include_inaccessible: bool = False,
        scan_limit: int = 1000,
    ) -> list[RetrievedChunk]:
        if not requester_user_id and not include_inaccessible:
            logger.info("[AI Search] authenticated user: no")
            return []
        requester_access_level = (
            self._requester_access_level(requester_user_id)
            if requester_user_id
            else 0
        )
        if requester_user_id:
            logger.info("[AI Search] authenticated user: yes user_id=%s", requester_user_id)
            if requester_access_level <= 0:
                logger.info("[AI Search] chunks returned after DB access filter: 0")
                return []
        tokens = _query_tokens(query)
        request = (
            self._client.table("document_chunks")
            .select(
                "id, document_id, content, chunk_index, page_number, sheet_name, "
                "heading, expert, topic, access_level, access_tag, metadata, "
                "documents(file_name, source, access_level, access_tag)"
            )
            .lt("access_level", 99)
            .limit(scan_limit)
        )
        if not include_inaccessible:
            request = request.lte("access_level", requester_access_level)
        if expert:
            request = request.eq("expert", expert)
        result = request.execute()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in result.data or []:
            score = _keyword_score(row.get("content") or "", tokens)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        logger.info(
            "[AI Search] chunks returned after DB access filter: %s",
            len(scored[:top_k]),
        )
        return [
            _to_retrieved_chunk(_keyword_row_to_retrieved(row, score))
            for score, row in scored[:top_k]
        ]

    def exact_search_documents(
        self,
        query: str,
        top_k: int,
        expert: str | None = None,
        requester_user_id: str | None = None,
        include_inaccessible: bool = False,
    ) -> list[RetrievedChunk]:
        phrase = query.strip()
        if len(phrase) < 3:
            return []
        requester_access_level = (
            self._requester_access_level(requester_user_id)
            if requester_user_id and not include_inaccessible
            else 0
        )
        request = (
            self._client.table("document_chunks")
            .select(
                "id, document_id, content, chunk_index, page_number, sheet_name, "
                "heading, expert, topic, access_level, access_tag, metadata, "
                "documents(file_name, source, access_level, access_tag)"
            )
            .ilike("content", f"%{_escape_like(phrase)}%")
            .lt("access_level", 99)
            .limit(top_k)
        )
        if not include_inaccessible:
            request = request.lte("access_level", requester_access_level)
        if expert:
            request = request.eq("expert", expert)
        result = request.execute()
        chunks = [
            _to_retrieved_chunk(_keyword_row_to_retrieved(row, 10.0))
            for row in result.data or []
        ]
        if chunks:
            return chunks

        scored_rows: list[tuple[float, dict[str, Any]]] = []
        seen_ids: set[str] = set()
        for token in _query_tokens(phrase):
            token_request = (
                self._client.table("document_chunks")
                .select(
                    "id, document_id, content, chunk_index, page_number, sheet_name, "
                    "heading, expert, topic, access_level, access_tag, metadata, "
                    "documents(file_name, source, access_level, access_tag)"
                )
                .ilike("content", f"%{_escape_like(token)}%")
                .lt("access_level", 99)
                .limit(top_k)
            )
            if not include_inaccessible:
                token_request = token_request.lte("access_level", requester_access_level)
            if expert:
                token_request = token_request.eq("expert", expert)
            token_result = token_request.execute()
            for row in token_result.data or []:
                row_id = str(row.get("id"))
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                scored_rows.append((_keyword_score(row.get("content") or "", _query_tokens(phrase)), row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [
            _to_retrieved_chunk(_keyword_row_to_retrieved(row, score))
            for score, row in scored_rows[:top_k]
            if score > 0
        ]

    def _requester_access_level(self, requester_user_id: str) -> int:
        try:
            result = (
                self._client.table("profiles")
                .select("access_level")
                .eq("user_id", requester_user_id)
                .limit(1)
                .execute()
            )
            if result.data:
                return int(result.data[0]["access_level"])
        except Exception:
            logger.debug(
                "Could not load requester access_level from profiles",
                exc_info=True,
            )

        try:
            result = (
                self._client.table("employee_accounts")
                .select("access_level")
                .eq("user_id", requester_user_id)
                .eq("active", True)
                .limit(1)
                .execute()
            )
            if not result.data:
                return 0
            return int(result.data[0]["access_level"])
        except Exception:
            logger.debug(
                "Could not load requester access_level from employee_accounts",
                exc_info=True,
            )
            return 0

    def insert_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "metadata": metadata or {},
        }
        self._client.table("chat_messages").insert(row).execute()

    def list_employee_accounts(self) -> list[EmployeeAccount]:
        result = (
            self._client.table("employee_accounts")
            .select("user_id, full_name, email, department, access_level")
            .eq("active", True)
            .order("access_level")
            .order("full_name")
            .execute()
        )
        return [employee_from_row(row) for row in result.data or []]


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        id=str(row["id"]),
        document_id=_optional_str(row.get("document_id")),
        content=row["content"],
        source=row.get("source"),
        file_name=row.get("file_name"),
        expert=row.get("expert"),
        topic=row.get("topic"),
        chunk_index=row.get("chunk_index"),
        page_number=row.get("page_number"),
        sheet_name=row.get("sheet_name"),
        heading=row.get("heading"),
        metadata=row.get("metadata") or {},
        similarity=float(row.get("similarity", 0.0)),
    )


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _query_tokens(query: str) -> list[str]:
    stop_words = {
        "about",
        "and",
        "are",
        "der",
        "die",
        "das",
        "for",
        "from",
        "how",
        "ist",
        "mit",
        "the",
        "und",
        "was",
        "what",
        "wie",
        "with",
    }
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", query.lower())
    return [token for token in tokens if token not in stop_words]


def _keyword_score(content: str, tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    lowered = content.lower()
    matches = sum(lowered.count(token) for token in tokens)
    coverage = sum(1 for token in set(tokens) if token in lowered)
    if matches == 0:
        return 0.0
    return float(matches + coverage * 2)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _keyword_row_to_retrieved(row: dict[str, Any], score: float) -> dict[str, Any]:
    document = row.get("documents") or {}
    access_level = int(row.get("access_level") or document.get("access_level") or 1)
    access_tag = row.get("access_tag") or document.get("access_tag") or "L1 Public"
    return {
        "id": row["id"],
        "document_id": row.get("document_id"),
        "content": row["content"],
        "source": document.get("source"),
        "file_name": document.get("file_name"),
        "expert": row.get("expert"),
        "topic": row.get("topic"),
        "chunk_index": row.get("chunk_index"),
        "page_number": row.get("page_number"),
        "sheet_name": row.get("sheet_name"),
        "heading": row.get("heading"),
        "metadata": {
            **(row.get("metadata") or {}),
            "retrieval_mode": "keyword",
            "access_level": access_level,
            "access_tag": access_tag,
        },
        "similarity": min(score / 10.0, 1.0),
    }
