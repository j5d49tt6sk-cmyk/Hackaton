from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

from supabase import Client, create_client

from company_brain.models import DocumentChunk, RetrievedChunk


logger = logging.getLogger(__name__)


class SupabaseDocumentStore:
    def __init__(self, url: str, service_role_key: str, bucket: str) -> None:
        self._client: Client = create_client(url, service_role_key)
        self._bucket = bucket

    def upload_file(self, path: Path, document_id: int) -> str:
        storage_path = f"documents/{document_id}/{path.name}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as handle:
            self._client.storage.from_(self._bucket).upload(
                storage_path,
                handle,
                {"content-type": content_type, "upsert": "true"},
            )
        return storage_path

    def create_document(
        self,
        path: Path,
        expert: str | None,
        topic: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
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
            "metadata": metadata or {},
            "status": "created",
        }
        result = self._client.table("documents").insert(row).execute()
        return int(result.data[0]["id"])

    def update_document(
        self,
        document_id: int,
        values: dict[str, Any],
    ) -> None:
        self._client.table("documents").update(values).eq("id", document_id).execute()

    def insert_document_text(
        self,
        document_id: int,
        raw_text: str,
        cleaned_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        row = {
            "document_id": document_id,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "metadata": metadata or {},
        }
        result = self._client.table("document_texts").insert(row).execute()
        return int(result.data[0]["id"])

    def insert_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> int:
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
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
                    "metadata": chunk.metadata,
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
    ) -> list[RetrievedChunk]:
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "match_expert": expert,
            "similarity_threshold": similarity_threshold,
        }
        result = self._client.rpc("match_documents", params).execute()
        return [_to_retrieved_chunk(row) for row in result.data or []]

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


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        id=int(row["id"]),
        document_id=_optional_int(row.get("document_id")),
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


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
