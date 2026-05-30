from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from supabase import Client, create_client

from company_brain.models import DocumentChunk, RetrievedChunk


logger = logging.getLogger(__name__)


class SupabaseDocumentStore:
    def __init__(self, url: str, service_role_key: str) -> None:
        self._client: Client = create_client(url, service_role_key)

    def insert_chunks(
        self, chunks: Iterable[DocumentChunk], embeddings: list[list[float]]
    ) -> int:
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            rows.append(
                {
                    "content": chunk.content,
                    "source": chunk.source,
                    "file_name": chunk.file_name,
                    "expert": chunk.expert,
                    "topic": chunk.topic,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                }
            )

        if not rows:
            return 0

        result = self._client.table("documents").insert(rows).execute()
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


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        id=int(row["id"]),
        content=row["content"],
        source=row.get("source"),
        file_name=row.get("file_name"),
        expert=row.get("expert"),
        topic=row.get("topic"),
        chunk_index=row.get("chunk_index"),
        metadata=row.get("metadata") or {},
        similarity=float(row.get("similarity", 0.0)),
    )

