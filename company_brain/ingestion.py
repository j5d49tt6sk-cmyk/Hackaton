from __future__ import annotations

import logging
from pathlib import Path

from company_brain.chunking import split_text
from company_brain.config import Settings
from company_brain.embeddings import EmbeddingClient
from company_brain.loaders import extract_text, iter_supported_files
from company_brain.metadata import infer_expert, infer_topic
from company_brain.models import DocumentChunk
from company_brain.supabase_store import SupabaseDocumentStore


logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embedding_client: EmbeddingClient | None = None,
        document_store: SupabaseDocumentStore | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_client = embedding_client or EmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
        self._document_store = document_store or SupabaseDocumentStore(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )

    def ingest_path(
        self,
        root: Path,
        expert: str | None = None,
        topic: str | None = None,
        batch_size: int = 64,
    ) -> int:
        files = iter_supported_files(root)
        logger.info("Found %s supported files under %s", len(files), root)
        total_inserted = 0

        for file_path in files:
            try:
                chunks = self._load_file_chunks(file_path, expert, topic)
                if not chunks:
                    logger.info("No text extracted from %s", file_path)
                    continue
                total_inserted += self._insert_in_batches(chunks, batch_size)
            except Exception:
                logger.exception("Failed to ingest %s", file_path)

        logger.info("Finished ingestion. Inserted %s chunks.", total_inserted)
        return total_inserted

    def _load_file_chunks(
        self, file_path: Path, expert: str | None, topic: str | None
    ) -> list[DocumentChunk]:
        text = extract_text(file_path)
        parts = split_text(
            text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        inferred_expert = infer_expert(file_path, expert)
        inferred_topic = infer_topic(file_path, topic)
        return [
            DocumentChunk(
                content=content,
                source=str(file_path),
                file_name=file_path.name,
                expert=inferred_expert,
                topic=inferred_topic,
                chunk_index=index,
                metadata={
                    "extension": file_path.suffix.lower(),
                    "relative_parent": file_path.parent.name,
                    "character_count": len(content),
                },
            )
            for index, content in enumerate(parts)
        ]

    def _insert_in_batches(self, chunks: list[DocumentChunk], batch_size: int) -> int:
        inserted = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self._embedding_client.embed_texts(
                [chunk.content for chunk in batch]
            )
            inserted += self._document_store.insert_chunks(batch, embeddings)
        return inserted

