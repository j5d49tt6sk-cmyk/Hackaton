from __future__ import annotations

import logging
from pathlib import Path

from company_brain.access_control import infer_document_access
from company_brain.chunking import chunk_sections, normalize_text
from company_brain.config import Settings
from company_brain.embeddings import EmbeddingClient
from company_brain.loaders import extract_sections, iter_supported_files
from company_brain.metadata import infer_expert, infer_topic
from company_brain.models import DocumentChunk, ExtractedSection
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
            settings.supabase_storage_bucket,
        )

    def ingest_path(
        self,
        root: Path,
        expert: str | None = None,
        topic: str | None = None,
        batch_size: int = 64,
        replace_existing: bool = True,
        access_level: int | None = None,
        access_tag: str | None = None,
    ) -> int:
        files = iter_supported_files(root)
        logger.info("Found %s supported files under %s", len(files), root)
        total_inserted = 0

        for file_path in files:
            try:
                total_inserted += self.ingest_file(
                    file_path,
                    expert=expert,
                    topic=topic,
                    batch_size=batch_size,
                    replace_existing=replace_existing,
                    access_level=access_level,
                    access_tag=access_tag,
                )
            except Exception:
                logger.exception("Failed to ingest %s", file_path)

        logger.info("Finished ingestion. Inserted %s chunks.", total_inserted)
        return total_inserted

    def ingest_file(
        self,
        file_path: Path,
        expert: str | None = None,
        topic: str | None = None,
        batch_size: int = 64,
        replace_existing: bool = True,
        access_level: int | None = None,
        access_tag: str | None = None,
    ) -> int:
        inferred_expert = infer_expert(file_path, expert)
        inferred_topic = infer_topic(file_path, topic)
        inferred_access_level, inferred_access_tag = infer_document_access(file_path)
        document_access_level = access_level or inferred_access_level
        document_access_tag = access_tag or inferred_access_tag
        if replace_existing:
            existing_document_id = self._document_store.find_existing_document_id(
                file_path
            )
            if existing_document_id is not None:
                logger.info(
                    "Replacing existing document %s for %s",
                    existing_document_id,
                    file_path.name,
                )
                self._document_store.delete_document(existing_document_id)

        document_id = self._document_store.create_document(
            file_path,
            expert=inferred_expert,
            topic=inferred_topic,
            access_level=document_access_level,
            access_tag=document_access_tag,
            metadata=_document_metadata(
                file_path,
                document_access_level,
                document_access_tag,
            ),
        )

        try:
            storage_path = self._document_store.upload_file(file_path, document_id)
            self._document_store.update_document(
                document_id,
                {"storage_path": storage_path, "status": "uploaded"},
            )

            sections = extract_sections(file_path)
            raw_text = _join_sections(sections, clean=False)
            cleaned_text = _join_sections(sections, clean=True)
            if not cleaned_text:
                self._document_store.update_document(
                    document_id,
                    {"status": "empty", "error_message": "No extractable text found"},
                )
                return 0

            document_text_id = self._document_store.insert_document_text(
                document_id=document_id,
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                access_level=document_access_level,
                access_tag=document_access_tag,
                metadata={
                    "section_count": len(sections),
                    "extractor": file_path.suffix.lower().lstrip("."),
                    "access_level": document_access_level,
                    "access_tag": document_access_tag,
                },
            )

            chunks = chunk_sections(
                sections=sections,
                document_id=document_id,
                document_text_id=document_text_id,
                expert=inferred_expert,
                topic=inferred_topic,
                chunk_size=self._settings.chunk_size,
                overlap=self._settings.chunk_overlap,
            )
            inserted = self._insert_in_batches(
                chunks,
                batch_size,
                access_level=document_access_level,
                access_tag=document_access_tag,
            )
            self._document_store.update_document(
                document_id,
                {
                    "status": "indexed",
                    "metadata": {
                        **_document_metadata(file_path),
                        "access_level": document_access_level,
                        "access_tag": document_access_tag,
                        "section_count": len(sections),
                        "chunk_count": inserted,
                    },
                },
            )
            return inserted
        except Exception as exc:
            try:
                self._document_store.update_document(
                    document_id,
                    {"status": "failed", "error_message": str(exc)},
                )
            except Exception:
                logger.exception("Could not write document error_message")
                self._document_store.update_document(document_id, {"status": "failed"})
            raise

    def _insert_in_batches(
        self,
        chunks: list[DocumentChunk],
        batch_size: int,
        access_level: int,
        access_tag: str,
    ) -> int:
        inserted = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            if self._settings.use_openai:
                try:
                    embeddings = self._embedding_client.embed_texts(
                        [chunk.content for chunk in batch]
                    )
                except Exception:
                    logger.exception(
                        "Could not generate embeddings; storing chunks without embeddings"
                    )
                    embeddings = [None] * len(batch)
            else:
                embeddings = [None] * len(batch)
            inserted += self._document_store.insert_chunks(
                batch,
                embeddings,
                access_level=access_level,
                access_tag=access_tag,
            )
        return inserted


def _join_sections(sections: list[ExtractedSection], clean: bool) -> str:
    parts: list[str] = []
    for section in sections:
        prefix_parts = []
        if section.page_number:
            prefix_parts.append(f"Page {section.page_number}")
        if section.sheet_name:
            prefix_parts.append(f"Sheet: {section.sheet_name}")
        if section.heading:
            prefix_parts.append(f"Heading: {section.heading}")
        prefix = " | ".join(prefix_parts)
        content = normalize_text(section.content) if clean else section.content
        parts.append(f"[{prefix}]\n{content}" if prefix else content)
    return "\n\n".join(part for part in parts if part.strip())


def _document_metadata(
    file_path: Path,
    access_level: int | None = None,
    access_tag: str | None = None,
) -> dict[str, object]:
    inferred_access_level, inferred_access_tag = infer_document_access(file_path)
    return {
        "extension": file_path.suffix.lower(),
        "source_name": file_path.name,
        "source_path": str(file_path),
        "access_level": access_level or inferred_access_level,
        "access_tag": access_tag or inferred_access_tag,
    }
