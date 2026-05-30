from __future__ import annotations

from company_brain.config import Settings
from company_brain.embeddings import EmbeddingClient
from company_brain.models import RetrievedChunk
from company_brain.supabase_store import SupabaseDocumentStore


EXPERT_OPTIONS = {
    "Ask Company Brain": None,
    "Ask Compliance Expert": "Compliance Expert",
    "Ask ESG Expert": "ESG Expert",
    "Ask Internal Expert": "Internal Expert",
}


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedding_client: EmbeddingClient | None = None,
        document_store: SupabaseDocumentStore | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_client = (
            embedding_client
            if embedding_client is not None
            else (
                EmbeddingClient(
                    api_key=settings.openai_api_key,
                    model=settings.embedding_model,
                )
                if settings.use_openai
                else None
            )
        )
        self._document_store = document_store or SupabaseDocumentStore(
            settings.supabase_url,
            settings.supabase_service_role_key,
            settings.supabase_storage_bucket,
        )

    def retrieve(
        self,
        question: str,
        expert: str | None = None,
        top_k: int | None = None,
        requester_access_level: int = 1,
    ) -> list[RetrievedChunk]:
        if not self._settings.use_openai:
            return self._document_store.keyword_search_documents(
                query=question,
                top_k=top_k or self._settings.retrieval_top_k,
                expert=expert,
                requester_access_level=requester_access_level,
            )
        if self._embedding_client is None:
            raise RuntimeError("OpenAI retrieval is enabled, but no embedding client exists.")
        query_embedding = self._embedding_client.embed_text(question)
        return self._document_store.match_documents(
            query_embedding=query_embedding,
            top_k=top_k or self._settings.retrieval_top_k,
            expert=expert,
            similarity_threshold=self._settings.similarity_threshold,
            requester_access_level=requester_access_level,
        )


def expert_for_ui_choice(choice: str) -> str | None:
    return EXPERT_OPTIONS.get(choice)
