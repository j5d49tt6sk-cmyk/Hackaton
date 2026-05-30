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
        self._embedding_client = embedding_client or EmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
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
    ) -> list[RetrievedChunk]:
        query_embedding = self._embedding_client.embed_text(question)
        return self._document_store.match_documents(
            query_embedding=query_embedding,
            top_k=top_k or self._settings.retrieval_top_k,
            expert=expert,
            similarity_threshold=self._settings.similarity_threshold,
        )


def expert_for_ui_choice(choice: str) -> str | None:
    return EXPERT_OPTIONS.get(choice)
