from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    supabase_url: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "rag-documents"
    embedding_model: str = "text-embedding-3-small"
    answer_model: str = "gpt-4.1-mini"
    chunk_size: int = 1200
    chunk_overlap: int = 180
    retrieval_top_k: int = 8
    similarity_threshold: float = 0.2
    use_openai: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=_required("OPENAI_API_KEY"),
            supabase_url=_required("SUPABASE_URL"),
            supabase_service_role_key=_required("SUPABASE_SERVICE_ROLE_KEY"),
            supabase_storage_bucket=os.getenv(
                "SUPABASE_STORAGE_BUCKET", cls.supabase_storage_bucket
            ),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", cls.embedding_model),
            answer_model=os.getenv("OPENAI_ANSWER_MODEL", cls.answer_model),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", cls.retrieval_top_k)),
            similarity_threshold=float(
                os.getenv("SIMILARITY_THRESHOLD", cls.similarity_threshold)
            ),
            use_openai=_bool_env("USE_OPENAI", cls.use_openai),
        )


def find_placeholder_settings(settings: Settings) -> list[str]:
    placeholders: list[str] = []
    if _looks_like_placeholder(settings.openai_api_key):
        placeholders.append("OPENAI_API_KEY")
    if _looks_like_placeholder(settings.supabase_url):
        placeholders.append("SUPABASE_URL")
    if _looks_like_placeholder(settings.supabase_service_role_key):
        placeholders.append("SUPABASE_SERVICE_ROLE_KEY")
    return placeholders


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return lowered in {"sk-...", "your-service-role-key"} or "your-project" in lowered


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
