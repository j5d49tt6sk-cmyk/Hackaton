from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field


load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = Field("rag-documents", env="SUPABASE_STORAGE_BUCKET")
    embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    answer_model: str = Field("gpt-4.1-mini", env="OPENAI_ANSWER_MODEL")
    chunk_size: int = Field(1200, env="CHUNK_SIZE")
    chunk_overlap: int = Field(180, env="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(8, env="RETRIEVAL_TOP_K")
    similarity_threshold: float = Field(0.2, env="SIMILARITY_THRESHOLD")
    use_openai: bool = Field(True, env="USE_OPENAI")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()


def find_placeholder_settings(settings: Settings) -> list[str]:
    placeholders: list[str] = []
    if _looks_like_placeholder(settings.openai_api_key):
        placeholders.append("OPENAI_API_KEY")
    if _looks_like_placeholder(settings.supabase_url):
        placeholders.append("SUPABASE_URL")
    if _looks_like_placeholder(settings.supabase_service_role_key):
        placeholders.append("SUPABASE_SERVICE_ROLE_KEY")
    return placeholders


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {"sk-...", "your-service-role-key"} or "your-project" in lowered
