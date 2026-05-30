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
    embedding_model: str = "text-embedding-3-small"
    answer_model: str = "gpt-4.1-mini"
    chunk_size: int = 1200
    chunk_overlap: int = 180
    retrieval_top_k: int = 8
    similarity_threshold: float = 0.2

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=_required("OPENAI_API_KEY"),
            supabase_url=_required("SUPABASE_URL"),
            supabase_service_role_key=_required("SUPABASE_SERVICE_ROLE_KEY"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", cls.embedding_model),
            answer_model=os.getenv("OPENAI_ANSWER_MODEL", cls.answer_model),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", cls.retrieval_top_k)),
            similarity_threshold=float(
                os.getenv("SIMILARITY_THRESHOLD", cls.similarity_threshold)
            ),
        )


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

