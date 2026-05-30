from __future__ import annotations

import argparse

from company_brain.config import Settings


PLACEHOLDER_MARKERS = ("...", "your-", "sk-...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate backend configuration.")
    parser.add_argument(
        "--skip-openai",
        action="store_true",
        help="Only check environment and Supabase connectivity.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    _validate_real_value("OPENAI_API_KEY", settings.openai_api_key)
    _validate_real_value("SUPABASE_URL", settings.supabase_url)
    _validate_real_value(
        "SUPABASE_SERVICE_ROLE_KEY", settings.supabase_service_role_key
    )

    from supabase import create_client

    supabase = create_client(
        settings.supabase_url,
        settings.supabase_service_role_key,
    )
    for table_name in (
        "documents",
        "document_texts",
        "document_chunks",
        "chat_messages",
        "employee_accounts",
    ):
        supabase.table(table_name).select("id").limit(1).execute()
        print(f"Supabase: {table_name} table is reachable.")

    try:
        supabase.storage.from_(settings.supabase_storage_bucket).list("", {"limit": 1})
    except Exception as exc:
        raise RuntimeError(
            "Supabase Storage bucket is not reachable: "
            f"{settings.supabase_storage_bucket}"
        ) from exc
    print(f"Supabase: storage bucket {settings.supabase_storage_bucket} is reachable.")

    if settings.use_openai and not args.skip_openai:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        client.embeddings.create(model=settings.embedding_model, input="backend check")
        print(f"OpenAI: embedding model {settings.embedding_model} is reachable.")
    elif not settings.use_openai:
        print("OpenAI: skipped because USE_OPENAI=false.")

    print("Backend configuration looks ready.")


def _validate_real_value(name: str, value: str) -> None:
    lowered = value.lower()
    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        raise RuntimeError(f"{name} still looks like a placeholder in .env")


if __name__ == "__main__":
    main()
