from __future__ import annotations

import pytest

from company_brain.config import Settings, find_placeholder_settings


def test_settings_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-role")
    monkeypatch.setenv("USE_OPENAI", "false")

    settings = Settings.from_env()

    assert settings.openai_api_key == "test-key"
    assert settings.supabase_url == "https://example.supabase.co"
    assert settings.supabase_service_role_key == "test-role"
    assert settings.use_openai is False
    assert settings.supabase_storage_bucket == "rag-documents"


def test_find_placeholder_settings_detects_placeholder_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-...")
    monkeypatch.setenv("SUPABASE_URL", "your-project.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "your-service-role-key")

    settings = Settings.from_env()

    assert set(find_placeholder_settings(settings)) == {
        "OPENAI_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    }
