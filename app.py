from __future__ import annotations

import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class UploadResult:
    local_chunks: int = 0
    database_chunks: int | None = None
    database_error: str | None = None

    @property
    def indexed_chunks(self) -> int:
        return self.local_chunks if self.local_chunks else self.database_chunks or 0

    @property
    def database_saved(self) -> bool:
        return self.database_chunks is not None and self.database_error is None


def _module_is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _find_free_port(start_port: int = 8501, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            try:
                handle.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("Could not find a free port for Streamlit.")


if __name__ == "__main__" and not os.getenv("COMPANY_BRAIN_STREAMLIT_BOOTSTRAPPED"):
    if not _module_is_available("streamlit"):
        print("Streamlit is not installed yet.")
        print("Run this once from the project folder:")
        print(f"{sys.executable} -m pip install -r requirements.txt")
        raise SystemExit(1)

    port = str(_find_free_port())
    env = os.environ.copy()
    env["COMPANY_BRAIN_STREAMLIT_BOOTSTRAPPED"] = "1"
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                __file__,
                "--server.port",
                port,
                "--server.headless",
                "false",
                "--browser.gatherUsageStats",
                "false",
            ],
            env=env,
        )
    )


import streamlit as st

from company_brain.access_control import (
    DEMO_EMPLOYEES,
    DEMO_EMPLOYEE_PASSWORDS,
    EmployeeAccount,
    access_label,
    access_tag,
    data_case_access_level,
)
from company_brain.answering import AnswerGenerator
from company_brain.config import Settings
from company_brain.ingestion import IngestionPipeline
from company_brain.local_knowledge import LocalKnowledgeStore
from company_brain.models import GeneratedAnswer, RetrievedChunk
from company_brain.ollama_answering import OllamaAnswerGenerator
from company_brain.retrieval import Retriever, expert_for_ui_choice
from company_brain.supabase_store import SupabaseDocumentStore


st.set_page_config(page_title="Company Brain", layout="wide")

_brand_logo_path = Path(__file__).with_name("six_group_logo.png")

st.markdown(
    """
    <style>
    [data-testid="stDecoration"],
    [data-testid="stHeader"] {
        display: none;
    }

    .stApp {
        background: #fafafa;
    }

    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .stMarkdown h5 {
        margin: 0.85rem 0 0.25rem;
        font-size: 1rem;
        font-weight: 650;
    }

    .mode-selector-spacer {
        width: 100%;
        margin: 0.45rem 0 1.8rem;
    }

    .six-header-divider {
        height: 4px;
        width: 100%;
        background: #e30613;
        margin: 0.8rem 0 1.15rem;
    }

    .six-top-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 20px;
        margin: 0 0 0.55rem;
    }

    .six-top-divider {
        height: 3px;
        width: 100%;
        background: #e30613;
        margin: 0 0 1.5rem;
    }

    .app-title-block {
        text-align: center;
        margin-bottom: 1.2rem;
    }

    .app-title-block h1 {
        margin-bottom: 0.25rem;
    }

    .app-title-block p {
        margin: 0;
        color: #555555;
        font-size: 0.98rem;
    }

    .top-account-row {
        display: flex;
        justify-content: flex-end;
        align-items: stretch;
        gap: 10px;
        margin: 0;
    }

    .st-key-top_account_area {
        display: flex;
        justify-content: flex-end;
    }

    .st-key-mode_case_guide button,
    .st-key-mode_quick_question button,
    .st-key-mode_add_case button {
        min-height: 62px;
        border-radius: 2px;
        font-size: 1.12rem;
        font-weight: 650;
        border-width: 1.5px;
        letter-spacing: 0;
    }

    .st-key-mode_case_guide button p,
    .st-key-mode_quick_question button p,
    .st-key-mode_add_case button p {
        font-size: 1.12rem;
        line-height: 1.1;
        font-weight: 650;
    }

    .login-badge {
        display: inline-block;
        width: fit-content;
        min-width: 0;
        max-width: 420px;
        background: rgba(255, 255, 255, 0.97);
        border: 1px solid #d9d9d9;
        border-top: 3px solid #e30613;
        border-radius: 2px;
        padding: 10px 14px 9px;
        color: #1f1f1f;
        font-size: 0.82rem;
        line-height: 1.35;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
        letter-spacing: 0;
    }

    .login-badge strong {
        font-weight: 650;
    }

    .profile-symbol {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 21px;
        height: 21px;
        margin-right: 7px;
        border: 1px solid #1f1f1f;
        border-radius: 50%;
        font-size: 0.78rem;
        vertical-align: -2px;
    }

    .st-key-top_logout_button {
        width: 120px;
    }

    .st-key-top_logout_button button {
        min-height: 48px;
        border-radius: 2px;
        border: 1px solid #1f1f1f;
        background: #ffffff;
        color: #1f1f1f;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .st-key-top_logout_button button:hover {
        border-color: #e30613;
        color: #e30613;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


BUSINESS_AREAS = [
    "Not sure",
    "Tax / FATCA",
    "MiFID / Product governance",
    "ESG / SFDR",
    "Client onboarding",
    "Reference data",
    "Internal process",
]

OUTPUT_FOCUS = [
    "Decision",
    "Reasoning",
    "Regulatory requirement",
    "Risks",
    "Sources",
]

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
]


def _uses_ollama_backend() -> bool:
    return os.getenv("COMPANY_BRAIN_BACKEND", "").lower() == "ollama"


def _is_authenticated() -> bool:
    return bool(st.session_state.get("employee_user_id"))


def _employee_accounts() -> list[EmployeeAccount]:
    if _uses_ollama_backend() or _missing_environment():
        return DEMO_EMPLOYEES
    try:
        accounts = _document_store().list_employee_accounts()
    except Exception:
        logging.exception("Failed to load employee accounts from Supabase")
        return DEMO_EMPLOYEES
    return accounts or DEMO_EMPLOYEES


def _current_employee() -> EmployeeAccount:
    user_id = st.session_state.get("employee_user_id")
    for employee in _employee_accounts():
        if employee.user_id == user_id:
            return employee
    return DEMO_EMPLOYEES[0]


def _normalize_login_value(value: str) -> str:
    return value.strip().lower().replace(" ", ".")


def _employee_login_names(employee: EmployeeAccount) -> set[str]:
    return {
        _normalize_login_value(employee.username),
        _normalize_login_value(employee.email),
        _normalize_login_value(employee.full_name),
    }


def _employee_passwords() -> dict[str, str]:
    raw_passwords = os.getenv("COMPANY_BRAIN_USER_PASSWORDS", "")
    if not raw_passwords:
        return DEMO_EMPLOYEE_PASSWORDS
    try:
        parsed = json.loads(raw_passwords)
    except json.JSONDecodeError:
        parsed = {
            key.strip(): value.strip()
            for item in raw_passwords.split(",")
            if "=" in item
            for key, value in [item.split("=", 1)]
        }
    if not isinstance(parsed, dict):
        return DEMO_EMPLOYEE_PASSWORDS
    return {
        _normalize_login_value(str(username)): str(password)
        for username, password in parsed.items()
    }


def _employee_for_login(username: str) -> EmployeeAccount | None:
    normalized = _normalize_login_value(username)
    for employee in _employee_accounts():
        if normalized in _employee_login_names(employee):
            return employee
    return None


def _password_is_valid(employee: EmployeeAccount, password: str) -> bool:
    passwords = _employee_passwords()
    for login_name in _employee_login_names(employee):
        if passwords.get(login_name) == password:
            return True
    return False


def _render_brand_logo(width: int = 120) -> None:
    if _brand_logo_path.exists():
        st.image(str(_brand_logo_path), width=width)
    else:
        st.markdown("### SIX")


def _render_login_gate() -> None:
    _render_brand_logo()
    st.markdown('<div class="six-top-divider"></div>', unsafe_allow_html=True)
    st.title("Company Brain")
    st.caption("Sign in with your employee account.")

    with st.form("employee_login_form"):
        username = st.text_input(
            "Username",
            placeholder="Example: anna.keller",
            autocomplete="username",
        )
        password = st.text_input(
            "Password",
            type="password",
            autocomplete="current-password",
        )
        submitted = st.form_submit_button("Log In", type="primary")

    if submitted:
        employee = _employee_for_login(username)
        if employee is None or not _password_is_valid(employee, password):
            st.error("Invalid username or password.")
            return
        st.session_state.employee_user_id = employee.user_id
        st.session_state.employee_access_level = employee.access_level
        st.session_state.employee_name = employee.full_name
        st.session_state.company_brain_authenticated = True
        st.rerun()


def _requester_user_id() -> str | None:
    user_id = st.session_state.get("employee_user_id")
    return str(user_id) if user_id else None


def _requester_access_level() -> int:
    return _current_employee().access_level


def _render_employee_badge() -> None:
    employee = _current_employee()
    _, details_column, logout_column = st.columns([0.42, 0.42, 0.16], gap="small")
    with details_column:
        st.markdown(
            (
                '<div class="login-badge">'
                '<span class="profile-symbol">&#128100;</span>'
                f"You're logged in as <strong>{employee.full_name}</strong><br>"
                f"{employee.department} | User ID: {employee.username} | "
                f"Access: {employee.access_label} (level {employee.access_level})"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with logout_column:
        if st.button("Log Out", key="top_logout_button", use_container_width=True):
            for key in (
                "employee_user_id",
                "employee_access_level",
                "employee_name",
                "company_brain_authenticated",
                "case_overview",
                "case_result",
                "quick_result",
            ):
                st.session_state.pop(key, None)
            st.rerun()


def _render_brand_strip() -> None:
    st.markdown('<div class="six-header-divider"></div>', unsafe_allow_html=True)


def _render_top_brand() -> None:
    logo_column, account_column = st.columns([0.28, 0.72], gap="large")
    with logo_column:
        _render_brand_logo()
    with account_column:
        with st.container(key="top_account_area"):
            _render_employee_badge()
    st.write("")
    st.markdown('<div class="six-top-divider"></div>', unsafe_allow_html=True)


@st.cache_resource
def _settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def _retriever() -> Retriever:
    return Retriever(_settings())


@st.cache_resource
def _answer_generator() -> AnswerGenerator:
    return AnswerGenerator(_settings())


@st.cache_resource
def _ollama_answer_generator() -> OllamaAnswerGenerator:
    return OllamaAnswerGenerator(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45")),
    )


@st.cache_resource
def _ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(_settings())


@st.cache_resource
def _local_knowledge_store() -> LocalKnowledgeStore:
    return LocalKnowledgeStore(Path(os.getenv("LOCAL_KNOWLEDGE_DIR", "local_knowledge")))


@st.cache_resource
def _document_store() -> SupabaseDocumentStore:
    settings = _settings()
    return SupabaseDocumentStore(
        settings.supabase_url,
        settings.supabase_service_role_key,
        settings.supabase_storage_bucket,
    )


def _missing_environment() -> list[str]:
    if _uses_ollama_backend():
        return []
    return _database_environment_issues()


def _database_environment_issues() -> list[str]:
    missing = []
    if not _env_value("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not _env_value("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    service_role_key = _env_value("SUPABASE_SERVICE_ROLE_KEY")
    if not service_role_key:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    elif _looks_like_publishable_supabase_key(service_role_key):
        missing.append("SUPABASE_SERVICE_ROLE_KEY_REAL_SERVICE_ROLE")
    return missing


def _supabase_is_configured() -> bool:
    return not _database_environment_issues()


def _use_local_upload_store() -> bool:
    return _uses_ollama_backend() or not _supabase_is_configured()


def _use_local_knowledge_store() -> bool:
    return _use_local_upload_store()


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and not _looks_like_placeholder(value):
            return value
    return None


def _looks_like_publishable_supabase_key(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered.startswith(("sb_publishable_", "sb_anon_")):
        return True
    if "." not in value:
        return False
    try:
        payload = value.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except Exception:
        return False
    return decoded.get("role") != "service_role"


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return lowered in {"sk-...", "your-service-role-key"} or "your-project" in lowered


def _format_answer(answer: GeneratedAnswer) -> str:
    source_lines = "\n".join(f"- {source}" for source in answer.sources)
    if not source_lines:
        source_lines = "- No sources found"
    return (
        f"{answer.answer}\n\n"
        f"### Sources\n{source_lines}\n\n"
        f"### Confidence\n{answer.confidence}"
    )


def _build_guided_question(
    keyword: str,
    situation: str,
) -> str:
    parts = [
        "Search the company knowledge base for relevant historical cases.",
        f"Keyword or topic: {keyword or 'not provided'}.",
        f"Situation: {situation or 'not provided'}.",
        (
            "Find cases that are similar to the keyword and situation. Prioritize "
            "case files over general reference documents."
        ),
    ]
    return "\n".join(parts)


def _build_open_case_question(original_question: str, case_title: str) -> str:
    return (
        f"Open the case named {case_title}.\n"
        f"Original search: {original_question}\n\n"
        "Use the selected case evidence for the case details and use only separate "
        "regulation/reference files for Regulation Source. Return exactly these Markdown sections "
        "in this order: Problem, Decision, Reasoning, Regulations Used, Regulation "
        "Source, Risks. In Regulation Source, never cite the opened case file itself. "
        "Cite the regulation/reference evidence file where the regulation appears. "
        "Keep Risks as the final section."
    )


def _run_company_brain(
    question: str,
    expert_choice: str,
    top_k: int,
    answer_style: str = "case",
) -> tuple[GeneratedAnswer, list[RetrievedChunk]]:
    chunks = _retrieve_company_brain(question, expert_choice, top_k)
    generated = _generate_company_brain_answer(
        question,
        chunks,
        expert_choice,
        answer_style=answer_style,
    )
    return generated, chunks


def _retrieve_company_brain(
    question: str,
    expert_choice: str,
    top_k: int,
) -> list[RetrievedChunk]:
    selected_expert = expert_for_ui_choice(expert_choice)
    if _use_local_knowledge_store():
        local_chunks = _local_knowledge_store().retrieve(
            question,
            expert=selected_expert,
            top_k=top_k,
            requester_access_level=_requester_access_level(),
        )
        if not _supabase_is_configured():
            return local_chunks
        try:
            database_chunks = _document_store().exact_search_documents(
                query=question,
                top_k=top_k,
                expert=selected_expert,
                requester_user_id=_requester_user_id(),
                include_inaccessible=True,
            )
            if not database_chunks:
                database_chunks = _document_store().keyword_search_documents(
                    query=question,
                    top_k=top_k,
                    expert=selected_expert,
                    requester_user_id=_requester_user_id(),
                    include_inaccessible=True,
                    scan_limit=5000,
                )
            database_chunks = [
                chunk
                for chunk in database_chunks
                if _chunk_access_level(chunk) <= _requester_access_level()
            ]
        except Exception:
            logging.exception("Failed to retrieve from Supabase; using local chunks")
            return local_chunks
        return _merge_retrieved_chunks(database_chunks, local_chunks, top_k)

    retrieved_chunks = _retriever().retrieve(
        question,
        expert=selected_expert,
        top_k=top_k,
        requester_user_id=_requester_user_id(),
    )
    return [
        chunk
        for chunk in retrieved_chunks
        if _chunk_access_level(chunk) <= _requester_access_level()
    ]


def _merge_retrieved_chunks(
    primary_chunks: list[RetrievedChunk],
    fallback_chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    merged: list[RetrievedChunk] = []
    seen: set[tuple[str | None, int | None, str | None]] = set()
    for chunk in [*primary_chunks, *fallback_chunks]:
        key = (chunk.file_name or chunk.source, chunk.chunk_index, chunk.content[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= top_k:
            break
    return merged


def _generate_company_brain_answer(
    question: str,
    chunks: list[RetrievedChunk],
    expert_choice: str,
    answer_style: str = "case",
) -> GeneratedAnswer:
    selected_expert = expert_for_ui_choice(expert_choice)
    if _uses_ollama_backend() or answer_style == "case_open":
        return _ollama_answer_generator().generate(
            question,
            chunks,
            selected_expert,
            answer_style=answer_style,
        )
    try:
        return _answer_generator().generate(
            question,
            chunks,
            selected_expert,
            answer_style=answer_style,
        )
    except Exception as exc:
        if "insufficient_quota" not in str(exc) and "RateLimitError" not in str(exc):
            raise
        logging.warning("OpenAI quota exceeded. Falling back to Ollama.")
        return _ollama_answer_generator().generate(
            question,
            chunks,
            selected_expert,
            answer_style=answer_style,
        )


def _case_key(chunk: RetrievedChunk) -> str:
    return chunk.file_name or chunk.source or f"chunk-{chunk.id}"


def _case_title(chunk: RetrievedChunk) -> str:
    file_name = chunk.file_name or ""
    topic = chunk.topic or ""
    if file_name.startswith("Case_") or topic.lower() in {"uploads", "cases"}:
        title = file_name or chunk.source or topic or "Untitled case"
    else:
        title = topic or file_name or chunk.source or "Untitled case"
    if title.endswith((".pdf", ".docx", ".xlsx", ".txt", ".md", ".csv")):
        title = Path(title).stem
    return title.replace("_", " ").strip()


def _is_case_chunk(chunk: RetrievedChunk) -> bool:
    file_name = chunk.file_name or ""
    source = chunk.source or ""
    return file_name.startswith("Case_") or "data_cases/" in source


def _is_regulation_source_chunk(chunk: RetrievedChunk) -> bool:
    if _is_case_chunk(chunk):
        return False
    haystack = " ".join(
        [
            chunk.file_name or "",
            chunk.source or "",
            chunk.topic or "",
            chunk.heading or "",
            chunk.content[:1200],
        ]
    ).lower()
    return any(
        token in haystack
        for token in (
            "regulation",
            "regulatory",
            "directive",
            "mifid",
            "mifir",
            "sfdr",
            "fatca",
            "taxonomy",
            "finma",
            "esma",
            "rts",
            "eet",
            "emt",
            "law",
            "compliance",
        )
    )


CASE_MATCH_STOP_WORDS = {
    "about",
    "and",
    "are",
    "case",
    "cases",
    "company",
    "der",
    "die",
    "das",
    "for",
    "from",
    "how",
    "knowledge",
    "not",
    "provided",
    "search",
    "similar",
    "situation",
    "the",
    "und",
    "was",
    "what",
    "with",
}


def _similarity_percent(value: float) -> int:
    return round(max(0.0, min(value, 1.0)) * 100)


def _build_similar_cases(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        key = _case_key(chunk)
        case = cases.setdefault(
            key,
            {
                "key": key,
                "title": _case_title(chunk),
                "best_similarity": chunk.similarity,
                "word_similarity": 0.0,
                "llm_similarity": None,
                "access_level": _chunk_access_level(chunk),
                "collaborators": _case_collaborators(chunk),
                "chunks": [],
            },
        )
        case["best_similarity"] = max(float(case["best_similarity"]), chunk.similarity)
        case["access_level"] = max(
            int(case.get("access_level") or 1),
            _chunk_access_level(chunk),
        )
        case_chunks = case["chunks"]
        if isinstance(case_chunks, list):
            case_chunks.append(chunk)

    for case in cases.values():
        case_chunks = case["chunks"]
        if isinstance(case_chunks, list):
            case_chunks.sort(key=lambda chunk: chunk.similarity, reverse=True)
            word_similarity = _case_word_similarity(
                question,
                str(case.get("title") or ""),
                case_chunks,
            )
            case["word_similarity"] = word_similarity
            case["best_similarity"] = word_similarity

    positive_cases = [
        case
        for case in cases.values()
        if _similarity_percent(float(case.get("best_similarity") or 0.0)) > 0
    ]
    ranked_cases = sorted(
        positive_cases,
        key=lambda case: float(case["best_similarity"]),
        reverse=True,
    )
    return _rerank_cases_with_ollama(question, ranked_cases)


def _case_word_similarity(
    question: str,
    title: str,
    chunks: list[RetrievedChunk],
) -> float:
    query_tokens = _case_match_tokens(question)
    if not query_tokens:
        return 0.0
    case_text = "\n".join(
        [title]
        + [
            f"{chunk.file_name or ''} {chunk.topic or ''} {chunk.heading or ''} {chunk.content}"
            for chunk in chunks[:8]
        ]
    )
    case_tokens = _case_match_tokens(case_text)
    if not case_tokens:
        return 0.0
    overlap = query_tokens & case_tokens
    coverage = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(case_tokens)
    return min((coverage * 0.82) + (precision * 0.18), 1.0)


def _case_match_tokens(text: str) -> set[str]:
    tokens = {
        match.group(0).lower()
        for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text)
    }
    return {token for token in tokens if token not in CASE_MATCH_STOP_WORDS}


def _rerank_cases_with_ollama(
    question: str,
    cases: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not cases:
        return cases
    prompt_cases = []
    for case in cases[:8]:
        chunks = case.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        evidence = " ".join(chunk.content[:500] for chunk in chunks[:2])
        prompt_cases.append(
            {
                "key": case.get("key"),
                "title": case.get("title"),
                "word_score": case.get("word_similarity"),
                "evidence": evidence,
            }
        )
    prompt = (
        "Rank how closely each case matches the user search. Use only these words "
        "and evidence. Return strict JSON as a list of objects with keys key and "
        "score, where score is between 0 and 1.\n\n"
        f"Search:\n{question}\n\nCases:\n{json.dumps(prompt_cases, ensure_ascii=True)}"
    )
    try:
        raw = _ask_ollama_for_case_scores(prompt)
        parsed = json.loads(raw)
    except Exception:
        return cases
    if not isinstance(parsed, list):
        return cases
    scores = {
        str(item.get("key")): float(item.get("score"))
        for item in parsed
        if isinstance(item, dict) and item.get("key") is not None
    }
    if not scores:
        return cases
    for case in cases:
        key = str(case.get("key") or "")
        if key in scores:
            llm_similarity = max(0.0, min(scores[key], 1.0))
            word_similarity = float(case.get("word_similarity") or 0.0)
            case["llm_similarity"] = llm_similarity
            case["best_similarity"] = (word_similarity * 0.45) + (llm_similarity * 0.55)
    return sorted(
        cases,
        key=lambda case: float(case["best_similarity"]),
        reverse=True,
    )


def _ask_ollama_for_case_scores(prompt: str) -> str:
    payload = json.dumps(
        {
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 500},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("response", ""))


def _chunk_access_level(chunk: RetrievedChunk) -> int:
    case_access_level = data_case_access_level(chunk.file_name or chunk.source or "")
    if case_access_level is not None:
        return case_access_level
    return int(chunk.metadata.get("access_level") or 1)


def _case_can_be_opened(case: dict[str, object]) -> bool:
    access_level = int(case.get("access_level") or 1)
    return access_level < 99 and access_level <= _requester_access_level()


def _case_collaborators(chunk: RetrievedChunk) -> list[dict[str, object]]:
    metadata_collaborators = chunk.metadata.get("collaborators")
    if isinstance(metadata_collaborators, list):
        collaborators = []
        for value in metadata_collaborators:
            if isinstance(value, dict):
                collaborators.append(value)
                continue
            employee = _employee_by_name(str(value))
            if employee:
                collaborators.append(_employee_profile(employee))
            elif str(value).strip():
                collaborators.append(
                    {
                        "name": str(value).strip(),
                        "email": "unknown",
                        "department": "Unknown",
                        "access_level": None,
                    }
                )
        return collaborators
    access_level = _chunk_access_level(chunk)
    if access_level >= 99:
        return [
            {
                "name": "Legal Archive Team",
                "email": "legal-archive@six-demo.local",
                "department": "Legal",
                "access_level": 99,
            },
            {
                "name": "Compliance Lead",
                "email": "compliance-lead@six-demo.local",
                "department": "Compliance",
                "access_level": 3,
            },
        ]
    employees = [
        employee
        for employee in DEMO_EMPLOYEES
        if employee.access_level >= access_level
    ]
    if not employees:
        employees = DEMO_EMPLOYEES[-1:]
    return [_employee_profile(employee) for employee in employees]


def _employee_by_name(name: str) -> EmployeeAccount | None:
    normalized = _normalize_login_value(name)
    for employee in DEMO_EMPLOYEES:
        if normalized in {
            _normalize_login_value(employee.full_name),
            _normalize_login_value(employee.username),
            _normalize_login_value(employee.email),
        }:
            return employee
    return None


def _employee_profile(employee: EmployeeAccount) -> dict[str, object]:
    return {
        "name": employee.full_name,
        "email": employee.email,
        "department": employee.department,
        "access_level": employee.access_level,
        "username": employee.username,
    }


def _remember_sent_mail(recipient: str, subject: str, message: str) -> None:
    sent_mail = st.session_state.setdefault("sent_contact_mail", [])
    if isinstance(sent_mail, list):
        sent_mail.append(
            {
                "recipient": recipient,
                "subject": subject,
                "message": message,
                "sender": _current_employee().full_name,
            }
        )


def _streamlit_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "item"


def _render_contact_profiles(collaborators: object, context_key: str) -> None:
    if not isinstance(collaborators, list) or not collaborators:
        return
    st.caption("Mitarbeiter kontaktieren:")
    columns = st.columns(min(len(collaborators), 3))
    for index, collaborator in enumerate(collaborators):
        if not isinstance(collaborator, dict):
            continue
        column = columns[index % len(columns)]
        name = str(collaborator.get("name") or "Unknown")
        with column:
            with st.popover(name, use_container_width=True, width=700):
                st.write(f"**{name}**")
                st.write(f"Department: {collaborator.get('department') or 'Unknown'}")
                level = collaborator.get("access_level")
                if level is not None:
                    st.write(f"Access level: {level}")
                recipient = str(collaborator.get("email") or "")
                if not recipient or recipient == "unknown":
                    st.info("No email address is available for this contact.")
                    continue
                form_key = _streamlit_key(
                    f"contact_mail_{context_key}_{index}_{recipient}"
                )
                with st.form(form_key):
                    subject = st.text_input(
                        "Subject",
                        value="Question about a similar case",
                    )
                    default_message = f"Dear {name.split()[0]},\n\n\nBest,"
                    message = st.text_area(
                        "Message",
                        value=default_message,
                        placeholder="Add your question or follow-up here.",
                        height=150,
                    )
                    submitted = st.form_submit_button("Send Mail", type="primary")
                if submitted:
                    if not subject.strip() or message.strip() == default_message.strip():
                        st.error("Please add a subject and message.")
                    else:
                        _remember_sent_mail(recipient, subject.strip(), message.strip())
                        st.success(f"Mail sent to {name}.")


def _retrieve_similar_case_chunks(
    question: str,
    expert_choice: str,
) -> list[RetrievedChunk]:
    selected_expert = expert_for_ui_choice(expert_choice)
    if _use_local_knowledge_store():
        chunks = _local_knowledge_store().retrieve(
            question,
            expert=selected_expert,
            top_k=5000,
            requester_access_level=_requester_access_level(),
            include_inaccessible=True,
        )
    else:
        chunks = _document_store().keyword_search_documents(
            query=question,
            top_k=5000,
            expert=selected_expert,
            requester_user_id=_requester_user_id(),
            include_inaccessible=True,
        )
    case_chunks = [chunk for chunk in chunks if _is_case_chunk(chunk)]
    return case_chunks


def _retrieve_regulation_source_chunks(
    case_title: str,
    case_chunks: list[RetrievedChunk],
    expert_choice: str,
) -> list[RetrievedChunk]:
    query = _build_regulation_source_query(case_title, case_chunks)
    selected_expert = expert_for_ui_choice(expert_choice)
    try:
        if _use_local_knowledge_store():
            chunks = _local_knowledge_store().retrieve(
                query,
                expert=selected_expert,
                top_k=24,
                requester_access_level=_requester_access_level(),
            )
        else:
            chunks = _document_store().keyword_search_documents(
                query=query,
                top_k=24,
                expert=selected_expert,
                requester_user_id=_requester_user_id(),
            )
    except Exception:
        logging.exception("Failed to retrieve regulation source chunks")
        return []
    seen: set[tuple[str | None, int | None]] = set()
    regulation_chunks: list[RetrievedChunk] = []
    for chunk in chunks:
        if not _is_regulation_source_chunk(chunk):
            continue
        key = (chunk.file_name or chunk.source, chunk.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        regulation_chunks.append(chunk)
        if len(regulation_chunks) >= 5:
            break
    return regulation_chunks


def _build_regulation_source_query(
    case_title: str,
    case_chunks: list[RetrievedChunk],
) -> str:
    evidence_text = "\n".join(chunk.content[:2500] for chunk in case_chunks[:3])
    sections = _extract_case_sections_for_ui(evidence_text)
    regulation_text = sections.get("regulatory_requirements") or sections.get("regulations") or ""
    if not regulation_text:
        regulation_text = evidence_text[:2000]
    tokens = sorted(_case_match_tokens(f"{case_title} {regulation_text}"))
    return " ".join(["regulation", "source", *tokens[:28]])


def _extract_case_sections_for_ui(content: str) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"(?im)^\s*(problem|decision|reasoning|regulatory requirements|regulations used|risks)\s*:?\s*$",
            content,
        )
    )
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1).strip().lower().replace(" ", "_")
        if key == "regulations_used":
            key = "regulatory_requirements"
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        value = content[start:end].strip()
        if value:
            sections[key] = value
    return sections


def _remember_case_overview(
    question: str,
    cases: list[dict[str, object]],
) -> None:
    st.session_state.case_overview = {
        "question": question,
        "cases": cases,
    }
    st.session_state.pop("case_result", None)


def _return_to_case_overview() -> None:
    st.session_state.pop("case_result", None)


def _close_open_case() -> None:
    _return_to_case_overview()


def _render_case_overview(
    overview: dict[str, object],
    expert_choice: str,
    exclude_key: str | None = None,
    heading: str = "Similar Cases",
) -> None:
    cases = overview.get("cases", [])
    question = str(overview.get("question", ""))

    st.subheader(heading)
    if not isinstance(cases, list) or not cases:
        st.info("No similar cases were found in the indexed documents.")
        return

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            continue
        key = str(case.get("key") or "")
        if exclude_key and key == exclude_key:
            continue
        chunks = case.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        title = str(case.get("title") or "Untitled case")
        if title.lower() == "uploads" and chunks:
            title = _case_title(chunks[0])
        similarity = _similarity_percent(float(case.get("best_similarity", 0.0)))
        access_level = int(case.get("access_level") or 1)
        collaborators = case.get("collaborators") or []
        can_open = _case_can_be_opened(case)

        title_column, match_column, action_column = st.columns([0.62, 0.18, 0.20])
        with title_column:
            st.markdown(f"**{index}. {title}**")
            if chunks:
                source = chunks[0].file_name or chunks[0].source or "Unknown source"
                st.caption(source)
            st.caption(f"Required access: {access_label(access_level)} (level {access_level})")
            contact_context_key = _streamlit_key(f"{heading}_{key or title}_{index}")
            _render_contact_profiles(collaborators, contact_context_key)
            if not can_open:
                st.caption("Locked: bitte einen der Mitarbeiter kontaktieren.")
        with match_column:
            st.metric("Match", f"{similarity}%")
        with action_column:
            if st.button(
                "Open",
                key=_streamlit_key(f"select_case_{heading}_{key}_{index}"),
                use_container_width=True,
                disabled=not can_open,
            ):
                open_question = _build_open_case_question(question, title)
                with st.spinner("Opening selected case..."):
                    regulation_chunks = _retrieve_regulation_source_chunks(
                        title,
                        chunks,
                        expert_choice,
                    )
                    answer_chunks = [*chunks, *regulation_chunks]
                    answer = _generate_company_brain_answer(
                        open_question,
                        answer_chunks,
                        expert_choice,
                        answer_style="case_open",
                    )
                _remember_result(
                    question,
                    answer,
                    answer_chunks,
                    mode="case",
                    extra={
                        "case_key": key,
                        "case_title": title,
                        "regulation_chunks": regulation_chunks,
                    },
                )
                _persist_chat_message("user", question)
                _persist_chat_message("assistant", answer.answer, answer_chunks)
                st.rerun()

        if can_open:
            with st.expander("Preview evidence"):
                _render_evidence(chunks[:3])


def _session_id() -> str:
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
    return st.session_state.chat_session_id


def _chunk_sources(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "file_name": chunk.file_name,
            "page_number": chunk.page_number,
            "sheet_name": chunk.sheet_name,
            "heading": chunk.heading,
            "similarity": chunk.similarity,
            "access_level": chunk.metadata.get("access_level"),
        }
        for chunk in chunks
    ]


def _persist_chat_message(
    role: str,
    content: str,
    chunks: list[RetrievedChunk] | None = None,
) -> None:
    if _uses_ollama_backend():
        return
    try:
        _document_store().insert_chat_message(
            session_id=_session_id(),
            role=role,
            content=content,
            sources=_chunk_sources(chunks or []),
            metadata={
                "employee_user_id": _current_employee().user_id,
                "employee_name": _current_employee().full_name,
                "employee_access_level": _current_employee().access_level,
            },
        )
    except Exception:
        logging.exception("Failed to persist chat message")


def _ingest_uploaded_file(
    uploaded_file,
    expert_choice: str,
    access_level: int,
    access_tag: str,
    collaborators: list[dict[str, object]],
) -> UploadResult:
    selected_expert = expert_for_ui_choice(expert_choice)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getbuffer())
        local_chunks = 0
        if _use_local_upload_store():
            local_chunks = _local_knowledge_store().ingest_file(
                temp_path,
                expert=selected_expert,
                topic=Path(uploaded_file.name).stem,
                chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
                overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
                access_level=access_level,
                access_tag=access_tag,
                collaborators=collaborators,
            )
        if not _supabase_is_configured():
            return UploadResult(
                local_chunks=local_chunks,
                database_error=", ".join(_database_environment_issues()),
            )
        try:
            database_chunks = _ingestion_pipeline().ingest_file(
                temp_path,
                expert=selected_expert,
                topic=Path(uploaded_file.name).stem,
                replace_existing=True,
                access_level=access_level,
                access_tag=access_tag,
                collaborators=collaborators,
            )
        except Exception as exc:
            logging.exception("Failed to persist uploaded file to Supabase")
            return UploadResult(
                local_chunks=local_chunks,
                database_error=str(exc),
            )
        return UploadResult(
            local_chunks=local_chunks,
            database_chunks=database_chunks,
        )


def _render_result(
    answer: GeneratedAnswer,
    chunks: list[RetrievedChunk],
    show_chunks: bool,
) -> None:
    st.markdown(_format_answer(answer))

    if answer.decision_trail:
        st.markdown("### Decision Trail")
        st.markdown(answer.decision_trail)

    if show_chunks:
        _render_evidence(chunks)


def _render_evidence(chunks: list[RetrievedChunk]) -> None:
    st.markdown("### Retrieved Evidence")
    if not chunks:
        st.info("No matching evidence was found in the indexed documents.")
        return
    for chunk in chunks:
        title = f"{chunk.file_name or chunk.source} | similarity {chunk.similarity:.3f}"
        with st.expander(title):
            st.write(chunk.content)
            st.caption(
                f"Topic: {chunk.topic or 'Unknown'} | "
                f"Chunk: {chunk.chunk_index if chunk.chunk_index is not None else 'Unknown'}"
            )


def _render_setup_check(missing_env: list[str]) -> bool:
    if _uses_ollama_backend():
        return True

    if not missing_env:
        return True

    return True


def _upload_signature(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"


def _render_upload_panel(is_configured: bool, expert_choice: str) -> None:
    st.markdown("### Add Case")
    st.caption("Choose case files to add them to Company Brain.")
    if _use_local_upload_store():
        st.caption("Upload destination: local knowledge store")
    else:
        st.caption("Upload destination: Supabase")
    selected_access_level = _requester_access_level()
    selected_access_tag = access_tag(selected_access_level)
    employee_options = {
        f"{employee.full_name} ({employee.department})": employee
        for employee in _employee_accounts()
    }
    current_employee = _current_employee()
    default_people = [
        label
        for label, employee in employee_options.items()
        if employee.user_id == current_employee.user_id
    ]
    selected_people = st.multiselect(
        "Associated people",
        list(employee_options.keys()),
        default=default_people,
    )
    selected_collaborators = [
        _employee_profile(employee_options[label])
        for label in selected_people
        if label in employee_options
    ]
    uploaded_files = st.file_uploader(
        "Case files",
        type=["pdf", "docx", "xlsx", "txt", "md", "csv"],
        accept_multiple_files=True,
        disabled=False,
        label_visibility="collapsed",
    )
    if "indexed_uploads" not in st.session_state:
        st.session_state.indexed_uploads = set()

    pending_files = [
        uploaded_file
        for uploaded_file in uploaded_files or []
        if _upload_signature(uploaded_file) not in st.session_state.indexed_uploads
    ]
    if pending_files:
        total_chunks = 0
        database_saved = 0
        database_errors: list[str] = []
        progress = st.progress(0)
        for index, uploaded_file in enumerate(pending_files, start=1):
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                result = _ingest_uploaded_file(
                    uploaded_file,
                    expert_choice,
                    selected_access_level,
                    selected_access_tag,
                    selected_collaborators,
                )
                total_chunks += result.indexed_chunks
                if result.database_saved:
                    database_saved += 1
                elif result.database_error:
                    database_errors.append(f"{uploaded_file.name}: {result.database_error}")
                st.session_state.indexed_uploads.add(_upload_signature(uploaded_file))
            progress.progress(index / len(pending_files))
        st.success(f"Indexed {total_chunks} chunks from {len(pending_files)} file(s).")
        if database_saved:
            st.success(f"Saved {database_saved} file(s) to Supabase.")
        if database_errors:
            st.warning(
                "Indexed locally, but not every file was saved to Supabase. "
                "Check the database credentials and schema."
            )
            for error in database_errors:
                st.caption(error)


def _stop_ollama_generation() -> None:
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    try:
        subprocess.run(
            ["ollama", "stop", model],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        st.success("Stopped the local Ollama model.")
    except Exception as exc:
        st.error(f"Could not stop Ollama: {exc}")


def _remember_result(
    question: str,
    answer: GeneratedAnswer,
    chunks: list[RetrievedChunk],
    mode: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    st.session_state.last_question = question
    st.session_state.last_answer = answer
    st.session_state.last_chunks = chunks
    if mode:
        result = {
            "question": question,
            "answer": answer,
            "chunks": chunks,
        }
        if extra:
            result.update(extra)
        st.session_state[f"{mode}_result"] = result


def _render_reasoning_panel() -> None:
    st.markdown("### Reasoning")
    question = st.session_state.get("last_question")
    answer = st.session_state.get("last_answer")
    chunks = st.session_state.get("last_chunks", [])
    if not question or not answer:
        st.caption("Ask through the case guide to see how Company Brain got there.")
        return

    st.caption("Question")
    st.write(question)
    st.caption("Model basis")
    st.write(
        "The answer was generated from the retrieved chunks below. "
        "Company Brain does not use information outside these indexed uploads."
    )
    st.caption("Confidence")
    st.write(answer.confidence)

    st.markdown("#### Evidence Trail")
    if not chunks:
        st.info("No evidence was retrieved.")
        return
    for index, chunk in enumerate(chunks[:5], start=1):
        title = f"{index}. {chunk.file_name or chunk.source} ({chunk.similarity:.3f})"
        with st.expander(title):
            st.write(chunk.content)
            st.caption(
                f"Expert: {chunk.expert or 'Company Brain'} | "
                f"Topic: {chunk.topic or 'Unknown'} | "
                f"Chunk: {chunk.chunk_index}"
            )


def _render_answer_block(
    answer: GeneratedAnswer,
    chunks: list[RetrievedChunk],
    case_title: str | None = None,
    regulation_chunks: list[RetrievedChunk] | None = None,
) -> None:
    if case_title:
        title_column, close_column = st.columns([0.78, 0.22])
        with title_column:
            st.subheader(case_title)
        with close_column:
            if st.button("Close", key="close_open_case", use_container_width=True):
                _close_open_case()
                st.rerun()
    rendered_answer = _format_open_case_answer(
        answer.answer,
        chunks,
        regulation_chunks or [],
    )
    st.markdown(rendered_answer)
    if case_title:
        with st.expander("Regulation Sources"):
            if regulation_chunks:
                _render_evidence(regulation_chunks)
            else:
                st.info("No separate regulation source file was found for this case.")
    with st.expander("Sources and Evidence"):
        source_text = ", ".join(answer.sources) if answer.sources else "No sources found"
        st.caption(f"Sources: {source_text}")
        st.caption(f"Confidence: {answer.confidence}")
        _render_evidence(chunks)


def _format_open_case_answer(
    answer_text: str,
    chunks: list[RetrievedChunk],
    regulation_chunks: list[RetrievedChunk],
) -> str:
    answer_text = _ensure_regulation_source_section(
        answer_text,
        chunks,
        regulation_chunks,
    )
    return re.sub(r"(?m)^###\s+", "##### ", answer_text)


def _ensure_regulation_source_section(
    answer_text: str,
    chunks: list[RetrievedChunk],
    regulation_chunks: list[RetrievedChunk] | None = None,
) -> str:
    if "### Regulations Used" not in answer_text:
        return answer_text
    source_text = _regulation_source_text(regulation_chunks or [])
    if "### Regulation Source" in answer_text:
        return re.sub(
            r"(?s)### Regulation Source\s*\n.*?(?=\n\n### Risks|\n### Risks|$)",
            f"### Regulation Source\n{source_text}",
            answer_text,
            count=1,
        )
    section = f"\n\n### Regulation Source\n{source_text}\n\n"
    if "### Risks" in answer_text:
        return answer_text.replace("\n\n### Risks", section + "### Risks", 1)
    return answer_text + section


def _regulation_source_text(regulation_chunks: list[RetrievedChunk]) -> str:
    sources = []
    for chunk in regulation_chunks:
        source = chunk.file_name or chunk.source
        if source and source not in sources:
            sources.append(source)
    if sources:
        return "\n".join(f"- {source}" for source in sources[:5])
    return (
        "No separate regulation source file was found. "
        "The opened case file is intentionally not used as its own regulation source."
    )


def _render_plain_answer(answer: GeneratedAnswer, chunks: list[RetrievedChunk]) -> None:
    st.markdown(answer.answer)
    source_text = ", ".join(answer.sources) if answer.sources else "No sources found"
    st.caption(f"Sources: {source_text}")
    st.caption(f"Confidence: {answer.confidence}")
    with st.expander("Retrieved Evidence"):
        _render_evidence(chunks)


def _stored_result(mode: str) -> dict[str, object] | None:
    result = st.session_state.get(f"{mode}_result")
    return result if isinstance(result, dict) else None


def _render_main_mode_selector() -> str:
    if "main_mode_selector" not in st.session_state:
        st.session_state.main_mode_selector = "Case Guide"

    options = [
        ("Case Guide", "mode_case_guide"),
        ("Quick Question", "mode_quick_question"),
        ("Add Case", "mode_add_case"),
    ]
    columns = st.columns(3, gap="small")
    for column, (label, key) in zip(columns, options):
        with column:
            selected = st.session_state.main_mode_selector == label
            if st.button(
                label,
                key=key,
                type="primary" if selected else "secondary",
                use_container_width=True,
            ):
                st.session_state.main_mode_selector = label
                st.rerun()
    st.markdown('<div class="mode-selector-spacer"></div>', unsafe_allow_html=True)
    return str(st.session_state.main_mode_selector)


if not _is_authenticated():
    _render_login_gate()
    st.stop()


_render_top_brand()
st.markdown(
    (
        '<div class="app-title-block">'
        "<h1>Company Brain</h1>"
        "<p>Case-first access to past decisions, regulations, reasoning, and risks.</p>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

is_configured = _render_setup_check(_missing_environment())
expert_choice = "Ask Company Brain"
top_k = 8

case_result = _stored_result("case")

if case_result:
    left_column, main_column = st.columns([0.33, 0.67], gap="large")
    with left_column:
        _render_reasoning_panel()
else:
    main_column = st.container()

with main_column:
    _render_brand_strip()
    mode = _render_main_mode_selector()

    if mode == "Case Guide":
        st.subheader("Case Guide")

        with st.form("guided_case_form"):
            keyword = st.text_input(
                "Keyword or topic",
                placeholder="Example: tax evasion, FATCA, MiFID, ESG disclosure",
            )
            situation = st.text_area(
                "Describe the situation",
                placeholder=(
                    "Example: A client setup looks like it might avoid tax reporting. "
                    "I want to know how similar cases were handled before."
                ),
                height=150,
            )
            submitted = st.form_submit_button(
                "Find Relevant Cases",
                type="primary",
                disabled=not is_configured,
            )

        if submitted and is_configured:
            question = _build_guided_question(
                keyword=keyword,
                situation=situation,
            )
            with st.spinner("Searching similar cases..."):
                try:
                    chunks = _retrieve_similar_case_chunks(
                        question,
                        expert_choice,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    _remember_case_overview(
                        question,
                        _build_similar_cases(question, chunks),
                    )

        if case_result:
            _render_answer_block(
                case_result["answer"],
                case_result["chunks"],
                case_title=str(case_result.get("case_title") or ""),
                regulation_chunks=case_result.get("regulation_chunks")
                if isinstance(case_result.get("regulation_chunks"), list)
                else [],
            )
            case_overview = st.session_state.get("case_overview")
            if isinstance(case_overview, dict):
                _render_case_overview(
                    case_overview,
                    expert_choice,
                    exclude_key=str(case_result.get("case_key") or ""),
                    heading="Other Similar Cases",
                )
        else:
            case_overview = st.session_state.get("case_overview")
            if isinstance(case_overview, dict):
                _render_case_overview(case_overview, expert_choice)

    elif mode == "Quick Question":
        st.subheader("Quick Question")
        st.write("Ask directly when you already know the exact question.")

        with st.form("quick_question_form"):
            direct_question = st.text_area(
                "Question",
                placeholder=(
                    "Example: What are the FATCA risks mentioned in the indexed "
                    "documents?"
                ),
                height=150,
            )
            ask_column, stop_column = st.columns([0.58, 0.42])
            with ask_column:
                direct_submitted = st.form_submit_button(
                    "Ask Company Brain",
                    type="primary",
                    disabled=not is_configured,
                    use_container_width=True,
                )
            with stop_column:
                stop_submitted = st.form_submit_button(
                    "Stop Generating",
                    use_container_width=True,
                )

        if stop_submitted:
            _stop_ollama_generation()
        elif direct_submitted and is_configured:
            if not direct_question.strip():
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Searching indexed knowledge and grounding an answer..."):
                    try:
                        _persist_chat_message("user", direct_question)
                        answer, chunks = _run_company_brain(
                            direct_question,
                            expert_choice,
                            max(top_k, 12),
                            answer_style="plain",
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        _remember_result(direct_question, answer, chunks, mode="quick")
                        _persist_chat_message("assistant", answer.answer, chunks)

        quick_result = _stored_result("quick")
        if quick_result:
            _render_plain_answer(
                quick_result["answer"],
                quick_result["chunks"],
            )

    else:
        _render_upload_panel(is_configured, expert_choice)
