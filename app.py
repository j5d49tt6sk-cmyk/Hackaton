from __future__ import annotations

import importlib.util
import logging
import os
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO)


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

from company_brain.answering import AnswerGenerator
from company_brain.config import Settings
from company_brain.ingestion import IngestionPipeline
from company_brain.local_knowledge import LocalKnowledgeStore
from company_brain.models import GeneratedAnswer, RetrievedChunk
from company_brain.ollama_answering import OllamaAnswerGenerator
from company_brain.retrieval import EXPERT_OPTIONS, Retriever, expert_for_ui_choice
from company_brain.supabase_store import SupabaseDocumentStore


st.set_page_config(page_title="Company Brain", layout="wide")


CASE_TYPES = [
    "Find a similar past case",
    "Check regulatory requirements",
    "Understand risks",
    "Reconstruct why a decision was made",
    "Prepare an escalation or handover",
]

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


def _entered_password_is_valid() -> bool:
    expected_password = os.getenv("COMPANY_BRAIN_PASSWORD", "realesthacks")
    entered_password = st.session_state.get("company_brain_password", "")
    return entered_password == expected_password


def _is_authenticated() -> bool:
    return bool(st.session_state.get("company_brain_authenticated"))


def _render_password_gate() -> None:
    st.title("Company Brain")
    st.caption("Protected access for the hackathon demo.")

    with st.form("password_form"):
        st.text_input(
            "Password",
            type="password",
            key="company_brain_password",
            placeholder="Enter access password",
        )
        submitted = st.form_submit_button("Unlock", type="primary")

    if submitted:
        if _entered_password_is_valid():
            st.session_state.company_brain_authenticated = True
            st.rerun()
        else:
            st.error("Wrong password.")


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
    return [
        name
        for name in REQUIRED_ENV_VARS
        if not os.getenv(name) or _looks_like_placeholder(os.getenv(name, ""))
    ]


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
    case_type: str,
    keyword: str,
    business_area: str,
    situation: str,
    known_regulation: str,
    focus: list[str],
) -> str:
    focus_text = ", ".join(focus) if focus else ", ".join(OUTPUT_FOCUS)
    parts = [
        "Search the company knowledge base for relevant historical cases.",
        f"Task: {case_type}.",
        f"Business area: {business_area}.",
        f"Keyword or topic: {keyword or 'not provided'}.",
        f"Situation: {situation or 'not provided'}.",
        f"Known regulation or policy: {known_regulation or 'not provided'}.",
        (
            "Return the answer as a practical case card with Problem, Decision, "
            "Reasoning, Regulatory Requirement, Risks, Similar Cases or Evidence, "
            f"and Sources. Emphasize: {focus_text}."
        ),
    ]
    return "\n".join(parts)


def _run_company_brain(
    question: str,
    expert_choice: str,
    top_k: int,
) -> tuple[GeneratedAnswer, list[RetrievedChunk]]:
    selected_expert = expert_for_ui_choice(expert_choice)
    if _uses_ollama_backend():
        chunks = _local_knowledge_store().retrieve(
            question,
            expert=selected_expert,
            top_k=top_k,
        )
        generated = _ollama_answer_generator().generate(
            question,
            chunks,
            selected_expert,
        )
        return generated, chunks

    chunks = _retriever().retrieve(question, expert=selected_expert, top_k=top_k)
    generated = _answer_generator().generate(question, chunks, selected_expert)
    return generated, chunks


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
        )
    except Exception:
        logging.exception("Failed to persist chat message")


def _ingest_uploaded_file(uploaded_file, expert_choice: str) -> int:
    selected_expert = expert_for_ui_choice(expert_choice)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getbuffer())
        if _uses_ollama_backend():
            return _local_knowledge_store().ingest_file(
                temp_path,
                expert=selected_expert,
                topic=Path(uploaded_file.name).stem,
                chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
                overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
            )
        return _ingestion_pipeline().ingest_file(
            temp_path,
            expert=selected_expert,
            topic=Path(uploaded_file.name).stem,
            replace_existing=True,
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
            st.json(
                {
                    "expert": chunk.expert,
                    "topic": chunk.topic,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
            )


def _render_sidebar() -> tuple[str, int, bool]:
    with st.sidebar:
        st.header("Search Settings")
        expert_choice = st.radio("Knowledge area", list(EXPERT_OPTIONS.keys()))
        top_k = st.slider("Evidence depth", min_value=3, max_value=15, value=8)
        show_chunks = st.toggle("Show retrieved evidence", value=False)
        st.divider()
        st.caption(
            "Use Company Brain for broad search, or choose an expert when the "
            "question clearly belongs to one knowledge area."
        )
    return expert_choice, top_k, show_chunks


def _render_setup_check(missing_env: list[str]) -> bool:
    if _uses_ollama_backend():
        st.info(
            "Running in local Ollama mode. Uploads are indexed locally and answers "
            "are generated by Ollama instead of OpenAI."
        )
        return True

    if not missing_env:
        return True

    st.warning("Company Brain is open, but the backend is not configured yet.")
    st.write(
        "The questionnaire UI is ready. To make the search work, add the missing "
        "credentials and ingest the SIX documents once."
    )
    st.code(
        "cp .env.example .env\n"
        "# then fill in OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY\n"
        "python3 ingest.py SIX_Hack_Zurich\n"
        "python3 -m streamlit run app.py --server.port 8501",
        language="bash",
    )
    st.write("Missing configuration:")
    for name in missing_env:
        st.markdown(f"- `{name}`")
    return False


if not _is_authenticated():
    _render_password_gate()
    st.stop()


expert_choice, top_k, show_chunks = _render_sidebar()

st.title("Company Brain")
st.caption("Guided search for past decisions, regulations, reasoning, and risks.")

is_configured = _render_setup_check(_missing_environment())

with st.expander("Upload Documents", expanded=False):
    st.write(
        "Upload scripts, transcripts, PDFs, Word documents, spreadsheets, or CSVs. "
        "After indexing, Direct Question will answer from this uploaded knowledge."
    )
    uploaded_files = st.file_uploader(
        "Documents",
        type=["pdf", "docx", "xlsx", "txt", "md", "csv"],
        accept_multiple_files=True,
        disabled=not is_configured,
    )
    if st.button("Upload and Index", disabled=not is_configured or not uploaded_files):
        total_chunks = 0
        progress = st.progress(0)
        for index, uploaded_file in enumerate(uploaded_files or [], start=1):
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                total_chunks += _ingest_uploaded_file(uploaded_file, expert_choice)
            progress.progress(index / len(uploaded_files))
        st.success(f"Indexed {total_chunks} chunks from {len(uploaded_files)} file(s).")

guided_tab, direct_tab = st.tabs(["Guided Case Finder", "Direct Question"])

with guided_tab:
    st.subheader("Case Questionnaire")
    st.write(
        "Fill in what you know. The app will turn it into a structured search "
        "and return a case-style answer with decision logic and sources."
    )

    with st.form("guided_case_form"):
        case_type = st.selectbox("What do you need?", CASE_TYPES)
        business_area = st.selectbox("Which area sounds closest?", BUSINESS_AREAS)
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
            height=140,
        )
        known_regulation = st.text_input(
            "Known regulation or policy, if any",
            placeholder="Example: FATCA, MiFID II, SFDR, internal tax navigator",
        )
        focus = st.multiselect(
            "What should the answer focus on?",
            OUTPUT_FOCUS,
            default=OUTPUT_FOCUS,
        )
        submitted = st.form_submit_button(
            "Find Relevant Cases",
            type="primary",
            disabled=not is_configured,
        )

    if submitted and is_configured:
        question = _build_guided_question(
            case_type=case_type,
            keyword=keyword,
            business_area=business_area,
            situation=situation,
            known_regulation=known_regulation,
            focus=focus,
        )
        with st.spinner("Searching indexed knowledge and building a case card..."):
            try:
                _persist_chat_message("user", question)
                answer, chunks = _run_company_brain(question, expert_choice, top_k)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                _render_result(answer, chunks, show_chunks)
                _persist_chat_message("assistant", answer.answer, chunks)

with direct_tab:
    st.subheader("Ask Uploaded Knowledge")
    st.write(
        "Ask a question and the AI will search the indexed uploads first, then answer "
        "only from the retrieved evidence with sources."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about the uploaded scripts, documents, decisions, or risks",
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not is_configured:
                response = (
                    "I can take your question, but the AI backend is not configured "
                    "yet. Add the OpenAI and Supabase values in `.env`, then upload "
                    "and index documents before asking."
                )
                st.warning(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.stop()

            _persist_chat_message("user", question)
            with st.spinner("Searching indexed knowledge and grounding an answer..."):
                try:
                    answer, chunks = _run_company_brain(question, expert_choice, top_k)
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    response = _format_answer(answer)
                    st.markdown(response)

                    if answer.decision_trail:
                        st.markdown("### Decision Trail")
                        st.markdown(answer.decision_trail)

                    if show_chunks:
                        _render_evidence(chunks)

                    full_response = response
                    if answer.decision_trail:
                        full_response += f"\n\n### Decision Trail\n{answer.decision_trail}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    _persist_chat_message("assistant", full_response, chunks)
