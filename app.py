from __future__ import annotations

import base64
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
from company_brain.retrieval import Retriever, expert_for_ui_choice
from company_brain.supabase_store import SupabaseDocumentStore


st.set_page_config(page_title="Company Brain", layout="wide")

_logo_path = Path(__file__).with_name("sixlogo.png")
_logo_background_css = ""
if _logo_path.exists():
    _logo_data = base64.b64encode(_logo_path.read_bytes()).decode("ascii")
    _logo_background_css = f"""
    .stApp::before {{
        content: "";
        position: fixed;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: min(156vw, 1960px);
        height: min(92vw, 1120px);
        background-image: url("data:image/png;base64,{_logo_data}");
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
        opacity: 0.09;
        z-index: 0;
        pointer-events: none;
    }}
    """
else:
    _logo_background_css = """
    .stApp::before {
        content: "SIX";
        position: fixed;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        font-size: min(38vw, 500px);
        font-weight: 800;
        letter-spacing: 0;
        color: rgba(230, 0, 18, 0.06);
        z-index: 0;
        pointer-events: none;
        line-height: 1;
    }
    """

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

    __LOGO_BACKGROUND_CSS__

    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """.replace("__LOGO_BACKGROUND_CSS__", _logo_background_css),
    unsafe_allow_html=True,
)


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
    if _uses_ollama_backend():
        return _local_knowledge_store().retrieve(
            question,
            expert=selected_expert,
            top_k=top_k,
        )

    return _retriever().retrieve(question, expert=selected_expert, top_k=top_k)


def _generate_company_brain_answer(
    question: str,
    chunks: list[RetrievedChunk],
    expert_choice: str,
    answer_style: str = "case",
) -> GeneratedAnswer:
    selected_expert = expert_for_ui_choice(expert_choice)
    if _uses_ollama_backend():
        return _ollama_answer_generator().generate(
            question,
            chunks,
            selected_expert,
            answer_style=answer_style,
        )
    return _answer_generator().generate(question, chunks, selected_expert)


def _case_key(chunk: RetrievedChunk) -> str:
    return chunk.file_name or chunk.source or f"chunk-{chunk.id}"


def _case_title(chunk: RetrievedChunk) -> str:
    title = chunk.topic or chunk.file_name or chunk.source or "Untitled case"
    if title.endswith((".pdf", ".docx", ".xlsx", ".txt", ".md", ".csv")):
        title = Path(title).stem
    return title.replace("_", " ").strip()


def _is_case_chunk(chunk: RetrievedChunk) -> bool:
    file_name = chunk.file_name or ""
    source = chunk.source or ""
    return file_name.startswith("Case_") or "data_cases/" in source


def _similarity_percent(value: float) -> int:
    return round(max(0.0, min(value, 1.0)) * 100)


def _build_similar_cases(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        key = _case_key(chunk)
        case = cases.setdefault(
            key,
            {
                "key": key,
                "title": _case_title(chunk),
                "best_similarity": chunk.similarity,
                "chunks": [],
            },
        )
        case["best_similarity"] = max(float(case["best_similarity"]), chunk.similarity)
        case_chunks = case["chunks"]
        if isinstance(case_chunks, list):
            case_chunks.append(chunk)

    for case in cases.values():
        case_chunks = case["chunks"]
        if isinstance(case_chunks, list):
            case_chunks.sort(key=lambda chunk: chunk.similarity, reverse=True)

    return sorted(
        cases.values(),
        key=lambda case: float(case["best_similarity"]),
        reverse=True,
    )


def _retrieve_similar_case_chunks(
    question: str,
    expert_choice: str,
) -> list[RetrievedChunk]:
    chunks = _retrieve_company_brain(
        question,
        expert_choice,
        top_k=5000 if _uses_ollama_backend() else 60,
    )
    case_chunks = [chunk for chunk in chunks if _is_case_chunk(chunk)]
    return case_chunks or chunks


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


def _render_case_overview(
    overview: dict[str, object],
    expert_choice: str,
) -> None:
    cases = overview.get("cases", [])
    question = str(overview.get("question", ""))

    st.subheader("Similar Cases")
    if not isinstance(cases, list) or not cases:
        st.info("No similar cases were found in the indexed documents.")
        return

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            continue
        chunks = case.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        title = str(case.get("title") or "Untitled case")
        similarity = _similarity_percent(float(case.get("best_similarity", 0.0)))

        title_column, match_column, action_column = st.columns([0.62, 0.18, 0.20])
        with title_column:
            st.markdown(f"**{index}. {title}**")
            if chunks:
                source = chunks[0].file_name or chunks[0].source or "Unknown source"
                st.caption(source)
        with match_column:
            st.metric("Match", f"{similarity}%")
        with action_column:
            if st.button("Select", key=f"select_case_{index}", use_container_width=True):
                with st.spinner("Building answer from the selected case..."):
                    answer = _generate_company_brain_answer(
                        question,
                        chunks,
                        expert_choice,
                        answer_style="case",
                    )
                _remember_result(question, answer, chunks, mode="case")
                _persist_chat_message("user", question)
                _persist_chat_message("assistant", answer.answer, chunks)
                st.rerun()

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


def _render_setup_check(missing_env: list[str]) -> bool:
    if _uses_ollama_backend():
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


def _upload_signature(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"


def _render_upload_panel(is_configured: bool, expert_choice: str) -> None:
    st.markdown("### Upload")
    st.caption("Files are chunked and added to Company Brain immediately.")
    uploaded_files = st.file_uploader(
        "Documents",
        type=["pdf", "docx", "xlsx", "txt", "md", "csv"],
        accept_multiple_files=True,
        disabled=not is_configured,
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
        progress = st.progress(0)
        for index, uploaded_file in enumerate(pending_files, start=1):
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                chunks = _ingest_uploaded_file(uploaded_file, expert_choice)
                total_chunks += chunks
                st.session_state.indexed_uploads.add(_upload_signature(uploaded_file))
            progress.progress(index / len(pending_files))
        st.success(f"Indexed {total_chunks} chunks from {len(pending_files)} file(s).")


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
) -> None:
    st.session_state.last_question = question
    st.session_state.last_answer = answer
    st.session_state.last_chunks = chunks
    if mode:
        st.session_state[f"{mode}_result"] = {
            "question": question,
            "answer": answer,
            "chunks": chunks,
        }


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


def _render_answer_block(answer: GeneratedAnswer, chunks: list[RetrievedChunk]) -> None:
    st.markdown(_format_answer(answer))
    if answer.decision_trail:
        st.markdown("### Decision Trail")
        st.markdown(answer.decision_trail)
    with st.expander("Retrieved Evidence"):
        _render_evidence(chunks)


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


if not _is_authenticated():
    _render_password_gate()
    st.stop()


st.title("Company Brain")
st.caption("Case-first access to past decisions, regulations, reasoning, and risks.")

is_configured = _render_setup_check(_missing_environment())
expert_choice = "Ask Company Brain"
top_k = 8

left_column, main_column = st.columns([0.33, 0.67], gap="large")

with left_column:
    _render_reasoning_panel()
    st.divider()
    _render_upload_panel(is_configured, expert_choice)

with main_column:
    mode = st.segmented_control(
        "Mode",
        ["Case Guide", "Quick Question"],
        default="Case Guide",
        label_visibility="collapsed",
    )

    if mode == "Case Guide":
        st.subheader("Case Guide")
        st.write(
            "Describe the issue in normal words. Company Brain will search indexed "
            "knowledge, build a case card, and show the evidence trail on the left."
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
                height=150,
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
            with st.spinner("Searching similar cases..."):
                try:
                    chunks = _retrieve_similar_case_chunks(
                        question,
                        expert_choice,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    _remember_case_overview(question, _build_similar_cases(chunks))

        case_result = _stored_result("case")
        if case_result:
            if st.button(
                "Back to Similar Cases",
                key="back_to_similar_cases",
                use_container_width=True,
            ):
                _return_to_case_overview()
                st.rerun()
            _render_answer_block(
                case_result["answer"],
                case_result["chunks"],
            )
        else:
            case_overview = st.session_state.get("case_overview")
            if isinstance(case_overview, dict):
                _render_case_overview(case_overview, expert_choice)

    else:
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
                            top_k,
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
