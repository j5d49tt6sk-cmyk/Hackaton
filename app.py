from __future__ import annotations

import logging

import streamlit as st

from company_brain.answering import AnswerGenerator
from company_brain.config import Settings
from company_brain.retrieval import EXPERT_OPTIONS, Retriever, expert_for_ui_choice


logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Company Brain", layout="wide")


@st.cache_resource
def _settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def _retriever() -> Retriever:
    return Retriever(_settings())


@st.cache_resource
def _answer_generator() -> AnswerGenerator:
    return AnswerGenerator(_settings())


def _format_answer(answer: str, sources: list[str], confidence: str) -> str:
    source_lines = "\n".join(f"- {source}" for source in sources) or "- No sources found"
    return f"### Answer\n{answer}\n\n### Sources\n{source_lines}\n\n### Confidence\n{confidence}"


st.title("Company Brain")
st.caption("Evidence-based organizational memory for SIX knowledge domains.")

with st.sidebar:
    expert_choice = st.radio("Expert Twin", list(EXPERT_OPTIONS.keys()))
    top_k = st.slider("Retrieved chunks", min_value=3, max_value=15, value=8)
    show_chunks = st.toggle("Show retrieved evidence", value=True)

question = st.chat_input("Ask about decisions, regulations, onboarding, or expert context")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    selected_expert = expert_for_ui_choice(expert_choice)
    with st.chat_message("assistant"):
        with st.spinner("Searching indexed knowledge and grounding an answer..."):
            chunks = _retriever().retrieve(question, expert=selected_expert, top_k=top_k)
            generated = _answer_generator().generate(question, chunks, selected_expert)

        response = _format_answer(generated.answer, generated.sources, generated.confidence)
        st.markdown(response)

        if generated.decision_trail:
            st.markdown("### Decision Trail")
            st.markdown(generated.decision_trail)

        if show_chunks:
            st.markdown("### Retrieved Evidence")
            for chunk in chunks:
                with st.expander(
                    f"{chunk.file_name or chunk.source} · similarity {chunk.similarity:.3f}"
                ):
                    st.write(chunk.content)
                    st.json(
                        {
                            "expert": chunk.expert,
                            "topic": chunk.topic,
                            "chunk_index": chunk.chunk_index,
                            "metadata": chunk.metadata,
                        }
                    )

        full_response = response
        if generated.decision_trail:
            full_response += f"\n\n### Decision Trail\n{generated.decision_trail}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
