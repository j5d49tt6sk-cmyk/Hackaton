from __future__ import annotations

import json
import logging
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from company_brain.config import Settings
from company_brain.models import GeneratedAnswer, RetrievedChunk


logger = logging.getLogger(__name__)


class _AnswerPayload(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: str
    decision_trail: Optional[str] = None


class AnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self._use_openai = settings.use_openai
        self._client = OpenAI(api_key=settings.openai_api_key) if self._use_openai else None
        self._model = settings.answer_model

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        expert: str | None = None,
        answer_style: str = "case",
    ) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer=(
                    "I could not find relevant information in the indexed documents."
                ),
                sources=[],
                confidence="Low",
            )

        if not self._use_openai:
            return _generate_extractive_answer(chunks)

        if self._client is None:
            raise RuntimeError("OpenAI answering is enabled, but no client exists.")
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _system_prompt(answer_style)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "selected_expert": expert or "Company Brain",
                            "retrieved_context": [_chunk_to_context(c) for c in chunks],
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            payload = _AnswerPayload.model_validate_json(raw)
        except ValidationError:
            logger.exception("Model returned invalid answer payload: %s", raw)
            payload = _AnswerPayload(
                answer=(
                    "I found relevant documents, but could not format a reliable "
                    "answer. Please inspect the retrieved sources."
                ),
                sources=_unique_sources(chunks),
                confidence="Low",
            )

        return GeneratedAnswer(
            answer=payload.answer.strip(),
            sources=payload.sources or _unique_sources(chunks),
            confidence=_normalize_confidence(payload.confidence),
            decision_trail=payload.decision_trail,
        )


def _system_prompt(answer_style: str = "case") -> str:
    base_prompt = (
        "You are Company Brain, an evidence-based organizational knowledge system. "
        "The retrieved_context contains excerpts from uploaded scripts, transcripts, "
        "documents, and spreadsheets. Answer only from the retrieved_context provided "
        "by the user. Never invent facts, numbers, decisions, or sources. If the "
        "answer is missing from the context, say that the indexed uploads do not "
        "contain enough information. "
        "The user is often looking for historical cases where former employees solved "
        "similar problems. "
    )
    if answer_style == "case_open":
        section_prompt = (
            "When a specific case is opened, structure the answer with exactly these "
            "Markdown sections in this order: Problem, Decision, Reasoning, "
            "Regulations Used, Risks. Keep Risks as the final section. Cite file "
            "names in the sources list. Use confidence High only when the selected "
            "case directly contains the decision, reasoning, and regulations, "
            "Medium when evidence is partial, and Low when evidence is weak or "
            "missing. Return strict JSON with keys answer, sources, confidence, "
            "decision_trail."
        )
    else:
        section_prompt = (
            "When evidence is available, structure the answer as a case card with "
            "these Markdown sections: Problem, Decision, Reasoning, Regulatory "
            "Requirement, Risks, and Similar Cases or Evidence. Cite file names in "
            "the sources list. Use confidence High only when several strong chunks "
            "directly answer the question, Medium when evidence is partial but "
            "useful, and Low when evidence is weak or missing. If the context "
            "contains decisions, alternatives, reasoning, or outcomes, include a "
            "decision_trail field with the sections Problem, Decision, Reasoning, "
            "Regulatory Requirement, Risks, and Outcome. Return strict JSON with "
            "keys answer, sources, confidence, decision_trail."
        )
    return base_prompt + section_prompt


def _chunk_to_context(chunk: RetrievedChunk) -> dict[str, object]:
    return {
        "id": chunk.id,
        "content": chunk.content,
        "file_name": chunk.file_name,
        "source": chunk.source,
        "expert": chunk.expert,
        "topic": chunk.topic,
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number,
        "sheet_name": chunk.sheet_name,
        "heading": chunk.heading,
        "similarity": round(chunk.similarity, 4),
        "metadata": chunk.metadata,
    }


def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        source = chunk.file_name or chunk.source
        if source and source not in sources:
            sources.append(source)
    return sources


def _normalize_confidence(value: str) -> str:
    normalized = value.strip().title()
    return normalized if normalized in {"High", "Medium", "Low"} else "Low"


def _generate_extractive_answer(chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    evidence = []
    for index, chunk in enumerate(chunks[:5], start=1):
        source = chunk.file_name or chunk.source or "Unknown source"
        location_parts = []
        if chunk.page_number:
            location_parts.append(f"page {chunk.page_number}")
        if chunk.sheet_name:
            location_parts.append(f"sheet {chunk.sheet_name}")
        if chunk.heading:
            location_parts.append(chunk.heading)
        location = f" ({', '.join(location_parts)})" if location_parts else ""
        excerpt = chunk.content.strip().replace("\n", " ")
        if len(excerpt) > 600:
            excerpt = excerpt[:597].rstrip() + "..."
        evidence.append(f"{index}. **{source}{location}**\n\n{excerpt}")

    return GeneratedAnswer(
        answer=(
            "OpenAI is disabled, so I cannot generate a synthesized answer. "
            "Here are the most relevant indexed excerpts I found:\n\n"
            + "\n\n".join(evidence)
        ),
        sources=_unique_sources(chunks),
        confidence="Medium" if len(chunks) >= 3 else "Low",
    )
