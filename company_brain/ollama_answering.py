from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request

from company_brain.models import GeneratedAnswer, RetrievedChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")
QUESTION_STOP_WORDS = {
    "about",
    "and",
    "are",
    "can",
    "does",
    "down",
    "for",
    "from",
    "how",
    "tell",
    "the",
    "what",
    "when",
    "where",
    "which",
    "with",
}


class OllamaAnswerGenerator:
    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 45,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        expert: str | None = None,
        answer_style: str = "case",
    ) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer="I could not find relevant information in the indexed uploads.",
                sources=[],
                confidence="Low",
            )
        if answer_style == "plain":
            chunks = _rerank_chunks_for_question(question, chunks)
        if answer_style == "plain" and _evidence_is_not_relevant(question, chunks):
            return GeneratedAnswer(
                answer=(
                    "I could not find relevant information about that in the "
                    "indexed company cases or documents."
                ),
                sources=[],
                confidence="Low",
            )

        missing_exact_term_answer = _answer_if_exact_term_missing(question, chunks)
        if missing_exact_term_answer:
            return missing_exact_term_answer

        prompt = _build_prompt(question, chunks, expert, answer_style)
        answer = self._generate(prompt).strip()
        if (
            not answer
            or _looks_like_refusal(answer)
            or _ollama_is_unreachable(answer)
            or _looks_like_weak_plain_answer(answer, answer_style, chunks)
        ):
            return _fallback_answer_from_evidence(chunks, answer_style)
        return GeneratedAnswer(
            answer=answer,
            sources=_unique_sources(chunks),
            confidence="Medium" if answer else "Low",
        )

    def _generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 350,
                    "num_ctx": 4096,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except socket.timeout as exc:
            raise RuntimeError(
                "Ollama took too long to answer. Try a more specific question or "
                "reduce Evidence Depth."
            ) from exc
        except urllib.error.URLError as exc:
            return (
                "Ollama is not reachable, so I cannot synthesize a local model "
                "answer yet. I found relevant evidence; inspect the retrieved "
                "sources below. To enable local generation, start Ollama with "
                f"`ollama serve` and pull the model with `ollama pull {self._model}`."
            )
        return str(data.get("response", ""))


def _build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    expert: str | None,
    answer_style: str,
) -> str:
    context = []
    max_chunks = 8 if answer_style == "plain" else 5
    max_chars = 950 if answer_style == "plain" else 1200
    for index, chunk in enumerate(chunks[:max_chunks], start=1):
        content = chunk.content[:max_chars]
        context.append(
            "\n".join(
                [
                    f"[Evidence {index}]",
                    f"File: {chunk.file_name or chunk.source or 'Unknown'}",
                    f"Expert: {chunk.expert or expert or 'Company Brain'}",
                    f"Topic: {chunk.topic or 'Unknown'}",
                    content,
                ]
            )
        )
    if answer_style == "plain":
        answer_instruction = (
            "Write a direct answer for a business user in plain text. Use 4 to 7 "
            "sentences. Start with the answer, then explain the evidence. Include "
            "specific names of regulations, controls, risks, decisions, or cases "
            "that appear in the evidence. If evidence conflicts, say what conflicts. "
            "Do not use headings, Markdown tables, or generic disclaimers. Do not "
            "answer from general knowledge."
        )
    elif answer_style == "case_open":
        answer_instruction = (
            "Return exactly these Markdown sections in this order: Problem, "
            "Decision, Reasoning, Regulations Used, Regulation Source, Risks. In "
            "Regulation Source, cite only a separate regulation/reference evidence "
            "file where the regulation reference appears; never cite the opened "
            "case file itself. Keep Risks as the final section. Use only the "
            "selected evidence and do not add general knowledge."
        )
    else:
        answer_instruction = (
            "Structure useful answers with Problem, Decision, Reasoning, Regulatory "
            "Requirement, Risks, and Sources when those sections fit the question."
        )

    return (
        "You are Company Brain, an internal knowledge assistant. Your job is to "
        "turn retrieved internal evidence into a useful answer. Use only the "
        "evidence below; do not add general knowledge about regulations, products, "
        "dates, or laws. If the evidence is insufficient, say exactly what is "
        f"missing. {answer_instruction}\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{chr(10).join(context)}"
    )


def _answer_if_exact_term_missing(
    question: str,
    chunks: list[RetrievedChunk],
) -> GeneratedAnswer | None:
    lowered_question = question.lower()
    if "micar" not in lowered_question:
        return None
    combined = "\n".join(chunk.content.lower() for chunk in chunks)
    if "micar" in combined:
        return None
    return GeneratedAnswer(
        answer=(
            "I did not find an exact MiCAR reference in the indexed uploads. "
            "The closest retrieved evidence mentions crypto regulations, digital "
            "assets, regulatory navigation, and related compliance topics. Based on "
            "the uploaded evidence, I cannot reliably state MiCAR-specific "
            "requirements."
        ),
        sources=_unique_sources(chunks),
        confidence="Low",
    )


def _evidence_is_not_relevant(
    question: str,
    chunks: list[RetrievedChunk],
) -> bool:
    best_similarity = max((chunk.similarity for chunk in chunks), default=0.0)
    if best_similarity >= 0.52:
        return False

    query_tokens = _meaningful_tokens(question)
    if not query_tokens:
        return False

    evidence_text = " ".join(
        " ".join(
            [
                chunk.file_name or "",
                chunk.topic or "",
                chunk.heading or "",
                chunk.content[:1200],
            ]
        )
        for chunk in chunks[:5]
    )
    evidence_tokens = _meaningful_tokens(evidence_text)
    overlap_ratio = len(query_tokens & evidence_tokens) / len(query_tokens)
    return overlap_ratio < 0.75


def _rerank_chunks_for_question(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    query_tokens = _meaningful_tokens(question)
    if not query_tokens:
        return chunks

    def score(chunk: RetrievedChunk) -> float:
        title_tokens = _meaningful_tokens(
            " ".join([chunk.file_name or "", chunk.topic or "", chunk.heading or ""])
        )
        content_tokens = _meaningful_tokens(chunk.content[:1600])
        title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
        content_overlap = len(query_tokens & content_tokens) / len(query_tokens)
        exact_phrase_bonus = 0.2 if _compact(question) in _compact(chunk.content[:2200]) else 0.0
        return chunk.similarity + (title_overlap * 0.35) + (content_overlap * 0.25) + exact_phrase_bonus

    return sorted(chunks, key=score, reverse=True)


def _meaningful_tokens(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in TOKEN_PATTERN.finditer(text)
        if match.group(0).lower() not in QUESTION_STOP_WORDS
    }


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    refusal_markers = [
        "i can't assist",
        "i cannot assist",
        "i'm sorry, but i can't",
        "i am sorry, but i can't",
    ]
    return any(marker in lowered for marker in refusal_markers)


def _looks_like_weak_plain_answer(
    answer: str,
    answer_style: str,
    chunks: list[RetrievedChunk],
) -> bool:
    if answer_style != "plain" or not chunks:
        return False
    lowered = answer.lower()
    weak_markers = [
        "i don't know",
        "i do not know",
        "cannot determine",
        "not enough information",
        "no relevant information",
        "unable to answer",
    ]
    formatting_drift = bool(re.search(r"(?m)^\s*\d+[\.)]\s+", answer)) or "**" in answer
    too_broad = len(answer.split()) > 180
    return (
        len(answer.split()) < 35
        or any(marker in lowered for marker in weak_markers)
        or formatting_drift
        or too_broad
    )


def _ollama_is_unreachable(answer: str) -> bool:
    return "ollama is not reachable" in answer.lower()


def _fallback_answer_from_evidence(
    chunks: list[RetrievedChunk],
    answer_style: str,
) -> GeneratedAnswer:
    if answer_style == "plain":
        answer = _plain_answer_from_evidence(chunks)
    elif answer_style == "case_open":
        sections = _extract_case_sections(chunks[0].content)
        problem = sections.get("problem", "Not specified in the selected case.")
        answer = "\n\n".join(
            [
                f"### Problem\n{problem}",
                (
                    "### Decision\n"
                    f"{sections.get('decision', 'Not specified in the selected case.')}"
                ),
                (
                    "### Reasoning\n"
                    f"{sections.get('reasoning', 'Not specified in the selected case.')}"
                ),
                (
                    "### Regulations Used\n"
                    f"{sections.get('regulatory_requirements', 'Not specified in the selected case.')}"
                ),
                f"### Regulation Source\n{_regulation_source(chunks)}",
                f"### Risks\n{sections.get('risks', 'Not specified in the selected case.')}",
            ]
        )
    else:
        sections = _extract_case_sections(chunks[0].content)
        answer = "\n\n".join(
            f"### {heading}\n{sections.get(key, 'Not specified in the selected case.')}"
            for key, heading in [
                ("problem", "Problem"),
                ("decision", "Decision"),
                ("reasoning", "Reasoning"),
                ("regulatory_requirements", "Regulatory Requirement"),
                ("risks", "Risks"),
            ]
        )
    return GeneratedAnswer(
        answer=answer,
        sources=_unique_sources(chunks),
        confidence="Medium",
    )


def _plain_answer_from_evidence(chunks: list[RetrievedChunk]) -> str:
    case_answer = _plain_case_answer(chunks)
    if case_answer:
        return case_answer

    selected_sentences: list[str] = []
    seen: set[str] = set()
    for chunk in chunks[:6]:
        for sentence in _evidence_sentences(chunk.content):
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected_sentences.append(sentence)
            if len(selected_sentences) >= 5:
                break
        if len(selected_sentences) >= 5:
            break

    if not selected_sentences:
        selected_sentences = [_clean_whitespace(chunks[0].content[:700])]

    source_names = _unique_sources(chunks[:4])
    source_text = ", ".join(source_names)
    return (
        "Based on the retrieved company evidence, "
        + " ".join(selected_sentences)
        + (f" The strongest sources are {source_text}." if source_text else "")
    )


def _regulation_source(chunks: list[RetrievedChunk]) -> str:
    sources = _unique_sources([chunk for chunk in chunks if not _is_case_chunk(chunk)])
    if not sources:
        return (
            "No separate regulation source file was found. "
            "The opened case file is intentionally not used as its own regulation source."
        )
    return ", ".join(sources[:3])


def _is_case_chunk(chunk: RetrievedChunk) -> bool:
    file_name = chunk.file_name or ""
    source = chunk.source or ""
    return file_name.startswith("Case_") or "data_cases/" in source


def _plain_case_answer(chunks: list[RetrievedChunk]) -> str | None:
    for chunk in chunks[:4]:
        sections = _extract_case_sections(chunk.content)
        if not sections:
            continue
        topic = chunk.topic or ""
        if topic.lower() in {"uploads", "cases"}:
            title = chunk.file_name or chunk.source or "the selected case"
        else:
            title = topic or chunk.file_name or "the selected case"
        if title.endswith((".pdf", ".docx", ".xlsx", ".txt", ".md", ".csv")):
            title = title.rsplit(".", 1)[0].replace("_", " ")
        problem = sections.get("problem")
        decision = sections.get("decision")
        reasoning = sections.get("reasoning")
        regulations = sections.get("regulatory_requirements")
        risks = sections.get("risks")
        parts = [f"The closest matching evidence is {title}."]
        if problem:
            parts.append(f"The problem was that {_sentence_fragment(problem)}.")
        if decision:
            parts.append(f"The recorded decision was to {_sentence_fragment(decision)}.")
        if reasoning:
            parts.append(f"The reasoning was that {_sentence_fragment(reasoning)}.")
        if regulations:
            parts.append(f"The regulations or requirements mentioned are {_clean_sentence(regulations)}.")
        if risks:
            parts.append(f"The main risks were {_sentence_fragment(risks)}.")
        return " ".join(parts)
    return None


def _sentence_fragment(value: str) -> str:
    cleaned = _clean_sentence(value)
    return cleaned[0].lower() + cleaned[1:] if cleaned else cleaned


def _clean_sentence(value: str) -> str:
    return _clean_whitespace(value).replace("- ", "-").rstrip(".")


def _evidence_sentences(content: str) -> list[str]:
    cleaned = _clean_whitespace(content)
    candidates = re.split(r"(?<=[.!?])\s+|(?:\s+-\s+)", cleaned)
    useful: list[str] = []
    for candidate in candidates:
        sentence = candidate.strip(" -")
        if len(sentence.split()) < 6:
            continue
        if len(sentence) > 320:
            sentence = sentence[:317].rsplit(" ", 1)[0] + "..."
        useful.append(sentence)
    return useful


def _extract_case_sections(content: str) -> dict[str, str]:
    labels = {
        "problem": "Problem",
        "regulatory_requirements": "Regulatory Requirements",
        "options_considered": "Options Considered",
        "decision": "Decision",
        "reasoning": "Reasoning",
        "risks": "Risks",
    }
    pattern = "|".join(re.escape(label) for label in labels.values())
    matches = list(re.finditer(rf"\b({pattern})\b", content))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        key = next(key for key, value in labels.items() if value == label)
        sections[key] = _clean_whitespace(content[start:end])
    return sections


def _clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        source = chunk.file_name or chunk.source
        if source and source not in sources:
            sources.append(source)
    return sources
