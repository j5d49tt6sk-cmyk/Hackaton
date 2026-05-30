from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from company_brain.models import GeneratedAnswer, RetrievedChunk


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

        missing_exact_term_answer = _answer_if_exact_term_missing(question, chunks)
        if missing_exact_term_answer:
            return missing_exact_term_answer

        prompt = _build_prompt(question, chunks, expert, answer_style)
        answer = self._generate(prompt).strip()
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
    for index, chunk in enumerate(chunks[:5], start=1):
        content = chunk.content[:1200]
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
            "Answer in plain text in 3 to 6 concise sentences. Do not use case-card "
            "section headings like Problem, Decision, Reasoning, Regulatory "
            "Requirement, or Risks. Mention uncertainty clearly if the evidence is "
            "thin."
        )
    else:
        answer_instruction = (
            "Structure useful answers with Problem, Decision, Reasoning, Regulatory "
            "Requirement, Risks, and Sources when those sections fit the question."
        )

    return (
        "You are Company Brain. Answer only from the evidence below. "
        "Do not add general knowledge about regulations, products, dates, or laws. "
        "If the evidence is insufficient or does not mention the user's exact term, "
        f"say so. {answer_instruction}\n\n"
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


def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        source = chunk.file_name or chunk.source
        if source and source not in sources:
            sources.append(source)
    return sources
