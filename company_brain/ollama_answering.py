from __future__ import annotations

import json
import urllib.error
import urllib.request

from company_brain.models import GeneratedAnswer, RetrievedChunk


class OllamaAnswerGenerator:
    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        expert: str | None = None,
    ) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer="I could not find relevant information in the indexed uploads.",
                sources=[],
                confidence="Low",
            )

        prompt = _build_prompt(question, chunks, expert)
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
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start it with `ollama serve` and pull "
                f"the model with `ollama pull {self._model}`."
            ) from exc
        return str(data.get("response", ""))


def _build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    expert: str | None,
) -> str:
    context = []
    for index, chunk in enumerate(chunks, start=1):
        context.append(
            "\n".join(
                [
                    f"[Evidence {index}]",
                    f"File: {chunk.file_name or chunk.source or 'Unknown'}",
                    f"Expert: {chunk.expert or expert or 'Company Brain'}",
                    f"Topic: {chunk.topic or 'Unknown'}",
                    chunk.content,
                ]
            )
        )
    return (
        "You are Company Brain. Answer only from the evidence below. "
        "If the evidence is insufficient, say so. Structure useful answers with "
        "Problem, Decision, Reasoning, Regulatory Requirement, Risks, and Sources "
        "when those sections fit the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{chr(10).join(context)}"
    )


def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        source = chunk.file_name or chunk.source
        if source and source not in sources:
            sources.append(source)
    return sources
