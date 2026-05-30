from __future__ import annotations

from pathlib import Path


EXPERT_KEYWORDS = {
    "Compliance Expert": ("mifid", "fatca", "sfdr", "compliance", "regulatory"),
    "ESG Expert": ("esg", "sustainability", "sustainable", "taxonomy", "climate"),
    "Internal Expert": ("meeting", "transcript", "minutes", "internal", "discussion"),
}


def infer_expert(path: Path, explicit_expert: str | None = None) -> str | None:
    if explicit_expert:
        return explicit_expert
    haystack = _path_haystack(path)
    for expert, keywords in EXPERT_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return expert
    return None


def infer_topic(path: Path, explicit_topic: str | None = None) -> str | None:
    if explicit_topic:
        return explicit_topic
    parts = [part for part in path.parts if part and part not in {".", ".."}]
    if len(parts) >= 2:
        return _humanize(parts[-2])
    return _humanize(path.stem)


def _path_haystack(path: Path) -> str:
    return " ".join(path.parts).replace("_", " ").replace("-", " ").lower()


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()

