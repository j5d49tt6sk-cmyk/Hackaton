from __future__ import annotations

import re

from company_brain.models import DocumentChunk, ExtractedSection


def split_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", clean) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            chunks.extend(_split_long_text(paragraph, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return _add_overlap(chunks, overlap, chunk_size)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_sections(
    sections: list[ExtractedSection],
    document_id: str,
    document_text_id: str | None,
    expert: str | None,
    topic: str | None,
    chunk_size: int = 1200,
    overlap: int = 180,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for section_index, section in enumerate(sections):
        heading_prefix = f"{section.heading}\n\n" if section.heading else ""
        section_chunks = split_text(
            f"{heading_prefix}{section.content}",
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for section_chunk_index, content in enumerate(section_chunks):
            chunks.append(
                DocumentChunk(
                    content=content,
                    document_id=document_id,
                    document_text_id=document_text_id,
                    chunk_index=len(chunks),
                    expert=expert,
                    topic=topic,
                    page_number=section.page_number,
                    sheet_name=section.sheet_name,
                    heading=section.heading,
                    metadata={
                        **section.metadata,
                        "section_index": section_index,
                        "section_chunk_index": section_chunk_index,
                        "character_count": len(content),
                    },
                )
            )
    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _add_overlap(chunks: list[str], overlap: int, chunk_size: int) -> list[str]:
    if overlap == 0 or len(chunks) <= 1:
        return chunks

    merged: list[str] = [chunks[0]]
    for index in range(1, len(chunks)):
        prefix = chunks[index - 1][-overlap:].strip()
        candidate = f"{prefix}\n\n{chunks[index]}".strip()
        merged.append(candidate[-chunk_size:])
    return merged
