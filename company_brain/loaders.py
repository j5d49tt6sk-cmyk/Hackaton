from __future__ import annotations

import csv
import logging
from pathlib import Path

from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".txt", ".md", ".csv", ".docx"}

logger = logging.getLogger(__name__)


def iter_supported_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".xlsx":
        return _read_xlsx(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".docx":
        return _read_docx(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {index}]\n{text}")
        except Exception as exc:  # pragma: no cover - defensive against malformed PDFs
            logger.warning("Failed to extract page %s from %s: %s", index, path, exc)
    return "\n\n".join(pages)


def _read_xlsx(path: Path) -> str:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sections: list[str] = []
    for sheet in workbook.worksheets:
        rows: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            values = [str(cell).strip() for cell in row if cell is not None]
            if values:
                rows.append(" | ".join(values))
        if rows:
            sections.append(f"[Sheet: {sheet.title}]\n" + "\n".join(rows))
    workbook.close()
    return "\n\n".join(sections)


def _read_csv(path: Path) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
        reader = csv.reader(handle, dialect)
        for row in reader:
            values = [value.strip() for value in row if value.strip()]
            if values:
                lines.append(" | ".join(values))
    return "\n".join(lines)


def _read_docx(path: Path) -> str:
    document = Document(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
