from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from src.loaders.utils import base_metadata, detect_language


MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def load_text_file(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    title = _extract_markdown_title(text) if suffix in MARKDOWN_EXTENSIONS else path.stem
    doc_type = "paper" if suffix in MARKDOWN_EXTENSIONS else "text"
    metadata = base_metadata(
        path,
        doc_type=doc_type,
        title=title,
        section=title or path.stem,
        language=detect_language(text),
        is_citable=True,
    )
    return [Document(page_content=text, metadata=metadata)]


def _extract_markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return None
