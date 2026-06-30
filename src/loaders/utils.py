from __future__ import annotations

import re
from pathlib import Path

from src.metadata.schema import DEFAULT_DOMAIN
from src.metadata.validators import normalize_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def relative_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def detect_language(text: str) -> str:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_ascii = bool(re.search(r"[A-Za-z]", text))
    if has_cjk and has_ascii:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_ascii:
        return "en"
    return "unknown"


def base_metadata(
    path: Path,
    *,
    doc_type: str,
    title: str | None = None,
    section: str = "default",
    language: str = "unknown",
    is_citable: bool = True,
    **extra: object,
) -> dict[str, object]:
    metadata = {
        "source": relative_source(path),
        "doc_type": doc_type,
        "domain": DEFAULT_DOMAIN,
        "title": title or path.stem,
        "section": section,
        "chunk_id": "pending",
        "language": language,
        "is_citable": is_citable,
    }
    metadata.update(extra)
    return normalize_metadata(metadata)
