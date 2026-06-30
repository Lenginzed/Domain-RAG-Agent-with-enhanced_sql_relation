from __future__ import annotations

import ast
import logging
from pathlib import Path

from langchain_core.documents import Document

from src.loaders.utils import base_metadata, detect_language


logger = logging.getLogger(__name__)


def load_python_file(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    names, parse_warning = _extract_top_level_names(text, path)
    section = ", ".join(names) if names else path.stem
    extra: dict[str, object] = {"file_role": "unknown"}
    if parse_warning:
        extra["parse_warning"] = parse_warning
    metadata = base_metadata(
        path,
        doc_type="code",
        title=path.stem,
        section=section,
        language=detect_language(text),
        is_citable=True,
        **extra,
    )
    return [Document(page_content=text, metadata=metadata)]


def _extract_top_level_names(text: str, path: Path) -> tuple[list[str], str | None]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        warning = f"Python AST parse failed: {exc}"
        logger.warning("%s in %s", warning, path)
        return [], warning

    names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    return names, None
