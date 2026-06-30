from __future__ import annotations

import logging
from pathlib import Path

import yaml
from langchain_core.documents import Document

from src.loaders.utils import base_metadata, detect_language


logger = logging.getLogger(__name__)


def load_yaml_file(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    extra: dict[str, object] = {}
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        warning = f"YAML parse failed: {exc}"
        logger.warning("%s in %s", warning, path)
        extra["parse_warning"] = warning

    metadata = base_metadata(
        path,
        doc_type="config",
        title=path.stem,
        section="yaml_config",
        language=detect_language(text),
        is_citable=True,
        **extra,
    )
    return [Document(page_content=text, metadata=metadata)]
