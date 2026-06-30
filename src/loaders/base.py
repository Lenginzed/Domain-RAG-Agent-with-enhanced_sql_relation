from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from src.loaders.code_loader import load_python_file
from src.loaders.csv_loader import load_csv_file
from src.loaders.pdf_loader import load_pdf_file
from src.loaders.text_loader import load_text_file
from src.loaders.yaml_loader import load_yaml_file


logger = logging.getLogger(__name__)

LoaderFunc = Callable[[Path], list[Document]]

LOADER_BY_EXTENSION: dict[str, LoaderFunc] = {
    ".txt": load_text_file,
    ".md": load_text_file,
    ".markdown": load_text_file,
    ".pdf": load_pdf_file,
    ".py": load_python_file,
    ".yaml": load_yaml_file,
    ".yml": load_yaml_file,
    ".csv": load_csv_file,
}

SUPPORTED_EXTENSIONS = set(LOADER_BY_EXTENSION)


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    loader = LOADER_BY_EXTENSION.get(suffix)
    if loader is None:
        logger.warning("Skipping unsupported file type: %s", path)
        return []

    try:
        documents = loader(path)
        logger.info("Loaded %s document(s) from %s", len(documents), path)
        return documents
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop ingestion.
        logger.error("Failed to load %s: %s", path, exc)
        return []


def is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
