from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from src.loaders.utils import base_metadata, detect_language


logger = logging.getLogger(__name__)


def load_pdf_file(path: Path) -> list[Document]:
    documents: list[Document] = []
    try:
        reader = PdfReader(str(path))
        title = _pdf_title(reader) or path.stem
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                logger.warning("PDF page has no extractable text: %s page %s", path, page_index)
                continue
            metadata = base_metadata(
                path,
                doc_type="paper",
                title=title,
                section=f"page_{page_index}",
                language=detect_language(text),
                is_citable=True,
                page=page_index,
            )
            documents.append(Document(page_content=text, metadata=metadata))
    except Exception as exc:  # noqa: BLE001 - loader must not stop ingestion.
        logger.error("Failed to read PDF %s: %s", path, exc)
    return documents


def _pdf_title(reader: PdfReader) -> str | None:
    metadata = getattr(reader, "metadata", None)
    if not metadata:
        return None
    title = getattr(metadata, "title", None)
    return str(title).strip() if title else None
