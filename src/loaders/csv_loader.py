from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from src.loaders.utils import base_metadata


logger = logging.getLogger(__name__)


def load_csv_file(path: Path) -> list[Document]:
    documents: list[Document] = []
    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - loader must not stop ingestion.
        logger.error("Failed to read CSV %s: %s", path, exc)
        return documents

    for row_index, row in dataframe.iterrows():
        content = _row_to_text(path, row_index, row)
        metadata = base_metadata(
            path,
            doc_type="experiment",
            title=path.stem,
            section=f"row_{row_index}",
            language="mixed",
            is_citable=True,
            row_index=int(row_index),
        )
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def _row_to_text(path: Path, row_index: int, row: pd.Series) -> str:
    lines = [
        f"CSV source: {path.name}",
        f"row_index: {row_index}",
    ]
    for column, value in row.items():
        lines.append(f"{column}: {value}")
    return "\n".join(lines)
