from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.metadata.validators import normalize_metadata


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval.yaml"


def load_chunk_config(config_path: Path = RETRIEVAL_CONFIG_PATH) -> dict[str, int]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return {
        "chunk_size": int(config.get("chunk_size", 800)),
        "chunk_overlap": int(config.get("chunk_overlap", 120)),
    }


def chunk_documents(documents: list[Document], config_path: Path = RETRIEVAL_CONFIG_PATH) -> list[Document]:
    config = load_chunk_config(config_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )

    chunks: list[Document] = []
    for document in documents:
        base_metadata = normalize_metadata(document.metadata)
        split_texts = splitter.split_text(document.page_content)
        for chunk_index, text in enumerate(split_texts):
            metadata = dict(base_metadata)
            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = make_chunk_id(metadata, chunk_index)
            chunks.append(Document(page_content=text, metadata=normalize_metadata(metadata)))

    logger.info("Generated %s chunks from %s documents.", len(chunks), len(documents))
    return chunks


def make_chunk_id(metadata: dict[str, Any], chunk_index: int) -> str:
    key_parts = [
        str(metadata.get("source", "unknown_source")),
        str(metadata.get("section", "default")),
        str(metadata.get("page", "")),
        str(metadata.get("row_index", "")),
        str(chunk_index),
    ]
    digest = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"
