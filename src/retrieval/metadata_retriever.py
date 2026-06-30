from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.indexes.vector_index import DEFAULT_COLLECTION_NAME, CHROMA_DIR, load_index


FILENAME_SCORE = 5.0
PATH_SCORE = 3.0
CATEGORY_SCORE = 2.0
DOC_TYPE_SCORE = 2.0
TITLE_SECTION_SCORE = 2.0
CHUNK_ID_SCORE = 1.0
MAX_METADATA_SCORE = 30.0


class MetadataRetriever:
    def __init__(
        self,
        *,
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.vector_store = load_index(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def retrieve(
        self,
        question: str,
        classification: dict[str, Any],
        *,
        top_k: int = 5,
    ) -> list[Document]:
        keywords = normalize_keywords(
            list(classification.get("keywords", []))
            + list(classification.get("target_categories", []))
            + [question]
        )
        raw = self.vector_store._collection.get(include=["documents", "metadatas"])  # noqa: SLF001
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        scored: list[Document] = []
        for content, metadata in zip(documents, metadatas, strict=False):
            metadata = dict(metadata or {})
            score, reasons = metadata_match_score(metadata, keywords)
            if score <= 0:
                continue
            metadata["metadata_score"] = round(min(score, MAX_METADATA_SCORE), 4)
            metadata["match_reason"] = ";".join(reasons[:12])
            metadata["retrieval_source"] = "metadata"
            scored.append(Document(page_content=str(content or ""), metadata=metadata))

        scored.sort(
            key=lambda doc: (
                -float(doc.metadata.get("metadata_score", 0)),
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("chunk_id", "")),
            )
        )
        return scored[:top_k]


def metadata_match_score(metadata: dict[str, Any], keywords: list[str]) -> tuple[float, list[str]]:
    source = str(metadata.get("source") or "")
    imported_source = str(metadata.get("imported_source") or "")
    original_source = str(metadata.get("original_source") or "")
    title = str(metadata.get("title") or "")
    section = str(metadata.get("section") or "")
    doc_type = str(metadata.get("doc_type") or "")
    category_dir = str(metadata.get("category_dir") or "")
    chunk_id = str(metadata.get("chunk_id") or "")

    filename_text = " ".join([Path(source).name, Path(imported_source).name, Path(original_source).name]).lower()
    path_text = " ".join([source, imported_source, original_source]).lower()
    title_section_text = " ".join([title, section]).lower()
    category_text = category_dir.lower()
    doc_type_text = doc_type.lower()
    chunk_id_text = chunk_id.lower()

    score = 0.0
    reasons: list[str] = []
    for keyword in keywords:
        key = keyword.lower().strip()
        if not key:
            continue
        if key in filename_text:
            score += FILENAME_SCORE
            reasons.append(f"filename:{key}")
        if key in path_text:
            score += PATH_SCORE
            reasons.append(f"path:{key}")
        if key in category_text:
            score += CATEGORY_SCORE
            reasons.append(f"category:{key}")
        if key in doc_type_text:
            score += DOC_TYPE_SCORE
            reasons.append(f"doc_type:{key}")
        if key in title_section_text:
            score += TITLE_SECTION_SCORE
            reasons.append(f"title_section:{key}")
        if key in chunk_id_text:
            score += CHUNK_ID_SCORE
            reasons.append(f"chunk_id:{key}")
        if score >= MAX_METADATA_SCORE:
            return MAX_METADATA_SCORE, reasons
    return score, reasons


def normalize_keywords(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if not text or len(text) > 120:
            continue
        if text not in normalized:
            normalized.append(text)
    return normalized
