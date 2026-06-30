from __future__ import annotations

import logging
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings.ollama_embedder import OllamaEmbedder


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"
DEFAULT_COLLECTION_NAME = "domain_rag_v1"


def chroma_settings(persist_directory: Path) -> Settings:
    return Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=str(persist_directory),
    )


def build_index(
    documents: list[Document],
    *,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = True,
) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    embedding_function = OllamaEmbedder()

    if reset:
        _delete_collection_if_exists(persist_directory, collection_name, embedding_function)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
        client_settings=chroma_settings(persist_directory),
    )

    if documents:
        ids = [str(doc.metadata["chunk_id"]) for doc in documents]
        vector_store.add_documents(documents=documents, ids=ids)
        persist = getattr(vector_store, "persist", None)
        if callable(persist):
            persist()
    logger.info("Indexed %s chunks into Chroma collection '%s'.", len(documents), collection_name)
    return vector_store


def load_index(
    *,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=OllamaEmbedder(),
        client_settings=chroma_settings(persist_directory),
    )


def similarity_search(
    query: str,
    *,
    top_k: int = 5,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[Document]:
    vector_store = load_index(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    results = vector_store.similarity_search_with_score(query, k=top_k)
    documents: list[Document] = []
    for document, score in results:
        document.metadata["score"] = float(score)
        documents.append(document)
    return documents


def collection_count(vector_store: Chroma) -> int:
    try:
        return int(vector_store._collection.count())  # noqa: SLF001 - Chroma exposes count here.
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read Chroma collection count: %s", exc)
        return 0


def _delete_collection_if_exists(
    persist_directory: Path,
    collection_name: str,
    embedding_function: OllamaEmbedder,
) -> None:
    try:
        existing = Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
            client_settings=chroma_settings(persist_directory),
        )
        existing.delete_collection()
    except Exception as exc:  # noqa: BLE001 - collection may not exist yet.
        logger.debug("No existing Chroma collection '%s' to delete: %s", collection_name, exc)
