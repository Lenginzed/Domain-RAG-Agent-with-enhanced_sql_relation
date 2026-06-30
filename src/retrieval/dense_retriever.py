from __future__ import annotations

import logging
from pathlib import Path

import yaml
from langchain_core.documents import Document

from src.indexes.vector_index import DEFAULT_COLLECTION_NAME, load_index


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval.yaml"


class DenseRetriever:
    def __init__(
        self,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        config_path: Path = RETRIEVAL_CONFIG_PATH,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = _load_top_k(config_path)
        self.vector_store = load_index(collection_name=collection_name)

    def retrieve(self, query: str) -> list[Document]:
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        documents: list[Document] = []
        for document, score in results:
            document.metadata["score"] = float(score)
            logger.info(
                "Dense hit source=%s chunk_id=%s section=%s score=%s",
                document.metadata.get("source"),
                document.metadata.get("chunk_id"),
                document.metadata.get("section"),
                score,
            )
            print(
                "[dense hit] "
                f"source={document.metadata.get('source')} "
                f"chunk_id={document.metadata.get('chunk_id')} "
                f"section={document.metadata.get('section')} "
                f"score={float(score):.6f}"
            )
            documents.append(document)
        return documents


def _load_top_k(config_path: Path) -> int:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return int(config.get("top_k", 5))
