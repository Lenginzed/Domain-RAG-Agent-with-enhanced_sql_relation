from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_ollama import OllamaEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"
DEFAULT_EMBEDDING_BATCH_SIZE = 16


class OllamaEmbedder:
    def __init__(self, config_path: Path = MODEL_CONFIG_PATH) -> None:
        config = _load_model_config(config_path)
        self.model = str(config.get("embedding_model") or "").strip()
        self.base_url = str(config.get("ollama_base_url") or "http://localhost:11434").strip()
        if not self.model:
            raise ValueError(
                "config/model.yaml has an empty embedding_model. "
                "Run scripts/check_ollama.py and pull qwen3-embedding or embeddinggemma if needed."
            )
        self.client = OllamaEmbeddings(model=self.model, base_url=self.base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings: list[list[float]] = []
            for start in range(0, len(texts), DEFAULT_EMBEDDING_BATCH_SIZE):
                batch = texts[start : start + DEFAULT_EMBEDDING_BATCH_SIZE]
                embeddings.extend(self.client.embed_documents(batch))
            return embeddings
        except Exception as exc:  # noqa: BLE001 - expose local model/API failures clearly.
            raise RuntimeError(
                f"Failed to embed documents with Ollama model '{self.model}' at {self.base_url}: {exc}"
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.client.embed_query(text)
        except Exception as exc:  # noqa: BLE001 - expose local model/API failures clearly.
            raise RuntimeError(
                f"Failed to embed query with Ollama model '{self.model}' at {self.base_url}: {exc}"
            ) from exc


def _load_model_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid model config: {config_path}")
    return config
