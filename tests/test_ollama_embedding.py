from __future__ import annotations

import pytest

from src.embeddings.ollama_embedder import OllamaEmbedder


@pytest.mark.smoke
def test_ollama_embedding_smoke() -> None:
    embedder = OllamaEmbedder()
    vector = embedder.embed_query("敌机意图预测")

    assert isinstance(vector, list)
    assert len(vector) == 4096
