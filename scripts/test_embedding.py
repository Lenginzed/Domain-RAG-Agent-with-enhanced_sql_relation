from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"
DEFAULT_TEXTS = [
    "敌机意图预测",
    "opponent tactical behavior prediction",
]


def load_model_config() -> dict[str, Any]:
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {MODEL_CONFIG_PATH}")
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML object in {MODEL_CONFIG_PATH}")
    return config


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        if not body:
            return {}
        return json.loads(body)


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    embed_url = f"{base_url.rstrip('/')}/api/embed"
    try:
        response = post_json(embed_url, {"model": model, "input": texts})
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == len(texts):
            return embeddings
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise RuntimeError(f"Ollama /api/embed failed with HTTP {exc.code}: {exc.reason}") from exc
    except Exception as exc:  # noqa: BLE001 - preserve the clear environment error.
        raise RuntimeError(f"Ollama /api/embed failed: {exc}") from exc

    legacy_url = f"{base_url.rstrip('/')}/api/embeddings"
    embeddings: list[list[float]] = []
    for text in texts:
        try:
            response = post_json(legacy_url, {"model": model, "prompt": text})
            embedding = response.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(f"Unexpected response from {legacy_url}: {response}")
            embeddings.append(embedding)
        except Exception as exc:  # noqa: BLE001 - preserve the clear environment error.
            raise RuntimeError(f"Ollama /api/embeddings failed: {exc}") from exc
    return embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} vs {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")
    return dot / (left_norm * right_norm)


def main() -> int:
    try:
        config = load_model_config()
        base_url = str(config.get("ollama_base_url") or "http://localhost:11434")
        embedding_model = str(config.get("embedding_model") or "").strip()
        if not embedding_model:
            raise ValueError(
                "config/model.yaml has an empty embedding_model. "
                "Run scripts/check_ollama.py and pull qwen3-embedding or embeddinggemma if needed."
            )

        print("== Embedding smoke test ==")
        print(f"base_url: {base_url}")
        print(f"embedding_model: {embedding_model}")

        embeddings = embed_texts(base_url, embedding_model, DEFAULT_TEXTS)
        if len(embeddings) != 2:
            raise RuntimeError(f"Expected 2 embeddings, received {len(embeddings)}")

        first_dim = len(embeddings[0])
        second_dim = len(embeddings[1])
        similarity = cosine_similarity(embeddings[0], embeddings[1])

        print(f"text_1: {DEFAULT_TEXTS[0]}")
        print(f"vector_1_dim: {first_dim}")
        print(f"text_2: {DEFAULT_TEXTS[1]}")
        print(f"vector_2_dim: {second_dim}")
        print(f"cosine_similarity: {similarity:.6f}")
        return 0
    except Exception as exc:  # noqa: BLE001 - this script should explain failures directly.
        print("Embedding smoke test failed.")
        print(f"reason: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
