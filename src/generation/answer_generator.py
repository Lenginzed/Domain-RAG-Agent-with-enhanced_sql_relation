from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.generation.llm import OllamaLLM
from src.generation.prompts import build_rag_prompt


def generate_answer(question: str, documents: list[Document]) -> dict[str, Any]:
    prompt = build_rag_prompt(question, documents)
    answer = OllamaLLM().generate(prompt)
    return {
        "answer": answer,
        "sources": format_sources(documents),
    }


def format_sources(documents: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        chunk_id = str(metadata.get("chunk_id", ""))
        key = chunk_id or f"{metadata.get('source')}:{index}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "label": f"S{index}",
                "source": metadata.get("source", "unknown_source"),
                "chunk_id": chunk_id,
                "section": metadata.get("section", "default"),
                "score": metadata.get("score"),
            }
        )
    return sources
