from __future__ import annotations

from langchain_core.documents import Document

from src.chunkers.text_chunker import chunk_documents


def test_chunker_adds_chunk_id() -> None:
    document = Document(
        page_content="敌机意图预测用于根据轨迹和战术动作推测可能行为。" * 80,
        metadata={
            "source": "data/raw/notes/demo.md",
            "doc_type": "paper",
            "domain": "multi_uav_air_combat",
            "title": "demo",
            "section": "intro",
            "chunk_id": "",
            "language": "zh",
            "is_citable": True,
        },
    )

    chunks = chunk_documents([document])

    assert chunks
    assert all(chunk.metadata.get("chunk_id") for chunk in chunks)
