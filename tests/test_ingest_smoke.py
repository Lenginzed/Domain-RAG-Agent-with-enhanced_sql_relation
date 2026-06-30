from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ingest_documents import ingest
from scripts.make_demo_data import create_demo_data


@pytest.mark.smoke
def test_ingest_smoke() -> None:
    create_demo_data()
    stats = ingest(collection_name="domain_rag_v1_test", reset=True, write_log=False)

    assert stats["loaded_documents"] > 0
    assert stats["generated_chunks"] > 0
    assert stats["indexed_chunks"] >= stats["generated_chunks"]
    assert Path("storage/chroma").exists()
