from __future__ import annotations

from src.metadata.schema import CORE_METADATA_FIELDS
from src.metadata.validators import normalize_metadata


def test_metadata_validator_fills_core_fields() -> None:
    metadata = normalize_metadata({"source": "demo.txt", "doc_type": "text"})

    for field in CORE_METADATA_FIELDS:
        assert field in metadata

    assert metadata["source"] == "demo.txt"
    assert metadata["doc_type"] == "text"
    assert metadata["is_citable"] is True
