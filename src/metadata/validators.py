from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from src.metadata.schema import CORE_METADATA_FIELDS, METADATA_DEFAULTS


logger = logging.getLogger(__name__)


def normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return metadata with all required fields present.

    Missing core fields are filled with conservative defaults and logged as
    warnings. This keeps ingestion moving while making metadata quality issues
    visible during development.
    """

    normalized: dict[str, Any] = dict(metadata or {})
    for field in CORE_METADATA_FIELDS:
        value = normalized.get(field)
        if value is None or value == "":
            normalized[field] = METADATA_DEFAULTS[field]
            logger.warning(
                "Metadata missing required field '%s'; using default '%s'.",
                field,
                METADATA_DEFAULTS[field],
            )

    normalized["source"] = str(normalized["source"])
    normalized["doc_type"] = str(normalized["doc_type"])
    normalized["domain"] = str(normalized["domain"])
    normalized["title"] = str(normalized["title"])
    normalized["section"] = str(normalized["section"])
    normalized["chunk_id"] = str(normalized["chunk_id"])
    normalized["language"] = str(normalized["language"])
    normalized["is_citable"] = _to_bool(normalized["is_citable"])

    return normalized


def validate_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compatibility wrapper for callers that prefer a validation name."""

    return normalize_metadata(metadata)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "none", ""}
    return bool(value)
