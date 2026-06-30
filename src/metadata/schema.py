from __future__ import annotations

CORE_METADATA_FIELDS = [
    "source",
    "doc_type",
    "domain",
    "title",
    "section",
    "chunk_id",
    "language",
    "is_citable",
]

OPTIONAL_METADATA_FIELDS = [
    "stage",
    "method",
    "experiment_id",
    "seed",
    "metric",
    "file_role",
    "page",
    "row_index",
    "parse_warning",
]

DEFAULT_DOMAIN = "multi_uav_air_combat"
DEFAULT_LANGUAGE = "unknown"

METADATA_DEFAULTS = {
    "source": "unknown_source",
    "doc_type": "unknown",
    "domain": DEFAULT_DOMAIN,
    "title": "untitled",
    "section": "default",
    "chunk_id": "",
    "language": DEFAULT_LANGUAGE,
    "is_citable": True,
}

SUPPORTED_DOC_TYPES = {
    "text",
    "paper",
    "note",
    "report",
    "code",
    "config",
    "experiment",
    "unknown",
}
