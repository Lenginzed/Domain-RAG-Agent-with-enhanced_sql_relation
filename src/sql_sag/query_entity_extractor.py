from __future__ import annotations

import re
from typing import Any

from src.sql_sag.entity_extractor import camel_to_snake, normalize_entity_name, split_camel_case


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./\\-]*|\d+v\d+|\d+")
STAGE_RE = re.compile(r"\bstage[\s_-]*(\d+(?:[._]\d+)?)\b", re.IGNORECASE)


def extract_query_entities(question: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    extraction_config = config.get("query_entity_extraction", config)
    ignored = {normalize_entity_name(item) for item in extraction_config.get("ignored_tokens", [])}
    min_len = int(extraction_config.get("min_token_len", 3))
    entities: dict[str, dict[str, Any]] = {}

    if extraction_config.get("enable_stage_entities", True):
        for match in STAGE_RE.finditer(question or ""):
            raw = match.group(0)
            normalized = normalize_entity_name(f"stage{match.group(1).replace('.', '_')}")
            add_entity(entities, raw, normalized, "stage", 0.95, ignored, min_len)

    if extraction_config.get("enable_filename_like_entities", True):
        for raw in TOKEN_RE.findall(question or ""):
            raw = raw.strip(".,;:!?，。；：！？()[]{}")
            if not raw:
                continue
            base = raw.replace("\\", "/").split("/")[-1]
            normalized = normalize_entity_name(base)
            add_entity(entities, base, normalized, "filename_like", 1.0 if "_" in normalized else 0.8, ignored, min_len)
            if extraction_config.get("enable_camel_case_split", True):
                camel = camel_to_snake(base)
                add_entity(entities, base, camel, "camel_case", 0.95, ignored, min_len)
                for piece in split_camel_case(base):
                    add_entity(entities, piece, normalize_entity_name(piece), "camel_piece", 0.7, ignored, min_len)
            for part in re.split(r"[_./\\-]+", normalized):
                add_entity(entities, part, normalize_entity_name(part), "token", 0.65, ignored, min_len)

    if extraction_config.get("enable_domain_terms", True):
        lowered = str(question or "").lower()
        for term in config.get("domain_terms", []):
            normalized = normalize_entity_name(term)
            if not normalized or normalized in ignored:
                continue
            pattern = re.escape(str(term).lower()).replace("_", r"[_\s-]?")
            if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", lowered):
                add_entity(entities, str(term), normalized, "domain_term", 0.9, ignored, min_len)

    sorted_entities = sorted(entities.values(), key=lambda row: (-float(row["confidence"]), row["normalized_name"]))
    return sorted_entities[: int(extraction_config.get("max_query_entities", config.get("retriever", {}).get("max_query_entities", 12)))]


def add_entity(
    entities: dict[str, dict[str, Any]],
    name: str,
    normalized: str,
    source: str,
    confidence: float,
    ignored: set[str],
    min_len: int,
) -> None:
    normalized = normalize_entity_name(normalized)
    if not normalized or normalized in ignored:
        return
    if len(normalized) < min_len and not re.fullmatch(r"\d+v\d+", normalized):
        return
    existing = entities.get(normalized)
    if existing and float(existing["confidence"]) >= confidence:
        return
    entities[normalized] = {
        "name": name,
        "normalized_name": normalized,
        "source": source,
        "confidence": confidence,
    }
