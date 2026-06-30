from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./\\-]*|\d+")
STAGE_PATTERNS = [
    re.compile(r"\bstage[\s_-]*(\d+(?:[._]\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bStage\s+(\d+(?:\.\d+)?)\b"),
]
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
YAML_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*:", re.MULTILINE)


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    normalized_name: str
    entity_type: str
    role: str
    source_hint: str
    confidence: float = 1.0


@dataclass(frozen=True)
class EntityRemoval:
    name: str
    normalized_name: str
    entity_type: str
    role: str
    source_hint: str
    reason: str


def normalize_entity_name(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"\.(py|yaml|yml|md|pdf|csv|json|toml|ini)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"[\s\-/\\.:]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def extract_entities_for_chunk(
    *,
    text: str,
    metadata: dict[str, Any],
    config: dict[str, Any],
    domain_terms: list[str],
) -> list[ExtractedEntity]:
    ignored = {normalize_entity_name(item) for item in config.get("ignored_tokens", [])}
    min_len = int(config.get("min_token_len", 3))
    max_entities = int(config.get("max_entities_per_chunk", 40))
    entities: list[ExtractedEntity] = []
    blob = "\n".join(
        [
            str(text or ""),
            str(metadata.get("source", "")),
            str(metadata.get("imported_source", "")),
            str(metadata.get("title", "")),
            str(metadata.get("section", "")),
        ]
    )

    if bool(config.get("enable_filename_entities", True)):
        entities.extend(extract_filename_entities(metadata, ignored, min_len))
    if bool(config.get("enable_stage_entities", True)):
        entities.extend(extract_stage_entities(blob, ignored))
    if bool(config.get("enable_code_symbol_entities", True)) and str(metadata.get("file_ext", "")).lower() == ".py":
        entities.extend(extract_code_symbols(text, ignored))
    if bool(config.get("enable_yaml_key_entities", True)) and str(metadata.get("file_ext", "")).lower() in {".yaml", ".yml"}:
        entities.extend(extract_yaml_keys(text, ignored))
    if bool(config.get("enable_domain_terms", True)):
        entities.extend(extract_domain_terms(blob, domain_terms, ignored))

    deduped: dict[tuple[str, str, str], ExtractedEntity] = {}
    for entity in entities:
        if not entity.normalized_name or len(entity.normalized_name) < min_len:
            continue
        if entity.normalized_name in ignored:
            continue
        key = (entity.normalized_name, entity.entity_type, entity.role)
        deduped.setdefault(key, entity)
    return list(deduped.values())[:max_entities]


def cleanup_extracted_entities(
    entities: list[ExtractedEntity],
    cleanup_config: dict[str, Any],
) -> tuple[list[ExtractedEntity], list[EntityRemoval]]:
    if not bool(cleanup_config.get("enabled", False)):
        return entities, []

    min_len = int(cleanup_config.get("min_token_len", 1))
    max_len = int(cleanup_config.get("max_token_len", 10_000))
    kept: list[ExtractedEntity] = []
    removed: list[EntityRemoval] = []
    for entity in entities:
        reason = entity_filter_reason(entity, cleanup_config, min_len=min_len, max_len=max_len)
        if reason:
            removed.append(
                EntityRemoval(
                    name=entity.name,
                    normalized_name=entity.normalized_name,
                    entity_type=entity.entity_type,
                    role=entity.role,
                    source_hint=entity.source_hint,
                    reason=reason,
                )
            )
        else:
            kept.append(entity)
    return kept, removed


def entity_filter_reason(
    entity: ExtractedEntity,
    cleanup_config: dict[str, Any],
    *,
    min_len: int | None = None,
    max_len: int | None = None,
) -> str | None:
    normalized = normalize_entity_name(entity.normalized_name or entity.name)
    if not normalized:
        return "empty_normalized_name"
    if is_preserved_entity(normalized, cleanup_config):
        return None

    min_len = int(cleanup_config.get("min_token_len", 1) if min_len is None else min_len)
    max_len = int(cleanup_config.get("max_token_len", 10_000) if max_len is None else max_len)
    if len(normalized) < min_len:
        return "too_short"
    if len(normalized) > max_len:
        return "too_long"

    generic_stoplist = {normalize_entity_name(item) for item in cleanup_config.get("generic_entity_stoplist", [])}
    if normalized in generic_stoplist:
        return "generic_entity_stoplist"

    yaml_stoplist = {normalize_entity_name(item) for item in cleanup_config.get("generic_yaml_key_stoplist", [])}
    if entity.entity_type == "config_key" and normalized in yaml_stoplist:
        return "generic_yaml_key_stoplist"

    if bool(cleanup_config.get("filter_hash_like_tokens", False)) and is_hash_like_token(normalized, cleanup_config):
        return "hash_like_token"
    return None


def is_preserved_entity(normalized_name: str, cleanup_config: dict[str, Any]) -> bool:
    normalized = normalize_entity_name(normalized_name)
    preserve = {normalize_entity_name(item) for item in cleanup_config.get("preserve_key_entities", [])}
    return normalized in preserve


def is_hash_like_token(normalized_name: str, cleanup_config: dict[str, Any]) -> bool:
    normalized = normalize_entity_name(normalized_name)
    if is_structural_token(normalized):
        return False
    patterns = [str(item) for item in cleanup_config.get("hash_like_patterns", [])]
    return any(re.fullmatch(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def is_structural_token(normalized_name: str) -> bool:
    normalized = normalize_entity_name(normalized_name)
    return bool(
        re.fullmatch(r"stage\d+(?:_\d+)?", normalized)
        or re.fullmatch(r"\d+v\d+", normalized)
    )


def extract_filename_entities(
    metadata: dict[str, Any],
    ignored: set[str],
    min_len: int,
) -> list[ExtractedEntity]:
    source = str(metadata.get("imported_source") or metadata.get("source") or "")
    filename = Path(source.replace("\\", "/")).name
    stem = Path(filename).stem
    file_ext = str(metadata.get("file_ext") or Path(filename).suffix).lower()
    entity_type = "file_stem"
    if file_ext == ".py":
        entity_type = "code_file"
    elif file_ext in {".yaml", ".yml"}:
        entity_type = "config"

    values = [stem, camel_to_snake(stem)]
    values.extend(split_source_tokens(stem, min_len=min_len))
    entities: list[ExtractedEntity] = []
    for value in values:
        normalized = normalize_entity_name(value)
        if normalized and normalized not in ignored and len(normalized) >= min_len:
            entities.append(
                ExtractedEntity(
                    name=value,
                    normalized_name=normalized,
                    entity_type=entity_type if value in {stem, camel_to_snake(stem)} else "domain_term",
                    role="filename",
                    source_hint=source,
                )
            )
    return entities


def extract_stage_entities(text: str, ignored: set[str]) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for pattern in STAGE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0)
            digits = match.group(1).replace(".", "_")
            if "_" in digits:
                normalized = f"stage{digits}"
            else:
                normalized = f"stage{digits}"
            normalized = normalize_entity_name(normalized)
            if normalized and normalized not in ignored:
                entities.append(
                    ExtractedEntity(
                        name=raw,
                        normalized_name=normalized,
                        entity_type="experiment_stage",
                        role="stage",
                        source_hint="text",
                    )
                )
    return entities


def extract_code_symbols(text: str, ignored: set[str]) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for class_name in CLASS_RE.findall(text or ""):
        add_symbol_entities(entities, class_name, "code_symbol", "class", ignored)
    for function_name in DEF_RE.findall(text or ""):
        add_symbol_entities(entities, function_name, "method", "function", ignored)
    return entities


def add_symbol_entities(
    entities: list[ExtractedEntity],
    name: str,
    entity_type: str,
    role: str,
    ignored: set[str],
) -> None:
    candidates = [name, camel_to_snake(name)]
    for candidate in candidates:
        normalized = normalize_entity_name(candidate)
        if normalized and normalized not in ignored:
            entities.append(
                ExtractedEntity(
                    name=candidate,
                    normalized_name=normalized,
                    entity_type=entity_type,
                    role=role,
                    source_hint="code",
                )
            )


def extract_yaml_keys(text: str, ignored: set[str]) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for key in YAML_KEY_RE.findall(text or ""):
        normalized = normalize_entity_name(key)
        if normalized and normalized not in ignored:
            entities.append(
                ExtractedEntity(
                    name=key,
                    normalized_name=normalized,
                    entity_type="config_key",
                    role="yaml_key",
                    source_hint="yaml",
                )
            )
    return entities


def extract_domain_terms(text: str, domain_terms: list[str], ignored: set[str]) -> list[ExtractedEntity]:
    lowered = str(text or "").lower()
    entities: list[ExtractedEntity] = []
    for term in domain_terms:
        normalized = normalize_entity_name(term)
        if not normalized or normalized in ignored:
            continue
        term_pattern = re.escape(str(term).lower()).replace("_", r"[_\s-]?")
        if re.search(rf"(?<![a-z0-9]){term_pattern}(?![a-z0-9])", lowered):
            entities.append(
                ExtractedEntity(
                    name=str(term),
                    normalized_name=normalized,
                    entity_type="domain_term",
                    role="mention",
                    source_hint="domain_terms",
                )
            )
    return entities


def split_source_tokens(text: str, *, min_len: int) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(str(text)):
        raw = Path(raw.replace("\\", "/")).name
        parts = re.split(r"[_./\\-]+", raw)
        for part in parts:
            pieces = [part, *split_camel_case(part)]
            for piece in pieces:
                normalized = normalize_entity_name(piece)
                if len(normalized) >= min_len and normalized not in tokens:
                    tokens.append(normalized)
    return tokens


def split_camel_case(text: str) -> list[str]:
    if not text or text.islower() or text.isupper():
        return []
    return [item.lower() for item in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", text)]


def camel_to_snake(text: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(text))
    return normalize_entity_name(value)
