from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.indexes.vector_index import load_index
from src.sql_sag.entity_extractor import (
    EntityRemoval,
    ExtractedEntity,
    cleanup_extracted_entities,
    extract_entities_for_chunk,
)
from src.sql_sag.schema import create_schema, reset_database, table_counts


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_sql_sag_index(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(resolve_project_path(config_path))
    sqlite_config = config.get("sqlite", {})
    db_path = resolve_project_path(sqlite_config["db_path"])
    if bool(sqlite_config.get("overwrite_existing", True)):
        connection = reset_database(db_path)
    else:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        create_schema(connection)

    manifest_rows = read_manifest(resolve_project_path(config["inputs"]["expanded_files_csv"]))
    allowed_sources = {row["source"] for row in manifest_rows if as_bool(row.get("ingested", True))}
    chroma_rows = read_chroma_chunks(config)
    selected_chunks = [row for row in chroma_rows if row["metadata"].get("source") in allowed_sources]
    if not selected_chunks:
        selected_chunks = chroma_rows

    now = datetime.now().isoformat(timespec="seconds")
    documents = build_documents(manifest_rows, selected_chunks, now)
    chunks = build_chunks(selected_chunks)
    events = build_events(chunks)
    entity_rows, event_entity_rows, removed_entities, entity_type_merges = build_entities_and_links(events, chunks, config)

    insert_documents(connection, documents)
    insert_chunks(connection, chunks)
    insert_events(connection, events)
    insert_entities(connection, entity_rows)
    insert_event_entities(connection, event_entity_rows)
    connection.commit()

    outputs = config.get("outputs", {})
    write_csv(resolve_project_path(outputs["documents_csv"]), documents)
    write_csv(resolve_project_path(outputs["chunks_csv"]), chunks)
    write_csv(resolve_project_path(outputs["events_csv"]), events)
    write_csv(resolve_project_path(outputs["entities_csv"]), entity_rows)
    write_csv(resolve_project_path(outputs["event_entities_csv"]), event_entity_rows)
    if outputs.get("removed_entities_csv"):
        write_csv(resolve_project_path(outputs["removed_entities_csv"]), removed_entities)
    if outputs.get("entity_type_merge_csv"):
        write_csv(resolve_project_path(outputs["entity_type_merge_csv"]), entity_type_merges)

    counts = table_counts(connection)
    summary = {
        "generated_at": now,
        "config_path": str(resolve_project_path(config_path)),
        "db_path": str(db_path),
        "collection_name": config["collection"]["collection_name"],
        "persist_directory": str(resolve_project_path(config["collection"]["persist_directory"])),
        "manifest_files": len(manifest_rows),
        "allowed_sources": len(allowed_sources),
        "chroma_chunks_read": len(chroma_rows),
        "selected_chunks": len(selected_chunks),
        "document_count": counts["documents"],
        "chunk_count": counts["chunks"],
        "event_count": counts["events"],
        "entity_count": counts["entities"],
        "event_entity_count": counts["event_entities"],
        "chunks_by_category": dict(Counter(row.get("category_dir", "") for row in chunks)),
        "chunks_by_extension": dict(Counter(row.get("file_ext", "") for row in documents)),
        "entity_type_counts": entity_type_counts(connection),
        "removed_entity_count": len(removed_entities),
        "removed_hash_like_count": count_reason(removed_entities, "hash_like_token"),
        "removed_generic_count": count_reason(removed_entities, "generic_entity_stoplist"),
        "removed_generic_yaml_key_count": count_reason(removed_entities, "generic_yaml_key_stoplist"),
        "entity_type_merge_count": len(entity_type_merges),
        "llm_called": False,
        "chroma_written": False,
        "raw_imported_modified": False,
        "external_source_modified": False,
    }
    write_json(resolve_project_path(outputs["build_summary_json"]), summary)
    connection.close()
    return summary


def read_chroma_chunks(config: dict[str, Any]) -> list[dict[str, Any]]:
    collection = config["collection"]
    vector_store = load_index(
        persist_directory=resolve_project_path(collection["persist_directory"]),
        collection_name=str(collection["collection_name"]),
    )
    data = vector_store._collection.get(include=["documents", "metadatas"])  # noqa: SLF001 - read-only Chroma export.
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    rows: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = dict(metadatas[index] or {})
        rows.append(
            {
                "chroma_id": str(chunk_id),
                "text": str(documents[index] or ""),
                "metadata": metadata,
            }
        )
    return rows


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def build_documents(manifest_rows: list[dict[str, Any]], chunks: list[dict[str, Any]], created_at: str) -> list[dict[str, Any]]:
    metadata_by_source: dict[str, dict[str, Any]] = {}
    for row in chunks:
        metadata = row["metadata"]
        source = str(metadata.get("source") or "")
        if source and source not in metadata_by_source:
            metadata_by_source[source] = metadata

    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in manifest_rows:
        source = str(row.get("source", ""))
        if not source or source in seen:
            continue
        seen.add(source)
        metadata = metadata_by_source.get(source, {})
        documents.append(
            {
                "doc_id": stable_id("doc", source),
                "source": source,
                "imported_source": str(metadata.get("imported_source") or source),
                "original_source": str(metadata.get("original_source") or ""),
                "category_dir": str(row.get("category_dir") or metadata.get("category_dir") or ""),
                "doc_type": str(row.get("doc_type") or metadata.get("doc_type") or ""),
                "file_ext": str(row.get("file_ext") or metadata.get("file_ext") or Path(source).suffix),
                "title": str(metadata.get("title") or Path(source).stem),
                "created_at": created_at,
            }
        )
    return documents


def build_chunks(chroma_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    source_counts: defaultdict[str, int] = defaultdict(int)
    for row in chroma_rows:
        metadata = row["metadata"]
        source = str(metadata.get("source") or "")
        chunk_id = str(metadata.get("chunk_id") or row["chroma_id"])
        chunk_index = metadata.get("chunk_index")
        if chunk_index is None:
            chunk_index = source_counts[source]
        source_counts[source] += 1
        text = str(row.get("text") or "")
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": stable_id("doc", source),
                "chunk_index": int(chunk_index),
                "source": source,
                "section": str(metadata.get("section") or ""),
                "text": text,
                "content_preview": summarize_text(text, 500),
                "chroma_chunk_id": str(row["chroma_id"]),
                "category_dir": str(metadata.get("category_dir") or ""),
                "doc_type": str(metadata.get("doc_type") or ""),
                "file_ext": str(metadata.get("file_ext") or Path(source).suffix),
                "title": str(metadata.get("title") or ""),
                "imported_source": str(metadata.get("imported_source") or source),
                "original_source": str(metadata.get("original_source") or ""),
            }
        )
    return chunks


def build_events(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for chunk in chunks:
        filename = Path(str(chunk["source"]).replace("\\", "/")).name
        event_text = f"{filename} | section={chunk.get('section', '')} | preview={chunk.get('content_preview', '')}"
        events.append(
            {
                "event_id": stable_id("evt", chunk["chunk_id"]),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "event_text": event_text,
                "event_type": "chunk_event",
                "source": chunk["source"],
                "confidence": 1.0,
            }
        )
    return events


def build_entities_and_links(
    events: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    entity_map: dict[tuple[str, str], dict[str, Any]] = {}
    links: dict[tuple[str, str, str], dict[str, Any]] = {}
    extractor_config = config.get("entity_extraction", {})
    cleanup_config = config.get("entity_cleanup", {})
    domain_terms = [str(item) for item in config.get("domain_terms", [])]
    removed_entities: list[dict[str, Any]] = []
    for event in events:
        chunk = chunk_by_id[event["chunk_id"]]
        metadata = {
            "source": chunk.get("source"),
            "imported_source": chunk.get("imported_source"),
            "original_source": chunk.get("original_source"),
            "title": chunk.get("title"),
            "section": chunk.get("section"),
            "file_ext": chunk.get("file_ext"),
            "category_dir": chunk.get("category_dir"),
            "doc_type": chunk.get("doc_type"),
        }
        extracted = extract_entities_for_chunk(
            text=str(chunk.get("text") or ""),
            metadata=metadata,
            config=extractor_config,
            domain_terms=domain_terms,
        )
        extracted, removed = cleanup_extracted_entities(extracted, cleanup_config)
        for removal in removed:
            removed_entities.append(removal_to_row(removal, event, chunk))
        for entity in extracted:
            entity_id = stable_entity_id(entity)
            entity_key = (entity.normalized_name, entity.entity_type)
            entity_map.setdefault(
                entity_key,
                {
                    "entity_id": entity_id,
                    "name": entity.name,
                    "normalized_name": entity.normalized_name,
                    "entity_type": entity.entity_type,
                    "source_hint": entity.source_hint,
                },
            )
            link_key = (event["event_id"], entity_id, entity.role)
            links.setdefault(
                link_key,
                {
                    "event_id": event["event_id"],
                    "entity_id": entity_id,
                    "role": entity.role,
                    "confidence": entity.confidence,
                },
            )
    entity_rows, event_entity_rows, merge_rows, merge_removals = apply_entity_type_priority(
        list(entity_map.values()),
        list(links.values()),
        cleanup_config,
    )
    removed_entities.extend(merge_removals)
    return entity_rows, event_entity_rows, removed_entities, merge_rows


def apply_entity_type_priority(
    entity_rows: list[dict[str, Any]],
    event_entity_rows: list[dict[str, Any]],
    cleanup_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(cleanup_config.get("enabled", False)):
        return entity_rows, event_entity_rows, [], []

    priority = {
        str(key): int(value)
        for key, value in (cleanup_config.get("entity_type_priority", {}) or {}).items()
    }
    if not priority:
        return entity_rows, event_entity_rows, [], []

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entity_rows:
        grouped[str(row.get("normalized_name", ""))].append(row)

    kept_rows: list[dict[str, Any]] = []
    entity_id_map: dict[str, str] = {}
    merge_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    for normalized_name, rows in grouped.items():
        if len({str(row.get("entity_type", "")) for row in rows}) <= 1:
            kept_rows.extend(rows)
            continue

        sorted_rows = sorted(
            rows,
            key=lambda row: (
                priority.get(str(row.get("entity_type", "")), 0),
                str(row.get("entity_type", "")),
            ),
            reverse=True,
        )
        kept = sorted_rows[0]
        kept_rows.append(kept)
        removed_types: list[str] = []
        for removed in sorted_rows[1:]:
            entity_id_map[str(removed["entity_id"])] = str(kept["entity_id"])
            removed_types.append(str(removed.get("entity_type", "")))
            removed_rows.append(
                {
                    "name": removed.get("name", ""),
                    "normalized_name": normalized_name,
                    "entity_type": removed.get("entity_type", ""),
                    "role": "",
                    "source_hint": removed.get("source_hint", ""),
                    "reason": "lower_priority_entity_type",
                    "event_id": "",
                    "chunk_id": "",
                    "source": "",
                }
            )
        merge_rows.append(
            {
                "normalized_name": normalized_name,
                "kept_entity_type": kept.get("entity_type", ""),
                "removed_entity_types": ";".join(sorted(set(removed_types))),
                "reason": "entity_type_priority",
            }
        )

    merged_links: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in event_entity_rows:
        new_row = dict(row)
        new_row["entity_id"] = entity_id_map.get(str(row["entity_id"]), str(row["entity_id"]))
        key = (str(new_row["event_id"]), str(new_row["entity_id"]), str(new_row["role"]))
        merged_links.setdefault(key, new_row)
    return kept_rows, list(merged_links.values()), merge_rows, removed_rows


def removal_to_row(removal: EntityRemoval, event: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": removal.name,
        "normalized_name": removal.normalized_name,
        "entity_type": removal.entity_type,
        "role": removal.role,
        "source_hint": removal.source_hint,
        "reason": removal.reason,
        "event_id": event.get("event_id", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "source": chunk.get("source", ""),
    }


def insert_documents(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    connection.executemany(
        """
        INSERT OR REPLACE INTO documents
        (doc_id, source, imported_source, original_source, category_dir, doc_type, file_ext, title, created_at)
        VALUES (:doc_id, :source, :imported_source, :original_source, :category_dir, :doc_type, :file_ext, :title, :created_at)
        """,
        rows,
    )


def insert_chunks(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    connection.executemany(
        """
        INSERT OR REPLACE INTO chunks
        (chunk_id, doc_id, chunk_index, source, section, text, content_preview, chroma_chunk_id)
        VALUES (:chunk_id, :doc_id, :chunk_index, :source, :section, :text, :content_preview, :chroma_chunk_id)
        """,
        rows,
    )


def insert_events(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    connection.executemany(
        """
        INSERT OR REPLACE INTO events
        (event_id, chunk_id, doc_id, event_text, event_type, source, confidence)
        VALUES (:event_id, :chunk_id, :doc_id, :event_text, :event_type, :source, :confidence)
        """,
        rows,
    )


def insert_entities(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    connection.executemany(
        """
        INSERT OR IGNORE INTO entities
        (entity_id, name, normalized_name, entity_type, source_hint)
        VALUES (:entity_id, :name, :normalized_name, :entity_type, :source_hint)
        """,
        rows,
    )


def insert_event_entities(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    connection.executemany(
        """
        INSERT OR IGNORE INTO event_entities
        (event_id, entity_id, role, confidence)
        VALUES (:event_id, :entity_id, :role, :confidence)
        """,
        rows,
    )


def entity_type_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("SELECT entity_type, COUNT(*) AS count FROM entities GROUP BY entity_type ORDER BY count DESC").fetchall()
    return {str(row["entity_type"]): int(row["count"]) for row in rows}


def count_reason(rows: list[dict[str, Any]], reason: str) -> int:
    return sum(1 for row in rows if row.get("reason") == reason)


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def stable_entity_id(entity: ExtractedEntity) -> str:
    return stable_id("ent", f"{entity.normalized_name}|{entity.entity_type}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    base_config = data.get("base_config")
    if base_config:
        base_path = resolve_project_path(base_config)
        base_data = load_yaml(base_path)
        data = deep_merge(base_data, {key: value for key, value in data.items() if key != "base_config"})
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def resolve_project_path(path: str | Path) -> Path:
    value = Path(str(path))
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def summarize_text(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
