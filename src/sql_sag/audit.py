from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from src.sql_sag.entity_extractor import normalize_entity_name
from src.sql_sag.schema import table_counts


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def audit_sql_sag_index(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(resolve_project_path(config_path))
    db_path = resolve_project_path(config["sqlite"]["db_path"])
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    outputs = config["outputs"]

    counts = table_counts(connection)
    entity_type_counts = query_counts(connection, "SELECT entity_type AS key, COUNT(*) AS count FROM entities GROUP BY entity_type ORDER BY count DESC")
    category_counts = query_counts(
        connection,
        "SELECT category_dir AS key, COUNT(*) AS count FROM documents GROUP BY category_dir ORDER BY count DESC",
    )
    top_entities = query_rows(
        connection,
        """
        SELECT e.normalized_name, e.name, e.entity_type, COUNT(*) AS frequency
        FROM entities e
        JOIN event_entities ee ON e.entity_id = ee.entity_id
        GROUP BY e.entity_id
        ORDER BY frequency DESC, e.normalized_name
        LIMIT 50
        """,
    )
    orphan_chunks = query_rows(
        connection,
        """
        SELECT c.chunk_id, c.source, c.section, c.content_preview
        FROM chunks c
        LEFT JOIN events ev ON c.chunk_id = ev.chunk_id
        LEFT JOIN event_entities ee ON ev.event_id = ee.event_id
        WHERE ev.event_id IS NULL OR ee.entity_id IS NULL
        GROUP BY c.chunk_id
        ORDER BY c.source, c.chunk_index
        """,
    )
    orphan_events = query_rows(
        connection,
        """
        SELECT ev.event_id, ev.chunk_id, ev.source
        FROM events ev
        LEFT JOIN event_entities ee ON ev.event_id = ee.event_id
        WHERE ee.entity_id IS NULL
        GROUP BY ev.event_id
        ORDER BY ev.source
        """,
    )
    key_entity_rows = audit_key_entities(connection, config.get("audit", {}).get("key_entities", []))
    removed_entities = read_csv(resolve_project_path(outputs["removed_entities_csv"])) if outputs.get("removed_entities_csv") else []
    merge_rows = read_csv(resolve_project_path(outputs["entity_type_merge_csv"])) if outputs.get("entity_type_merge_csv") else []

    event_count = max(1, counts["events"])
    event_entity_count = counts["event_entities"]
    summary = {
        "db_path": str(db_path),
        "document_count": counts["documents"],
        "chunk_count": counts["chunks"],
        "event_count": counts["events"],
        "entity_count": counts["entities"],
        "event_entity_count": event_entity_count,
        "avg_entities_per_event": round(event_entity_count / event_count, 4),
        "orphan_chunk_count": len(orphan_chunks),
        "orphan_event_count": len(orphan_events),
        "orphan_chunk_ratio": round(len(orphan_chunks) / max(1, counts["chunks"]), 6),
        "entity_type_counts": entity_type_counts,
        "category_counts": category_counts,
        "top_entities": top_entities[:20],
        "key_entity_hits": sum(1 for row in key_entity_rows if row["found"]),
        "key_entity_total": len(key_entity_rows),
        "removed_entity_count": len(removed_entities),
        "removed_hash_like_count": count_reason(removed_entities, "hash_like_token"),
        "removed_generic_count": count_reason(removed_entities, "generic_entity_stoplist"),
        "removed_generic_yaml_key_count": count_reason(removed_entities, "generic_yaml_key_stoplist"),
        "entity_type_merge_count": len(merge_rows),
        "audit_passed": build_audit_passed(counts, len(orphan_chunks), config),
        "llm_called": False,
        "chroma_written": False,
        "raw_imported_modified": False,
        "external_source_modified": False,
    }

    write_json(resolve_project_path(outputs["build_summary_json"]), summary)
    write_csv(resolve_project_path(outputs["entity_frequency_csv"]), top_entities)
    write_csv(resolve_project_path(outputs["key_entity_audit_csv"]), key_entity_rows)
    write_csv(resolve_project_path(outputs["orphan_chunks_csv"]), orphan_chunks)
    write_markdown(resolve_project_path(outputs["audit_markdown"]), summary, key_entity_rows, orphan_chunks)
    if config.get("audit", {}).get("compare_with_v4a"):
        compare_summary = build_compare_summary(connection, config, summary, key_entity_rows, removed_entities, merge_rows)
        write_json(resolve_project_path(outputs["compare_summary_json"]), compare_summary)
        write_compare_markdown(resolve_project_path(outputs["compare_markdown"]), compare_summary)
    connection.close()
    return summary


def audit_key_entities(connection: sqlite3.Connection, keys: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        name = str(key)
        normalized = normalize_entity_name(name)
        matches = query_rows(
            connection,
            """
            SELECT e.normalized_name, e.name, e.entity_type, COUNT(ee.event_id) AS frequency
            FROM entities e
            LEFT JOIN event_entities ee ON e.entity_id = ee.entity_id
            WHERE e.normalized_name = ?
            GROUP BY e.entity_id
            ORDER BY frequency DESC
            """,
            (normalized,),
        )
        rows.append(
            {
                "key_entity": name,
                "normalized_name": normalized,
                "found": bool(matches),
                "match_count": len(matches),
                "total_frequency": sum(int(item.get("frequency", 0) or 0) for item in matches),
                "matched_entity_types": ";".join(str(item.get("entity_type", "")) for item in matches),
            }
        )
    return rows


def build_audit_passed(counts: dict[str, int], orphan_chunk_count: int, config: dict[str, Any]) -> bool:
    audit_config = config.get("audit", {})
    min_document_count = int(audit_config.get("min_document_count", 1))
    min_chunk_count = int(audit_config.get("min_chunk_count", 1))
    min_entity_count = int(audit_config.get("min_entity_count", 1))
    max_orphan_ratio = float(audit_config.get("max_orphan_chunk_ratio", 1.0))
    orphan_ratio = orphan_chunk_count / max(1, counts["chunks"])
    return (
        counts["documents"] >= min_document_count
        and counts["chunks"] >= min_chunk_count
        and counts["events"] >= counts["chunks"]
        and counts["entities"] >= min_entity_count
        and counts["event_entities"] > 0
        and orphan_ratio <= max_orphan_ratio
    )


def build_compare_summary(
    v4b_connection: sqlite3.Connection,
    config: dict[str, Any],
    v4b_summary: dict[str, Any],
    v4b_key_entities: list[dict[str, Any]],
    removed_entities: list[dict[str, Any]],
    merge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    audit_config = config.get("audit", {})
    v4a_db_path = resolve_project_path(audit_config["v4a_db_path"])
    v4a_connection = sqlite3.connect(v4a_db_path)
    v4a_connection.row_factory = sqlite3.Row
    v4a_counts = table_counts(v4a_connection)
    v4b_counts = table_counts(v4b_connection)
    v4a_key_entities = audit_key_entities(v4a_connection, audit_config.get("key_entities", []))
    v4a_top_entities = top_entity_rows(v4a_connection, 20)
    v4b_top_entities = top_entity_rows(v4b_connection, 20)

    entity_drop_count = max(0, v4a_counts["entities"] - v4b_counts["entities"])
    entity_drop_ratio = round(entity_drop_count / max(1, v4a_counts["entities"]), 6)
    v4a_avg_entities_per_event = round(v4a_counts["event_entities"] / max(1, v4a_counts["events"]), 4)
    v4b_avg_entities_per_event = round(v4b_counts["event_entities"] / max(1, v4b_counts["events"]), 4)
    v4b_key_hits = sum(1 for row in v4b_key_entities if row["found"])
    v4b_key_total = len(v4b_key_entities)
    key_hit_rate = v4b_key_hits / max(1, v4b_key_total)
    max_drop_ratio = float(audit_config.get("max_entity_drop_ratio_allowed", 1.0))
    required_key_hit_rate = float(audit_config.get("key_entity_required_hit_rate", 0.0))
    noise_entities = ["c276357917", "readme_c276357917", "readme", "paper", "enabled"]
    noise_checks = [
        {
            "entity": item,
            "normalized_name": normalize_entity_name(item),
            "v4a_frequency": entity_frequency(v4a_connection, item),
            "v4b_frequency": entity_frequency(v4b_connection, item),
        }
        for item in noise_entities
    ]
    compare_summary = {
        "v4a_db_path": str(v4a_db_path),
        "v4b_db_path": v4b_summary["db_path"],
        "v4a_document_count": v4a_counts["documents"],
        "v4b_document_count": v4b_counts["documents"],
        "v4a_chunk_count": v4a_counts["chunks"],
        "v4b_chunk_count": v4b_counts["chunks"],
        "v4a_event_count": v4a_counts["events"],
        "v4b_event_count": v4b_counts["events"],
        "v4a_entity_count": v4a_counts["entities"],
        "v4b_entity_count": v4b_counts["entities"],
        "entity_drop_count": entity_drop_count,
        "entity_drop_ratio": entity_drop_ratio,
        "v4a_event_entity_count": v4a_counts["event_entities"],
        "v4b_event_entity_count": v4b_counts["event_entities"],
        "avg_entities_per_event_before": v4a_avg_entities_per_event,
        "avg_entities_per_event_after": v4b_avg_entities_per_event,
        "orphan_chunks": v4b_summary["orphan_chunk_count"],
        "orphan_events": v4b_summary["orphan_event_count"],
        "v4a_key_entity_hits": sum(1 for row in v4a_key_entities if row["found"]),
        "v4a_key_entity_total": len(v4a_key_entities),
        "v4b_key_entity_hits": v4b_key_hits,
        "v4b_key_entity_total": v4b_key_total,
        "key_entity_hit_rate": round(key_hit_rate, 6),
        "removed_entity_count": len(removed_entities),
        "removed_hash_like_count": count_reason(removed_entities, "hash_like_token"),
        "removed_generic_count": count_reason(removed_entities, "generic_entity_stoplist"),
        "removed_generic_yaml_key_count": count_reason(removed_entities, "generic_yaml_key_stoplist"),
        "entity_type_merge_count": len(merge_rows),
        "noise_entity_checks": noise_checks,
        "v4a_top_entities": v4a_top_entities,
        "v4b_top_entities": v4b_top_entities,
        "audit_passed": (
            entity_drop_ratio <= max_drop_ratio
            and key_hit_rate >= required_key_hit_rate
            and v4b_summary["orphan_chunk_ratio"] <= float(audit_config.get("max_orphan_chunk_ratio", 1.0))
        ),
    }
    v4a_connection.close()
    return compare_summary


def top_entity_rows(connection: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    return query_rows(
        connection,
        """
        SELECT e.normalized_name, e.name, e.entity_type, COUNT(*) AS frequency
        FROM entities e
        JOIN event_entities ee ON e.entity_id = ee.entity_id
        GROUP BY e.entity_id
        ORDER BY frequency DESC, e.normalized_name
        LIMIT ?
        """,
        (limit,),
    )


def entity_frequency(connection: sqlite3.Connection, name: str) -> int:
    normalized = normalize_entity_name(name)
    row = connection.execute(
        """
        SELECT COUNT(ee.event_id) AS frequency
        FROM entities e
        LEFT JOIN event_entities ee ON e.entity_id = ee.entity_id
        WHERE e.normalized_name = ?
        """,
        (normalized,),
    ).fetchone()
    return int(row["frequency"] or 0)


def count_reason(rows: list[dict[str, Any]], reason: str) -> int:
    return sum(1 for row in rows if row.get("reason") == reason)


def query_counts(connection: sqlite3.Connection, sql: str) -> dict[str, int]:
    return {str(row["key"]): int(row["count"]) for row in connection.execute(sql).fetchall()}


def query_rows(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, params).fetchall()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_markdown(path: Path, summary: dict[str, Any], key_entities: list[dict[str, Any]], orphan_chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SQL SAG-lite V4a Audit",
        "",
        "## Summary",
        "",
        f"- documents: {summary['document_count']}",
        f"- chunks: {summary['chunk_count']}",
        f"- events: {summary['event_count']}",
        f"- entities: {summary['entity_count']}",
        f"- event_entities: {summary['event_entity_count']}",
        f"- avg_entities_per_event: {summary['avg_entities_per_event']}",
        f"- orphan_chunks: {summary['orphan_chunk_count']}",
        f"- orphan_events: {summary['orphan_event_count']}",
        f"- removed_entities: {summary.get('removed_entity_count', 0)}",
        f"- entity_type_merges: {summary.get('entity_type_merge_count', 0)}",
        f"- audit_passed: {summary['audit_passed']}",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in summary.get("category_counts", {}).items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Entity Type Counts", ""])
    for entity_type, count in summary.get("entity_type_counts", {}).items():
        lines.append(f"- {entity_type}: {count}")
    lines.extend(["", "## Key Entity Audit", ""])
    for row in key_entities:
        lines.append(
            f"- {row['key_entity']} -> found={row['found']} frequency={row['total_frequency']} types={row['matched_entity_types']}"
        )
    lines.extend(["", "## Top Entities", ""])
    for row in summary.get("top_entities", [])[:20]:
        lines.append(f"- {row['normalized_name']} ({row['entity_type']}): {row['frequency']}")
    lines.extend(["", "## Orphan Chunk Sample", ""])
    for row in orphan_chunks[:20]:
        lines.append(f"- {row.get('chunk_id')} | {row.get('source')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compare_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SQL SAG-lite V4a vs V4b Compare",
        "",
        "## Summary",
        "",
        f"- v4a_entities: {summary['v4a_entity_count']}",
        f"- v4b_entities: {summary['v4b_entity_count']}",
        f"- entity_drop_count: {summary['entity_drop_count']}",
        f"- entity_drop_ratio: {summary['entity_drop_ratio']}",
        f"- v4a_event_entities: {summary['v4a_event_entity_count']}",
        f"- v4b_event_entities: {summary['v4b_event_entity_count']}",
        f"- avg_entities_per_event_before: {summary['avg_entities_per_event_before']}",
        f"- avg_entities_per_event_after: {summary['avg_entities_per_event_after']}",
        f"- removed_hash_like_count: {summary['removed_hash_like_count']}",
        f"- removed_generic_count: {summary['removed_generic_count']}",
        f"- removed_generic_yaml_key_count: {summary['removed_generic_yaml_key_count']}",
        f"- entity_type_merge_count: {summary['entity_type_merge_count']}",
        f"- v4b_key_entity_hits: {summary['v4b_key_entity_hits']}/{summary['v4b_key_entity_total']}",
        f"- orphan_chunks: {summary['orphan_chunks']}",
        f"- orphan_events: {summary['orphan_events']}",
        f"- audit_passed: {summary['audit_passed']}",
        "",
        "## Noise Entity Checks",
        "",
    ]
    for row in summary.get("noise_entity_checks", []):
        lines.append(
            f"- {row['normalized_name']}: v4a_frequency={row['v4a_frequency']} v4b_frequency={row['v4b_frequency']}"
        )
    lines.extend(["", "## Top Entities Before", ""])
    for row in summary.get("v4a_top_entities", [])[:20]:
        lines.append(f"- {row['normalized_name']} ({row['entity_type']}): {row['frequency']}")
    lines.extend(["", "## Top Entities After", ""])
    for row in summary.get("v4b_top_entities", [])[:20]:
        lines.append(f"- {row['normalized_name']} ({row['entity_type']}): {row['frequency']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    base_config = data.get("base_config")
    if base_config:
        base_data = load_yaml(resolve_project_path(base_config))
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
