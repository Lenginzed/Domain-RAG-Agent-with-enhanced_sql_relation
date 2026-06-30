from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.sql_sag.entity_extractor import normalize_entity_name
from src.sql_sag.query_entity_extractor import extract_query_entities as extract_query_entities_from_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_sql_sag_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {resolved}")
    return data


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = resolve_project_path(db_path)
    if not path.exists():
        raise FileNotFoundError(path)
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def extract_query_entities(question: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return extract_query_entities_from_text(question, config)


def lookup_seed_entities(conn: sqlite3.Connection, query_entities: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    max_seed_entities = int(config.get("retriever", {}).get("max_seed_entities", 20))
    rows: dict[str, dict[str, Any]] = {}
    for query_entity in query_entities:
        normalized = str(query_entity["normalized_name"])
        matches = conn.execute(
            """
            SELECT entity_id, name, normalized_name, entity_type, source_hint,
              CASE
                WHEN normalized_name = ? THEN 'exact'
                WHEN normalized_name LIKE ? THEN 'contains'
                WHEN ? LIKE '%' || normalized_name || '%' THEN 'contained_by'
                ELSE 'none'
              END AS match_type
            FROM entities
            WHERE normalized_name = ?
               OR normalized_name LIKE ?
               OR ? LIKE '%' || normalized_name || '%'
            ORDER BY
              CASE
                WHEN normalized_name = ? THEN 0
                WHEN normalized_name LIKE ? THEN 1
                ELSE 2
              END,
              LENGTH(normalized_name) DESC,
              normalized_name
            LIMIT ?
            """,
            (
                normalized,
                f"%{normalized}%",
                normalized,
                normalized,
                f"%{normalized}%",
                normalized,
                normalized,
                f"%{normalized}%",
                max_seed_entities,
            ),
        ).fetchall()
        for match in matches:
            row = dict(match)
            row["query_entity"] = normalized
            row["query_entity_name"] = query_entity.get("name", normalized)
            row["query_confidence"] = query_entity.get("confidence", 0.0)
            key = f"{row['entity_id']}|{row['query_entity']}"
            rows.setdefault(key, row)
    return list(rows.values())[:max_seed_entities]


def lookup_seed_events(conn: sqlite3.Connection, seed_entities: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    max_seed_events = int(config.get("retriever", {}).get("max_seed_events", 40))
    rows: dict[str, dict[str, Any]] = {}
    for seed in seed_entities:
        matches = conn.execute(
            """
            SELECT
              ev.event_id, ev.chunk_id, ev.doc_id, ev.event_text, ev.source AS event_source,
              e.entity_id, e.normalized_name AS seed_entity, e.entity_type AS seed_entity_type,
              ee.role AS seed_role,
              c.source, c.section, c.content_preview, c.text,
              d.category_dir, d.doc_type, d.file_ext, d.title, d.imported_source, d.original_source
            FROM entities e
            JOIN event_entities ee ON e.entity_id = ee.entity_id
            JOIN events ev ON ee.event_id = ev.event_id
            JOIN chunks c ON ev.chunk_id = c.chunk_id
            JOIN documents d ON ev.doc_id = d.doc_id
            WHERE e.entity_id = ?
            ORDER BY d.source, c.chunk_index
            LIMIT ?
            """,
            (seed["entity_id"], max_seed_events),
        ).fetchall()
        for match in matches:
            row = dict(match)
            row["query_entity"] = seed.get("query_entity", "")
            row["match_type"] = seed.get("match_type", "")
            row["is_seed_event"] = True
            rows.setdefault(row["event_id"], row)
    return list(rows.values())[:max_seed_events]


def expand_events_one_hop(conn: sqlite3.Connection, seed_events: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    max_expanded = int(config.get("retriever", {}).get("max_expanded_events", 80))
    rows: dict[str, dict[str, Any]] = {}
    for seed_event in seed_events:
        matches = conn.execute(
            """
            SELECT
              ev2.event_id, ev2.chunk_id, ev2.doc_id, ev2.event_text, ev2.source AS event_source,
              shared.entity_id AS shared_entity_id,
              shared.normalized_name AS shared_entity,
              shared.entity_type AS shared_entity_type,
              c.source, c.section, c.content_preview, c.text,
              d.category_dir, d.doc_type, d.file_ext, d.title, d.imported_source, d.original_source
            FROM event_entities seed_ee
            JOIN entities shared ON seed_ee.entity_id = shared.entity_id
            JOIN event_entities ee2 ON shared.entity_id = ee2.entity_id
            JOIN events ev2 ON ee2.event_id = ev2.event_id
            JOIN chunks c ON ev2.chunk_id = c.chunk_id
            JOIN documents d ON ev2.doc_id = d.doc_id
            WHERE seed_ee.event_id = ?
              AND ev2.event_id != ?
            ORDER BY shared.normalized_name, d.source, c.chunk_index
            LIMIT ?
            """,
            (seed_event["event_id"], seed_event["event_id"], max_expanded),
        ).fetchall()
        for match in matches:
            row = dict(match)
            row["seed_event_id"] = seed_event["event_id"]
            row["seed_chunk_id"] = seed_event["chunk_id"]
            row["query_entity"] = seed_event.get("query_entity", "")
            row["seed_entity"] = seed_event.get("seed_entity", "")
            row["is_seed_event"] = False
            rows.setdefault(row["event_id"], row)
            if len(rows) >= max_expanded:
                return list(rows.values())
    return list(rows.values())


def map_events_to_chunks(conn: sqlite3.Connection, events: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event["event_id"])
        if event_id in candidates:
            continue
        candidates[event_id] = {
            "rank": 0,
            "source": event.get("source", ""),
            "chunk_id": event.get("chunk_id", ""),
            "event_id": event_id,
            "doc_id": event.get("doc_id", ""),
            "category_dir": event.get("category_dir", ""),
            "doc_type": event.get("doc_type", ""),
            "file_ext": event.get("file_ext", ""),
            "title": event.get("title", ""),
            "section": event.get("section", ""),
            "imported_source": event.get("imported_source", ""),
            "original_source": event.get("original_source", ""),
            "content_preview": event.get("content_preview", ""),
            "text": event.get("text", ""),
            "is_seed_event": bool(event.get("is_seed_event")),
            "seed_event_id": event.get("seed_event_id", event_id),
            "query_entity": event.get("query_entity", ""),
            "seed_entity": event.get("seed_entity", ""),
            "shared_entity": event.get("shared_entity", ""),
            "relation_score": 0.0,
            "relation_reasons": [],
            "relation_path": [],
        }
    return list(candidates.values())


def score_sql_candidates(
    candidates: list[dict[str, Any]],
    query_entities: list[dict[str, Any]],
    seed_entities: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    scoring = config.get("scoring", {})
    query_names = {str(item["normalized_name"]) for item in query_entities}
    seed_names = {str(item["normalized_name"]) for item in seed_entities}
    key_entities = {"missile_engine", "event_driven_reward", "hierarchy_selfplay", "opd", "hint", "world_model", "mappo", "reward", "risk"}

    for candidate in candidates:
        candidate_entities = load_event_entities_for_candidate(candidate, config)
        candidate_entity_names = {row["normalized_name"] for row in candidate_entities}
        score = 0.0
        reasons: list[str] = []

        exact = sorted(candidate_entity_names & query_names)
        if exact:
            delta = float(scoring.get("exact_entity_match", 30.0)) * len(exact)
            score += delta
            reasons.append(f"exact_entity_match:{','.join(exact)}:+{delta:g}")

        seed_exact = sorted(candidate_entity_names & seed_names)
        if seed_exact:
            delta = float(scoring.get("normalized_entity_match", 20.0)) * len(seed_exact)
            score += delta
            reasons.append(f"normalized_entity_match:{','.join(seed_exact)}:+{delta:g}")

        if candidate.get("shared_entity"):
            delta = float(scoring.get("shared_entity_match", 10.0))
            score += delta
            reasons.append(f"shared_entity_match:{candidate['shared_entity']}:+{delta:g}")

        category = str(candidate.get("category_dir", ""))
        if category and category in infer_query_categories(query_names):
            delta = float(scoring.get("category_match", 5.0))
            score += delta
            reasons.append(f"category_match:{category}:+{delta:g}")

        if candidate.get("is_seed_event"):
            delta = float(scoring.get("seed_event_bonus", 8.0))
            score += delta
            reasons.append(f"seed_event_bonus:+{delta:g}")
        else:
            delta = float(scoring.get("expanded_event_bonus", 4.0))
            score += delta
            reasons.append(f"expanded_event_bonus:+{delta:g}")

        key_hits = sorted(candidate_entity_names & key_entities)
        if key_hits:
            delta = float(scoring.get("key_entity_bonus", 5.0))
            score += delta
            reasons.append(f"key_entity_bonus:{','.join(key_hits[:5])}:+{delta:g}")

        if candidate.get("seed_event_id") and candidate.get("is_seed_event") is False:
            delta = float(scoring.get("same_doc_bonus", 3.0)) if same_doc_seed(candidate) else 0.0
            if delta:
                score += delta
                reasons.append(f"same_doc_bonus:+{delta:g}")

        candidate["relation_score"] = round(score, 6)
        candidate["relation_reasons"] = reasons
        candidate["relation_path"] = build_relation_path(candidate)
        candidate["matched_entities"] = sorted(candidate_entity_names)

    ranked = sorted(candidates, key=lambda item: (-float(item["relation_score"]), str(item.get("source", "")), str(item.get("chunk_id", ""))))
    for index, candidate in enumerate(ranked, start=1):
        candidate["rank"] = index
    return ranked[: int(config.get("retriever", {}).get("final_top_k", 8))]


def load_event_entities_for_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    db_path = config["sqlite"]["db_path"]
    with connect_readonly(db_path) as conn:
        rows = conn.execute(
            """
            SELECT e.name, e.normalized_name, e.entity_type, ee.role
            FROM event_entities ee
            JOIN entities e ON ee.entity_id = e.entity_id
            WHERE ee.event_id = ?
            ORDER BY e.normalized_name
            """,
            (candidate["event_id"],),
        ).fetchall()
        return [dict(row) for row in rows]


def retrieve_sql_relation(question: str, config: dict[str, Any]) -> dict[str, Any]:
    with connect_readonly(config["sqlite"]["db_path"]) as conn:
        query_entities = extract_query_entities(question, config)
        seed_entities = lookup_seed_entities(conn, query_entities, config)
        seed_events = lookup_seed_events(conn, seed_entities, config) if config.get("retriever", {}).get("include_seed_events", True) else []
        expanded_events = (
            expand_events_one_hop(conn, seed_events, config)
            if config.get("retriever", {}).get("include_expanded_events", True)
            else []
        )
        candidates = map_events_to_chunks(conn, [*seed_events, *expanded_events], config)
    candidates = score_sql_candidates(candidates, query_entities, seed_entities, config)
    return {
        "question": question,
        "query_entities": query_entities,
        "seed_entities": seed_entities,
        "seed_events": summarize_events(seed_events),
        "expanded_events": summarize_events(expanded_events),
        "candidates": strip_candidate_text(candidates),
        "debug": {
            "seed_entity_count": len(seed_entities),
            "seed_event_count": len(seed_events),
            "expanded_event_count": len(expanded_events),
            "candidate_count": len(candidates),
        },
    }


def summarize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": row.get("event_id", ""),
            "chunk_id": row.get("chunk_id", ""),
            "source": row.get("source", ""),
            "category_dir": row.get("category_dir", ""),
            "seed_entity": row.get("seed_entity", ""),
            "shared_entity": row.get("shared_entity", ""),
        }
        for row in events
    ]


def strip_candidate_text(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row.pop("text", None)
        stripped.append(row)
    return stripped


def build_relation_path(candidate: dict[str, Any]) -> list[str]:
    query_entity = candidate.get("query_entity") or "unknown"
    seed_entity = candidate.get("seed_entity") or query_entity
    source = Path(str(candidate.get("source", "")).replace("\\", "/")).name
    chunk_id = candidate.get("chunk_id", "")
    if candidate.get("is_seed_event"):
        return [
            f"query_entity:{query_entity}",
            f"seed_entity:{seed_entity}",
            f"seed_event:{source}:{chunk_id}",
        ]
    shared = candidate.get("shared_entity") or seed_entity
    return [
        f"query_entity:{query_entity}",
        f"seed_entity:{seed_entity}",
        f"seed_event:{candidate.get('seed_event_id', '')}",
        f"shared_entity:{shared}",
        f"candidate_event:{source}:{chunk_id}",
    ]


def infer_query_categories(query_names: set[str]) -> set[str]:
    categories: set[str] = set()
    if {"reward", "event_driven_reward"} & query_names:
        categories.add("code")
    if {"hierarchy_selfplay", "selfplay", "curriculum"} & query_names:
        categories.add("configs")
    if {"opd", "hint", "missile_engine", "missile", "engine"} & query_names:
        categories.add("notes")
    if {"risk", "reward"} & query_names:
        categories.update({"experiments", "thesis_or_reports"})
    return categories


def same_doc_seed(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("seed_event_id")) and candidate.get("seed_event_id") == candidate.get("event_id")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_project_path(path: str | Path) -> Path:
    value = Path(str(path))
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value
