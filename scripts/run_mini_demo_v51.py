from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from scripts.build_mini_demo_index_v51 import PROJECT_ROOT, build_index, load_config, normalize_name, resolve_project_path, split_entity_parts


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        build_index()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def extract_query_entities(question: str) -> list[str]:
    entities: list[str] = []
    for token in question.replace("/", " ").replace("?", " ").replace("？", " ").split():
        normalized = normalize_name(token)
        if not normalized:
            continue
        entities.extend(split_entity_parts(normalized))
    compact = normalize_name(question)
    for key in ["event_driven_reward", "hierarchy_selfplay", "reward", "risk", "missile_engine", "missile", "engine"]:
        if key in compact:
            entities.append(key)
    return sorted({item for item in entities if len(item) >= 3}, key=lambda item: (-len(item), item))


def infer_category_intent(question: str) -> dict[str, Any]:
    compact = question.lower()
    if "代码" in question or "code" in compact or "class" in compact:
        return {"intent": "code", "preferred": ["code"], "penalized": ["configs", "experiments"]}
    if "配置" in question or "config" in compact or "yaml" in compact:
        return {"intent": "config", "preferred": ["configs"], "penalized": ["code", "experiments"]}
    if "报告" in question or "实验" in question or "report" in compact or "evaluation" in compact:
        return {"intent": "experiments", "preferred": ["experiments"], "penalized": ["configs"]}
    if "note" in compact or "说明" in question:
        return {"intent": "notes", "preferred": ["notes"], "penalized": []}
    return {"intent": "general", "preferred": [], "penalized": []}


def entity_rows_for_event(connection: sqlite3.Connection, event_id: str) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT e.entity_id, e.name, e.normalized_name, e.entity_type, ee.role
        FROM event_entities ee
        JOIN entities e ON ee.entity_id = e.entity_id
        WHERE ee.event_id = ?
        ORDER BY e.normalized_name
        """,
        (event_id,),
    ).fetchall()


def retrieve(question: str, config: dict[str, Any]) -> dict[str, Any]:
    db_path = resolve_project_path(config["mini_demo"]["sqlite_db_path"])
    top_k = int(config.get("retrieval", {}).get("top_k", 5))
    clip_max = float(config.get("retrieval", {}).get("relation_score_clip_max", 120.0))
    query_entities = extract_query_entities(question)
    intent = infer_category_intent(question)
    connection = connect(db_path)

    seed_entities_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for query_entity in query_entities:
        rows = connection.execute(
            """
            SELECT entity_id, name, normalized_name, entity_type
            FROM entities
            WHERE normalized_name = ?
               OR normalized_name LIKE ?
               OR ? LIKE '%' || normalized_name || '%'
            ORDER BY
              CASE WHEN normalized_name = ? THEN 0 ELSE 1 END,
              LENGTH(normalized_name) DESC
            """,
            (query_entity, f"%{query_entity}%", query_entity, query_entity),
        ).fetchall()
        for row in rows:
            item = dict(row)
            item["query_entity"] = query_entity
            seed_entities_by_key.setdefault((item["entity_id"], query_entity), item)
    seed_entities = list(seed_entities_by_key.values())

    candidates: dict[str, dict[str, Any]] = {}
    relation_paths_by_event: dict[str, list[list[str]]] = defaultdict(list)

    for seed in seed_entities:
        event_rows = connection.execute(
            """
            SELECT ev.event_id, ev.chunk_id, ev.doc_id, ev.event_text, ev.source,
                   c.content_preview, d.category_dir, d.doc_type, d.file_ext, d.title
            FROM event_entities ee
            JOIN events ev ON ee.event_id = ev.event_id
            JOIN chunks c ON ev.chunk_id = c.chunk_id
            JOIN documents d ON ev.doc_id = d.doc_id
            WHERE ee.entity_id = ?
            """,
            (seed["entity_id"],),
        ).fetchall()
        for event in event_rows:
            add_candidate(candidates, event)
            path = [
                f"query_entity:{seed['query_entity']}",
                f"seed_entity:{seed['normalized_name']}",
                f"seed_event:{event['source']}",
            ]
            append_path(relation_paths_by_event[event["event_id"]], path)
            candidates[event["event_id"]]["relation_score_raw"] += 30.0
            append_unique(candidates[event["event_id"]]["relation_reasons"], f"exact_or_contained_entity_match:{seed['normalized_name']}")

            for shared in entity_rows_for_event(connection, event["event_id"]):
                shared_name = str(shared["normalized_name"])
                if shared_name in {"demo", "example", "small", "synthetic", "this"}:
                    continue
                expanded = connection.execute(
                    """
                    SELECT ev.event_id, ev.chunk_id, ev.doc_id, ev.event_text, ev.source,
                           c.content_preview, d.category_dir, d.doc_type, d.file_ext, d.title
                    FROM event_entities ee
                    JOIN events ev ON ee.event_id = ev.event_id
                    JOIN chunks c ON ev.chunk_id = c.chunk_id
                    JOIN documents d ON ev.doc_id = d.doc_id
                    WHERE ee.entity_id = ?
                    """,
                    (shared["entity_id"],),
                ).fetchall()
                for expanded_event in expanded:
                    add_candidate(candidates, expanded_event)
                    if expanded_event["event_id"] != event["event_id"]:
                        candidates[expanded_event["event_id"]]["relation_score_raw"] += 8.0
                        append_unique(
                            candidates[expanded_event["event_id"]]["relation_reasons"],
                            f"one_hop_shared_entity:{shared_name}",
                        )
                        append_path(
                            relation_paths_by_event[expanded_event["event_id"]],
                            [
                                f"query_entity:{seed['query_entity']}",
                                f"seed_entity:{seed['normalized_name']}",
                                f"seed_event:{event['source']}",
                                f"shared_entity:{shared_name}",
                                f"candidate_event:{expanded_event['source']}",
                            ],
                        )

    for event_id, candidate in candidates.items():
        source_stem = normalize_name(Path(candidate["source"]).stem)
        if source_stem in query_entities:
            candidate["relation_score_raw"] += 40.0
            append_unique(candidate["relation_reasons"], f"filename_entity_match:{source_stem}")

        category = candidate.get("category_dir", "")
        if category in intent["preferred"]:
            candidate["category_intent_adjustment"] = 35.0
            append_unique(candidate["fusion_reasons"], f"category_intent_bonus:{category}")
        elif category in intent["penalized"]:
            candidate["category_intent_adjustment"] = -25.0
            append_unique(candidate["fusion_reasons"], f"category_intent_penalty:{category}")
        else:
            candidate["category_intent_adjustment"] = 0.0

        raw = float(candidate["relation_score_raw"])
        normalized = min(raw, clip_max) / clip_max * 100.0
        candidate["relation_score_normalized"] = round(normalized, 6)
        candidate["final_score"] = round(normalized + float(candidate["category_intent_adjustment"]), 6)
        candidate["relation_path"] = relation_paths_by_event.get(event_id, [])[:3]
        candidate["relation_reasons"] = candidate["relation_reasons"][:8]
        append_unique(candidate["fusion_reasons"], "relation_score_normalized")
        candidate["retrieval_channels"] = ["sql_relation", "enhanced_sql_relation_mini"]

    final_sources = sorted(candidates.values(), key=lambda item: (-item["final_score"], item["source"]))[:top_k]
    for rank, candidate in enumerate(final_sources, start=1):
        candidate["rank"] = rank

    connection.close()
    return {
        "question": question,
        "query_entities": query_entities,
        "category_intent": intent,
        "final_sources": final_sources,
    }


def add_candidate(candidates: dict[str, dict[str, Any]], event: sqlite3.Row) -> None:
    event_id = str(event["event_id"])
    if event_id in candidates:
        return
    candidates[event_id] = {
        "event_id": event_id,
        "chunk_id": event["chunk_id"],
        "doc_id": event["doc_id"],
        "source": event["source"],
        "category_dir": event["category_dir"],
        "doc_type": event["doc_type"],
        "file_ext": event["file_ext"],
        "title": event["title"],
        "content_preview": event["content_preview"],
        "relation_score_raw": 0.0,
        "relation_score_normalized": 0.0,
        "category_intent_adjustment": 0.0,
        "final_score": 0.0,
        "relation_reasons": [],
        "relation_path": [],
        "fusion_reasons": [],
        "retrieval_channels": [],
    }


def append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def append_path(paths: list[list[str]], value: list[str]) -> None:
    if value not in paths:
        paths.append(value)


def run_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config() if config is None else config
    db_path = resolve_project_path(config["mini_demo"]["sqlite_db_path"])
    if not db_path.exists():
        build_index(config)

    results = []
    for query in config.get("queries", []):
        result = retrieve(str(query["question"]), config)
        top_source = result["final_sources"][0]["source"] if result["final_sources"] else ""
        expected_source = str(query.get("expected_source", ""))
        result["id"] = query.get("id", "")
        result["expected_source"] = expected_source
        result["top_source"] = top_source
        result["expected_source_hit"] = top_source.endswith(expected_source)
        results.append(result)

    payload = {"results": results}
    output_dir = resolve_project_path(config["mini_demo"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mini_demo_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "mini_demo_results.md").write_text(render_markdown(payload), encoding="utf-8")
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Mini Demo Results", ""]
    for result in payload["results"]:
        top = result["final_sources"][0] if result["final_sources"] else {}
        lines.extend(
            [
                f"## {result['id']}",
                "",
                f"- Question: {result['question']}",
                f"- Top source: {result.get('top_source', '')}",
                f"- Expected source: {result.get('expected_source', '')}",
                f"- Expected source hit: {result.get('expected_source_hit', False)}",
                f"- Relation reasons: {', '.join(top.get('relation_reasons', []))}",
                f"- Fusion reasons: {', '.join(top.get('fusion_reasons', []))}",
                "- Relation path:",
            ]
        )
        for path in top.get("relation_path", [])[:1]:
            lines.append(f"  - {' -> '.join(path)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = run_demo()
    for result in payload["results"]:
        print(
            f"{result['id']}: {result['question']} -> {result['top_source']} "
            f"(expected: {result['expected_source']}, hit={result['expected_source_hit']})"
        )
        top = result["final_sources"][0]
        if top.get("relation_path"):
            print("  relation_path:", " -> ".join(top["relation_path"][0]))
        print("  fusion_reasons:", ", ".join(top.get("fusion_reasons", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
