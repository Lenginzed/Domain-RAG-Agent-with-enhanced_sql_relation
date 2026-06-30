from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sql_sag.relation_retriever import load_sql_sag_config, retrieve_sql_relation, write_json  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4c.yaml"
DEFAULT_QUESTION = "missile engine 相关说明在哪份 notes 文件中？"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQL SAG-lite relation retriever V4c.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    config = load_sql_sag_config(args.config)
    result = retrieve_sql_relation(args.question, config)
    outputs = config["outputs"]
    write_json(outputs["single_debug_json"], result)
    write_debug_markdown(PROJECT_ROOT / outputs["single_debug_md"], result)

    print("=== SQL Relation Retriever V4c ===")
    print(f"question: {result['question']}")
    print(f"query_entities: {[row['normalized_name'] for row in result['query_entities']]}")
    print(f"seed_entities: {result['debug']['seed_entity_count']}")
    print(f"seed_events: {result['debug']['seed_event_count']}")
    print(f"expanded_events: {result['debug']['expanded_event_count']}")
    print(f"candidates: {result['debug']['candidate_count']}")
    for candidate in result["candidates"][:8]:
        print(
            f"rank={candidate['rank']} score={candidate['relation_score']} "
            f"source={candidate['source']} category={candidate['category_dir']}"
        )
        print(f"  reasons: {'; '.join(candidate['relation_reasons'])}")
        print(f"  path: {' -> '.join(candidate['relation_path'])}")
    print(f"debug_json: {PROJECT_ROOT / outputs['single_debug_json']}")
    print(f"debug_md: {PROJECT_ROOT / outputs['single_debug_md']}")


def write_debug_markdown(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SQL Relation Retriever V4c Single Debug",
        "",
        f"- question: {result['question']}",
        f"- query_entities: {', '.join(row['normalized_name'] for row in result['query_entities'])}",
        f"- seed_entity_count: {result['debug']['seed_entity_count']}",
        f"- seed_event_count: {result['debug']['seed_event_count']}",
        f"- expanded_event_count: {result['debug']['expanded_event_count']}",
        f"- candidate_count: {result['debug']['candidate_count']}",
        "",
        "## Top Candidates",
        "",
    ]
    for candidate in result["candidates"]:
        lines.extend(
            [
                f"### Rank {candidate['rank']} | score={candidate['relation_score']}",
                "",
                f"- source: `{candidate['source']}`",
                f"- category_dir: `{candidate['category_dir']}`",
                f"- doc_type: `{candidate['doc_type']}`",
                f"- chunk_id: `{candidate['chunk_id']}`",
                f"- reasons: {'; '.join(candidate['relation_reasons'])}",
                f"- relation_path: {' -> '.join(candidate['relation_path'])}",
                f"- preview: {candidate.get('content_preview', '')[:500]}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
