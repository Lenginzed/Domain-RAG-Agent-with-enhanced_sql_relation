from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sql_sag.entity_extractor import normalize_entity_name  # noqa: E402
from src.sql_sag.relation_retriever import load_sql_sag_config, retrieve_sql_relation, write_json  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4c.yaml"


def main() -> None:
    config = load_sql_sag_config(CONFIG_PATH)
    cases = load_or_write_cases(config)
    per_case: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    for case in cases:
        result = retrieve_sql_relation(str(case["question"]), config)
        metrics = evaluate_case(case, result)
        per_case.append(metrics)
        jsonl_rows.append({"case": case, "metrics": metrics, "result": result})

    summary = summarize_eval(per_case)
    output = {
        **summary,
        "cases": per_case,
        "llm_called": False,
        "chroma_written": False,
        "sqlite_readonly": True,
    }

    outputs = config["outputs"]
    write_json(outputs["eval_json"], output)
    write_jsonl(PROJECT_ROOT / outputs["eval_jsonl"], jsonl_rows)
    write_csv(PROJECT_ROOT / outputs["eval_csv"], per_case)
    write_eval_markdown(PROJECT_ROOT / outputs["eval_markdown"], output)

    print("=== SQL Relation Retriever V4c Eval ===")
    for key in [
        "total_questions",
        "source_evaluable_questions",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "expected_entity_hit_rate",
        "expected_category_hit_rate",
        "relation_path_found_rate",
        "avg_seed_entities",
        "avg_seed_events",
        "avg_expanded_events",
    ]:
        print(f"{key}: {output[key]}")
    print(f"eval_json: {PROJECT_ROOT / outputs['eval_json']}")
    print(f"eval_markdown: {PROJECT_ROOT / outputs['eval_markdown']}")


def load_or_write_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    regression = config["regression_cases"]
    output_path = PROJECT_ROOT / regression["output_jsonl"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = [dict(row) for row in regression["cases"]]
    write_jsonl(output_path, cases)
    return cases


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    candidates = result["candidates"]
    expected_sources = [str(item) for item in case.get("expected_sources_contains", [])]
    expected_entities = {normalize_entity_name(item) for item in case.get("expected_entities", [])}
    expected_categories = {str(item) for item in case.get("expected_categories", [])}

    first_hit_rank = 0
    if expected_sources:
        for candidate in candidates:
            source_blob = " ".join(
                [
                    str(candidate.get("source", "")),
                    str(candidate.get("imported_source", "")),
                    str(candidate.get("original_source", "")),
                ]
            ).lower()
            if any(expected.lower() in source_blob for expected in expected_sources):
                first_hit_rank = int(candidate["rank"])
                break

    query_entities = {row["normalized_name"] for row in result["query_entities"]}
    seed_entities = {row["normalized_name"] for row in result["seed_entities"]}
    candidate_entities = {name for candidate in candidates for name in candidate.get("matched_entities", [])}
    categories = {str(candidate.get("category_dir", "")) for candidate in candidates}
    expected_entity_hit = bool(expected_entities & (query_entities | seed_entities | candidate_entities))
    expected_category_hit = bool(expected_categories & categories) if expected_categories else False

    return {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "expected_sources_contains": ";".join(expected_sources),
        "expected_entities": ";".join(sorted(expected_entities)),
        "expected_categories": ";".join(sorted(expected_categories)),
        "query_entities": ";".join(row["normalized_name"] for row in result["query_entities"]),
        "seed_entity_count": result["debug"]["seed_entity_count"],
        "seed_event_count": result["debug"]["seed_event_count"],
        "expanded_event_count": result["debug"]["expanded_event_count"],
        "candidate_count": result["debug"]["candidate_count"],
        "first_source_hit_rank": first_hit_rank,
        "hit_at_1": bool(first_hit_rank and first_hit_rank <= 1),
        "hit_at_3": bool(first_hit_rank and first_hit_rank <= 3),
        "hit_at_5": bool(first_hit_rank and first_hit_rank <= 5),
        "mrr": round(1 / first_hit_rank, 6) if first_hit_rank else 0.0,
        "expected_entity_hit": expected_entity_hit,
        "expected_category_hit": expected_category_hit,
        "relation_path_found": any(candidate.get("relation_path") for candidate in candidates),
        "top_source": candidates[0]["source"] if candidates else "",
        "top_relation_score": candidates[0]["relation_score"] if candidates else 0.0,
        "top_relation_path": " -> ".join(candidates[0]["relation_path"]) if candidates else "",
    }


def summarize_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    source_rows = [row for row in rows if row.get("expected_sources_contains")]
    source_total = len(source_rows)
    return {
        "total_questions": total,
        "source_evaluable_questions": source_total,
        "hit_at_1": rate(source_rows, "hit_at_1"),
        "hit_at_3": rate(source_rows, "hit_at_3"),
        "hit_at_5": rate(source_rows, "hit_at_5"),
        "mrr": round(sum(float(row["mrr"]) for row in source_rows) / max(1, source_total), 6),
        "expected_entity_hit_rate": rate(rows, "expected_entity_hit"),
        "expected_category_hit_rate": rate(rows, "expected_category_hit"),
        "relation_path_found_rate": rate(rows, "relation_path_found"),
        "avg_seed_entities": round(sum(int(row["seed_entity_count"]) for row in rows) / max(1, total), 4),
        "avg_seed_events": round(sum(int(row["seed_event_count"]) for row in rows) / max(1, total), 4),
        "avg_expanded_events": round(sum(int(row["expanded_event_count"]) for row in rows) / max(1, total), 4),
    }


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return round(sum(1 for row in rows if row.get(field)) / max(1, len(rows)), 6)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_eval_markdown(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SQL Relation Retriever V4c Smoke Eval",
        "",
        f"- total_questions: {output['total_questions']}",
        f"- source_evaluable_questions: {output['source_evaluable_questions']}",
        f"- Hit@1: {output['hit_at_1']}",
        f"- Hit@3: {output['hit_at_3']}",
        f"- Hit@5: {output['hit_at_5']}",
        f"- MRR: {output['mrr']}",
        f"- expected_entity_hit_rate: {output['expected_entity_hit_rate']}",
        f"- expected_category_hit_rate: {output['expected_category_hit_rate']}",
        f"- relation_path_found_rate: {output['relation_path_found_rate']}",
        "",
        "## Cases",
        "",
    ]
    for row in output["cases"]:
        lines.extend(
            [
                f"### {row['id']}",
                "",
                f"- question: {row['question']}",
                f"- first_source_hit_rank: {row['first_source_hit_rank']}",
                f"- expected_entity_hit: {row['expected_entity_hit']}",
                f"- expected_category_hit: {row['expected_category_hit']}",
                f"- top_source: `{row['top_source']}`",
                f"- top_relation_score: {row['top_relation_score']}",
                f"- top_relation_path: {row['top_relation_path']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
