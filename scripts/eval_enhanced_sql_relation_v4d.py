from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.real_smoke_retriever import retrieve_real_smoke  # noqa: E402
from src.sql_sag.fusion_retriever import (  # noqa: E402
    load_v4d_config,
    resolve_project_path,
    retrieve_enhanced_sql_relation,
    write_csv,
    write_json,
    write_jsonl,
)
from src.sql_sag.relation_retriever import retrieve_sql_relation  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4d.yaml"


def main() -> None:
    config = load_v4d_config(CONFIG_PATH)
    relation_cases = load_jsonl(PROJECT_ROOT / config["eval"]["relation_eval_jsonl"])
    v25c_cases = select_v25c_cases(PROJECT_ROOT / config["eval"]["retrieval_eval_v25c_jsonl"], int(config["eval"].get("selected_v25c_case_limit", 12)))
    all_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []

    for case in relation_cases:
        for mode in ["enhanced_keyword", "sql_relation", "enhanced_sql_relation"]:
            result = run_mode(case["question"], mode, config)
            row = evaluate_result(case, result, mode, "v4c_relation")
            all_rows.append(row)
            jsonl_rows.append({"case": case, "mode": mode, "metrics": row, "result": result})

    for case in v25c_cases:
        for mode in ["enhanced_keyword", "enhanced_sql_relation"]:
            result = run_mode(case["question"], mode, config)
            row = evaluate_result(case, result, mode, "v25c_subset")
            all_rows.append(row)
            jsonl_rows.append({"case": case, "mode": mode, "metrics": row, "result": result})

    summary = summarize_rows(all_rows)
    comparison = build_compare(summary, all_rows)
    outputs = config["outputs"]
    write_json(outputs["eval_json"], {"summary": summary, "rows": all_rows, "llm_called": False, "chroma_written": False})
    write_jsonl(outputs["eval_jsonl"], jsonl_rows)
    write_csv(outputs["eval_csv"], all_rows)
    write_json(outputs["compare_json"], comparison)
    write_csv(outputs["compare_csv"], comparison["rows"])
    write_report(PROJECT_ROOT / outputs["report_md"], summary, comparison, all_rows)

    print("=== Enhanced SQL Relation V4d Eval ===")
    for mode, metrics in summary["metrics_by_mode"].items():
        print(f"[{mode}] {metrics}")
    print("reward_risk_case:", comparison["reward_risk_case"])
    print("regression_degradation_count:", comparison["regression_degradation_count"])
    print(f"eval_json: {PROJECT_ROOT / outputs['eval_json']}")
    print(f"compare_json: {PROJECT_ROOT / outputs['compare_json']}")


def run_mode(question: str, mode: str, config: dict[str, Any]) -> dict[str, Any]:
    if mode == "enhanced_sql_relation":
        result = retrieve_enhanced_sql_relation(question, config)
        return {"final_sources": result["final_sources"], "debug": result}
    if mode == "sql_relation":
        sql_result = retrieve_sql_relation(question, config["sql_relation_config"])
        return {"final_sources": sql_result["candidates"], "debug": sql_result}
    if mode == "enhanced_keyword":
        collection = config["collection"]
        retrieval_config = config["retrieval"]
        result = retrieve_real_smoke(
            question=question,
            persist_directory=resolve_project_path(collection["persist_directory"]),
            collection_name=collection["collection_name"],
            retrieval_mode="enhanced_keyword",
            dense_top_k=int(retrieval_config.get("dense_top_k", 5)),
            metadata_top_k=int(retrieval_config.get("metadata_top_k", 8)),
            keyword_top_k=int(retrieval_config.get("keyword_top_k", 8)),
            final_top_k=int(retrieval_config.get("final_top_k", 8)),
            max_chunks_per_source=int(retrieval_config.get("max_chunks_per_source", 2)),
            enable_source_diversity=bool(retrieval_config.get("enable_source_diversity", True)),
            source_key=str(retrieval_config.get("source_key", "imported_source")),
            calibration_config=config.get("calibration"),
        )
        return {"final_sources": result.debug["final_sources"], "debug": result.debug}
    raise ValueError(mode)


def evaluate_result(case: dict[str, Any], result: dict[str, Any], mode: str, case_group: str) -> dict[str, Any]:
    final_sources = result.get("final_sources", [])
    expected_sources = [str(item) for item in case.get("expected_sources_contains", [])]
    expected_categories = [str(item) for item in case.get("expected_categories", [])]
    required_all = [str(item) for item in case.get("required_all_categories", [])]
    first_rank = first_source_rank(final_sources, expected_sources)
    categories = {str(row.get("category_dir", "")) for row in final_sources}
    relation_path_found = any(row.get("relation_path") for row in final_sources)
    sql_only = any("sql_relation" in row.get("retrieval_channels", []) and "enhanced_keyword" not in row.get("retrieval_channels", []) for row in final_sources)
    diversity_pass = source_diversity_pass(final_sources)
    return {
        "case_group": case_group,
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "mode": mode,
        "expected_sources_contains": ";".join(expected_sources),
        "expected_categories": ";".join(expected_categories),
        "first_source_hit_rank": first_rank,
        "hit_at_1": bool(first_rank and first_rank <= 1),
        "hit_at_3": bool(first_rank and first_rank <= 3),
        "hit_at_5": bool(first_rank and first_rank <= 5),
        "hit_at_8": bool(first_rank and first_rank <= 8),
        "mrr": round(1 / first_rank, 6) if first_rank else 0.0,
        "category_hit": bool(set(expected_categories) & categories) if expected_categories else False,
        "required_all_hit": set(required_all).issubset(categories) if required_all else True,
        "source_diversity": diversity_pass,
        "relation_path_found": relation_path_found,
        "sql_only_candidate_present": sql_only,
        "top_source": final_sources[0].get("source", "") if final_sources else "",
        "top_category": final_sources[0].get("category_dir", "") if final_sources else "",
        "top_score": final_sources[0].get("final_score", final_sources[0].get("relation_score", 0)) if final_sources else 0,
    }


def first_source_rank(final_sources: list[dict[str, Any]], expected_sources: list[str]) -> int:
    if not expected_sources:
        return 0
    for idx, row in enumerate(final_sources, start=1):
        blob = " ".join(str(row.get(key, "")) for key in ["source", "imported_source", "original_source"]).lower()
        if any(expected.lower() in blob for expected in expected_sources):
            return idx
    return 0


def source_diversity_pass(final_sources: list[dict[str, Any]]) -> bool:
    counts: dict[str, int] = {}
    for row in final_sources:
        source = str(row.get("imported_source") or row.get("source") or "")
        counts[source] = counts.get(source, 0) + 1
    return all(count <= 2 for count in counts.values())


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        source_rows = [row for row in mode_rows if row["expected_sources_contains"]]
        metrics_by_mode[mode] = {
            "total_cases": len(mode_rows),
            "source_evaluable_cases": len(source_rows),
            "hit_at_1": rate(source_rows, "hit_at_1"),
            "hit_at_3": rate(source_rows, "hit_at_3"),
            "hit_at_5": rate(source_rows, "hit_at_5"),
            "hit_at_8": rate(source_rows, "hit_at_8"),
            "mrr": round(sum(float(row["mrr"]) for row in source_rows) / max(1, len(source_rows)), 6),
            "category_hit": rate(mode_rows, "category_hit"),
            "required_all": rate(mode_rows, "required_all_hit"),
            "source_diversity": rate(mode_rows, "source_diversity"),
            "relation_path_found_rate": rate(mode_rows, "relation_path_found"),
            "sql_only_candidate_contribution_rate": rate(mode_rows, "sql_only_candidate_present"),
        }
    return {"metrics_by_mode": metrics_by_mode, "total_rows": len(rows)}


def build_compare(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    reward_rows = [row for row in rows if row["id"] == "sql_v4c_q005"]
    by_mode = {row["mode"]: row for row in reward_rows}
    regressions = [row for row in rows if row["case_group"] == "v25c_subset"]
    degradation_count = 0
    for case_id in sorted({row["id"] for row in regressions}):
        enhanced = next((row for row in regressions if row["id"] == case_id and row["mode"] == "enhanced_keyword"), None)
        fused = next((row for row in regressions if row["id"] == case_id and row["mode"] == "enhanced_sql_relation"), None)
        if enhanced and fused and enhanced["hit_at_5"] and not fused["hit_at_5"]:
            degradation_count += 1
    compare_rows: list[dict[str, Any]] = []
    for mode, metrics in summary["metrics_by_mode"].items():
        compare_rows.append({"mode": mode, **metrics})
    return {
        "rows": compare_rows,
        "reward_risk_case": {
            mode: {
                "top_source": row["top_source"],
                "top_category": row["top_category"],
                "category_hit": row["category_hit"],
                "top_score": row["top_score"],
            }
            for mode, row in by_mode.items()
        },
        "regression_degradation_count": degradation_count,
    }


def select_v25c_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    cases = [row for row in load_jsonl(path) if row.get("query_type") != "negative" and row.get("expected_sources_contains")]
    preferred_ids = {"v25c_q014", "v25c_q001", "v25c_q002", "v25c_q003", "v25c_q005"}
    selected = [row for row in cases if row.get("id") in preferred_ids]
    for row in cases:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)
    return selected[:limit]


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(1 for row in rows if row.get(key)) / max(1, len(rows)), 6)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_report(path: Path, summary: dict[str, Any], comparison: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Enhanced SQL Relation V4d Eval", "", "## Metrics By Mode", ""]
    for mode, metrics in summary["metrics_by_mode"].items():
        lines.append(f"### {mode}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## reward/risk Case", ""])
    for mode, row in comparison["reward_risk_case"].items():
        lines.append(f"- {mode}: top={row['top_source']} category={row['top_category']} category_hit={row['category_hit']}")
    lines.extend(["", "## Per Case Rows", ""])
    for row in rows:
        lines.append(
            f"- {row['case_group']} / {row['id']} / {row['mode']}: "
            f"hit@5={row['hit_at_5']} category_hit={row['category_hit']} top={row['top_source']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
