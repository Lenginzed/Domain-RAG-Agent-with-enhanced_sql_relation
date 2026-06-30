from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_eval_v25c.yaml"

METRICS = [
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_8",
    "mrr",
    "category_hit_rate",
    "required_all_categories_hit_rate",
    "source_diversity_pass_rate",
    "expected_source_absent_count",
]


def main() -> int:
    configure_stdout()
    try:
        result = compare_retrieval_eval()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001
        print("V2.5b vs V2.5c retrieval eval comparison failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compare_retrieval_eval(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    v25b = load_json(resolve_project_path(config["inputs"]["old_v25b_eval_json"]))
    v25c = load_json(resolve_project_path(config["outputs"]["eval_results_json"]))
    rows: list[dict[str, Any]] = []
    for mode in v25c.get("modes", []):
        old_metrics = v25b.get("metrics_by_mode", {}).get(mode, {})
        new_metrics = v25c.get("metrics_by_mode", {}).get(mode, {})
        for metric in METRICS:
            old = metric_value(metric, old_metrics)
            new = metric_value(metric, new_metrics)
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "v25b_value": old,
                    "v25c_value": new,
                    "delta": round(new - old, 6),
                    "change": classify_change(metric, new - old),
                    "note": metric_note(metric, old_metrics, new_metrics),
                }
            )

    enhanced_keyword = [
        row
        for row in rows
        if row["mode"] == "enhanced_keyword"
        and row["metric"] in {"hit_at_5", "mrr", "source_diversity_pass_rate", "expected_source_absent_count"}
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_scope_note": (
            "V2.5b and V2.5c use different evaluation sets. Deltas show evaluation-set alignment effects, "
            "not strict model or retrieval algorithm improvement."
        ),
        "v25b_json": str(resolve_project_path(config["inputs"]["old_v25b_eval_json"])),
        "v25c_json": str(resolve_project_path(config["outputs"]["eval_results_json"])),
        "v25b_collection": v25b.get("collection_name", ""),
        "v25c_collection": v25c.get("collection_name", ""),
        "v25b_total_questions": v25b.get("total_questions", 0),
        "v25c_total_questions": v25c.get("total_questions", 0),
        "v25b_expected_source_absent_question_ids": v25b.get("expected_source_absent_question_ids", []),
        "v25c_expected_source_absent_question_ids": v25c.get("expected_source_absent_question_ids", []),
        "comparison_rows": rows,
        "enhanced_keyword_summary": enhanced_keyword,
        "alignment_fixed_expected_source_absent": (
            len(v25b.get("expected_source_absent_question_ids", [])) > 0
            and len(v25c.get("expected_source_absent_question_ids", [])) == 0
        ),
    }


def metric_value(metric: str, metrics: dict[str, Any]) -> float:
    if metric == "expected_source_absent_count":
        return float(metrics.get(metric, 0))
    return float(metrics.get(metric, 0.0))


def classify_change(metric: str, delta: float) -> str:
    if metric == "expected_source_absent_count":
        delta = -delta
    if delta > 1e-9:
        return "improved"
    if delta < -1e-9:
        return "degraded"
    return "unchanged"


def metric_note(metric: str, old_metrics: dict[str, Any], new_metrics: dict[str, Any]) -> str:
    if metric.startswith("hit_at") or metric == "mrr":
        return (
            "different_eval_sets; "
            f"v25b_source_metric_eligible={old_metrics.get('source_metric_eligible_count', '')}; "
            f"v25c_source_metric_eligible={new_metrics.get('source_metric_eligible_count', '')}"
        )
    if metric == "expected_source_absent_count":
        return "V2.5c should be 0 by construction."
    return "different_eval_sets"


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    output_json = resolve_project_path(config["outputs"]["compare_json"])
    output_csv = resolve_project_path(config["outputs"]["compare_csv"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["mode", "metric", "v25b_value", "v25c_value", "delta", "change", "note"],
        )
        writer.writeheader()
        for row in result["comparison_rows"]:
            writer.writerow(row)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return data


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def print_summary(result: dict[str, Any]) -> None:
    print("V2.5b vs V2.5c retrieval eval comparison completed.")
    print(
        json.dumps(
            {
                "v25b_collection": result["v25b_collection"],
                "v25c_collection": result["v25c_collection"],
                "v25b_expected_source_absent_question_ids": result["v25b_expected_source_absent_question_ids"],
                "v25c_expected_source_absent_question_ids": result["v25c_expected_source_absent_question_ids"],
                "alignment_fixed_expected_source_absent": result["alignment_fixed_expected_source_absent"],
                "enhanced_keyword_summary": result["enhanced_keyword_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
