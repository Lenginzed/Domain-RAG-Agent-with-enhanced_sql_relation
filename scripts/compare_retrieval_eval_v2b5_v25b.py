from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "expanded_ingest_v25b.yaml"
BASELINE_JSON = PROJECT_ROOT / "storage" / "logs" / "retrieval_eval_v2b5.json"
METRICS = [
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_8",
    "mrr",
    "category_hit_rate",
    "required_all_categories_hit_rate",
    "source_diversity_pass_rate",
]


def main() -> int:
    configure_stdout()
    try:
        result = compare_retrieval_eval()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001
        print("Retrieval eval comparison failed.")
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
    v25b_json = resolve_project_path(config["outputs"]["retrieval_eval_json"])
    baseline = load_json(BASELINE_JSON)
    current = load_json(v25b_json)
    rows: list[dict[str, Any]] = []
    for mode in current.get("modes", []):
        baseline_metrics = baseline.get("metrics_by_mode", {}).get(mode, {})
        current_metrics = current.get("metrics_by_mode", {}).get(mode, {})
        for metric in METRICS:
            old = float(baseline_metrics.get(metric, 0.0))
            new = float(current_metrics.get(metric, 0.0))
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "v2b5_value": old,
                    "v25b_value": new,
                    "delta": round(new - old, 6),
                    "change": classify_change(new - old),
                    "note": metric_note(metric, current_metrics),
                }
            )
    enhanced_keyword = [
        row for row in rows if row["mode"] == "enhanced_keyword" and row["metric"] in {"hit_at_5", "mrr", "source_diversity_pass_rate"}
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_json": str(BASELINE_JSON),
        "v25b_json": str(v25b_json),
        "baseline_collection": baseline.get("collection_name", ""),
        "v25b_collection": current.get("collection_name", ""),
        "baseline_persist_directory": baseline.get("persist_directory", ""),
        "v25b_persist_directory": current.get("persist_directory", ""),
        "expected_source_absent_question_ids": current.get("expected_source_absent_question_ids", []),
        "expected_category_absent_question_ids": current.get("expected_category_absent_question_ids", []),
        "comparison_rows": rows,
        "enhanced_keyword_summary": enhanced_keyword,
        "enhanced_keyword_stable": all(
            row["change"] in {"improved", "unchanged"} for row in enhanced_keyword if row["metric"] != "mrr"
        ),
    }


def classify_change(delta: float) -> str:
    if delta > 1e-9:
        return "improved"
    if delta < -1e-9:
        return "degraded"
    return "unchanged"


def metric_note(metric: str, current_metrics: dict[str, Any]) -> str:
    if metric.startswith("hit_at") or metric == "mrr":
        return f"source_metric_eligible_count={current_metrics.get('source_metric_eligible_count', '')}; expected_source_absent_count={current_metrics.get('expected_source_absent_count', '')}"
    if metric == "required_all_categories_hit_rate":
        return f"required_all_categories_eligible_count={current_metrics.get('required_all_categories_eligible_count', '')}"
    return ""


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
            fieldnames=["mode", "metric", "v2b5_value", "v25b_value", "delta", "change", "note"],
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
    print("Retrieval eval comparison completed.")
    print(json.dumps({
        "baseline_collection": result["baseline_collection"],
        "v25b_collection": result["v25b_collection"],
        "expected_source_absent_question_ids": result["expected_source_absent_question_ids"],
        "enhanced_keyword_summary": result["enhanced_keyword_summary"],
        "enhanced_keyword_stable": result["enhanced_keyword_stable"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
