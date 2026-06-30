from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_calibration_v25e.yaml"

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
        print("V25c vs V25e retrieval eval comparison failed.")
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
    before = load_json(resolve_project_path(config["inputs"]["retrieval_eval_json"]))
    after = load_json(resolve_project_path(config["outputs"]["calibrated_eval_json"]))
    rows: list[dict[str, Any]] = []
    for mode in after.get("modes", []):
        old_metrics = before.get("metrics_by_mode", {}).get(mode, {})
        new_metrics = after.get("metrics_by_mode", {}).get(mode, {})
        for metric in METRICS:
            old = metric_value(metric, old_metrics)
            new = metric_value(metric, new_metrics)
            delta = round(new - old, 6)
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "v25c_value": old,
                    "v25e_value": new,
                    "delta": delta,
                    "change": classify_change(metric, delta),
                    "note": metric_note(metric),
                }
            )

    target_case_rows = compare_target_cases(before, after, config.get("target_cases", []))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_scope_note": (
            "V25e uses the same V25c eval set and same 80-file collection, with light score calibration "
            "enabled only for configured fusion modes."
        ),
        "v25c_json": str(resolve_project_path(config["inputs"]["retrieval_eval_json"])),
        "v25e_json": str(resolve_project_path(config["outputs"]["calibrated_eval_json"])),
        "collection_name": after.get("collection_name", ""),
        "calibration_apply_to_modes": after.get("calibration_apply_to_modes", []),
        "comparison_rows": rows,
        "target_case_comparison": target_case_rows,
        "enhanced_keyword_summary": [
            row
            for row in rows
            if row["mode"] == "enhanced_keyword"
            and row["metric"] in {"hit_at_1", "hit_at_5", "hit_at_8", "mrr", "source_diversity_pass_rate"}
        ],
        "obvious_degradation_detected": detect_obvious_degradation(rows),
    }


def compare_target_cases(before: dict[str, Any], after: dict[str, Any], target_cases: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for case in target_cases:
        case_id = str(case.get("id"))
        output[case_id] = {}
        for mode in after.get("modes", []):
            before_record = find_record(before, case_id, mode)
            after_record = find_record(after, case_id, mode)
            output[case_id][mode] = {
                "expected_source": case.get("expected_source", ""),
                "before_first_source_hit_rank": (before_record or {}).get("first_source_hit_rank"),
                "after_first_source_hit_rank": (after_record or {}).get("first_source_hit_rank"),
                "before_hit_at_8": (before_record or {}).get("source_hit_at_8"),
                "after_hit_at_8": (after_record or {}).get("source_hit_at_8"),
                "before_hit_at_5": (before_record or {}).get("source_hit_at_5"),
                "after_hit_at_5": (after_record or {}).get("source_hit_at_5"),
                "calibration_applied": (after_record or {}).get("calibration_applied"),
                "before_top8": summarize_top_sources((before_record or {}).get("final_sources", [])),
                "after_top8": summarize_top_sources((after_record or {}).get("final_sources", [])),
            }
    return output


def summarize_top_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources[:8]:
        rows.append(
            {
                "rank": source.get("rank"),
                "source": source.get("source"),
                "category_dir": source.get("category_dir"),
                "final_score": source.get("final_score"),
                "calibration_score": source.get("calibration_score", 0.0),
                "calibration_reasons": source.get("calibration_reasons", []),
            }
        )
    return rows


def find_record(data: dict[str, Any], item_id: str, mode: str) -> dict[str, Any] | None:
    for record in data.get("per_question_results", []):
        if str(record.get("id")) == item_id and str(record.get("retrieval_mode")) == mode:
            return record
    return None


def metric_value(metric: str, metrics: dict[str, Any]) -> float:
    return float(metrics.get(metric, 0.0) or 0.0)


def classify_change(metric: str, delta: float) -> str:
    if metric == "expected_source_absent_count":
        delta = -delta
    if delta > 1e-9:
        return "improved"
    if delta < -1e-9:
        return "degraded"
    return "unchanged"


def metric_note(metric: str) -> str:
    if metric in {"hit_at_1", "hit_at_3", "hit_at_5", "hit_at_8", "mrr"}:
        return "same_eval_set_same_collection; differences are from calibration only"
    if metric == "expected_source_absent_count":
        return "should remain 0 for V25c aligned eval"
    return "same_eval_set_same_collection"


def detect_obvious_degradation(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row["metric"] in {"hit_at_5", "hit_at_8", "source_diversity_pass_rate"} and row["delta"] < -0.05:
            return True
    return False


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
            fieldnames=["mode", "metric", "v25c_value", "v25e_value", "delta", "change", "note"],
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
    print("V25c vs V25e retrieval eval comparison completed.")
    print(
        json.dumps(
            {
                "calibration_apply_to_modes": result["calibration_apply_to_modes"],
                "enhanced_keyword_summary": result["enhanced_keyword_summary"],
                "target_case_comparison": result["target_case_comparison"],
                "obvious_degradation_detected": result["obvious_degradation_detected"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
