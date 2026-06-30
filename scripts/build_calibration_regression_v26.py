from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_calibration_v26.yaml"

SELECTED_EXISTING_IDS = [
    "v25c_q001",
    "v25c_q002",
    "v25c_q005",
    "v25c_q006",
    "v25c_q009",
    "v25c_q013",
    "v25c_q014",
    "v25c_q015",
    "v25c_q017",
    "v25c_q018",
    "v25c_q021",
    "v25c_q022",
]


def main() -> int:
    configure_stdout()
    try:
        result = build_regression_set()
    except Exception as exc:  # noqa: BLE001
        print("V2.6 calibration regression set build failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_regression_set(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    eval_items = load_jsonl(resolve_project_path(config["inputs"]["base_eval_set_jsonl"]))
    expanded_sources = load_expanded_sources(resolve_project_path(config["inputs"]["expanded_files_csv"]))
    by_id = {str(item["id"]): item for item in eval_items}

    selected: list[dict[str, Any]] = []
    missing_ids = [item_id for item_id in SELECTED_EXISTING_IDS if item_id not in by_id]
    if missing_ids:
        raise ValueError(f"Selected eval ids missing from V25c set: {missing_ids}")

    for item_id in SELECTED_EXISTING_IDS:
        item = dict(by_id[item_id])
        missing_sources = missing_expected_sources(item, expanded_sources)
        if missing_sources:
            raise ValueError(f"Expected source absent for {item_id}: {missing_sources}")
        item["calibration_case_type"] = "source_filename"
        item["calibration_focus"] = infer_case_type(item)
        item["notes"] = f"{item.get('notes', '')} calibration regression case".strip()
        selected.append(item)

    output_jsonl = resolve_project_path(config["outputs"]["regression_set_jsonl"])
    output_manifest = resolve_project_path(config["outputs"]["regression_manifest_json"])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as file:
        for item in selected:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "base_eval_set": str(resolve_project_path(config["inputs"]["base_eval_set_jsonl"])),
        "expanded_files_csv": str(resolve_project_path(config["inputs"]["expanded_files_csv"])),
        "total_questions": len(selected),
        "selected_existing_ids": SELECTED_EXISTING_IDS,
        "contains_v25c_q014": any(str(item["id"]) == "v25c_q014" for item in selected),
        "query_type_counts": dict(Counter(str(item.get("query_type", "")) for item in selected)),
        "case_type_counts": dict(Counter(str(item.get("calibration_case_type", "")) for item in selected)),
        "focus_counts": dict(Counter(str(item.get("calibration_focus", "")) for item in selected)),
        "expected_category_counts": dict(
            Counter(category for item in selected for category in item.get("expected_categories", []))
        ),
        "expected_source_absent_count": 0,
        "questions": [
            {
                "id": item["id"],
                "query_type": item["query_type"],
                "calibration_case_type": item["calibration_case_type"],
                "calibration_focus": item["calibration_focus"],
                "expected_sources_contains": item["expected_sources_contains"],
                "expected_categories": item["expected_categories"],
            }
            for item in selected
        ],
        "outputs": {
            "regression_set_jsonl": str(output_jsonl),
            "regression_manifest_json": str(output_manifest),
        },
    }
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def infer_case_type(item: dict[str, Any]) -> str:
    categories = set(str(category) for category in item.get("expected_categories", []))
    if "code" in categories:
        return "code_file"
    if "configs" in categories:
        return "config_file"
    if "notes" in categories:
        return "note_file"
    if categories.intersection({"papers", "thesis_or_reports"}):
        return "paper_or_report"
    return "source_filename"


def missing_expected_sources(item: dict[str, Any], expanded_sources: set[str]) -> list[str]:
    source_blob = "\n".join(expanded_sources).lower()
    return [
        str(source).lower()
        for source in item.get("expected_sources_contains", [])
        if str(source).lower() not in source_blob
    ]


def load_expanded_sources(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {str(row.get("source", "")) for row in csv.DictReader(file)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def print_summary(result: dict[str, Any]) -> None:
    print("V2.6 calibration regression set built.")
    print(
        json.dumps(
            {
                "total_questions": result["total_questions"],
                "contains_v25c_q014": result["contains_v25c_q014"],
                "query_type_counts": result["query_type_counts"],
                "case_type_counts": result["case_type_counts"],
                "expected_source_absent_count": result["expected_source_absent_count"],
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
