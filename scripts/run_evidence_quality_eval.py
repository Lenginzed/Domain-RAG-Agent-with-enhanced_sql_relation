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
CONFIG_PATH = PROJECT_ROOT / "config" / "evidence_quality.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evidence_quality import score_answer_record


def main() -> int:
    configure_stdout()
    try:
        result = run_evidence_quality_eval()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001 - evidence scoring should fail explicitly.
        print("Evidence quality evaluation failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_evidence_quality_eval(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    input_json = PROJECT_ROOT / str(config["input_answer_eval_json"])
    input_jsonl = PROJECT_ROOT / str(config["input_answer_eval_jsonl"])
    input_csv = PROJECT_ROOT / str(config["input_answer_eval_csv"])
    signal_inventory = PROJECT_ROOT / str(config["input_signal_inventory_json"])
    design_doc = PROJECT_ROOT / str(config["input_design_doc"])
    ensure_inputs_exist([input_json, input_jsonl, input_csv, signal_inventory, design_doc])

    answer_eval = load_json(input_json)
    records = answer_eval.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"No answer eval records found in {input_json}")

    scoring_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        quality = score_answer_record(record, config)
        scoring_records.append(
            {
                **quality,
                "evidence_sufficient": bool(record.get("evidence_sufficient", True)),
                "llm_called": bool(record.get("llm_called", False)),
                "supported_claim_ratio": record.get("supported_claim_ratio", 0),
                "unsupported_claim_count": record.get("unsupported_claim_count", 0),
                "weak_claim_count": record.get("weak_claim_count", 0),
                "citation_check_passed": bool(record.get("citation_check_passed", True)),
                "final_source_count": len(record.get("final_sources", []) or []),
            }
        )

    summary = summarize(scoring_records)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "input_answer_eval_json": str(input_json),
        "input_answer_eval_jsonl": str(input_jsonl),
        "input_answer_eval_csv": str(input_csv),
        "input_signal_inventory_json": str(signal_inventory),
        "input_design_doc": str(design_doc),
        "summary": summary,
        "records": scoring_records,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def ensure_inputs_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required evidence quality input files: {missing}")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts = dict(Counter(str(record["evidence_quality_level"]) for record in records))
    warnings = Counter(
        warning
        for record in records
        for warning in record.get("evidence_quality_warnings", [])
    )
    return {
        "total_records": len(records),
        "level_counts": level_counts,
        "avg_evidence_quality_score": average(record["evidence_quality_score"] for record in records),
        "avg_retrieval_confidence_score": average(record["retrieval_confidence_score"] for record in records),
        "avg_citation_support_score": average(record["citation_support_score"] for record in records),
        "avg_coverage_score": average(record["coverage_score"] for record in records),
        "avg_source_diversity_score": average(record["source_diversity_score"] for record in records),
        "avg_risk_penalty": average(record["risk_penalty"] for record in records),
        "insufficient_count": sum(1 for record in records if record["evidence_quality_level"] == "insufficient"),
        "needs_human_review_count": sum(1 for record in records if record["needs_human_review"]),
        "warning_counts": dict(warnings),
        "needs_human_review_ids": [
            record["id"]
            for record in records
            if record["needs_human_review"]
        ],
        "insufficient_ids": [
            record["id"]
            for record in records
            if record["evidence_quality_level"] == "insufficient"
        ],
    }


def average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    outputs = config["outputs"]
    output_json = PROJECT_ROOT / str(outputs["json"])
    output_jsonl = PROJECT_ROOT / str(outputs["jsonl"])
    output_csv = PROJECT_ROOT / str(outputs["csv"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_jsonl.open("w", encoding="utf-8") as file:
        for record in result["records"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(output_csv, result["records"])


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question",
        "query_type",
        "evidence_quality_score",
        "evidence_quality_level",
        "retrieval_confidence_score",
        "citation_support_score",
        "coverage_score",
        "source_diversity_score",
        "risk_penalty",
        "needs_human_review",
        "insufficient_answer",
        "evidence_sufficient",
        "llm_called",
        "supported_claim_ratio",
        "unsupported_claim_count",
        "weak_claim_count",
        "citation_check_passed",
        "final_source_count",
        "evidence_quality_reasons",
        "evidence_quality_warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key, "") for key in fieldnames},
                    "evidence_quality_reasons": "|".join(record.get("evidence_quality_reasons", [])),
                    "evidence_quality_warnings": "|".join(record.get("evidence_quality_warnings", [])),
                }
            )


def print_summary(result: dict[str, Any]) -> None:
    print("Evidence quality evaluation completed.")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
