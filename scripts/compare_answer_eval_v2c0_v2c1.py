from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2C0_PATH = PROJECT_ROOT / "storage" / "logs" / "answer_eval_v2c0.json"
V2C1_PATH = PROJECT_ROOT / "storage" / "logs" / "answer_eval_v2c1.json"
OUTPUT_JSON = PROJECT_ROOT / "storage" / "logs" / "answer_eval_v2c0_vs_v2c1_compare.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "eval" / "answer_eval_v2c0_vs_v2c1_compare.csv"


def main() -> int:
    configure_stdout()
    try:
        result = compare_answer_eval()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001 - comparison should fail explicitly.
        print("Answer eval comparison failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compare_answer_eval(v2c0_path: Path = V2C0_PATH, v2c1_path: Path = V2C1_PATH) -> dict[str, Any]:
    v2c0 = load_json(v2c0_path)
    v2c1 = load_json(v2c1_path)
    metric_names = [
        "unsupported_claim_count",
        "avg_supported_claim_ratio",
        "citation_check_pass_rate",
        "negative_refusal_rate",
        "insufficient_answer_count",
    ]
    metric_compare = []
    for name in metric_names:
        old_value = float(v2c0["metrics"].get(name, 0))
        new_value = float(v2c1["metrics"].get(name, 0))
        metric_compare.append(
            {
                "metric": name,
                "v2c0": old_value,
                "v2c1": new_value,
                "delta": round(new_value - old_value, 6),
            }
        )

    v2c0_by_id = {record["id"]: record for record in v2c0["records"]}
    v2c1_by_id = {record["id"]: record for record in v2c1["records"]}
    per_question = []
    for question_id in sorted(set(v2c0_by_id).intersection(v2c1_by_id)):
        old = v2c0_by_id[question_id]
        new = v2c1_by_id[question_id]
        per_question.append(
            {
                "id": question_id,
                "query_type": new.get("query_type", old.get("query_type", "")),
                "is_negative": bool(new.get("is_negative", old.get("is_negative", False))),
                "v2c0_supported_claim_ratio": old.get("supported_claim_ratio", 0),
                "v2c1_supported_claim_ratio": new.get("supported_claim_ratio", 0),
                "supported_claim_ratio_delta": round(
                    float(new.get("supported_claim_ratio", 0)) - float(old.get("supported_claim_ratio", 0)),
                    6,
                ),
                "v2c0_unsupported_claim_count": old.get("unsupported_claim_count", 0),
                "v2c1_unsupported_claim_count": new.get("unsupported_claim_count", 0),
                "unsupported_claim_count_delta": int(new.get("unsupported_claim_count", 0))
                - int(old.get("unsupported_claim_count", 0)),
                "v2c1_ignored_claim_count": new.get("ignored_claim_count", 0),
                "v2c1_weak_claim_count": new.get("weak_claim_count", 0),
                "v2c1_needs_human_review": new.get("needs_human_review", False),
                "v2c1_claim_type_counts": new.get("claim_type_counts", {}),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "v2c0_path": str(v2c0_path),
        "v2c1_path": str(v2c1_path),
        "metrics_compare": metric_compare,
        "per_question_compare": per_question,
        "summary": {
            "unsupported_claim_count_v2c0": v2c0["metrics"].get("unsupported_claim_count", 0),
            "unsupported_claim_count_v2c1": v2c1["metrics"].get("unsupported_claim_count", 0),
            "unsupported_claim_count_delta": int(v2c1["metrics"].get("unsupported_claim_count", 0))
            - int(v2c0["metrics"].get("unsupported_claim_count", 0)),
            "avg_supported_claim_ratio_v2c0": v2c0["metrics"].get("avg_supported_claim_ratio", 0),
            "avg_supported_claim_ratio_v2c1": v2c1["metrics"].get("avg_supported_claim_ratio", 0),
            "negative_refusal_rate_v2c0": v2c0["metrics"].get("negative_refusal_rate", 0),
            "negative_refusal_rate_v2c1": v2c1["metrics"].get("negative_refusal_rate", 0),
            "citation_check_pass_rate_v2c0": v2c0["metrics"].get("citation_check_pass_rate", 0),
            "citation_check_pass_rate_v2c1": v2c1["metrics"].get("citation_check_pass_rate", 0),
            "needs_human_review_count_v2c1": v2c1["metrics"].get("needs_human_review_count", 0),
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing answer eval file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid answer eval JSON: {path}")
    return data


def write_outputs(result: dict[str, Any]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "query_type",
                "is_negative",
                "v2c0_supported_claim_ratio",
                "v2c1_supported_claim_ratio",
                "supported_claim_ratio_delta",
                "v2c0_unsupported_claim_count",
                "v2c1_unsupported_claim_count",
                "unsupported_claim_count_delta",
                "v2c1_ignored_claim_count",
                "v2c1_weak_claim_count",
                "v2c1_needs_human_review",
                "v2c1_claim_type_counts",
            ],
        )
        writer.writeheader()
        for row in result["per_question_compare"]:
            output = dict(row)
            output["v2c1_claim_type_counts"] = json.dumps(
                output["v2c1_claim_type_counts"],
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(output)


def print_summary(result: dict[str, Any]) -> None:
    print("Answer eval comparison completed.")
    print("summary:")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("metrics_compare:")
    for row in result["metrics_compare"]:
        print(row)


if __name__ == "__main__":
    sys.exit(main())
