from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v25d.yaml"
SUBSET_PATH = PROJECT_ROOT / "data" / "eval" / "answer_eval_v25d_subset.jsonl"
ANSWER_EVAL_PATH = PROJECT_ROOT / "storage" / "logs" / "answer_eval_v25d.json"
QUALITY_PATH = PROJECT_ROOT / "storage" / "logs" / "evidence_quality_eval_v25d.json"
REVIEW_TEMPLATE_PATH = PROJECT_ROOT / "data" / "eval" / "answer_review_v25d_template.csv"


def test_answer_eval_v25d_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_answer_eval_v25d_subset_exists_and_size_is_controlled() -> None:
    assert SUBSET_PATH.exists()
    rows = load_jsonl(SUBSET_PATH)

    assert 10 <= len(rows) <= 14


def test_answer_eval_v25d_subset_has_negative_and_failure_case() -> None:
    rows = load_jsonl(SUBSET_PATH)
    ids = {row["id"] for row in rows}
    negative_count = sum(1 for row in rows if row.get("query_type") == "negative")

    assert negative_count >= 2
    assert "v25c_q014" in ids


def test_answer_eval_v25d_outputs_exist_after_run() -> None:
    assert ANSWER_EVAL_PATH.exists()
    assert QUALITY_PATH.exists()
    assert REVIEW_TEMPLATE_PATH.exists()


def test_answer_eval_v25d_quality_has_expected_records() -> None:
    answer_eval = json.loads(ANSWER_EVAL_PATH.read_text(encoding="utf-8"))
    quality_eval = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))

    assert answer_eval["metrics"]["total_eval_questions"] == len(answer_eval["records"])
    assert quality_eval["summary"]["total_records"] == len(answer_eval["records"])
    assert "v25c_q014" in {record["id"] for record in answer_eval["records"]}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
