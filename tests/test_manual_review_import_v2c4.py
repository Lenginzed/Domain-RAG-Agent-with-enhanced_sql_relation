from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from scripts.import_answer_review_labels import (
    CONFIG_PATH,
    PROJECT_ROOT,
    import_answer_review_labels,
    read_text_with_encoding_fallback,
)


REVIEW_TEMPLATE = PROJECT_ROOT / "data" / "eval" / "answer_review_v2c2_template.csv"
SUMMARY_JSON = PROJECT_ROOT / "storage" / "logs" / "manual_review_summary_v2c4.json"
MERGED_JSON = PROJECT_ROOT / "storage" / "logs" / "manual_review_merged_v2c4.json"
MERGED_JSONL = PROJECT_ROOT / "storage" / "logs" / "manual_review_merged_v2c4.jsonl"
MERGED_CSV = PROJECT_ROOT / "data" / "eval" / "manual_review_merged_v2c4.csv"
SUMMARY_MD = PROJECT_ROOT / "data" / "eval" / "manual_review_summary_v2c4.md"


def test_manual_review_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_import_manual_review_labels_outputs_files() -> None:
    result = import_answer_review_labels()
    summary = result["summary"]

    assert MERGED_JSON.exists()
    assert MERGED_JSONL.exists()
    assert MERGED_CSV.exists()
    assert SUMMARY_JSON.exists()
    assert SUMMARY_MD.exists()
    assert summary["total_records"] == 10


def test_empty_or_current_review_state_is_counted() -> None:
    result = import_answer_review_labels()
    summary = result["summary"]
    expected_unreviewed = count_unreviewed_rows(REVIEW_TEMPLATE)

    assert summary["unreviewed_count"] == expected_unreviewed
    assert summary["reviewed_count"] == summary["total_records"] - expected_unreviewed


def test_needs_human_review_ids_are_preserved() -> None:
    result = import_answer_review_labels()
    summary = result["summary"]

    assert summary["needs_human_review_count"] == 4
    assert {"q003", "q005", "q006", "q007"}.issubset(set(summary["needs_human_review_ids"]))


def test_summary_json_matches_returned_summary() -> None:
    result = import_answer_review_labels()
    persisted = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    assert persisted["total_records"] == result["summary"]["total_records"]
    assert persisted["unreviewed_count"] == result["summary"]["unreviewed_count"]


def count_unreviewed_rows(path: Path) -> int:
    manual_fields = [
        "manual_judgment",
        "manual_citation_judgment",
        "manual_should_refuse",
        "manual_notes",
        "reviewer",
        "reviewed_at",
    ]
    text, _encoding = read_text_with_encoding_fallback(path, ["utf-8-sig", "gb18030"])
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = list(reader)
    return sum(not any((row.get(field) or "").strip() for field in manual_fields) for row in rows)
