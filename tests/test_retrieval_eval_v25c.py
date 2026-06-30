from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_eval_v25c.yaml"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_v25c.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_v25c_manifest.json"


def test_retrieval_eval_v25c_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_retrieval_eval_v25c_eval_set_exists_and_has_enough_questions() -> None:
    assert EVAL_PATH.exists()
    items = load_eval_items()

    assert len(items) >= 25


def test_retrieval_eval_v25c_covers_required_categories_and_types() -> None:
    items = load_eval_items()
    categories = {
        category
        for item in items
        for category in item.get("expected_categories", [])
    }
    query_types = {item.get("query_type") for item in items}

    assert {"code", "configs", "experiments", "notes", "papers", "thesis_or_reports"}.issubset(categories)
    assert "mixed" in query_types
    assert "negative" in query_types


def test_non_negative_questions_have_expected_sources() -> None:
    items = load_eval_items()
    missing = [
        item["id"]
        for item in items
        if item.get("query_type") != "negative" and not item.get("expected_sources_contains")
    ]

    assert missing == []


def test_manifest_reports_no_expected_source_absent() -> None:
    assert MANIFEST_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["expected_source_absent_count"] == 0


def load_eval_items() -> list[dict]:
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
