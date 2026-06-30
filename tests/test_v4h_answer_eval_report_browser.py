from __future__ import annotations

from pathlib import Path

from src.ui.answer_eval_report_service import (
    extract_failure_options,
    extract_mode_options,
    filter_answer_eval_records,
    format_run_detail,
    load_answer_eval_compare,
    load_answer_eval_records,
    load_answer_eval_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "ui_v4h.yaml"
APP_PATH = PROJECT_ROOT / "apps" / "streamlit_app_v25f.py"


def test_v4h_config_and_app_exist() -> None:
    assert CONFIG_PATH.exists()
    assert APP_PATH.exists()


def test_load_v4g_eval_json_and_jsonl_records() -> None:
    report = load_answer_eval_report(CONFIG_PATH, "v4g")
    assert report["eval_json"]
    assert report["metrics"]["total_runs"] == 12

    records = load_answer_eval_records(CONFIG_PATH, "v4g")
    assert len(records) == 12
    assert {record["retrieval_mode"] for record in records} >= {"enhanced_keyword", "enhanced_sql_relation"}


def test_load_v4f_v4g_compare_json() -> None:
    compare = load_answer_eval_compare(CONFIG_PATH, "v4g")
    assert compare["v4f_empty_answer_count"] == 3
    assert compare["v4g_empty_answer_count"] == 2
    assert compare["negative_refusal_pass_rate"] == 1.0


def test_filter_failure_taxonomy_records() -> None:
    records = load_answer_eval_records(CONFIG_PATH, "v4g")
    failures = extract_failure_options(records)
    assert "llm_empty_answer" in failures
    assert "retrieval_insufficient" in failures

    empty_runs = filter_answer_eval_records(records, failure_type="llm_empty_answer")
    assert len(empty_runs) >= 1
    assert all("llm_empty_answer" in format_run_detail(record)["failure_types"] for record in empty_runs)

    insufficient_runs = filter_answer_eval_records(records, failure_type="retrieval_insufficient")
    assert any(record["case_id"] == "v4f_negative_missing" for record in insufficient_runs)
    assert all(not record.get("llm_called", False) for record in insufficient_runs)


def test_filter_enhanced_sql_relation_relation_paths() -> None:
    records = load_answer_eval_records(CONFIG_PATH, "v4g")
    assert "enhanced_sql_relation" in extract_mode_options(records)

    sql_runs = filter_answer_eval_records(records, retrieval_mode="enhanced_sql_relation")
    assert len(sql_runs) == 6
    assert any(format_run_detail(record)["relation_paths"] for record in sql_runs)

