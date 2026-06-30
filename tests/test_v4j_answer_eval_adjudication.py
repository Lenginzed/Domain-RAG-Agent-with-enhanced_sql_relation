from __future__ import annotations

import csv
import py_compile
from pathlib import Path

from src.ui.answer_eval_adjudication_service import (
    build_calibration_notes,
    enrich_adjudication_tasks,
    export_adjudication_outputs,
    load_adjudication_config,
    load_manual_review_csv,
    load_review_tasks,
    merge_manual_review,
    summarize_adjudication,
    validate_manual_fields,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "review_adjudication_v4j.yaml"
APP_PATH = PROJECT_ROOT / "apps" / "streamlit_app_v25f.py"
TASKS_JSON = PROJECT_ROOT / "data" / "eval" / "answer_eval_review_tasks_v4i.json"


def test_v4j_config_and_load_tasks() -> None:
    assert CONFIG_PATH.exists()
    config = load_adjudication_config(CONFIG_PATH)
    tasks = load_review_tasks(TASKS_JSON)
    assert len(tasks) == 8
    assert config["inputs"]["default_review_tasks_json"].endswith("answer_eval_review_tasks_v4i.json")


def test_empty_manual_fields_generate_pending_summary() -> None:
    config = load_adjudication_config(CONFIG_PATH)
    tasks = enrich_adjudication_tasks(load_review_tasks(TASKS_JSON), config)
    validation = validate_manual_fields(tasks, config)
    summary = summarize_adjudication(tasks, config)
    assert summary["total_tasks"] == 8
    assert summary["pending_count"] == 8
    assert summary["adjudicated_count"] == 0
    assert validation["pending_count"] == 8
    assert validation["missing_required_high_priority_judgments"]


def test_filled_csv_generates_adjudication_and_fix_categories(tmp_path: Path) -> None:
    config = load_adjudication_config(CONFIG_PATH)
    tasks = load_review_tasks(TASKS_JSON)
    llm_empty = first_task_with_failure(tasks, "llm_empty_answer")
    unsupported = first_task_with_failure(tasks, "unsupported_claims_present")
    insufficient = first_task_with_failure(tasks, "retrieval_insufficient")

    manual_csv = tmp_path / "filled.csv"
    with manual_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "manual_judgment",
                "manual_citation_judgment",
                "manual_should_refuse",
                "manual_fix_suggestion",
                "manual_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": llm_empty["task_id"],
                "manual_judgment": "rerun_llm",
                "manual_citation_judgment": "not_applicable",
                "manual_should_refuse": "not_applicable",
                "manual_fix_suggestion": "Retry with reliability wrapper or fallback model.",
                "manual_notes": "empty answer example",
            }
        )
        writer.writerow(
            {
                "task_id": unsupported["task_id"],
                "manual_judgment": "checker_false_positive",
                "manual_citation_judgment": "checker_false_positive",
                "manual_should_refuse": "not_applicable",
                "manual_fix_suggestion": "Tune citation checker for this pattern.",
                "manual_notes": "citation checker false positive example",
            }
        )
        writer.writerow(
            {
                "task_id": insufficient["task_id"],
                "manual_judgment": "keep_refusal",
                "manual_citation_judgment": "not_applicable",
                "manual_should_refuse": "not_applicable",
                "manual_fix_suggestion": "Keep refusal.",
                "manual_notes": "valid refusal example",
            }
        )

    merged = merge_manual_review(tasks, load_manual_review_csv(manual_csv))
    enriched = enrich_adjudication_tasks(merged, config)
    summary = summarize_adjudication(enriched, config)
    by_id = {task["task_id"]: task for task in enriched}
    assert by_id[llm_empty["task_id"]]["adjudication_status"] == "needs_prompt_fix"
    assert set(by_id[llm_empty["task_id"]]["fix_categories"]) >= {"prompt_fix", "llm_reliability_fix"}
    assert by_id[unsupported["task_id"]]["adjudication_status"] == "needs_citation_checker_fix"
    assert "citation_checker_fix" in by_id[unsupported["task_id"]]["fix_categories"]
    assert by_id[insufficient["task_id"]]["adjudication_status"] == "valid_refusal"
    assert "no_action_needed" in by_id[insufficient["task_id"]]["fix_categories"]
    assert summary["adjudicated_count"] == 3


def test_export_adjudication_outputs(tmp_path: Path) -> None:
    config = load_adjudication_config(CONFIG_PATH)
    tasks = enrich_adjudication_tasks(load_review_tasks(TASKS_JSON), config)
    summary = summarize_adjudication(tasks, config)
    notes = build_calibration_notes(tasks, summary, config)
    config = dict(config)
    config["outputs"] = {
        "adjudicated_tasks_json": str(tmp_path / "adjudicated.json"),
        "adjudicated_tasks_jsonl": str(tmp_path / "adjudicated.jsonl"),
        "adjudicated_tasks_csv": str(tmp_path / "adjudicated.csv"),
        "adjudication_summary_json": str(tmp_path / "summary.json"),
        "adjudication_report_md": str(tmp_path / "report.md"),
        "calibration_notes_md": str(tmp_path / "notes.md"),
    }
    result = export_adjudication_outputs(tasks, summary, notes, config)
    for path in result["output_paths"].values():
        assert Path(path).exists()


def test_streamlit_app_py_compile() -> None:
    py_compile.compile(str(APP_PATH), doraise=True)


def first_task_with_failure(tasks: list[dict], failure_type: str) -> dict:
    for task in tasks:
        if failure_type in (task.get("failure_types") or []):
            return task
    raise AssertionError(f"No task with failure type {failure_type}")

