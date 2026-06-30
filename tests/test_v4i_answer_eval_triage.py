from __future__ import annotations

import py_compile
from pathlib import Path

from src.ui.answer_eval_triage_service import (
    build_review_tasks,
    export_review_tasks,
    infer_failure_priority,
    load_eval_records_for_triage,
    load_triage_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "review_triage_v4i.yaml"
APP_PATH = PROJECT_ROOT / "apps" / "streamlit_app_v25f.py"


def test_v4i_config_and_imports() -> None:
    assert CONFIG_PATH.exists()
    assert APP_PATH.exists()


def test_build_v4g_review_tasks() -> None:
    config = load_triage_config(CONFIG_PATH)
    records = load_eval_records_for_triage(config, "v4g")
    assert len(records) == 12

    tasks = build_review_tasks(records, config)
    assert tasks
    labels = {label for task in tasks for label in task.get("failure_types", [])}
    assert "llm_empty_answer" in labels
    assert "unsupported_claims_present" in labels
    assert "retrieval_insufficient" in labels
    assert all("priority" in task for task in tasks)
    assert all("suggested_review_action" in task for task in tasks)
    assert all("manual_judgment" in task for task in tasks)
    assert all("manual_citation_judgment" in task for task in tasks)
    assert all("manual_should_refuse" in task for task in tasks)
    assert all("manual_fix_suggestion" in task for task in tasks)
    assert all("manual_notes" in task for task in tasks)


def test_priority_rules() -> None:
    config = load_triage_config(CONFIG_PATH)
    assert infer_failure_priority({"failure_types": ["llm_empty_answer"]}, config) == "high"
    assert infer_failure_priority({"failure_types": ["unsupported_claims_present"]}, config) == "high"
    assert infer_failure_priority({"failure_types": ["medium_or_lower_quality"]}, config) == "medium"
    assert infer_failure_priority({"failure_types": ["retrieval_insufficient"]}, config) == "low"


def test_export_review_tasks_to_all_formats(tmp_path: Path) -> None:
    config = load_triage_config(CONFIG_PATH)
    records = load_eval_records_for_triage(config, "v4g")
    tasks = build_review_tasks(records, config)
    config = dict(config)
    config["_triage_total_records"] = len(records)
    config["_triage_version"] = "v4g"
    config["outputs"] = {
        "review_tasks_json": str(tmp_path / "tasks.json"),
        "review_tasks_jsonl": str(tmp_path / "tasks.jsonl"),
        "review_tasks_csv": str(tmp_path / "tasks.csv"),
        "review_tasks_md": str(tmp_path / "tasks.md"),
        "summary_json": str(tmp_path / "summary.json"),
    }

    result = export_review_tasks(tasks, config)
    for path in result["output_paths"].values():
        assert Path(path).exists()
    assert result["summary"]["total_records"] == 12
    assert result["summary"]["review_task_count"] == len(tasks)


def test_streamlit_app_py_compile() -> None:
    py_compile.compile(str(APP_PATH), doraise=True)

