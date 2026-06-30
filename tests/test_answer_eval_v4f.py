from __future__ import annotations

from pathlib import Path

from scripts.build_answer_eval_v4f_cases import build_cases, load_yaml as load_build_yaml
from scripts.run_answer_eval_v4f import run_answer_eval_v4f


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v4f.yaml"
CASES_PATH = PROJECT_ROOT / "data" / "eval" / "answer_eval_v4f_cases.jsonl"


def test_v4f_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_build_cases_contains_required_items() -> None:
    config = load_build_yaml(CONFIG_PATH)
    cases = build_cases(config)
    ids = {case["id"] for case in cases}
    assert "v4f_q014_missile_engine" in ids
    assert "v4f_event_driven_reward" in ids
    assert "v4f_hierarchy_selfplay" in ids
    assert "v4f_reward_risk" in ids
    assert "v4f_negative_missing" in ids
    assert any(case["query_type"] == "negative" for case in cases)


def test_cases_jsonl_exists_after_build() -> None:
    assert CASES_PATH.exists()
    lines = [line for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 5


def test_answer_eval_runner_dry_run_no_llm() -> None:
    result = run_answer_eval_v4f(CONFIG_PATH, dry_run=True)
    assert result["external_cloud_api_called"] is False
    assert result["answer_eval_status"] == "dry_run_retrieval_only"
    assert result["metrics"]["total_cases"] >= 5
    assert result["metrics"]["total_runs"] >= 10
    assert result["metrics"]["llm_called_count"] == 0
    assert {record["retrieval_mode"] for record in result["records"]} == {
        "enhanced_keyword",
        "enhanced_sql_relation",
    }
    assert any(record["case_id"] == "v4f_negative_missing" for record in result["records"])
