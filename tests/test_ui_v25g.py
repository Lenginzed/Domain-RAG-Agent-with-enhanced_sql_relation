from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.ui.rag_inspection_service import (
    load_demo_cases,
    load_interaction_logs,
    load_manual_review_exports,
    load_ui_config,
    summarize_interaction_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "ui_v25f.yaml"


def test_demo_case_config_contains_highlighted_q014() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    demo_cases = config.get("demo_cases", {})

    assert demo_cases.get("retrieval_eval_v25c")
    assert demo_cases.get("answer_eval_v25d")
    assert demo_cases.get("calibration_regression_v26")
    assert any(item.get("id") == "v25c_q014" for item in demo_cases.get("highlighted_cases", []))


def test_load_demo_cases_reads_cases() -> None:
    config = load_ui_config(CONFIG_PATH)
    demo = load_demo_cases(config)

    loaded_case_count = sum(len(source.get("cases", [])) for source in demo.get("sources", {}).values())
    assert loaded_case_count > 0
    assert "v25c_q014" in demo.get("all_cases_by_id", {})
    assert any(item.get("id") == "v25c_q014" and item.get("found") for item in demo.get("highlighted_cases", []))


def test_load_interaction_logs_missing_file_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing_ui_log.jsonl"
    assert load_interaction_logs(missing) == []


def test_load_interaction_logs_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "ui_log.jsonl"
    rows = [
        {
            "timestamp": "2026-06-30T10:00:00",
            "question": "missile engine 相关说明在哪份 notes 文件中？",
            "run_mode": "retrieval_only",
            "retrieval_mode": "enhanced_keyword",
            "llm_called": False,
            "evidence_quality_level": None,
            "top_sources": [{"source": "missile_engine.md", "rank": 1}],
        },
        {
            "timestamp": "2026-06-30T10:01:00",
            "question": "EventDrivenReward 是什么？",
            "run_mode": "full_rag_answer",
            "retrieval_mode": "enhanced_keyword",
            "llm_called": True,
            "evidence_quality_level": "high",
            "evidence_quality_score": 0.9,
            "final_sources": [{"source": "event_driven_reward.py", "rank": 1}],
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    loaded = load_interaction_logs(path, limit=10)
    summaries = [summarize_interaction_record(row) for row in loaded]

    assert len(loaded) == 2
    assert summaries[0]["llm_called"] is False
    assert summaries[1]["evidence_quality_level"] == "high"
    assert summaries[1]["top_source"] == "event_driven_reward.py"


def test_manual_review_exports_missing_file_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing_review_exports.csv"
    assert load_manual_review_exports(missing) == []
