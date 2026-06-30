from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_v2b5.jsonl"
RESULT_PATH = PROJECT_ROOT / "storage" / "logs" / "retrieval_eval_v2b5.json"


def test_retrieval_eval_dataset_exists_and_has_enough_questions() -> None:
    assert EVAL_PATH.exists(), "retrieval_eval_v2b5.jsonl is missing"
    rows = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 16
    query_types = {row["query_type"] for row in rows}
    required_types = {
        "code_entry",
        "code_general",
        "config",
        "experiment",
        "note_or_report",
        "paper",
        "thesis_or_reports",
        "mixed",
        "negative",
    }
    assert required_types.issubset(query_types)


def test_retrieval_eval_latest_results_smoke() -> None:
    assert RESULT_PATH.exists(), "Run scripts/run_retrieval_eval.py before this lightweight check"
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["total_questions"] >= 16
    assert set(result["modes"]) == {"dense", "enhanced", "keyword", "enhanced_keyword"}

    metrics = result["metrics_by_mode"]
    assert metrics["enhanced_keyword"]["hit_at_5"] >= metrics["dense"]["hit_at_5"]

    for mode, mode_metrics in metrics.items():
        assert mode_metrics["source_diversity_pass_rate"] == 1.0, mode
