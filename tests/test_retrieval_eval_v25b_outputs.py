from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "storage" / "logs" / "retrieval_eval_v25b.json"


def test_retrieval_eval_v25b_json_exists() -> None:
    assert RESULT_PATH.exists()


def test_retrieval_eval_v25b_has_all_modes() -> None:
    result = load_result()

    assert set(result["modes"]) == {"dense", "enhanced", "keyword", "enhanced_keyword"}
    assert set(result["metrics_by_mode"]) == {"dense", "enhanced", "keyword", "enhanced_keyword"}


def test_enhanced_keyword_metrics_include_source_diversity() -> None:
    result = load_result()
    metrics = result["metrics_by_mode"]["enhanced_keyword"]

    assert "source_diversity_pass_rate" in metrics
    assert "hit_at_5" in metrics
    assert "mrr" in metrics


def test_expected_source_absent_field_is_present() -> None:
    result = load_result()
    assert "expected_source_absent_question_ids" in result
    assert isinstance(result["expected_source_absent_question_ids"], list)


def load_result() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
