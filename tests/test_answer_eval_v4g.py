from __future__ import annotations

from pathlib import Path

from scripts.run_answer_eval_v4g import run_answer_eval_v4g
from src.answering.llm_reliability import call_local_llm_with_reliability, is_empty_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v4g.yaml"


def test_v4g_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_empty_answer_detection() -> None:
    config = {"min_answer_chars": 20, "strip_thinking_tags": True, "treat_whitespace_as_empty": True}
    assert is_empty_answer("", config)
    assert is_empty_answer("   \n\t  ", config)
    assert is_empty_answer("<think>reasoning only</think>", config)
    assert not is_empty_answer("<think>reasoning</think>这是一个足够长的回答正文，用于确认不会被误判为空。", config)


def test_retry_records_structure() -> None:
    calls = {"count": 0}

    def fake_generate(prompt: str, model: str, config: dict) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"raw_response": "<think>only hidden reasoning</think>"}
        return {"raw_response": "这是重试后生成的有效回答，长度足够通过空回答检测。"}

    result = call_local_llm_with_reliability(
        prompt="test prompt",
        config={
            "model": "local-test",
            "max_retries": 1,
            "retry_on_empty": True,
            "retry_with_same_model": True,
            "min_answer_chars": 20,
            "strip_thinking_tags": True,
        },
        case_id="case",
        retrieval_mode="enhanced_sql_relation",
        generate_func=fake_generate,
    )
    assert result["answer_generated"] is True
    assert result["status"] == "answered_after_retry"
    assert result["retry_count"] == 1
    assert result["attempt_count"] == 2
    assert result["attempts"][0]["status"] == "empty_answer"
    assert result["attempts"][1]["status"] == "success"


def test_v4g_runner_dry_run_no_llm_and_negative_refusal() -> None:
    result = run_answer_eval_v4g(CONFIG_PATH, dry_run=True)
    assert result["external_cloud_api_called"] is False
    assert result["answer_eval_status"] == "dry_run_retrieval_only"
    assert result["metrics"]["llm_called_count"] == 0
    negative = [record for record in result["records"] if record["case_id"] == "v4f_negative_missing"]
    assert negative
    assert all(record["llm_called"] is False for record in negative)
    assert all(record["negative_refusal_pass"] is True for record in negative)
