from __future__ import annotations

from pathlib import Path

from src.agent.trace import generate_run_id
from src.agent.workflow import load_graph_trace, load_recent_graph_runs, run_rag_graph_v3b


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "langgraph_v3b.yaml"


def test_langgraph_v3b_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_generate_run_id_unique() -> None:
    first = generate_run_id("lg")
    second = generate_run_id("lg")
    assert first
    assert second
    assert first != second
    assert first.startswith("lg_")
    assert second.startswith("lg_")


def test_q014_retrieval_only_v3b_has_run_id_trace_and_review_decision() -> None:
    result = run_rag_graph_v3b(
        question="missile engine 相关说明在哪份 notes 文件中？",
        run_mode="retrieval_only",
        retrieval_mode="enhanced_keyword",
        enable_calibration=True,
        config_path=CONFIG_PATH,
    )
    summary = result["summary"]
    state = result["state"]
    run_id = summary["run_id"]

    assert run_id
    assert summary["llm_called"] is False
    assert state["llm_called"] is False
    assert state["sources"]
    assert state["trace"]
    assert state["review_decision"]
    assert "needs_human_review" in state["review_decision"]

    recent_runs = load_recent_graph_runs(CONFIG_PATH)
    assert any(row.get("run_id") == run_id for row in recent_runs)

    trace_record = load_graph_trace(run_id, CONFIG_PATH)
    assert trace_record is not None
    assert trace_record["summary"]["run_id"] == run_id
    assert trace_record["summary"]["llm_called"] is False
    assert any(item.get("node") == "human_review_decision_node" for item in trace_record["summary"].get("trace", []))
