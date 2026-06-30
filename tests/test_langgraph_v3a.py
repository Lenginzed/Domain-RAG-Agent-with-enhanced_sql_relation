from __future__ import annotations

import json
from pathlib import Path

from src.agent.workflow import build_rag_graph, run_rag_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "langgraph_v3a.yaml"


def test_langgraph_v3a_files_exist() -> None:
    assert CONFIG_PATH.exists()
    assert (PROJECT_ROOT / "src/agent/state.py").exists()
    assert (PROJECT_ROOT / "src/agent/nodes.py").exists()
    assert (PROJECT_ROOT / "src/agent/workflow.py").exists()
    assert (PROJECT_ROOT / "src/agent/trace.py").exists()


def test_build_graph_succeeds() -> None:
    graph = build_rag_graph()
    assert graph is not None


def test_q014_retrieval_only_graph_run_no_llm() -> None:
    result = run_rag_graph(
        question="missile engine 相关说明在哪份 notes 文件中？",
        run_mode="retrieval_only",
        retrieval_mode="enhanced_keyword",
        enable_calibration=True,
        config_path=CONFIG_PATH,
    )
    summary = result["summary"]
    state = result["state"]

    assert summary["llm_called"] is False
    assert state["llm_called"] is False
    assert state["sources"]
    assert state["evidence_sufficient"] in {True, False}
    assert state["trace"]
    assert any(item["node"] == "retrieve_node" for item in state["trace"])
    assert any(item["node"] == "evidence_gate_node" for item in state["trace"])
    assert summary.get("trace_path")

    last_run = PROJECT_ROOT / "storage/logs/langgraph_last_run_v3a.json"
    assert last_run.exists()
    parsed = json.loads(last_run.read_text(encoding="utf-8"))
    assert parsed["summary"]["llm_called"] is False
