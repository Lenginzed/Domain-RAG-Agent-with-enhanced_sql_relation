from __future__ import annotations

from pathlib import Path

from src.agent.workflow import run_rag_graph_v3b
from src.ui.rag_inspection_service import run_retrieval_inspection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGGRAPH_CONFIG = PROJECT_ROOT / "config" / "langgraph_v4e.yaml"
UI_CONFIG = PROJECT_ROOT / "config" / "ui_v4e.yaml"


def test_v4e_configs_exist() -> None:
    assert LANGGRAPH_CONFIG.exists()
    assert UI_CONFIG.exists()


def test_enhanced_sql_relation_ui_inspection_returns_sources() -> None:
    result = run_retrieval_inspection(
        question="missile engine 相关说明在哪份 notes 文件中？",
        retrieval_mode="enhanced_sql_relation",
        enable_calibration=True,
        final_top_k=8,
        config_path=UI_CONFIG,
        log_interaction=False,
    )
    sources = result["final_sources"]
    assert sources
    assert any(source.get("retrieval_channels") for source in sources)
    assert any(source.get("relation_path") or source.get("fusion_reasons") for source in sources)


def test_langgraph_retrieval_only_enhanced_sql_relation_no_llm() -> None:
    result = run_rag_graph_v3b(
        question="missile engine 相关说明在哪份 notes 文件中？",
        run_mode="retrieval_only",
        retrieval_mode="enhanced_sql_relation",
        enable_calibration=True,
        config_path=LANGGRAPH_CONFIG,
    )
    summary = result["summary"]
    state = result["state"]
    assert summary["llm_called"] is False
    assert state["llm_called"] is False
    assert state["sources"]
    assert state.get("sql_relation_debug")
    assert any(source.get("relation_path") for source in state["sources"])


def test_reward_risk_category_intent_experiments() -> None:
    result = run_retrieval_inspection(
        question="reward 和 risk 相关的实验或报告有哪些？",
        retrieval_mode="enhanced_sql_relation",
        enable_calibration=True,
        final_top_k=8,
        config_path=UI_CONFIG,
        log_interaction=False,
    )
    intent = result.get("category_intent", {})
    assert intent.get("intent") == "experiments"
    assert "experiments" in intent.get("preferred_categories", [])
    assert result["final_sources"]
