from __future__ import annotations

from pathlib import Path

from src.sql_sag.fusion_retriever import (
    infer_category_intent,
    load_v4d_config,
    normalize_relation_score,
    retrieve_enhanced_sql_relation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4d.yaml"


def test_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_category_intent_rules() -> None:
    config = load_v4d_config(CONFIG_PATH)
    experiment_intent = infer_category_intent("reward 和 risk 相关的实验或报告有哪些？", config)
    assert "experiments" in experiment_intent["preferred_categories"]
    assert "thesis_or_reports" in experiment_intent["preferred_categories"]
    assert "configs" in experiment_intent["penalized_categories"]

    config_intent = infer_category_intent("HierarchySelfplay 对应哪个配置文件？", config)
    assert "configs" in config_intent["preferred_categories"]


def test_relation_score_normalization() -> None:
    config = load_v4d_config(CONFIG_PATH)
    assert normalize_relation_score(0, config) == 0.0
    assert normalize_relation_score(120, config) == 100.0
    assert normalize_relation_score(263, config) == 100.0


def test_enhanced_sql_relation_returns_sql_aware_sources() -> None:
    config = load_v4d_config(CONFIG_PATH)
    result = retrieve_enhanced_sql_relation("missile engine 相关说明在哪份 notes 文件中？", config)
    assert result["final_sources"]
    assert any("sql_relation" in row.get("retrieval_channels", []) for row in result["final_sources"])
    assert any(row.get("relation_path") for row in result["final_sources"])


def test_reward_risk_category_gate_does_not_let_configs_unconditionally_win() -> None:
    config = load_v4d_config(CONFIG_PATH)
    result = retrieve_enhanced_sql_relation("reward 和 risk 相关的实验或报告有哪些？", config)
    final_sources = result["final_sources"]
    assert final_sources
    top_categories = {row.get("category_dir") for row in final_sources[:3]}
    assert {"experiments", "thesis_or_reports"} & top_categories
    config_rows = [row for row in final_sources if row.get("category_dir") == "configs"]
    assert all(row.get("category_intent_adjustment", 0) <= 0 for row in config_rows)
