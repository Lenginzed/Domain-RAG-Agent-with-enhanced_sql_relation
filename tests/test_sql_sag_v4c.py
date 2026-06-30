from __future__ import annotations

from pathlib import Path

from src.sql_sag.relation_retriever import (
    connect_readonly,
    extract_query_entities,
    load_sql_sag_config,
    lookup_seed_entities,
    retrieve_sql_relation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4c.yaml"


def test_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_query_entity_extraction() -> None:
    config = load_sql_sag_config(CONFIG_PATH)
    q014_entities = {row["normalized_name"] for row in extract_query_entities("missile engine 相关说明在哪份 notes 文件中？", config)}
    assert {"missile", "engine"} & q014_entities

    reward_entities = {row["normalized_name"] for row in extract_query_entities("EventDrivenReward 奖励逻辑在哪个代码文件中？", config)}
    assert {"event_driven_reward", "reward"} & reward_entities

    hierarchy_entities = {row["normalized_name"] for row in extract_query_entities("HierarchySelfplay 对应哪个配置文件？", config)}
    assert {"hierarchy_selfplay", "selfplay"} & hierarchy_entities


def test_seed_entity_lookup_hits_v4b_db() -> None:
    config = load_sql_sag_config(CONFIG_PATH)
    query_entities = extract_query_entities("missile engine 相关说明在哪份 notes 文件中？", config)
    with connect_readonly(config["sqlite"]["db_path"]) as conn:
        seed_entities = lookup_seed_entities(conn, query_entities, config)
    seed_names = {row["normalized_name"] for row in seed_entities}
    assert {"missile", "engine", "missile_engine"} & seed_names


def test_retrieve_sql_relation_returns_explainable_candidates() -> None:
    config = load_sql_sag_config(CONFIG_PATH)
    result = retrieve_sql_relation("missile engine 相关说明在哪份 notes 文件中？", config)
    assert result["candidates"]
    candidate = result["candidates"][0]
    assert "relation_score" in candidate
    assert "relation_reasons" in candidate
    assert "relation_path" in candidate
    assert candidate["relation_path"]

    sources = " ".join(row["source"] for row in result["candidates"])
    assert "missile_engine.md" in sources
