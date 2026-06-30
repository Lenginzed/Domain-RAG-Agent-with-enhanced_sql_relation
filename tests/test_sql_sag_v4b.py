from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import yaml

from src.sql_sag.entity_extractor import (
    ExtractedEntity,
    cleanup_extracted_entities,
    is_hash_like_token,
    normalize_entity_name,
)
from src.sql_sag.schema import TABLES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4b.yaml"
DB_PATH = PROJECT_ROOT / "storage/sqlite/domain_rag_sag_lite_v4b.db"


def load_cleanup_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config["entity_cleanup"]


def entity(name: str, entity_type: str = "domain_term") -> ExtractedEntity:
    return ExtractedEntity(
        name=name,
        normalized_name=normalize_entity_name(name),
        entity_type=entity_type,
        role="test",
        source_hint="test",
    )


def test_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_hash_like_token_filtering() -> None:
    cleanup_config = load_cleanup_config()
    assert is_hash_like_token("c276357917", cleanup_config)
    assert is_hash_like_token("readme_c276357917", cleanup_config)
    assert not is_hash_like_token("stage125", cleanup_config)
    assert not is_hash_like_token("stage18_0", cleanup_config)
    assert not is_hash_like_token("2v2", cleanup_config)


def test_generic_and_yaml_stoplists_preserve_domain_terms() -> None:
    cleanup_config = load_cleanup_config()
    candidates = [
        entity("readme"),
        entity("paper"),
        entity("reward"),
        entity("risk"),
        entity("enabled", "config_key"),
        entity("missile_engine"),
        entity("EventDrivenReward", "code_symbol"),
    ]
    kept, removed = cleanup_extracted_entities(candidates, cleanup_config)
    kept_names = {item.normalized_name for item in kept}
    removed_names = {item.normalized_name for item in removed}

    assert "readme" in removed_names
    assert "paper" in removed_names
    assert "enabled" in removed_names
    assert "reward" in kept_names
    assert "risk" in kept_names
    assert "missile_engine" in kept_names
    assert "event_driven_reward" in kept_names


def test_v4b_sqlite_outputs_exist_and_nonempty() -> None:
    assert DB_PATH.exists()
    connection = sqlite3.connect(DB_PATH)
    for table in TABLES:
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert count > 0, table
    connection.close()

    key_audit_path = PROJECT_ROOT / "data/eval/sql_sag_v4b_key_entity_audit.csv"
    removed_path = PROJECT_ROOT / "data/eval/sql_sag_v4b_removed_entities.csv"
    assert key_audit_path.exists()
    assert removed_path.exists()

    with key_audit_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert sum(1 for row in rows if row.get("found") == "True") == 13
