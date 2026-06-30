from __future__ import annotations

import sqlite3
from pathlib import Path

from src.sql_sag.entity_extractor import extract_entities_for_chunk, normalize_entity_name
from src.sql_sag.schema import TABLES, create_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4a.yaml"
DB_PATH = PROJECT_ROOT / "storage/sqlite/domain_rag_sag_lite_v4a.db"


def test_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_schema_can_create_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "schema_smoke.db"
    connection = sqlite3.connect(db_path)
    create_schema(connection)
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    connection.close()
    assert set(TABLES).issubset(tables)


def test_entity_normalization_and_filename_extraction() -> None:
    assert normalize_entity_name("missile-engine.md") == "missile_engine"
    assert normalize_entity_name("EventDrivenReward") == "event_driven_reward"

    missile_entities = extract_entities_for_chunk(
        text="Missile guidance and engine notes.",
        metadata={"source": "data/raw_imported/notes/missile_engine.md", "file_ext": ".md"},
        config={"min_token_len": 3, "max_entities_per_chunk": 40},
        domain_terms=["missile", "engine", "reward"],
    )
    missile_names = {entity.normalized_name for entity in missile_entities}
    assert "missile_engine" in missile_names
    assert "missile" in missile_names
    assert "engine" in missile_names

    reward_entities = extract_entities_for_chunk(
        text="class EventDrivenReward:\n    def compute_reward(self):\n        return 0",
        metadata={"source": "data/raw_imported/code/event_driven_reward.py", "file_ext": ".py"},
        config={"min_token_len": 3, "max_entities_per_chunk": 40},
        domain_terms=["reward"],
    )
    reward_names = {entity.normalized_name for entity in reward_entities}
    assert "event_driven_reward" in reward_names
    assert "reward" in reward_names


def test_built_sqlite_outputs_exist_and_nonempty() -> None:
    assert DB_PATH.exists()
    connection = sqlite3.connect(DB_PATH)
    for table in TABLES:
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert count > 0, table
    connection.close()
    assert (PROJECT_ROOT / "data/eval/sql_sag_v4a_key_entity_audit.csv").exists()
    assert (PROJECT_ROOT / "data/eval/sql_sag_v4a_audit.md").exists()
