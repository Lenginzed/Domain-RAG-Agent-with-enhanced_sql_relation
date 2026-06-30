from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_calibration_v26_config_exists() -> None:
    assert (PROJECT_ROOT / "config" / "retrieval_calibration_v26.yaml").exists()


def test_regression_set_exists_and_contains_q014() -> None:
    path = PROJECT_ROOT / "data" / "eval" / "retrieval_calibration_regression_v26.jsonl"
    assert path.exists()
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(items) >= 10
    ids = {str(item.get("id")) for item in items}
    expected_sources = [source for item in items for source in item.get("expected_sources_contains", [])]
    assert "v25c_q014" in ids or any("missile_engine.md" in str(source) for source in expected_sources)


def test_sensitivity_and_recommended_outputs_exist() -> None:
    assert (PROJECT_ROOT / "storage" / "logs" / "retrieval_calibration_sensitivity_v26.json").exists()
    assert (PROJECT_ROOT / "config" / "retrieval_calibration_recommended_v26.yaml").exists()


def test_recommended_config_has_non_empty_boosts() -> None:
    path = PROJECT_ROOT / "config" / "retrieval_calibration_recommended_v26.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    calibration = data["calibration"]
    assert calibration["exact_filename_boost"] is not None
    assert calibration["stem_token_boost"] is not None
    assert float(calibration["exact_filename_boost"]) > 0
    assert float(calibration["stem_token_boost"]) > 0
