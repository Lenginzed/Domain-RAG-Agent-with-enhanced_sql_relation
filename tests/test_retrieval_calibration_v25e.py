from __future__ import annotations

from pathlib import Path

from src.retrieval.retrieval_calibration import calibrate_document_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_calibration_config_exists() -> None:
    assert (PROJECT_ROOT / "config" / "retrieval_calibration_v25e.yaml").exists()


def test_missile_engine_query_matches_filename_stem() -> None:
    result = calibrate_document_score(
        question="missile engine 相关说明在哪份 notes 文件中？",
        metadata={
            "source": "data/raw_imported/notes/missile_engine.md",
            "imported_source": "data/raw_imported/notes/missile_engine.md",
            "category_dir": "notes",
            "doc_type": "note",
            "title": "Missile Engine Notes",
            "section": "missile engine",
        },
        classification={"target_categories": ["notes"]},
        config={
            "enabled": True,
            "exact_filename_boost": 8.0,
            "stem_token_boost": 3.0,
            "metadata_title_boost": 3.0,
            "category_match_boost": 1.5,
            "matched_source_token_min_len": 3,
        },
    )
    assert result["calibration_score"] > 0
    assert result["calibration_breakdown"]
    assert any("source_stem_token_overlap" in reason for reason in result["calibration_reasons"])


def test_unrelated_source_does_not_receive_high_score() -> None:
    config = {
        "enabled": True,
        "exact_filename_boost": 8.0,
        "stem_token_boost": 3.0,
        "metadata_title_boost": 3.0,
        "category_match_boost": 1.5,
        "matched_source_token_min_len": 3,
    }
    matched = calibrate_document_score(
        question="missile engine",
        metadata={
            "source": "data/raw_imported/notes/missile_engine.md",
            "category_dir": "notes",
            "title": "missile engine",
        },
        classification={"target_categories": ["notes"]},
        config=config,
    )
    unrelated = calibrate_document_score(
        question="missile engine",
        metadata={
            "source": "data/raw_imported/notes/stage125_opd_hint_future_route.md",
            "category_dir": "notes",
            "title": "OPD HINT future route",
        },
        classification={"target_categories": ["notes"]},
        config=config,
    )
    assert unrelated["calibration_score"] < matched["calibration_score"]
    assert unrelated["calibration_score"] <= config["category_match_boost"]


def test_disabled_calibration_returns_zero_with_breakdown_field() -> None:
    result = calibrate_document_score(
        question="missile engine",
        metadata={"source": "data/raw_imported/notes/missile_engine.md"},
        classification={},
        config={"enabled": False},
    )
    assert result["calibration_score"] == 0.0
    assert "calibration_breakdown" in result
