from __future__ import annotations

from pathlib import Path

from src.evaluation.evidence_quality import score_answer_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "evidence_quality.yaml"


def high_quality_record() -> dict:
    return {
        "id": "q_test",
        "question": "训练入口文件是什么？",
        "query_type": "code_entry",
        "evidence_sufficient": True,
        "insufficient_answer": False,
        "supported_claim_ratio": 1.0,
        "citation_check_passed": True,
        "unsupported_claim_count": 0,
        "weak_claim_count": 0,
        "needs_human_review": False,
        "evidence_gate_signals": {
            "source_hits": ["train_jsbsim.py"],
            "category_hits": ["code"],
            "required_all_categories": [],
            "missing_required_categories": [],
        },
        "final_sources": [
            {
                "rank": 1,
                "imported_source": "data/raw_imported/code/train_jsbsim.py",
                "retrieval_channels": ["dense", "metadata", "keyword"],
                "matched_terms": ["train_jsbsim"],
                "matched_fields": ["filename", "content"],
            }
        ],
    }


def test_evidence_quality_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_high_quality_record_scores_in_range_and_high_or_medium() -> None:
    result = score_answer_record(high_quality_record())
    assert 0.0 <= result["evidence_quality_score"] <= 1.0
    assert result["evidence_quality_level"] in {"high", "medium"}


def test_insufficient_answer_gets_insufficient_level() -> None:
    record = high_quality_record()
    record["evidence_sufficient"] = False
    record["insufficient_answer"] = True
    result = score_answer_record(record)
    assert result["evidence_quality_level"] == "insufficient"
    assert "evidence_insufficient" in result["evidence_quality_warnings"]


def test_needs_human_review_adds_warning() -> None:
    record = high_quality_record()
    record["needs_human_review"] = True
    result = score_answer_record(record)
    assert "needs_human_review" in result["evidence_quality_warnings"]
    assert result["risk_penalty"] > 0


def test_unsupported_claims_lower_citation_or_increase_risk() -> None:
    clean = score_answer_record(high_quality_record())
    risky_record = high_quality_record()
    risky_record["unsupported_claim_count"] = 2
    risky_record["supported_claim_ratio"] = 0.8
    risky = score_answer_record(risky_record)
    assert risky["citation_support_score"] < clean["citation_support_score"]
    assert risky["risk_penalty"] > clean["risk_penalty"]
    assert "unsupported_claims_present" in risky["evidence_quality_warnings"]
