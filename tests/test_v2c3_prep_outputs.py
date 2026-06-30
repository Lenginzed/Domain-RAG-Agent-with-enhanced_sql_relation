from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MILESTONE_MD = PROJECT_ROOT / "docs" / "v2_milestone_summary.md"
SIGNAL_MD = PROJECT_ROOT / "docs" / "evidence_signal_inventory_v2c3_prep.md"
DESIGN_MD = PROJECT_ROOT / "docs" / "design_v2c3_evidence_quality_scoring.md"
INVENTORY_JSON = PROJECT_ROOT / "data" / "eval" / "evidence_signal_inventory_v2c3_prep.json"


def test_v2c3_prep_docs_exist() -> None:
    assert MILESTONE_MD.exists()
    assert SIGNAL_MD.exists()
    assert DESIGN_MD.exists()
    assert INVENTORY_JSON.exists()


def test_v2c3_signal_inventory_has_required_signals() -> None:
    data = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    signals = data.get("signals", [])
    assert len(signals) >= 12
    names = {signal["name"] for signal in signals}
    required = {
        "keyword_score",
        "metadata_score",
        "supported_claim_ratio",
        "unsupported_claim_count",
        "needs_human_review",
        "source_diversity",
    }
    assert required.issubset(names)


def test_v2c3_signal_inventory_separates_score_and_diagnostic() -> None:
    data = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    signals = data["signals"]
    assert any(signal["use_in_v2c3"] for signal in signals)
    assert any(not signal["use_in_v2c3"] for signal in signals)
    assert data["signal_counts"]["total"] == len(signals)
