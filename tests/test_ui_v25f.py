from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from src.ui import rag_inspection_service as service


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ui_v25f_files_exist() -> None:
    assert (PROJECT_ROOT / "config" / "ui_v25f.yaml").exists()
    assert (PROJECT_ROOT / "apps" / "streamlit_app_v25f.py").exists()
    assert (PROJECT_ROOT / "src" / "ui" / "rag_inspection_service.py").exists()


def test_run_retrieval_inspection_returns_sources_without_llm(monkeypatch) -> None:
    def fake_retrieve_real_smoke(**kwargs):
        document = Document(
            page_content="EventDrivenReward is implemented in event_driven_reward.py.",
            metadata={
                "rank": 1,
                "source": "data/raw_imported/code/event_driven_reward.py",
                "imported_source": "data/raw_imported/code/event_driven_reward.py",
                "category_dir": "code",
                "doc_type": "code",
                "chunk_id": "chunk_test",
                "retrieval_channels": ["metadata", "keyword"],
                "matched_terms": ["eventdrivenreward", "reward"],
                "matched_fields": ["filename", "content"],
                "final_score": 123.0,
                "calibration_score": 60.0,
                "calibration_reasons": ["exact_filename_match:event_driven_reward.py"],
            },
        )
        final_source = dict(document.metadata)
        return SimpleNamespace(
            documents=[document],
            debug={
                "query_type": "code_general",
                "target_categories": ["code"],
                "expanded_query": "EventDrivenReward event_driven_reward.py",
                "final_sources": [final_source],
                "source_diversity": {"passed": True, "violations": {}},
            },
        )

    class FailingLLM:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Retrieval-only path must not instantiate OllamaLLM.")

    monkeypatch.setattr(service, "retrieve_real_smoke", fake_retrieve_real_smoke)
    monkeypatch.setattr(service, "OllamaLLM", FailingLLM)

    result = service.run_retrieval_inspection(
        question="EventDrivenReward 在哪个文件里？",
        retrieval_mode="enhanced_keyword",
        enable_calibration=True,
        final_top_k=8,
        log_interaction=False,
    )
    assert result["llm_called"] is False
    assert result["final_sources"]
    assert result["final_sources"][0]["source"].endswith("event_driven_reward.py")
    assert result["final_sources"][0]["calibration_reasons"]


def test_manual_review_export_appends_csv(tmp_path) -> None:
    output = tmp_path / "ui_manual_review_exports.csv"
    record = {
        "question": "missile engine 相关说明在哪份 notes 文件中？",
        "run_mode": "retrieval_only",
        "retrieval_mode": "enhanced_keyword",
        "collection_name": "domain_rag_expanded_v25b_80",
        "calibration_enabled": True,
        "llm_called": False,
        "evidence_gate": {"evidence_sufficient": True, "reason": "test"},
        "final_sources": [
            {
                "rank": 1,
                "category_dir": "notes",
                "source": "data/raw_imported/notes/missile_engine.md",
            }
        ],
        "manual_judgment": "correct",
        "manual_citation_judgment": "",
        "manual_should_refuse": "false",
        "manual_notes": "lightweight check",
    }
    service.append_manual_review_export(record, output)
    service.append_manual_review_export(record, output)
    text = output.read_text(encoding="utf-8-sig")
    assert "manual_judgment" in text
    assert text.count("missile_engine.md") == 2
