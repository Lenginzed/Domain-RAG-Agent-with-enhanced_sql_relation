from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.generation.citation_checker import check_citations
from src.generation.evidence_gate import evaluate_evidence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_verification.yaml"


def test_answer_verification_config_exists() -> None:
    assert CONFIG_PATH.exists()


def test_evidence_gate_negative_eval_item_is_insufficient() -> None:
    retrieval_result = {
        "final_sources": [
            {
                "source": "data/raw_imported/experiments/experiment_protocol_summary.md",
                "imported_source": "data/raw_imported/experiments/experiment_protocol_summary.md",
                "category_dir": "experiments",
            }
        ]
    }
    eval_item = {
        "query_type": "negative",
        "expected_sources_contains": [],
        "expected_categories": [],
        "required_all_categories": [],
    }
    result = evaluate_evidence(question="当前知识库里有没有完整的 3v3 空战训练结果？", retrieval_result=retrieval_result, eval_item=eval_item)
    assert result["evidence_sufficient"] is False
    assert "negative" in result["reason"]


def test_evidence_gate_missing_expected_source_is_insufficient() -> None:
    retrieval_result = {
        "final_sources": [
            {
                "source": "data/raw_imported/code/event_driven_reward.py",
                "imported_source": "data/raw_imported/code/event_driven_reward.py",
                "category_dir": "code",
            }
        ]
    }
    eval_item = {
        "query_type": "code_entry",
        "expected_sources_contains": ["train_jsbsim.py"],
        "expected_categories": ["code"],
        "required_all_categories": [],
    }
    result = evaluate_evidence(question="训练入口文件是什么？", retrieval_result=retrieval_result, eval_item=eval_item)
    assert result["evidence_sufficient"] is False
    assert "expected_sources_contains" in result["reason"]


def test_citation_checker_supports_train_jsbsim_claim() -> None:
    answer = "训练入口文件是 train_jsbsim.py，用于启动 JSBSim 训练流程 [S1]。"
    sources = [
        Document(
            page_content="train_jsbsim.py defines the training script entry point for JSBSim experiments.",
            metadata={
                "source": "data/raw_imported/code/train_jsbsim.py",
                "imported_source": "data/raw_imported/code/train_jsbsim.py",
                "category_dir": "code",
                "doc_type": "code",
                "chunk_id": "chunk_train_jsbsim",
            },
        )
    ]
    result = check_citations(answer=answer, sources=sources)
    assert result["citation_check_passed"] is True
    assert result["supported_claim_ratio"] == 1.0
