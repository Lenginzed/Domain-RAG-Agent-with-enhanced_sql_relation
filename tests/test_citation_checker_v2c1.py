from __future__ import annotations

from langchain_core.documents import Document

from src.generation.citation_checker import check_citations


def test_heading_claim_is_ignored() -> None:
    result = check_citations(answer="实验结果文件", sources=[])
    assert result["unsupported_claims"] == []
    assert result["ignored_claim_count"] == 1
    assert result["ignored_claims"][0]["claim_type"] == "heading"


def test_list_intro_claim_is_ignored() -> None:
    answer = "这些资料中包含以下实验结果文件和配置文件："
    result = check_citations(answer=answer, sources=[])
    assert result["unsupported_claims"] == []
    assert result["ignored_claim_count"] == 1
    assert result["ignored_claims"][0]["claim_type"] == "list_intro"


def test_insufficient_evidence_answer_is_special_case() -> None:
    answer = "当前知识库没有足够依据回答这个问题。已检索到一些相近资料，但它们不足以支持该结论。"
    result = check_citations(answer=answer, sources=[])
    assert result["insufficient_answer"] is True
    assert result["citation_check_passed"] is True
    assert result["ignored_claims"][0]["claim_type"] == "insufficient_answer"


def test_train_jsbsim_fact_is_still_supported() -> None:
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
    assert result["supported_claims"][0]["claim_type"] == "factual_claim"


def test_obviously_wrong_filename_remains_unsupported() -> None:
    answer = "训练入口文件是 imaginary_launcher.py [S1]。"
    sources = [
        Document(
            page_content="train_jsbsim.py defines the training script entry point for JSBSim experiments.",
            metadata={
                "source": "data/raw_imported/code/train_jsbsim.py",
                "imported_source": "data/raw_imported/code/train_jsbsim.py",
            },
        )
    ]
    result = check_citations(answer=answer, sources=sources)
    assert result["citation_check_passed"] is False
    assert len(result["unsupported_claims"]) == 1
    assert result["unsupported_claims"][0]["checker_error_hint"] == "important_identifier_or_filename_not_found"
