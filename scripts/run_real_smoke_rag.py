from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "real_ingest_smoke.yaml"
REPAIR_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_repair.yaml"
KEYWORD_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_keyword.yaml"
ANSWER_VERIFICATION_CONFIG_PATH = PROJECT_ROOT / "config" / "answer_verification.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.generation.citation_checker import check_citations
from src.generation.evidence_gate import evaluate_evidence
from src.generation.llm import OllamaLLM
from src.retrieval.real_smoke_retriever import retrieve_real_smoke


INSUFFICIENT_EVIDENCE_ANSWER = (
    "当前知识库没有足够依据回答这个问题。"
    "已检索到一些相近资料，但它们不足以支持该结论。"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a RAG query against the real smoke Chroma collection.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "enhanced", "keyword", "enhanced_keyword"],
        default="enhanced",
        help="dense keeps V2b.1 behavior; enhanced adds metadata-aware recall; keyword uses BM25-lite; enhanced_keyword combines all channels.",
    )
    parser.add_argument(
        "--enable-evidence-gate",
        action="store_true",
        help="Apply V2c.0 rule-based evidence sufficiency gate before calling the LLM.",
    )
    parser.add_argument(
        "--enable-citation-check",
        action="store_true",
        help="Run V2c.0 rule-based citation support check after answer generation.",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        result = run_query(
            args.question,
            config,
            retrieval_mode=args.retrieval_mode,
            enable_evidence_gate=args.enable_evidence_gate,
            enable_citation_check=args.enable_citation_check,
        )
    except Exception as exc:  # noqa: BLE001 - RAG failures should be explicit.
        print("Real smoke RAG query failed.")
        print(f"reason: {exc}")
        return 1

    print_result(result)
    return 0


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    repair_config = load_optional_repair_config()
    if repair_config:
        config.update(repair_config)
    keyword_config = load_optional_keyword_config()
    if keyword_config:
        config.update(keyword_config)
    answer_verification_config = load_optional_answer_verification_config()
    if answer_verification_config:
        config["answer_verification"] = answer_verification_config
    return config


def load_optional_repair_config(path: Path = REPAIR_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {path}")
    retrieval = raw.get("retrieval") or {}
    if not isinstance(retrieval, dict):
        retrieval = {}
    merged: dict[str, Any] = {
        "collection_name": raw.get("real_smoke_collection_name"),
        "persist_directory": raw.get("real_smoke_persist_directory"),
    }
    merged.update(retrieval)
    return {key: value for key, value in merged.items() if value is not None}


def load_optional_keyword_config(path: Path = KEYWORD_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {path}")
    keyword = raw.get("keyword_retrieval") or {}
    enhanced_keyword = raw.get("enhanced_keyword") or {}
    fields = keyword.get("fields") if isinstance(keyword, dict) else {}
    merged: dict[str, Any] = {
        "collection_name": raw.get("collection_name"),
        "persist_directory": raw.get("persist_directory"),
        "keyword_top_k": keyword.get("top_k") if isinstance(keyword, dict) else None,
        "keyword_lowercase": keyword.get("lowercase", True) if isinstance(keyword, dict) else True,
        "keyword_min_token_len": keyword.get("min_token_len", 2) if isinstance(keyword, dict) else 2,
        "keyword_max_chunks_per_source": keyword.get("max_chunks_per_source") if isinstance(keyword, dict) else None,
        "keyword_field_weights": normalize_keyword_field_weights(fields if isinstance(fields, dict) else {}),
        "enhanced_keyword": enhanced_keyword if isinstance(enhanced_keyword, dict) else {},
    }
    return {key: value for key, value in merged.items() if value is not None}


def load_optional_answer_verification_config(path: Path = ANSWER_VERIFICATION_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {path}")
    return raw


def normalize_keyword_field_weights(fields: dict[str, Any]) -> dict[str, float]:
    mapping = {
        "content_weight": "content",
        "filename_weight": "filename",
        "path_weight": "path",
        "title_section_weight": "title_section",
        "category_doc_type_weight": "category_doc_type",
    }
    weights: dict[str, float] = {}
    for source_key, target_key in mapping.items():
        if source_key in fields:
            weights[target_key] = float(fields[source_key])
    return weights


def run_query(
    question: str,
    config: dict[str, Any],
    *,
    retrieval_mode: str,
    enable_evidence_gate: bool = False,
    enable_citation_check: bool = False,
) -> dict[str, Any]:
    collection_name = str(config["collection_name"])
    persist_directory = PROJECT_ROOT / str(config["persist_directory"])
    retrieval_kwargs = build_retrieval_kwargs(config, retrieval_mode)
    retrieval = retrieve_real_smoke(
        question=question,
        persist_directory=persist_directory,
        collection_name=collection_name,
        retrieval_mode=retrieval_mode,
        **retrieval_kwargs,
    )

    answer_config = config.get("answer_verification") or {}
    evidence_gate_result: dict[str, Any] | None = None
    if enable_evidence_gate:
        evidence_gate_result = evaluate_evidence(
            question=question,
            retrieval_result=retrieval,
            eval_item=None,
            config=answer_config.get("evidence_gate") if isinstance(answer_config, dict) else None,
        )

    llm_called = True
    if evidence_gate_result and not evidence_gate_result["evidence_sufficient"]:
        answer = INSUFFICIENT_EVIDENCE_ANSWER
        llm_called = False
    else:
        answer = OllamaLLM().generate(build_prompt(question, retrieval.documents))
        if not answer.strip():
            answer = build_empty_answer_fallback(retrieval.documents)

    citation_check_result: dict[str, Any] | None = None
    if enable_citation_check:
        citation_check_result = check_citations(
            answer=answer,
            sources=retrieval.documents,
            config=answer_config.get("citation_check") if isinstance(answer_config, dict) else None,
        )

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "question": question,
        "retrieval_mode": retrieval_mode,
        "query_type": retrieval.debug["query_type"],
        "target_categories": retrieval.debug["target_categories"],
        "expanded_query": retrieval.debug["expanded_query"],
        "dense_sources": retrieval.debug["dense_sources"],
        "metadata_sources": retrieval.debug["metadata_sources"],
        "keyword_sources": retrieval.debug["keyword_sources"],
        "final_sources": retrieval.debug["final_sources"],
        "source_diversity": retrieval.debug["source_diversity"],
        "evidence_gate_enabled": enable_evidence_gate,
        "evidence_gate": evidence_gate_result,
        "citation_check_enabled": enable_citation_check,
        "citation_check": citation_check_result,
        "llm_called": llm_called,
        "answer": answer,
        "answer_summary": summarize_answer(answer),
        "sources": format_sources(retrieval.documents),
    }
    success_notes, failure_notes = assess_query_result(result)
    result["success_notes"] = success_notes
    result["failure_notes"] = failure_notes
    append_query_log(PROJECT_ROOT / str(config["output_retrieval_compare_log"]), result)
    return result


def build_retrieval_kwargs(config: dict[str, Any], retrieval_mode: str) -> dict[str, Any]:
    values = {
        "dense_top_k": int(config.get("dense_top_k", config.get("top_k", 5))),
        "metadata_top_k": int(config.get("metadata_top_k", 5)),
        "keyword_top_k": int(config.get("keyword_top_k", 8)),
        "final_top_k": int(config.get("final_top_k", 8)),
        "max_chunks_per_source": int(config.get("max_chunks_per_source", config.get("keyword_max_chunks_per_source", 2))),
        "enable_source_diversity": bool(config.get("enable_source_diversity", True)),
        "source_key": str(config.get("source_key", "imported_source")),
        "keyword_lowercase": bool(config.get("keyword_lowercase", True)),
        "keyword_min_token_len": int(config.get("keyword_min_token_len", 2)),
        "keyword_field_weights": config.get("keyword_field_weights") or None,
    }
    if retrieval_mode == "enhanced_keyword":
        enhanced_keyword = config.get("enhanced_keyword") or {}
        if isinstance(enhanced_keyword, dict):
            for key in ["dense_top_k", "metadata_top_k", "keyword_top_k", "final_top_k", "max_chunks_per_source"]:
                if key in enhanced_keyword:
                    values[key] = int(enhanced_keyword[key])
    if retrieval_mode == "keyword":
        values["final_top_k"] = int(config.get("keyword_top_k", values["final_top_k"]))
        values["max_chunks_per_source"] = int(config.get("keyword_max_chunks_per_source", values["max_chunks_per_source"]))
    return values


def print_result(result: dict[str, Any]) -> None:
    print("Real smoke RAG query completed.")
    print(f"collection_name: {result['collection_name']}")
    print(f"persist_directory: {result['persist_directory']}")
    print(f"retrieval_mode: {result['retrieval_mode']}")
    print(f"query_type: {result['query_type']}")
    print(f"target_categories: {result['target_categories']}")
    print(f"expanded_query: {result['expanded_query']}")
    print(f"question: {result['question']}")
    print_sources("Dense sources", result["dense_sources"])
    print_sources("Metadata sources", result["metadata_sources"])
    print_sources("Keyword sources", result["keyword_sources"])
    print_sources("Final sources", result["final_sources"])
    print(f"\nSource diversity: {result['source_diversity']}")
    if result.get("evidence_gate_enabled"):
        print("\nEvidence gate:")
        print(json.dumps(result.get("evidence_gate"), ensure_ascii=False, indent=2))
        print(f"llm_called: {result.get('llm_called')}")
    print("\nAnswer:")
    print(result["answer"])
    if result.get("citation_check_enabled"):
        print("\nCitation check:")
        print(json.dumps(result.get("citation_check"), ensure_ascii=False, indent=2))
    print("\nSources:")
    for source in result["sources"]:
        print(
            f"[{source['label']}] score={source['score']} final_score={source['final_score']} "
            f"source={source['source']} category_dir={source['category_dir']} "
            f"doc_type={source['doc_type']} chunk_id={source['chunk_id']} "
            f"source_rank_within_file={source['source_rank_within_file']}"
        )
        print(f"    imported_source={source['imported_source']}")
        print(f"    original_source={source['original_source']}")
        print(f"    keyword_score={source['keyword_score']}")
        print(f"    matched_terms={source['matched_terms']}")
        print(f"    matched_fields={source['matched_fields']}")
        print(f"    retrieval_channels={source['retrieval_channels']}")
        print(f"    why_selected={source['why_selected']}")
        print(f"    match_reason={source['match_reason']}")
    print(f"\nsuccess_notes: {result['success_notes']}")
    print(f"failure_notes: {result['failure_notes']}")


def print_sources(title: str, sources: list[dict[str, Any]]) -> None:
    print(f"\n{title}:")
    if not sources:
        print("  []")
        return
    for item in sources:
        print(
            "  "
            f"rank={item.get('rank')} source={item.get('source')} "
            f"category_dir={item.get('category_dir')} doc_type={item.get('doc_type')} "
            f"chunk_id={item.get('chunk_id')} final_score={item.get('final_score')} "
            f"metadata_score={item.get('metadata_score')} "
            f"keyword_score={item.get('keyword_score')} "
            f"source_rank_within_file={item.get('source_rank_within_file')} "
            f"channels={item.get('retrieval_channels')} "
            f"why={item.get('why_selected')}"
        )


def build_prompt(question: str, documents: list[Document]) -> str:
    evidence = format_evidence(documents)
    return f"""/no_think
你是 Domain-RAG Agent 的真实资料 smoke test 问答模块。
只能根据下面的检索资料回答问题。
如果资料不足，必须明确说明“当前知识库没有足够依据”。
回答事实、代码位置、配置含义或实验结果时必须引用来源编号，例如 [S1]。
不要编造没有出现在资料中的文件、指标、实验结果或代码行为。
回答要简洁，最多 5 条要点，总字数尽量不超过 350 字，并确保句子完整。

检索资料：
{evidence}

用户问题：{question}

请输出：
1. 回答
2. 来源依据"""


def build_empty_answer_fallback(documents: list[Document]) -> str:
    if not documents:
        return "1. 回答\n当前知识库没有足够依据。\n\n2. 来源依据\n没有检索到资料。"
    lines = [
        "1. 回答",
        "LLM 未生成有效文本；以下仅列出检索证据的保守摘要，不扩展推断。",
    ]
    for index, document in enumerate(documents[:3], start=1):
        snippet = " ".join(document.page_content.strip().split())[:220]
        lines.append(f"- [S{index}] {snippet}")
    lines.append("\n2. 来源依据")
    lines.append("以上内容均来自 Top-K 检索片段。")
    return "\n".join(lines)


def format_evidence(documents: list[Document]) -> str:
    if not documents:
        return "没有检索到资料。"
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        content = document.page_content.strip()
        if len(content) > 1600:
            content = f"{content[:1600]}\n...[truncated]"
        blocks.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"source: {metadata.get('source', 'unknown_source')}",
                    f"category_dir: {metadata.get('category_dir', '')}",
                    f"doc_type: {metadata.get('doc_type', '')}",
                    f"chunk_id: {metadata.get('chunk_id', '')}",
                    f"imported_source: {metadata.get('imported_source', '')}",
                    f"original_source: {metadata.get('original_source', '')}",
                    f"section: {metadata.get('section', '')}",
                    f"matched_terms: {metadata.get('matched_terms', [])}",
                    f"matched_fields: {metadata.get('matched_fields', [])}",
                    f"why_selected: {metadata.get('why_selected', '')}",
                    "content:",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks)


def format_sources(documents: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        sources.append(
            {
                "label": f"S{index}",
                "score": metadata.get("dense_score"),
                "final_score": metadata.get("final_score"),
                "metadata_score": metadata.get("metadata_score"),
                "keyword_score": metadata.get("keyword_score"),
                "matched_terms": metadata.get("matched_terms", []),
                "matched_fields": metadata.get("matched_fields", []),
                "retrieval_channels": metadata.get("retrieval_channels", []),
                "source": metadata.get("source", ""),
                "category_dir": metadata.get("category_dir", ""),
                "doc_type": metadata.get("doc_type", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "imported_source": metadata.get("imported_source", ""),
                "original_source": metadata.get("original_source", ""),
                "section": metadata.get("section", ""),
                "why_selected": metadata.get("why_selected", ""),
                "match_reason": metadata.get("match_reason", ""),
                "source_rank_within_file": metadata.get("source_rank_within_file"),
                "source_diversity_key": metadata.get("source_diversity_key", ""),
                "source_diversity_relaxed": metadata.get("source_diversity_relaxed", False),
            }
        )
    return sources


def assess_query_result(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    question = str(result["question"])
    final_sources = result.get("final_sources", [])
    source_text = " ".join(
        " ".join([str(item.get("source", "")), str(item.get("imported_source", "")), str(item.get("original_source", ""))])
        for item in final_sources
    ).lower()
    categories = {str(item.get("category_dir", "")) for item in final_sources}
    success_notes: list[str] = []
    failure_notes: list[str] = []

    if "训练入口" in question or "入口" in question:
        if "train_jsbsim.py" in source_text:
            success_notes.append("train_jsbsim.py recalled")
        else:
            failure_notes.append("train_jsbsim.py not recalled")

    if "实验" in question and "配置" in question:
        if "experiments" in categories and "configs" in categories:
            success_notes.append("experiments and configs both recalled")
        else:
            failure_notes.append(f"missing mixed categories; final_categories={sorted(categories)}")

    if "奖励" in question or "任务配置" in question:
        if "code" in categories and "configs" in categories:
            success_notes.append("reward/task query kept code and configs")
        else:
            failure_notes.append(f"reward/task query categories limited: {sorted(categories)}")

    if "OPD" in question or "对手预测" in question:
        if "notes" in categories or "thesis_or_reports" in categories:
            success_notes.append("OPD query recalled notes/reports")
        else:
            failure_notes.append(f"OPD query missed notes/reports: {sorted(categories)}")

    return success_notes, failure_notes


def summarize_answer(answer: str) -> str:
    return " ".join(answer.split())[:300]


def append_query_log(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
