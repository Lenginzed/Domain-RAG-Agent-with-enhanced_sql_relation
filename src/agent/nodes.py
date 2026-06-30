from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document

from src.agent.trace import append_trace, summarize_text
from src.evaluation.evidence_quality import score_answer_record
from src.generation.citation_checker import check_citations
from src.generation.evidence_gate import evaluate_evidence
from src.generation.llm import OllamaLLM
from src.retrieval.query_classifier import classify_query
from src.retrieval.query_expansion import expand_query
from src.retrieval.real_smoke_retriever import retrieve_real_smoke
from src.sql_sag.ui_adapter import run_sql_sag_retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSUFFICIENT_EVIDENCE_ANSWER = (
    "当前知识库没有足够依据回答这个问题。已检索到一些相近资料，但它们不足以支持该结论。"
)
RETRIEVAL_ONLY_ANSWER = "Retrieval-only run: answer generation was skipped, and no LLM was called."


def classify_query_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "classify_query_node"
    try:
        question = str(state.get("question", ""))
        classification = classify_query(question)
        expanded_query = expand_query(question, classification)
        state.update(
            {
                "query_type": classification.get("query_type"),
                "target_categories": classification.get("target_categories", []),
                "expanded_query": expanded_query,
            }
        )
        return append_trace(
            state,
            node=node,
            status="ok",
            input_summary={"question": summarize_text(question, 160)},
            output_summary={
                "query_type": state.get("query_type"),
                "target_categories": state.get("target_categories", []),
                "expanded_query": summarize_text(expanded_query, 220),
            },
        )
    except Exception as exc:  # noqa: BLE001 - graph should surface node failures as trace.
        return append_trace(state, node=node, status="error", error=str(exc))


def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "retrieve_node"
    try:
        config = state.get("config") or {}
        collection = config.get("collection", {})
        retrieval_config = {**(config.get("retrieval") or {})}
        mode = str(state.get("retrieval_mode") or retrieval_config.get("mode", "enhanced_keyword"))
        retrieval_config["mode"] = mode
        calibration_config = None
        if bool(state.get("enable_calibration", retrieval_config.get("enable_calibration", True))):
            calibration_config = load_calibration_config(retrieval_config, mode)

        if mode in {"sql_relation", "enhanced_sql_relation"}:
            return retrieve_sql_sag_node(state, node=node, mode=mode, retrieval_config=retrieval_config)

        result = retrieve_real_smoke(
            question=str(state.get("question", "")),
            persist_directory=resolve_project_path(collection.get("persist_directory", "")),
            collection_name=str(collection.get("collection_name", "")),
            retrieval_mode=mode,
            dense_top_k=int(retrieval_config.get("dense_top_k", 5)),
            metadata_top_k=int(retrieval_config.get("metadata_top_k", 8)),
            keyword_top_k=int(retrieval_config.get("keyword_top_k", 8)),
            final_top_k=int(retrieval_config.get("final_top_k", 8)),
            max_chunks_per_source=int(retrieval_config.get("max_chunks_per_source", 2)),
            enable_source_diversity=bool(retrieval_config.get("enable_source_diversity", True)),
            source_key=str(retrieval_config.get("source_key", "imported_source")),
            calibration_config=calibration_config,
        )
        sources = attach_source_content(result.debug.get("final_sources", []), result.documents)
        state.update(
            {
                "retrieval_mode": mode,
                "sources": sources,
                "source_diversity": result.debug.get("source_diversity", {}),
                "query_type": result.debug.get("query_type") or state.get("query_type"),
                "target_categories": result.debug.get("target_categories", state.get("target_categories", [])),
                "expanded_query": result.debug.get("expanded_query", state.get("expanded_query", "")),
                "retrieval_debug": make_retrieval_debug_summary(result.debug),
            }
        )
        return append_trace(
            state,
            node=node,
            status="ok",
            input_summary={
                "question": summarize_text(str(state.get("question", "")), 160),
                "retrieval_mode": mode,
                "enable_calibration": bool(calibration_config),
            },
            output_summary={
                "source_count": len(sources),
                "top_sources": summarize_source_rows(sources),
                "source_diversity": state.get("source_diversity", {}),
            },
        )
    except Exception as exc:  # noqa: BLE001
        state["sources"] = []
        state["source_diversity"] = {}
        return append_trace(state, node=node, status="error", error=str(exc))


def evidence_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "evidence_gate_node"
    try:
        retrieval_like = {"final_sources": state.get("sources", []), "debug": {"final_sources": state.get("sources", [])}}
        gate = evaluate_evidence(
            question=str(state.get("question", "")),
            retrieval_result=retrieval_like,
            eval_item=None,
            config=(state.get("config") or {}).get("evidence_gate") or {},
        )
        state.update(
            {
                "evidence_sufficient": bool(gate.get("evidence_sufficient", False)),
                "evidence_gate_reason": str(gate.get("reason", "")),
                "evidence_gate_signals": gate.get("signals", {}),
                "insufficient_answer": not bool(gate.get("evidence_sufficient", False)),
            }
        )
        return append_trace(
            state,
            node=node,
            status="ok",
            input_summary={"source_count": len(state.get("sources", []))},
            output_summary={
                "evidence_sufficient": state.get("evidence_sufficient"),
                "reason": state.get("evidence_gate_reason"),
                "final_categories": (gate.get("signals") or {}).get("final_categories", []),
            },
        )
    except Exception as exc:  # noqa: BLE001
        state.update({"evidence_sufficient": False, "evidence_gate_reason": str(exc), "insufficient_answer": True})
        return append_trace(state, node=node, status="error", error=str(exc))


def retrieve_sql_sag_node(
    state: dict[str, Any],
    *,
    node: str,
    mode: str,
    retrieval_config: dict[str, Any],
) -> dict[str, Any]:
    config = state.get("config") or {}
    sql_config = config.get("enhanced_sql_relation") or {}
    config_path = sql_config.get("config_path", "config/sql_sag_v4d.yaml")
    result = run_sql_sag_retrieval(
        question=str(state.get("question", "")),
        retrieval_mode=mode,
        config_path=config_path,
        final_top_k=int(retrieval_config.get("final_top_k", 8)),
    )
    sources = result.get("final_sources", [])
    source_diversity = summarize_source_diversity_for_rows(
        sources,
        source_key=str(retrieval_config.get("source_key", "imported_source")),
        max_chunks_per_source=int(retrieval_config.get("max_chunks_per_source", 2)),
    )
    sql_debug = result.get("sql_relation_debug", {})
    state.update(
        {
            "retrieval_mode": mode,
            "sources": sources,
            "source_diversity": source_diversity,
            "category_intent": result.get("category_intent", {}),
            "sql_relation_debug": sql_debug,
            "fusion_debug": result.get("fusion_debug", {}),
            "retrieval_debug": {
                "enhanced_keyword_candidate_count": len(result.get("enhanced_keyword_candidates", [])),
                "sql_relation_candidate_count": sql_debug.get("candidate_count", 0),
                "relation_path_found_count": sql_debug.get("relation_path_found_count", 0),
                "sql_only_candidate_count": sum(1 for item in sources if item.get("sql_only_candidate")),
            },
        }
    )
    return append_trace(
        state,
        node=node,
        status="ok",
        input_summary={
            "question": summarize_text(str(state.get("question", "")), 160),
            "retrieval_mode": mode,
            "sql_config": str(config_path),
        },
        output_summary={
            "retrieval_mode": mode,
            "source_count": len(sources),
            "top_sources": summarize_source_rows(sources),
            "category_intent": state.get("category_intent", {}),
            "sql_relation_candidate_count": sql_debug.get("candidate_count", 0),
            "relation_path_found_count": sql_debug.get("relation_path_found_count", 0),
            "sql_only_candidate_count": state["retrieval_debug"].get("sql_only_candidate_count", 0),
            "source_diversity": source_diversity,
        },
    )


def refusal_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "refusal_node"
    state.update({"answer": INSUFFICIENT_EVIDENCE_ANSWER, "llm_called": False, "insufficient_answer": True})
    return append_trace(
        state,
        node=node,
        status="ok",
        input_summary={"evidence_sufficient": state.get("evidence_sufficient")},
        output_summary={"answer_preview": summarize_text(INSUFFICIENT_EVIDENCE_ANSWER, 180), "llm_called": False},
    )


def answer_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "answer_node"
    run_mode = str(state.get("run_mode", "retrieval_only"))
    if run_mode == "retrieval_only":
        state.update({"answer": None, "llm_called": False})
        return append_trace(
            state,
            node=node,
            status="skipped",
            input_summary={"run_mode": run_mode},
            output_summary={"llm_called": False, "reason": "retrieval_only"},
        )
    if not bool(state.get("evidence_sufficient", False)):
        return refusal_node(state)
    try:
        prompt = build_answer_prompt(str(state.get("question", "")), state.get("sources", []))
        answer = OllamaLLM().generate(prompt)
        state.update({"answer": answer, "llm_called": True, "insufficient_answer": False})
        return append_trace(
            state,
            node=node,
            status="ok",
            input_summary={"run_mode": run_mode, "source_count": len(state.get("sources", []))},
            output_summary={"llm_called": True, "answer_preview": summarize_text(answer, 240)},
        )
    except Exception as exc:  # noqa: BLE001
        state.update({"answer": None, "llm_called": False})
        return append_trace(state, node=node, status="error", error=str(exc))


def citation_check_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "citation_check_node"
    try:
        if state.get("run_mode") == "retrieval_only":
            citation = skipped_citation_result("retrieval_only")
            state["citation_result"] = citation
            return append_trace(
                state,
                node=node,
                status="skipped",
                input_summary={"run_mode": state.get("run_mode")},
                output_summary={"reason": citation["reason"]},
            )
        if bool(state.get("insufficient_answer", False)):
            citation = skipped_citation_result("insufficient_answer", insufficient=True)
        else:
            citation = check_citations(
                answer=str(state.get("answer") or ""),
                sources=state.get("sources", []),
                config=(state.get("config") or {}).get("citation_check") or {},
            )
        state["citation_result"] = citation
        return append_trace(
            state,
            node=node,
            status="ok",
            input_summary={"answer_present": bool(state.get("answer"))},
            output_summary={
                "citation_check_passed": citation.get("citation_check_passed"),
                "supported_claim_ratio": citation.get("supported_claim_ratio"),
                "unsupported_claim_count": len(citation.get("unsupported_claims", [])),
                "weak_claim_count": citation.get("weak_claim_count", 0),
            },
        )
    except Exception as exc:  # noqa: BLE001
        state["citation_result"] = skipped_citation_result("citation_error")
        return append_trace(state, node=node, status="error", error=str(exc))


def evidence_quality_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "evidence_quality_node"
    try:
        record = build_answer_record_for_quality(state)
        quality = score_answer_record(record, (state.get("config") or {}).get("evidence_quality") or {})
        state["evidence_quality"] = quality
        state["needs_human_review"] = bool(record.get("needs_human_review", False) or quality.get("needs_human_review", False))
        return append_trace(
            state,
            node=node,
            status="ok",
            input_summary={"run_mode": state.get("run_mode"), "source_count": len(state.get("sources", []))},
            output_summary={
                "evidence_quality_score": quality.get("evidence_quality_score"),
                "evidence_quality_level": quality.get("evidence_quality_level"),
                "warnings": quality.get("evidence_quality_warnings", []),
            },
        )
    except Exception as exc:  # noqa: BLE001
        state["evidence_quality"] = {}
        return append_trace(state, node=node, status="error", error=str(exc))


def review_suggestion_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "review_suggestion_node"
    suggestions: list[str] = []
    citation = state.get("citation_result") if isinstance(state.get("citation_result"), dict) else {}
    quality = state.get("evidence_quality") if isinstance(state.get("evidence_quality"), dict) else {}
    if not bool(state.get("evidence_sufficient", False)):
        suggestions.append("Review whether the retrieved sources are sufficient before answering.")
    if citation.get("unsupported_claims"):
        suggestions.append("Inspect unsupported claims before accepting the answer.")
    if int(citation.get("weak_claim_count", 0) or 0) > 0:
        suggestions.append("Inspect weak citation claims.")
    for warning in quality.get("evidence_quality_warnings", []) or []:
        suggestions.append(f"Quality warning: {warning}")
    if bool(state.get("needs_human_review", False)) and not suggestions:
        suggestions.append("Manual review suggested by evidence quality workflow.")
    state["review_suggestions"] = unique_values(suggestions)
    return append_trace(
        state,
        node=node,
        status="ok",
        input_summary={"needs_human_review": state.get("needs_human_review", False)},
        output_summary={"review_suggestions": state["review_suggestions"]},
    )


def human_review_decision_node(state: dict[str, Any]) -> dict[str, Any]:
    node = "human_review_decision_node"
    decision = build_human_review_decision(state)
    state["review_decision"] = decision
    state["needs_human_review"] = bool(decision.get("needs_human_review", False))
    state["review_suggestions"] = list(decision.get("review_reasons", []))
    return append_trace(
        state,
        node=node,
        status="ok",
        input_summary={
            "evidence_sufficient": state.get("evidence_sufficient"),
            "quality": (state.get("evidence_quality") or {}).get("evidence_quality_level")
            if isinstance(state.get("evidence_quality"), dict)
            else None,
        },
        output_summary={
            "needs_human_review": decision.get("needs_human_review", False),
            "review_reasons": decision.get("review_reasons", []),
            "recommended_action": decision.get("recommended_action", ""),
        },
    )


def build_human_review_decision(state: dict[str, Any]) -> dict[str, Any]:
    citation = state.get("citation_result") if isinstance(state.get("citation_result"), dict) else {}
    quality = state.get("evidence_quality") if isinstance(state.get("evidence_quality"), dict) else {}
    sources = state.get("sources") if isinstance(state.get("sources"), list) else []
    reasons: list[str] = []

    if not bool(state.get("evidence_sufficient", False)):
        reasons.append("evidence_insufficient")
    if citation.get("unsupported_claims"):
        reasons.append("unsupported_claims_present")
    if int(citation.get("weak_claim_count", 0) or 0) > 0:
        reasons.append("weak_claims_present")
    quality_level = str(quality.get("evidence_quality_level", "") or "")
    if quality_level in {"medium", "low", "insufficient"}:
        reasons.append("medium_or_lower_quality")
    if any(float(source.get("calibration_score", 0) or 0) > 0 for source in sources[:3] if isinstance(source, dict)):
        reasons.append("calibration_applied")

    blocking_reasons = {
        "evidence_insufficient",
        "unsupported_claims_present",
        "weak_claims_present",
        "medium_or_lower_quality",
    }
    needs_review = bool(set(reasons).intersection(blocking_reasons))
    if "evidence_insufficient" in reasons:
        recommended_action = "rerun_with_more_sources"
    elif needs_review:
        recommended_action = "manual_review"
    else:
        recommended_action = "accept"

    return {
        "run_id": state.get("run_id", ""),
        "needs_human_review": needs_review,
        "review_reasons": unique_values(reasons),
        "recommended_action": recommended_action,
        "manual_review_fields": {
            "manual_judgment": "",
            "manual_citation_judgment": "",
            "manual_should_refuse": "",
            "manual_notes": "",
        },
    }


def route_after_evidence_gate(state: dict[str, Any]) -> str:
    return "insufficient" if not bool(state.get("evidence_sufficient", False)) else "sufficient"


def load_calibration_config(retrieval_config: dict[str, Any], mode: str) -> dict[str, Any] | None:
    calibration_path = retrieval_config.get("calibration_config")
    if not calibration_path:
        return None
    data = load_yaml(resolve_project_path(calibration_path))
    calibration = dict(data.get("calibration", data))
    apply_to_modes = {str(item) for item in calibration.get("apply_to_modes", [])}
    if apply_to_modes and mode not in apply_to_modes:
        return None
    return calibration if calibration.get("enabled", True) else None


def attach_source_content(sources: list[dict[str, Any]], documents: list[Document]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, document in zip(sources, documents):
        row = dict(source)
        content = str(document.page_content or "")
        row["content_preview"] = summarize_text(content, 900)
        row["page_content"] = summarize_text(content, 2200)
        rows.append(row)
    return rows


def make_retrieval_debug_summary(debug: dict[str, Any]) -> dict[str, Any]:
    return {
        "dense_sources": debug.get("dense_sources", []),
        "metadata_sources": debug.get("metadata_sources", []),
        "keyword_sources": debug.get("keyword_sources", []),
        "metadata_candidate_count": debug.get("metadata_candidate_count", 0),
        "keyword_candidate_count": debug.get("keyword_candidate_count", 0),
    }


def summarize_source_rows(sources: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources[:limit]:
        rows.append(
            {
                "rank": source.get("rank"),
                "source": source.get("source", ""),
                "category_dir": source.get("category_dir", ""),
                "chunk_id": source.get("chunk_id", ""),
                "final_score": source.get("final_score"),
                "calibration_score": source.get("calibration_score", 0.0),
                "retrieval_channels": source.get("retrieval_channels", []),
                "relation_score": source.get("relation_score", 0.0),
                "relation_score_normalized": source.get("relation_score_normalized", 0.0),
                "relation_path_found": bool(source.get("relation_path")),
                "category_intent_adjustment": source.get("category_intent_adjustment", 0.0),
                "high_frequency_entity_penalty": source.get("high_frequency_entity_penalty", 0.0),
            }
        )
    return rows


def summarize_source_diversity_for_rows(
    sources: list[dict[str, Any]],
    *,
    source_key: str,
    max_chunks_per_source: int,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for source in sources:
        key = str(source.get(source_key) or source.get("imported_source") or source.get("source") or "")
        counts[key] = counts.get(key, 0) + 1
    violations = {key: count for key, count in counts.items() if count > max_chunks_per_source}
    return {
        "enabled": True,
        "source_key": source_key,
        "max_chunks_per_source": max_chunks_per_source,
        "counts": counts,
        "violations": violations,
        "relaxed_sources": [],
        "relaxed": False,
    }


def build_answer_prompt(question: str, sources: list[dict[str, Any]]) -> str:
    evidence = format_evidence(sources)
    return f"""/no_think
You are the answer node in a controlled Domain-RAG workflow. Answer only from the retrieved evidence below.
If the evidence is insufficient, say "当前知识库没有足够依据".
Do not invent files, metrics, experiments, or code behavior that are not present in the evidence.
When stating facts, cite sources using [S1], [S2], etc. Keep the answer concise.

Retrieved evidence:
{evidence}

User question:
{question}

Output:
1. Answer
2. Sources
"""


def format_evidence(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "No retrieved evidence."
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"source: {source.get('source', '')}",
                    f"imported_source: {source.get('imported_source', '')}",
                    f"category_dir: {source.get('category_dir', '')}",
                    f"doc_type: {source.get('doc_type', '')}",
                    f"chunk_id: {source.get('chunk_id', '')}",
                    f"retrieval_channels: {source.get('retrieval_channels', [])}",
                    f"matched_terms: {source.get('matched_terms', [])}",
                    f"calibration_reasons: {source.get('calibration_reasons', [])}",
                    "content:",
                    str(source.get("page_content") or source.get("content_preview") or ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def skipped_citation_result(reason: str, *, insufficient: bool = False) -> dict[str, Any]:
    return {
        "insufficient_answer": insufficient,
        "skipped": True,
        "supported_claims": [],
        "unsupported_claims": [],
        "ignored_claims": [],
        "weak_claims": [],
        "checked_claims": [],
        "supported_claim_ratio": 1.0,
        "checked_claim_count": 0,
        "ignored_claim_count": 0,
        "weak_claim_count": 0,
        "claim_type_counts": {},
        "citation_check_passed": True,
        "reason": f"citation check skipped: {reason}",
    }


def build_answer_record_for_quality(state: dict[str, Any]) -> dict[str, Any]:
    citation = state.get("citation_result") if isinstance(state.get("citation_result"), dict) else {}
    gate_signals = state.get("evidence_gate_signals") if isinstance(state.get("evidence_gate_signals"), dict) else {}
    unsupported_count = len(citation.get("unsupported_claims", []) or [])
    weak_count = int(citation.get("weak_claim_count", 0) or 0)
    supported_ratio = float(citation.get("supported_claim_ratio", 0.0) or 0.0)
    insufficient_answer = bool(state.get("insufficient_answer", False) or citation.get("insufficient_answer", False))
    needs_human_review = (
        unsupported_count > 0
        or weak_count > 0
        or (not insufficient_answer and state.get("run_mode") == "full_rag" and supported_ratio < 0.8)
        or not bool(state.get("evidence_sufficient", False))
    )
    return {
        "id": "langgraph_v3a",
        "question": state.get("question", ""),
        "query_type": state.get("query_type", ""),
        "retrieval_mode": state.get("retrieval_mode", ""),
        "evidence_sufficient": bool(state.get("evidence_sufficient", False)),
        "evidence_gate_reason": state.get("evidence_gate_reason", ""),
        "evidence_gate_signals": gate_signals,
        "llm_called": bool(state.get("llm_called", False)),
        "answer": state.get("answer") or "",
        "insufficient_answer": insufficient_answer,
        "supported_claim_ratio": supported_ratio,
        "citation_check_passed": bool(citation.get("citation_check_passed", True)),
        "checked_claim_count": int(citation.get("checked_claim_count", 0) or 0),
        "ignored_claim_count": int(citation.get("ignored_claim_count", 0) or 0),
        "weak_claim_count": weak_count,
        "unsupported_claim_count": unsupported_count,
        "claim_type_counts": citation.get("claim_type_counts", {}),
        "needs_human_review": needs_human_review,
        "unsupported_claims": citation.get("unsupported_claims", []),
        "weak_claims": citation.get("weak_claims", []),
        "source_diversity": state.get("source_diversity", {}),
        "final_sources": state.get("sources", []),
    }


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def unique_values(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
