from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypedDict


RunMode = Literal["retrieval_only", "full_rag"]


class RagWorkflowState(TypedDict, total=False):
    run_id: str
    created_at: str
    completed_at: str | None
    graph_version: str
    trace_path: str | None
    review_decision: dict[str, Any] | None

    question: str
    run_mode: str
    retrieval_mode: str
    enable_calibration: bool

    config: dict[str, Any]
    collection_name: str
    persist_directory: str

    query_type: str | None
    target_categories: list[str]
    expanded_query: str

    sources: list[dict[str, Any]]
    source_diversity: dict[str, Any]
    category_intent: dict[str, Any]
    sql_relation_debug: dict[str, Any]
    fusion_debug: dict[str, Any]
    retrieval_debug: dict[str, Any]
    evidence_sufficient: bool | None
    evidence_gate_reason: str | None
    evidence_gate_signals: dict[str, Any]
    insufficient_answer: bool

    answer: str | None
    llm_called: bool

    citation_result: dict[str, Any] | None
    evidence_quality: dict[str, Any] | None

    needs_human_review: bool
    review_suggestions: list[str]

    errors: list[str]
    trace: list[dict[str, Any]]


def initial_state(
    *,
    question: str,
    run_mode: str,
    retrieval_mode: str,
    enable_calibration: bool,
    config: dict[str, Any],
    run_id: str = "",
    graph_version: str = "v3a",
) -> RagWorkflowState:
    collection = config.get("collection", {})
    return {
        "run_id": run_id,
        "created_at": now_iso(),
        "completed_at": None,
        "graph_version": graph_version,
        "trace_path": None,
        "review_decision": None,
        "question": question,
        "run_mode": run_mode,
        "retrieval_mode": retrieval_mode,
        "enable_calibration": enable_calibration,
        "config": config,
        "collection_name": str(collection.get("collection_name", "")),
        "persist_directory": str(collection.get("persist_directory", "")),
        "query_type": None,
        "target_categories": [],
        "expanded_query": "",
        "sources": [],
        "source_diversity": {},
        "category_intent": {},
        "sql_relation_debug": {},
        "fusion_debug": {},
        "retrieval_debug": {},
        "evidence_sufficient": None,
        "evidence_gate_reason": None,
        "evidence_gate_signals": {},
        "insufficient_answer": False,
        "answer": None,
        "llm_called": False,
        "citation_result": None,
        "evidence_quality": None,
        "needs_human_review": False,
        "review_suggestions": [],
        "errors": [],
        "trace": [],
    }


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
