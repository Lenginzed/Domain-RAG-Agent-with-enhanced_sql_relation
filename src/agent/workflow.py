from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    answer_node,
    citation_check_node,
    classify_query_node,
    evidence_gate_node,
    evidence_quality_node,
    human_review_decision_node,
    refusal_node,
    retrieve_node,
    review_suggestion_node,
    route_after_evidence_gate,
)
from src.agent.state import RagWorkflowState, initial_state
from src.agent.trace import (
    generate_run_id,
    load_trace_by_run_id,
    load_trace_index,
    now_iso,
    resolve_project_path,
    summarize_graph_run,
    write_graph_trace,
    write_last_run,
    write_trace_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "langgraph_v3a.yaml"
DEFAULT_V3B_CONFIG_PATH = PROJECT_ROOT / "config" / "langgraph_v3b.yaml"
DEFAULT_V4E_CONFIG_PATH = PROJECT_ROOT / "config" / "langgraph_v4e.yaml"


def load_langgraph_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid LangGraph config: {path}")
    config["_config_path"] = str(path)
    return config


def load_langgraph_v3b_config(config_path: str | Path = DEFAULT_V3B_CONFIG_PATH) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid LangGraph V3b config: {path}")
    base_path = config.get("base_config")
    if base_path:
        base = load_langgraph_v3b_config(resolve_project_path(base_path))
        merged = deep_merge(base, config)
    else:
        merged = config
    merged["_config_path"] = str(path)
    return merged


def build_rag_graph(config: dict[str, Any] | None = None) -> Any:
    graph = StateGraph(RagWorkflowState)
    graph.add_node("classify_query", classify_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evidence_gate", evidence_gate_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("answer_generation", answer_node)
    graph.add_node("citation_check", citation_check_node)
    graph.add_node("evidence_quality_scoring", evidence_quality_node)
    graph.add_node("review_suggestion", review_suggestion_node)

    graph.add_edge(START, "classify_query")
    graph.add_edge("classify_query", "retrieve")
    graph.add_edge("retrieve", "evidence_gate")
    graph.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {
            "insufficient": "refusal",
            "sufficient": "answer_generation",
        },
    )
    graph.add_edge("refusal", "citation_check")
    graph.add_edge("answer_generation", "citation_check")
    graph.add_edge("citation_check", "evidence_quality_scoring")
    graph.add_edge("evidence_quality_scoring", "review_suggestion")
    graph.add_edge("review_suggestion", END)
    return graph.compile()


def build_rag_graph_v3b(config: dict[str, Any] | None = None) -> Any:
    graph = StateGraph(RagWorkflowState)
    graph.add_node("classify_query", classify_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evidence_gate", evidence_gate_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("answer_generation", answer_node)
    graph.add_node("citation_check", citation_check_node)
    graph.add_node("evidence_quality_scoring", evidence_quality_node)
    graph.add_node("human_review_decision", human_review_decision_node)

    graph.add_edge(START, "classify_query")
    graph.add_edge("classify_query", "retrieve")
    graph.add_edge("retrieve", "evidence_gate")
    graph.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {
            "insufficient": "refusal",
            "sufficient": "answer_generation",
        },
    )
    graph.add_edge("refusal", "citation_check")
    graph.add_edge("answer_generation", "citation_check")
    graph.add_edge("citation_check", "evidence_quality_scoring")
    graph.add_edge("evidence_quality_scoring", "human_review_decision")
    graph.add_edge("human_review_decision", END)
    return graph.compile()


def run_rag_graph(
    *,
    question: str,
    run_mode: str = "retrieval_only",
    retrieval_mode: str | None = None,
    enable_calibration: bool | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_langgraph_config(config_path)
    validate_run_mode(run_mode)
    retrieval_config = config.get("retrieval", {})
    mode = retrieval_mode or str(retrieval_config.get("mode", "enhanced_keyword"))
    calibration_enabled = (
        bool(retrieval_config.get("enable_calibration", True))
        if enable_calibration is None
        else bool(enable_calibration)
    )
    state = initial_state(
        question=question,
        run_mode=run_mode,
        retrieval_mode=mode,
        enable_calibration=calibration_enabled,
        config=config,
    )
    compiled = build_rag_graph(config)
    final_state = compiled.invoke(state)
    summary = summarize_graph_run(final_state)
    record = {"summary": summary, "state": serialize_state(final_state)}
    outputs = config.get("outputs", {})
    if bool(config.get("graph", {}).get("enable_trace", True)):
        trace_path = write_graph_trace(record, outputs.get("graph_trace_jsonl", "storage/logs/langgraph_traces_v3a.jsonl"))
        summary["trace_path"] = str(trace_path)
        record["summary"] = summary
    last_run_path = write_last_run(record, outputs.get("graph_run_json", "storage/logs/langgraph_last_run_v3a.json"))
    summary["last_run_path"] = str(last_run_path)
    return {"summary": summary, "state": final_state, "record": record}


def run_rag_graph_v3b(
    *,
    question: str,
    run_mode: str = "retrieval_only",
    retrieval_mode: str | None = None,
    enable_calibration: bool | None = None,
    config_path: str | Path = DEFAULT_V3B_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_langgraph_v3b_config(config_path)
    validate_run_mode(run_mode)
    retrieval_config = config.get("retrieval", {})
    mode = retrieval_mode or str(retrieval_config.get("mode", "enhanced_keyword"))
    calibration_enabled = (
        bool(retrieval_config.get("enable_calibration", True))
        if enable_calibration is None
        else bool(enable_calibration)
    )
    run_config = config.get("run", {})
    run_id = generate_run_id(str(run_config.get("run_id_prefix", "lg")))
    state = initial_state(
        question=question,
        run_mode=run_mode,
        retrieval_mode=mode,
        enable_calibration=calibration_enabled,
        config=config,
        run_id=run_id,
        graph_version=str(run_config.get("graph_version", "v3b")),
    )
    compiled = build_rag_graph_v3b(config)
    final_state = compiled.invoke(state)
    final_state["completed_at"] = now_iso()
    summary = summarize_graph_run(final_state)
    record = {"summary": summary, "state": serialize_state(final_state)}
    trace_config = config.get("trace", {})
    trace_path = write_graph_trace(record, trace_config.get("trace_jsonl", "storage/logs/langgraph_traces_v3b.jsonl"))
    summary["trace_path"] = str(trace_path)
    final_state["trace_path"] = str(trace_path)
    record["summary"] = summary
    last_run_path = write_last_run(record, trace_config.get("last_run_json", "storage/logs/langgraph_last_run_v3b.json"))
    summary["last_run_path"] = str(last_run_path)
    record["summary"] = summary
    index_path = write_trace_index(
        summary,
        trace_config.get("trace_index_json", "storage/logs/langgraph_trace_index_v3b.json"),
        max_recent_runs=int(trace_config.get("max_recent_runs", 50)),
    )
    summary["trace_index_path"] = str(index_path)
    record["summary"] = summary
    write_last_run(record, trace_config.get("last_run_json", "storage/logs/langgraph_last_run_v3b.json"))
    if bool(config.get("human_review", {}).get("enabled", True)):
        write_review_decision_outputs(final_state, summary, config)
    return {"summary": summary, "state": final_state, "record": record}


def load_recent_graph_runs(config_path: str | Path = DEFAULT_V3B_CONFIG_PATH) -> list[dict[str, Any]]:
    config = load_langgraph_v3b_config(config_path)
    return load_trace_index(config.get("trace", {}).get("trace_index_json", "storage/logs/langgraph_trace_index_v3b.json"))


def load_graph_trace(run_id: str, config_path: str | Path = DEFAULT_V3B_CONFIG_PATH) -> dict[str, Any] | None:
    config = load_langgraph_v3b_config(config_path)
    return load_trace_by_run_id(run_id, config.get("trace", {}).get("trace_jsonl", "storage/logs/langgraph_traces_v3b.jsonl"))


def validate_run_mode(run_mode: str) -> None:
    if run_mode not in {"retrieval_only", "full_rag"}:
        raise ValueError("run_mode must be either 'retrieval_only' or 'full_rag'.")


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": state.get("run_id", ""),
        "created_at": state.get("created_at", ""),
        "completed_at": state.get("completed_at"),
        "graph_version": state.get("graph_version", ""),
        "trace_path": state.get("trace_path"),
        "review_decision": state.get("review_decision"),
        "question": state.get("question", ""),
        "run_mode": state.get("run_mode", ""),
        "retrieval_mode": state.get("retrieval_mode", ""),
        "enable_calibration": state.get("enable_calibration", False),
        "query_type": state.get("query_type"),
        "target_categories": state.get("target_categories", []),
        "expanded_query": state.get("expanded_query", ""),
        "sources": state.get("sources", []),
        "source_diversity": state.get("source_diversity", {}),
        "category_intent": state.get("category_intent", {}),
        "sql_relation_debug": state.get("sql_relation_debug", {}),
        "fusion_debug": state.get("fusion_debug", {}),
        "retrieval_debug": state.get("retrieval_debug", {}),
        "evidence_sufficient": state.get("evidence_sufficient"),
        "evidence_gate_reason": state.get("evidence_gate_reason"),
        "evidence_gate_signals": state.get("evidence_gate_signals", {}),
        "insufficient_answer": state.get("insufficient_answer", False),
        "answer": state.get("answer"),
        "llm_called": state.get("llm_called", False),
        "citation_result": state.get("citation_result"),
        "evidence_quality": state.get("evidence_quality"),
        "needs_human_review": state.get("needs_human_review", False),
        "review_suggestions": state.get("review_suggestions", []),
        "errors": state.get("errors", []),
        "trace": state.get("trace", []),
    }


def write_review_decision_outputs(state: dict[str, Any], summary: dict[str, Any], config: dict[str, Any]) -> None:
    decision = state.get("review_decision") if isinstance(state.get("review_decision"), dict) else {}
    if not decision:
        return
    human_review = config.get("human_review", {})
    jsonl_path = resolve_project_path(human_review.get("review_decisions_jsonl", "storage/logs/langgraph_review_decisions_v3b.jsonl"))
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": now_iso(),
        "run_id": state.get("run_id", ""),
        "question": state.get("question", ""),
        "run_mode": state.get("run_mode", ""),
        "retrieval_mode": state.get("retrieval_mode", ""),
        "evidence_sufficient": state.get("evidence_sufficient"),
        "evidence_quality_level": (state.get("evidence_quality") or {}).get("evidence_quality_level")
        if isinstance(state.get("evidence_quality"), dict)
        else None,
        "needs_human_review": decision.get("needs_human_review", False),
        "review_reasons": decision.get("review_reasons", []),
        "recommended_action": decision.get("recommended_action", ""),
        "trace_path": summary.get("trace_path", ""),
    }
    with jsonl_path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = resolve_project_path(human_review.get("review_export_csv", "data/eval/langgraph_human_review_exports_v3b.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "run_id",
        "question",
        "run_mode",
        "retrieval_mode",
        "evidence_sufficient",
        "evidence_quality_level",
        "needs_human_review",
        "review_reasons",
        "recommended_action",
        "manual_judgment",
        "manual_citation_judgment",
        "manual_should_refuse",
        "manual_notes",
        "reviewed_at",
    ]
    manual_fields = decision.get("manual_review_fields", {}) if isinstance(decision.get("manual_review_fields"), dict) else {}
    with csv_path.open("a", encoding="utf-8-sig" if not exists else "utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": state.get("run_id", ""),
                "question": state.get("question", ""),
                "run_mode": state.get("run_mode", ""),
                "retrieval_mode": state.get("retrieval_mode", ""),
                "evidence_sufficient": state.get("evidence_sufficient"),
                "evidence_quality_level": row["evidence_quality_level"],
                "needs_human_review": decision.get("needs_human_review", False),
                "review_reasons": ";".join(str(item) for item in decision.get("review_reasons", [])),
                "recommended_action": decision.get("recommended_action", ""),
                "manual_judgment": manual_fields.get("manual_judgment", ""),
                "manual_citation_judgment": manual_fields.get("manual_citation_judgment", ""),
                "manual_should_refuse": manual_fields.get("manual_should_refuse", ""),
                "manual_notes": manual_fields.get("manual_notes", ""),
                "reviewed_at": "",
            }
        )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in override.items():
        if key == "base_config":
            merged[key] = value
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
