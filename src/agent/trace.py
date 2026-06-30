from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def append_trace(
    state: dict[str, Any],
    *,
    node: str,
    status: str,
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    trace = list(state.get("trace") or [])
    trace.append(
        {
            "timestamp": now_iso(),
            "run_id": state.get("run_id", ""),
            "node": node,
            "status": status,
            "input_summary": make_json_safe(input_summary or {}),
            "output_summary": make_json_safe(output_summary or {}),
            "error": error,
        }
    )
    state["trace"] = trace
    if error:
        errors = list(state.get("errors") or [])
        errors.append(f"{node}: {error}")
        state["errors"] = errors
    return state


def write_graph_trace(record: dict[str, Any], path: str | Path) -> Path:
    output = resolve_project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(make_json_safe(record), ensure_ascii=False) + "\n")
    return output


def write_last_run(record: dict[str, Any], path: str | Path) -> Path:
    output = resolve_project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(make_json_safe(record), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def generate_run_id(prefix: str = "lg") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    clean_prefix = str(prefix or "lg").strip().replace(" ", "_")
    return f"{clean_prefix}_{timestamp}_{suffix}"


def write_trace_index(summary: dict[str, Any], index_path: str | Path, max_recent_runs: int = 50) -> Path:
    path = resolve_project_path(index_path)
    rows = load_trace_index(path)
    index_row = build_trace_index_row(summary)
    rows = [row for row in rows if row.get("run_id") != index_row.get("run_id")]
    rows.append(index_row)
    rows = rows[-max(1, int(max_recent_runs)) :]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_trace_index(index_path: str | Path) -> list[dict[str, Any]]:
    path = resolve_project_path(index_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    return []


def load_trace_by_run_id(run_id: str, trace_jsonl_path: str | Path) -> dict[str, Any] | None:
    path = resolve_project_path(trace_jsonl_path)
    if not path.exists():
        return None
    target = str(run_id)
    for line in reversed([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary = record.get("summary", {}) if isinstance(record, dict) else {}
        if str(summary.get("run_id", "")) == target:
            return record
    return None


def summarize_node_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(trace, start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "step": index,
                "node": item.get("node", ""),
                "status": item.get("status", ""),
                "input_summary": item.get("input_summary", {}),
                "output_summary": item.get("output_summary", {}),
                "error": item.get("error"),
            }
        )
    return rows


def summarize_graph_run(state: dict[str, Any]) -> dict[str, Any]:
    sources = state.get("sources") if isinstance(state.get("sources"), list) else []
    quality = state.get("evidence_quality") if isinstance(state.get("evidence_quality"), dict) else {}
    citation = state.get("citation_result") if isinstance(state.get("citation_result"), dict) else {}
    return {
        "run_id": state.get("run_id", ""),
        "created_at": state.get("created_at", ""),
        "completed_at": state.get("completed_at"),
        "graph_version": state.get("graph_version", ""),
        "timestamp": now_iso(),
        "question": state.get("question", ""),
        "run_mode": state.get("run_mode", ""),
        "retrieval_mode": state.get("retrieval_mode", ""),
        "enable_calibration": state.get("enable_calibration", False),
        "collection_name": state.get("collection_name", ""),
        "persist_directory": state.get("persist_directory", ""),
        "query_type": state.get("query_type"),
        "target_categories": state.get("target_categories", []),
        "category_intent": state.get("category_intent", {}),
        "sql_relation_debug": state.get("sql_relation_debug", {}),
        "fusion_debug": state.get("fusion_debug", {}),
        "retrieval_debug": state.get("retrieval_debug", {}),
        "evidence_sufficient": state.get("evidence_sufficient"),
        "evidence_gate_reason": state.get("evidence_gate_reason"),
        "insufficient_answer": state.get("insufficient_answer", False),
        "llm_called": state.get("llm_called", False),
        "answer_preview": summarize_text(str(state.get("answer") or ""), 320),
        "evidence_quality_level": quality.get("evidence_quality_level"),
        "evidence_quality_score": quality.get("evidence_quality_score"),
        "citation_check_passed": citation.get("citation_check_passed"),
        "needs_human_review": state.get("needs_human_review", False),
        "review_suggestions": state.get("review_suggestions", []),
        "review_decision": state.get("review_decision"),
        "top_sources": summarize_sources(sources),
        "trace": state.get("trace", []),
        "errors": state.get("errors", []),
    }


def build_trace_index_row(summary: dict[str, Any]) -> dict[str, Any]:
    top_sources = summary.get("top_sources") if isinstance(summary.get("top_sources"), list) else []
    first_source = top_sources[0] if top_sources and isinstance(top_sources[0], dict) else {}
    review_decision = summary.get("review_decision") if isinstance(summary.get("review_decision"), dict) else {}
    return {
        "run_id": summary.get("run_id", ""),
        "timestamp": summary.get("timestamp", ""),
        "created_at": summary.get("created_at", ""),
        "completed_at": summary.get("completed_at", ""),
        "graph_version": summary.get("graph_version", ""),
        "question": summarize_text(str(summary.get("question", "")), 300),
        "run_mode": summary.get("run_mode", ""),
        "retrieval_mode": summary.get("retrieval_mode", ""),
        "evidence_sufficient": summary.get("evidence_sufficient"),
        "llm_called": summary.get("llm_called", False),
        "evidence_quality_level": summary.get("evidence_quality_level"),
        "evidence_quality_score": summary.get("evidence_quality_score"),
        "needs_human_review": summary.get("needs_human_review", False),
        "review_reasons": review_decision.get("review_reasons", []),
        "recommended_action": review_decision.get("recommended_action", ""),
        "top_source": first_source.get("source", ""),
        "trace_path": summary.get("trace_path", ""),
        "last_run_path": summary.get("last_run_path", ""),
    }


def summarize_sources(sources: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources[:limit]:
        if not isinstance(source, dict):
            continue
        rows.append(
            {
                "rank": source.get("rank"),
                "source": source.get("source", ""),
                "imported_source": source.get("imported_source", ""),
                "category_dir": source.get("category_dir", ""),
                "doc_type": source.get("doc_type", ""),
                "chunk_id": source.get("chunk_id", ""),
                "final_score": source.get("final_score"),
                "calibration_score": source.get("calibration_score"),
                "retrieval_channels": source.get("retrieval_channels", []),
                "relation_score": source.get("relation_score", 0.0),
                "relation_score_normalized": source.get("relation_score_normalized", 0.0),
                "relation_path": source.get("relation_path", []),
                "relation_reasons": source.get("relation_reasons", []),
                "fusion_reasons": source.get("fusion_reasons", []),
                "category_intent_adjustment": source.get("category_intent_adjustment", 0.0),
                "high_frequency_entity_penalty": source.get("high_frequency_entity_penalty", 0.0),
                "sql_only_candidate": source.get("sql_only_candidate", False),
            }
        )
    return rows


def summarize_text(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def resolve_project_path(path: str | Path) -> Path:
    value = Path(str(path))
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(make_json_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
