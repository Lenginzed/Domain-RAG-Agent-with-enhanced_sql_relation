from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document

from src.evaluation.evidence_quality import score_answer_record
from src.generation.citation_checker import check_citations
from src.generation.evidence_gate import evaluate_evidence
from src.generation.llm import OllamaLLM
from src.indexes.vector_index import collection_count, load_index
from src.retrieval.real_smoke_retriever import retrieve_real_smoke
from src.sql_sag.ui_adapter import run_sql_sag_retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "ui_v25f.yaml"

INSUFFICIENT_EVIDENCE_ANSWER = (
    "当前知识库没有足够依据回答这个问题。已检索到一些相近资料，但它们不足以支持该结论。"
)


def load_ui_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid UI config: {path}")
    base_config = config.get("base_config")
    if base_config:
        base = load_ui_config(resolve_project_path(base_config))
        config = deep_merge(base, {key: value for key, value in config.items() if key != "base_config"})
    config["_config_path"] = str(path)
    return config


def get_collection_info(config: dict[str, Any]) -> dict[str, Any]:
    collection = config["collection"]
    persist_directory = resolve_project_path(collection["persist_directory"])
    collection_name = str(collection["collection_name"])
    count: int | None = None
    error = ""
    try:
        vector_store = load_index(persist_directory=persist_directory, collection_name=collection_name)
        count = collection_count(vector_store)
    except Exception as exc:  # noqa: BLE001 - UI should show read failures rather than crash.
        error = str(exc)
    return {
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "chunk_count": count,
        "error": error,
    }


def load_demo_cases(config: dict[str, Any]) -> dict[str, Any]:
    demo_config = config.get("demo_cases", {}) or {}
    sources: dict[str, Any] = {}
    all_cases_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(demo_config, list):
        cases = [dict(item) for item in demo_config if isinstance(item, dict)]
        for case in cases:
            case_id = str(case.get("id", ""))
            if case_id:
                all_cases_by_id[case_id] = case
        return {
            "sources": {
                "v4e_demo_cases": {
                    "path": "config/ui_v4e.yaml::demo_cases",
                    "exists": True,
                    "warning": "",
                    "cases": cases,
                }
            },
            "highlighted_cases": [
                {"id": case.get("id", ""), "label": str(case.get("expected_focus", case.get("id", ""))), "case": case, "found": True}
                for case in cases
            ],
            "all_cases_by_id": all_cases_by_id,
        }
    for name, value in demo_config.items():
        if name == "highlighted_cases":
            continue
        path = resolve_project_path(value)
        cases: list[dict[str, Any]] = []
        warning = ""
        if path.exists():
            try:
                cases = load_jsonl(path)
            except Exception as exc:  # noqa: BLE001 - UI should show malformed files gently.
                warning = str(exc)
        else:
            warning = f"Missing demo case file: {path}"
        for case in cases:
            case_id = str(case.get("id", ""))
            if case_id and case_id not in all_cases_by_id:
                all_cases_by_id[case_id] = case
        sources[name] = {
            "path": str(path),
            "exists": path.exists(),
            "warning": warning,
            "cases": cases,
        }

    highlighted = []
    for item in demo_config.get("highlighted_cases", []) or []:
        case_id = str(item.get("id", ""))
        highlighted.append(
            {
                "id": case_id,
                "label": str(item.get("label", case_id)),
                "case": all_cases_by_id.get(case_id),
                "found": case_id in all_cases_by_id,
            }
        )
    return {
        "sources": sources,
        "highlighted_cases": highlighted,
        "all_cases_by_id": all_cases_by_id,
    }


def load_interaction_logs(path: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    log_path = resolve_project_path(path)
    if not log_path.exists():
        return []
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)) :]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"timestamp": "", "question": "", "parse_error": line[:200]})
    return rows


def summarize_interaction_record(record: dict[str, Any]) -> dict[str, Any]:
    quality = record.get("evidence_quality") if isinstance(record.get("evidence_quality"), dict) else {}
    gate = record.get("evidence_gate") if isinstance(record.get("evidence_gate"), dict) else {}
    sources = record.get("final_sources") if isinstance(record.get("final_sources"), list) else []
    if not sources and isinstance(record.get("top_sources"), list):
        sources = record.get("top_sources", [])
    top_source = ""
    if sources:
        first = sources[0] if isinstance(sources[0], dict) else {}
        top_source = str(first.get("imported_source") or first.get("source") or "")
    return {
        "timestamp": record.get("timestamp", ""),
        "question": record.get("question", ""),
        "run_mode": record.get("run_mode", ""),
        "retrieval_mode": record.get("retrieval_mode", ""),
        "enable_calibration": record.get("enable_calibration", record.get("calibration_enabled", "")),
        "llm_called": record.get("llm_called", False),
        "evidence_sufficient": record.get("evidence_sufficient", gate.get("evidence_sufficient", "")),
        "insufficient_answer": record.get("insufficient_answer", False),
        "evidence_quality_level": record.get("evidence_quality_level", quality.get("evidence_quality_level", "")),
        "evidence_quality_score": record.get("evidence_quality_score", quality.get("evidence_quality_score", "")),
        "top_source": top_source,
        "needs_human_review": record.get("needs_human_review", False),
    }


def load_manual_review_exports(path: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    csv_path = resolve_project_path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[-max(1, int(limit)) :]


def run_retrieval_inspection(
    *,
    question: str,
    retrieval_mode: str | None = None,
    enable_calibration: bool | None = None,
    final_top_k: int | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    log_interaction: bool = True,
) -> dict[str, Any]:
    config = load_ui_config(config_path)
    retrieval_config = dict(config.get("default_retrieval", {}))
    mode = retrieval_mode or str(retrieval_config.get("mode", "enhanced_keyword"))
    if final_top_k is not None:
        retrieval_config["final_top_k"] = int(final_top_k)
    calibration_enabled = (
        bool(retrieval_config.get("enable_calibration", True))
        if enable_calibration is None
        else bool(enable_calibration)
    )
    collection = config["collection"]
    persist_directory = resolve_project_path(collection["persist_directory"])
    collection_name = str(collection["collection_name"])
    calibration_config = load_calibration_config(retrieval_config, mode) if calibration_enabled else None

    if mode in {"sql_relation", "enhanced_sql_relation"}:
        result = run_sql_sag_retrieval(
            question=question,
            retrieval_mode=mode,
            config_path=config.get("enhanced_sql_relation", {}).get("config_path", "config/sql_sag_v4d.yaml"),
            final_top_k=int(retrieval_config.get("final_top_k", 8)),
        )
        evidence_gate = evaluate_evidence(
            question=question,
            retrieval_result={"debug": {"final_sources": result.get("final_sources", [])}, "documents": []},
            eval_item=None,
            config=config.get("evidence_gate") or {},
        )
        inspection = {
            "timestamp": now_iso(),
            "question": question,
            "run_mode": "retrieval_only",
            "retrieval_mode": mode,
            "collection_name": collection_name,
            "persist_directory": str(persist_directory),
            "calibration_enabled": calibration_enabled,
            "llm_called": False,
            "external_cloud_api_called": False,
            "query_type": "",
            "target_categories": [],
            "expanded_query": "",
            "category_intent": result.get("category_intent", {}),
            "evidence_gate": evidence_gate,
            "final_sources": result.get("final_sources", []),
            "source_diversity": summarize_source_diversity(result.get("final_sources", [])),
            "debug": result,
            "sql_relation_summary": result.get("sql_relation_debug", {}),
            "fusion_debug": result.get("fusion_debug", {}),
        }
        if log_interaction:
            append_interaction_log(inspection, config)
        return inspection

    retrieval = retrieve_real_smoke(
        question=question,
        persist_directory=persist_directory,
        collection_name=collection_name,
        retrieval_mode=mode,
        calibration_config=calibration_config,
        **build_retrieval_kwargs(retrieval_config, mode),
    )
    evidence_gate = evaluate_evidence(
        question=question,
        retrieval_result=retrieval,
        eval_item=None,
        config=config.get("evidence_gate") or {},
    )
    result = {
        "timestamp": now_iso(),
        "question": question,
        "run_mode": "retrieval_only",
        "retrieval_mode": mode,
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "calibration_enabled": bool(calibration_config),
        "llm_called": False,
        "external_cloud_api_called": False,
        "query_type": retrieval.debug.get("query_type"),
        "target_categories": retrieval.debug.get("target_categories", []),
        "expanded_query": retrieval.debug.get("expanded_query", ""),
        "evidence_gate": evidence_gate,
        "final_sources": attach_source_previews(retrieval.debug.get("final_sources", []), retrieval.documents),
        "source_diversity": retrieval.debug.get("source_diversity", {}),
        "debug": retrieval.debug,
    }
    if log_interaction:
        append_interaction_log(result, config)
    return result


def run_full_rag_inspection(
    *,
    question: str,
    retrieval_mode: str | None = None,
    enable_calibration: bool | None = None,
    final_top_k: int | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_ui_config(config_path)
    retrieval_result = run_retrieval_inspection(
        question=question,
        retrieval_mode=retrieval_mode,
        enable_calibration=enable_calibration,
        final_top_k=final_top_k,
        config_path=config_path,
        log_interaction=False,
    )
    documents = documents_from_debug_sources(retrieval_result)
    evidence_gate = retrieval_result["evidence_gate"]
    llm_called = False
    if evidence_gate.get("evidence_sufficient"):
        llm_called = True
        answer = OllamaLLM().generate(build_answer_prompt(question, documents))
    else:
        answer = INSUFFICIENT_EVIDENCE_ANSWER

    citation = check_citations(
        answer=answer,
        sources=documents,
        config=config.get("citation_check") or {},
    )
    answer_record = build_answer_record(
        question=question,
        retrieval_result=retrieval_result,
        answer=answer,
        llm_called=llm_called,
        citation=citation,
    )
    quality = score_answer_record(answer_record, config.get("evidence_quality") or {})
    result = {
        **retrieval_result,
        "timestamp": now_iso(),
        "run_mode": "full_rag_answer",
        "llm_called": llm_called,
        "local_llm_call": llm_called,
        "answer": answer,
        "insufficient_answer": bool(citation.get("insufficient_answer", False)),
        "citation_check": citation,
        "evidence_quality": quality,
        "answer_record": answer_record,
    }
    append_interaction_log(result, config)
    return result


def append_manual_review_export(record: dict[str, Any], output_csv: str | Path) -> Path:
    path = resolve_project_path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exported_at",
        "question",
        "run_mode",
        "retrieval_mode",
        "collection_name",
        "calibration_enabled",
        "llm_called",
        "evidence_sufficient",
        "evidence_gate_reason",
        "answer",
        "evidence_quality_score",
        "evidence_quality_level",
        "supported_claim_ratio",
        "unsupported_claim_count",
        "weak_claim_count",
        "top_sources",
        "manual_judgment",
        "manual_citation_judgment",
        "manual_should_refuse",
        "manual_notes",
        "reviewer",
        "reviewed_at",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig" if not exists else "utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: format_review_field(record, field) for field in fieldnames})
    return path


def build_retrieval_kwargs(config: dict[str, Any], mode: str) -> dict[str, Any]:
    values = {
        "dense_top_k": int(config.get("dense_top_k", 5)),
        "metadata_top_k": int(config.get("metadata_top_k", 8)),
        "keyword_top_k": int(config.get("keyword_top_k", 8)),
        "final_top_k": int(config.get("final_top_k", 8)),
        "max_chunks_per_source": int(config.get("max_chunks_per_source", 2)),
        "enable_source_diversity": bool(config.get("enable_source_diversity", True)),
        "source_key": str(config.get("source_key", "imported_source")),
    }
    if mode == "dense":
        values["final_top_k"] = min(values["final_top_k"], values["dense_top_k"])
    return values


def load_calibration_config(retrieval_config: dict[str, Any], mode: str) -> dict[str, Any] | None:
    calibration_path = retrieval_config.get("calibration_config")
    if not calibration_path:
        return None
    data = load_yaml(resolve_project_path(calibration_path))
    calibration = dict(data.get("calibration", {}))
    apply_to_modes = {str(item) for item in calibration.get("apply_to_modes", [])}
    if apply_to_modes and mode not in apply_to_modes:
        return None
    return calibration if calibration.get("enabled", True) else None


def build_answer_prompt(question: str, documents: list[Document]) -> str:
    evidence = format_evidence(documents)
    return f"""/no_think
你是 Domain-RAG Agent 的回答模块。只能根据下面的检索资料回答问题。
如果资料不足，必须明确说明“当前知识库没有足够依据”。
回答事实、代码位置、配置含义或实验结论时必须给来源编号，例如 [S1]。
不要编造未出现在资料中的文件、指标、实验结果或代码行为。
回答请保持简洁，最多 5 条要点。

检索资料：
{evidence}

用户问题：{question}

请输出：
1. 回答
2. 来源依据
"""


def format_evidence(documents: list[Document]) -> str:
    if not documents:
        return "没有检索到资料。"
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = dict(document.metadata or {})
        content = str(document.page_content or "").strip()
        if len(content) > 1400:
            content = f"{content[:1400]}\n...[truncated]"
        blocks.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"source: {metadata.get('source', '')}",
                    f"category_dir: {metadata.get('category_dir', '')}",
                    f"doc_type: {metadata.get('doc_type', '')}",
                    f"chunk_id: {metadata.get('chunk_id', '')}",
                    f"imported_source: {metadata.get('imported_source', '')}",
                    f"section: {metadata.get('section', '')}",
                    f"retrieval_channels: {metadata.get('retrieval_channels', [])}",
                    f"matched_terms: {metadata.get('matched_terms', [])}",
                    f"calibration_reasons: {metadata.get('calibration_reasons', [])}",
                    "content:",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks)


def attach_source_previews(sources: list[dict[str, Any]], documents: list[Document]) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for source, document in zip(sources, documents):
        row = dict(source)
        row["content_preview"] = summarize_text(str(document.page_content or ""), 900)
        previews.append(row)
    return previews


def documents_from_debug_sources(result: dict[str, Any]) -> list[Document]:
    documents: list[Document] = []
    for source in result.get("final_sources", []):
        content = str(source.get("content_preview", ""))
        metadata = {key: value for key, value in source.items() if key != "content_preview"}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_answer_record(
    *,
    question: str,
    retrieval_result: dict[str, Any],
    answer: str,
    llm_called: bool,
    citation: dict[str, Any],
) -> dict[str, Any]:
    gate = retrieval_result.get("evidence_gate", {})
    unsupported_count = len(citation.get("unsupported_claims", []))
    weak_count = int(citation.get("weak_claim_count", 0))
    supported_ratio = float(citation.get("supported_claim_ratio", 0.0))
    insufficient_answer = bool(citation.get("insufficient_answer", False))
    needs_human_review = (
        unsupported_count > 0
        or weak_count > 0
        or (not insufficient_answer and supported_ratio < 0.8)
        or (not gate.get("evidence_sufficient", True))
    )
    return {
        "id": f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "question": question,
        "query_type": retrieval_result.get("query_type", ""),
        "retrieval_mode": retrieval_result.get("retrieval_mode", ""),
        "evidence_sufficient": bool(gate.get("evidence_sufficient", False)),
        "evidence_gate_reason": gate.get("reason", ""),
        "evidence_gate_signals": gate.get("signals", {}),
        "llm_called": llm_called,
        "answer": answer,
        "insufficient_answer": insufficient_answer,
        "supported_claim_ratio": supported_ratio,
        "citation_check_passed": bool(citation.get("citation_check_passed", False)),
        "checked_claim_count": int(citation.get("checked_claim_count", 0)),
        "ignored_claim_count": int(citation.get("ignored_claim_count", 0)),
        "weak_claim_count": weak_count,
        "unsupported_claim_count": unsupported_count,
        "claim_type_counts": citation.get("claim_type_counts", {}),
        "needs_human_review": needs_human_review,
        "unsupported_claims": citation.get("unsupported_claims", []),
        "ignored_claims": citation.get("ignored_claims", []),
        "weak_claims": citation.get("weak_claims", []),
        "citation_check": citation,
        "source_diversity": retrieval_result.get("source_diversity", {}),
        "final_sources": retrieval_result.get("final_sources", []),
    }


def append_interaction_log(result: dict[str, Any], config: dict[str, Any]) -> None:
    output = resolve_project_path(config["outputs"]["interaction_log_jsonl"])
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = make_json_safe(build_interaction_log_record(result))
    with output.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def build_interaction_log_record(result: dict[str, Any]) -> dict[str, Any]:
    quality = result.get("evidence_quality") if isinstance(result.get("evidence_quality"), dict) else {}
    citation = result.get("citation_check") if isinstance(result.get("citation_check"), dict) else {}
    gate = result.get("evidence_gate") if isinstance(result.get("evidence_gate"), dict) else {}
    sources = result.get("final_sources") if isinstance(result.get("final_sources"), list) else []
    summary = {
        "timestamp": result.get("timestamp", now_iso()),
        "question": result.get("question", ""),
        "run_mode": result.get("run_mode", ""),
        "retrieval_mode": result.get("retrieval_mode", ""),
        "enable_calibration": result.get("calibration_enabled", False),
        "calibration_enabled": result.get("calibration_enabled", False),
        "llm_called": result.get("llm_called", False),
        "evidence_sufficient": gate.get("evidence_sufficient", ""),
        "insufficient_answer": result.get("insufficient_answer", False),
        "evidence_quality_level": quality.get("evidence_quality_level"),
        "evidence_quality_score": quality.get("evidence_quality_score"),
        "supported_claim_ratio": citation.get("supported_claim_ratio"),
        "top_sources": summarize_top_sources_for_log(sources),
        "needs_human_review": (result.get("answer_record") or {}).get("needs_human_review", False),
    }
    return {**summary, "detail": result}


def summarize_top_sources_for_log(sources: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for source in sources[:5]:
        if not isinstance(source, dict):
            continue
        rows.append(
            {
                "rank": source.get("rank"),
                "source": source.get("source", ""),
                "category_dir": source.get("category_dir", ""),
                "doc_type": source.get("doc_type", ""),
                "final_score": source.get("final_score"),
                "calibration_score": source.get("calibration_score"),
            }
        )
    return rows


def format_review_field(record: dict[str, Any], field: str) -> Any:
    if field == "exported_at":
        return now_iso()
    if field == "evidence_sufficient":
        return (record.get("evidence_gate") or {}).get("evidence_sufficient", "")
    if field == "evidence_gate_reason":
        return (record.get("evidence_gate") or {}).get("reason", "")
    if field == "evidence_quality_score":
        return (record.get("evidence_quality") or {}).get("evidence_quality_score", "")
    if field == "evidence_quality_level":
        return (record.get("evidence_quality") or {}).get("evidence_quality_level", "")
    if field == "supported_claim_ratio":
        return (record.get("citation_check") or {}).get("supported_claim_ratio", "")
    if field == "unsupported_claim_count":
        return len((record.get("citation_check") or {}).get("unsupported_claims", []))
    if field == "weak_claim_count":
        return int((record.get("citation_check") or {}).get("weak_claim_count", 0))
    if field == "top_sources":
        return " | ".join(
            f"{source.get('rank', '')}:{source.get('category_dir', '')}:{source.get('source', '')}"
            for source in record.get("final_sources", [])[:5]
        )
    return record.get(field, "")


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def summarize_source_diversity(sources: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for source in sources:
        key = str(source.get("imported_source") or source.get("source") or "")
        counts[key] = counts.get(key, 0) + 1
    return {
        "enabled": True,
        "source_key": "imported_source",
        "max_chunks_per_source": 2,
        "counts": counts,
        "violations": {key: count for key, count in counts.items() if count > 2},
        "relaxed_sources": [],
        "relaxed": False,
    }


def summarize_text(text: str, limit: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: value for key, value in base.items() if not key.startswith("_")}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
