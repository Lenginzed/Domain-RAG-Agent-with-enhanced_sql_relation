from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v4f.yaml"
INSUFFICIENT_EVIDENCE_ANSWER = (
    "当前知识库没有足够依据回答这个问题。已检索到一些相近资料，但它们不足以支持该结论。"
    " insufficient evidence."
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.nodes import build_human_review_decision  # noqa: E402
from src.evaluation.evidence_quality import score_answer_record  # noqa: E402
from src.generation.citation_checker import check_citations  # noqa: E402
from src.generation.evidence_gate import evaluate_evidence  # noqa: E402
from src.ui.rag_inspection_service import run_retrieval_inspection  # noqa: E402


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Run V4f small answer eval for enhanced_sql_relation.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--dry-run", action="store_true", help="Do retrieval/evidence diagnostics without Ollama calls.")
    args = parser.parse_args()

    result = run_answer_eval_v4f(Path(args.config), dry_run=args.dry_run)
    write_outputs(result, Path(args.config))
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_answer_eval_v4f(config_path: Path = CONFIG_PATH, *, dry_run: bool = False) -> dict[str, Any]:
    config = load_yaml(resolve_project_path(config_path))
    case_path = resolve_project_path(config["cases"]["output_jsonl"])
    if not case_path.exists():
        raise FileNotFoundError(f"Missing V4f case file: {case_path}. Run scripts/build_answer_eval_v4f_cases.py first.")
    cases = load_jsonl(case_path)
    modes = [str(mode) for mode in (config.get("answer_eval") or {}).get("modes", [])]
    if not modes:
        raise ValueError("answer_eval.modes is empty")

    llm_info = prepare_local_llm(config, dry_run=dry_run)
    records: list[dict[str, Any]] = []
    llm_call_errors: list[dict[str, str]] = []

    for case in cases:
        for mode in modes:
            record, error = run_case_mode(case, mode, config, llm_info, dry_run=dry_run)
            records.append(record)
            if error:
                llm_call_errors.append(error)

    metrics = build_metrics(records, modes)
    compare = build_compare(records, modes)
    return {
        "generated_at": now_iso(),
        "config_path": str(resolve_project_path(config_path)),
        "case_path": str(case_path),
        "collection_name": config["collection"]["collection_name"],
        "persist_directory": config["collection"]["persist_directory"],
        "external_cloud_api_called": False,
        "dry_run": dry_run,
        "answer_eval_status": determine_eval_status(records, llm_info, dry_run),
        "llm_availability": llm_info,
        "llm_call_errors": llm_call_errors,
        "metrics": metrics,
        "mode_compare_summary": compare,
        "records": records,
    }


def run_case_mode(
    case: dict[str, Any],
    mode: str,
    config: dict[str, Any],
    llm_info: dict[str, Any],
    *,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    retrieval = run_retrieval_inspection(
        question=str(case["question"]),
        retrieval_mode=mode,
        enable_calibration=True,
        final_top_k=8,
        config_path=resolve_project_path(config["ui_config"]),
        log_interaction=False,
    )
    final_sources = retrieval.get("final_sources", [])
    gate = evaluate_evidence(
        question=str(case["question"]),
        retrieval_result={"final_sources": final_sources, "debug": {"final_sources": final_sources}},
        eval_item=case,
        config=(config.get("evidence_gate") or {}),
    )
    expected_source_hit = source_hit(final_sources, case.get("expected_sources_contains", []))
    expected_category_hit = category_hit(final_sources, case.get("expected_categories", []))
    relation_path_found = any(source.get("relation_path") for source in final_sources if isinstance(source, dict))

    answer = ""
    llm_called = False
    answer_generated = False
    status = "retrieval_only_diagnostic" if dry_run else ""
    llm_model = ""
    llm_error: dict[str, str] | None = None

    if not bool(gate.get("evidence_sufficient", False)):
        answer = INSUFFICIENT_EVIDENCE_ANSWER
        status = "insufficient_evidence"
    elif dry_run:
        answer = ""
        status = "dry_run_retrieval_only"
    elif not llm_info.get("available"):
        answer = ""
        status = "skipped_llm_unavailable"
    else:
        llm_model = str(llm_info.get("selected_model", ""))
        try:
            answer = generate_with_ollama(
                build_answer_prompt(str(case["question"]), final_sources),
                model=llm_model,
                llm_config=config.get("local_llm") or {},
            )
            llm_called = True
            answer_generated = bool(answer.strip())
            status = "answered" if answer_generated else "llm_empty_answer"
        except Exception as exc:  # noqa: BLE001 - keep eval auditable and continue.
            status = "skipped_llm_error"
            llm_error = {"case_id": str(case["id"]), "retrieval_mode": mode, "error": str(exc)}

    if answer:
        citation = check_citations(answer=answer, sources=final_sources, config=config.get("citation_check") or {})
    else:
        citation = skipped_citation_result(status)

    record = build_record(
        case=case,
        mode=mode,
        retrieval=retrieval,
        gate=gate,
        answer=answer,
        llm_called=llm_called,
        llm_model=llm_model,
        answer_generated=answer_generated,
        status=status,
        citation=citation,
        expected_source_hit=expected_source_hit,
        expected_category_hit=expected_category_hit,
        relation_path_found=relation_path_found,
    )
    quality = score_answer_record(record, config.get("evidence_quality") or {})
    record["evidence_quality"] = quality
    record.update(
        {
            "evidence_quality_score": quality.get("evidence_quality_score", 0.0),
            "evidence_quality_level": quality.get("evidence_quality_level", ""),
            "evidence_quality_warnings": quality.get("evidence_quality_warnings", []),
        }
    )
    decision = build_human_review_decision(
        {
            "run_id": f"v4f::{case['id']}::{mode}",
            "question": case["question"],
            "run_mode": "full_rag" if llm_called else "retrieval_only_fallback",
            "retrieval_mode": mode,
            "evidence_sufficient": record["evidence_sufficient"],
            "insufficient_answer": record["insufficient_answer"],
            "citation_result": citation,
            "evidence_quality": quality,
            "sources": final_sources,
        }
    )
    record["review_decision"] = decision
    record["needs_human_review"] = bool(decision.get("needs_human_review", record.get("needs_human_review", False)))
    record["review_reasons"] = decision.get("review_reasons", [])
    record["recommended_action"] = decision.get("recommended_action", "")
    return record, llm_error


def build_record(
    *,
    case: dict[str, Any],
    mode: str,
    retrieval: dict[str, Any],
    gate: dict[str, Any],
    answer: str,
    llm_called: bool,
    llm_model: str,
    answer_generated: bool,
    status: str,
    citation: dict[str, Any],
    expected_source_hit: bool | None,
    expected_category_hit: bool | None,
    relation_path_found: bool,
) -> dict[str, Any]:
    final_sources = retrieval.get("final_sources", [])
    unsupported_count = len(citation.get("unsupported_claims", []))
    weak_count = int(citation.get("weak_claim_count", 0) or 0)
    supported_ratio = float(citation.get("supported_claim_ratio", 0.0) or 0.0)
    insufficient_answer = bool(citation.get("insufficient_answer", False) or status == "insufficient_evidence")
    needs_human_review = (
        unsupported_count > 0
        or weak_count > 0
        or (answer_generated and supported_ratio < 0.8)
        or not bool(gate.get("evidence_sufficient", False))
        or status not in {"answered", "insufficient_evidence"}
    )
    return {
        "id": str(case["id"]),
        "case_id": str(case["id"]),
        "question": str(case["question"]),
        "query_type": str(case.get("query_type", "")),
        "difficulty": str(case.get("difficulty", "")),
        "expected_focus": str(case.get("expected_focus", "")),
        "expected_behavior": str(case.get("expected_behavior", "")),
        "expected_sources_contains": case.get("expected_sources_contains", []),
        "expected_categories": case.get("expected_categories", []),
        "expected_answer_should_mention": case.get("expected_answer_should_mention", []),
        "retrieval_mode": mode,
        "answer_eval_status": status,
        "llm_called": llm_called,
        "llm_model": llm_model,
        "answer_generated": answer_generated,
        "evidence_sufficient": bool(gate.get("evidence_sufficient", False)),
        "evidence_gate_reason": gate.get("reason", ""),
        "evidence_gate_signals": gate.get("signals", {}),
        "answer": answer,
        "answer_preview": summarize_text(answer, 360),
        "insufficient_answer": insufficient_answer,
        "citation_check_passed": bool(citation.get("citation_check_passed", False)),
        "supported_claim_ratio": supported_ratio,
        "supported_claim_count": len(citation.get("supported_claims", [])),
        "checked_claim_count": int(citation.get("checked_claim_count", 0) or 0),
        "ignored_claim_count": int(citation.get("ignored_claim_count", 0) or 0),
        "unsupported_claim_count": unsupported_count,
        "weak_claim_count": weak_count,
        "claim_type_counts": citation.get("claim_type_counts", {}),
        "unsupported_claims": citation.get("unsupported_claims", []),
        "weak_claims": citation.get("weak_claims", []),
        "ignored_claims": citation.get("ignored_claims", []),
        "citation_check": citation,
        "needs_human_review": needs_human_review,
        "top_sources": summarize_sources(final_sources),
        "final_sources": final_sources,
        "source_diversity": retrieval.get("source_diversity", {}),
        "expected_source_hit": expected_source_hit,
        "expected_category_hit": expected_category_hit,
        "relation_path_found": relation_path_found,
        "category_intent": retrieval.get("category_intent", {}),
        "sql_relation_summary": retrieval.get("sql_relation_summary", {}),
        "fusion_debug": retrieval.get("fusion_debug", {}),
        "answer_should_mention_hit": answer_should_mention_hit(answer, case.get("expected_answer_should_mention", [])),
    }


def prepare_local_llm(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    llm_config = config.get("local_llm") or {}
    candidates = [str(llm_config.get("model", "")).strip()]
    candidates.extend(str(item).strip() for item in llm_config.get("fallback_models", []) or [])
    candidates = [item for item in candidates if item]
    if dry_run:
        return {
            "provider": str(llm_config.get("provider", "ollama")),
            "available": False,
            "selected_model": "",
            "candidate_models": candidates,
            "available_models": [],
            "reason": "dry_run_no_llm_check",
        }
    if not bool((config.get("answer_eval") or {}).get("allow_local_llm", True)):
        return {
            "provider": str(llm_config.get("provider", "ollama")),
            "available": False,
            "selected_model": "",
            "candidate_models": candidates,
            "available_models": [],
            "reason": "local_llm_disabled_by_config",
        }
    try:
        available_models = fetch_ollama_models(
            str(llm_config.get("base_url", "http://localhost:11434")),
            timeout_sec=min(10, int(llm_config.get("timeout_sec", 120))),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "provider": str(llm_config.get("provider", "ollama")),
            "available": False,
            "selected_model": "",
            "candidate_models": candidates,
            "available_models": [],
            "reason": f"ollama_unavailable: {exc}",
        }
    selected = next((model for model in candidates if model in available_models), "")
    return {
        "provider": str(llm_config.get("provider", "ollama")),
        "available": bool(selected),
        "selected_model": selected,
        "candidate_models": candidates,
        "available_models": available_models,
        "reason": "selected_model_available" if selected else "candidate_models_not_found_in_ollama_tags",
    }


def fetch_ollama_models(base_url: str, *, timeout_sec: int) -> list[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    with urllib.request.urlopen(url, timeout=timeout_sec) as response:  # noqa: S310 - local Ollama only.
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    names = []
    for item in parsed.get("models", []) or []:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return sorted(set(names))


def generate_with_ollama(prompt: str, *, model: str, llm_config: dict[str, Any]) -> str:
    base_url = str(llm_config.get("base_url", "http://localhost:11434"))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(llm_config.get("temperature", 0)),
            "num_predict": int(llm_config.get("num_predict", 512)),
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    retries = max(1, int(llm_config.get("max_retries", 1)))
    timeout_sec = int(llm_config.get("timeout_sec", 120))
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - local Ollama only.
                body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            return strip_thinking(str(parsed.get("response", "")).strip())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"local Ollama generation failed for model={model}: {last_error}")


def build_answer_prompt(question: str, sources: list[dict[str, Any]]) -> str:
    return f"""/no_think
你是 Domain-RAG Agent 的回答模块。只能根据下面的检索证据回答问题。
如果证据不足，必须明确说“当前知识库没有足够依据”。
回答事实、代码位置、配置名称、实验或报告结论时必须使用来源编号，例如 [S1]。
不要编造未出现在证据中的文件、指标、实验结果或代码行为。
请用中文回答，最多 5 条要点。

检索证据：
{format_sources_for_prompt(sources)}

用户问题：{question}

请输出：
1. 回答
2. 来源依据
"""


def format_sources_for_prompt(sources: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources[:8], start=1):
        content = str(source.get("content_preview") or source.get("page_content") or "").strip()
        if len(content) > 1200:
            content = f"{content[:1200]}\n...[truncated]"
        relation_path = source.get("relation_path", [])
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
                    f"relation_path: {relation_path}",
                    "content:",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "没有检索到证据。"


def skipped_citation_result(reason: str) -> dict[str, Any]:
    return {
        "insufficient_answer": reason == "insufficient_evidence",
        "supported_claims": [],
        "unsupported_claims": [],
        "ignored_claims": [],
        "weak_claims": [],
        "checked_claims": [],
        "supported_claim_ratio": 0.0,
        "checked_claim_count": 0,
        "ignored_claim_count": 0,
        "weak_claim_count": 0,
        "claim_type_counts": {},
        "citation_check_passed": False,
        "reason": f"citation check skipped: {reason}",
    }


def source_hit(sources: list[dict[str, Any]], expected_terms: list[Any]) -> bool | None:
    terms = [str(term).lower() for term in expected_terms if str(term).strip()]
    if not terms:
        return None
    blob = " ".join(
        str(source.get(key, ""))
        for source in sources
        for key in ["source", "imported_source", "original_source", "title", "section", "chunk_id"]
    ).lower()
    return any(term in blob for term in terms)


def category_hit(sources: list[dict[str, Any]], expected_categories: list[Any]) -> bool | None:
    categories = {str(item) for item in expected_categories if str(item).strip()}
    if not categories:
        return None
    final_categories = {str(source.get("category_dir", "")) for source in sources if source.get("category_dir")}
    return bool(categories.intersection(final_categories))


def answer_should_mention_hit(answer: str, expected_terms: list[Any]) -> bool | None:
    terms = [str(term).lower() for term in expected_terms if str(term).strip()]
    if not terms or not answer:
        return None if not terms else False
    lowered = answer.lower()
    return all(term in lowered for term in terms)


def build_metrics(records: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    non_negative = [r for r in records if r.get("expected_behavior") != "refuse_or_insufficient"]
    negative = [r for r in records if r.get("expected_behavior") == "refuse_or_insufficient"]
    ordinary = [r for r in records if r.get("answer_generated")]
    return {
        "total_cases": len({record["case_id"] for record in records}),
        "total_runs": len(records),
        "modes": modes,
        "llm_called_count": sum(1 for record in records if record["llm_called"]),
        "answer_generated_count": sum(1 for record in records if record["answer_generated"]),
        "expected_source_hit_rate": average_bool(record["expected_source_hit"] for record in records),
        "expected_category_hit_rate": average_bool(record["expected_category_hit"] for record in records),
        "citation_check_pass_rate": average_bool(record["citation_check_passed"] for record in ordinary),
        "avg_supported_claim_ratio": average(record["supported_claim_ratio"] for record in ordinary),
        "unsupported_claim_count_total": sum(int(record["unsupported_claim_count"]) for record in records),
        "weak_claim_count_total": sum(int(record["weak_claim_count"]) for record in records),
        "evidence_quality_avg": average(record["evidence_quality_score"] for record in records),
        "high_quality_count": sum(1 for record in records if record.get("evidence_quality_level") == "high"),
        "medium_or_lower_quality_count": sum(
            1 for record in records if record.get("evidence_quality_level") in {"medium", "low", "insufficient"}
        ),
        "needs_human_review_count": sum(1 for record in records if record["needs_human_review"]),
        "negative_refusal_pass_rate": average_bool(record["insufficient_answer"] and not record["llm_called"] for record in negative),
        "relation_path_found_rate": average_bool(record["relation_path_found"] for record in records),
        "non_negative_answered_rate": average_bool(record["answer_generated"] for record in non_negative),
        "metrics_by_mode": {mode: build_mode_metrics([r for r in records if r["retrieval_mode"] == mode]) for mode in modes},
    }


def build_mode_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary = [r for r in records if r.get("answer_generated")]
    negative = [r for r in records if r.get("expected_behavior") == "refuse_or_insufficient"]
    return {
        "runs": len(records),
        "llm_called_count": sum(1 for record in records if record["llm_called"]),
        "answer_generated_count": sum(1 for record in records if record["answer_generated"]),
        "expected_source_hit_rate": average_bool(record["expected_source_hit"] for record in records),
        "expected_category_hit_rate": average_bool(record["expected_category_hit"] for record in records),
        "citation_check_pass_rate": average_bool(record["citation_check_passed"] for record in ordinary),
        "avg_supported_claim_ratio": average(record["supported_claim_ratio"] for record in ordinary),
        "evidence_quality_avg": average(record["evidence_quality_score"] for record in records),
        "needs_human_review_count": sum(1 for record in records if record["needs_human_review"]),
        "negative_refusal_pass_rate": average_bool(record["insufficient_answer"] and not record["llm_called"] for record in negative),
        "relation_path_found_rate": average_bool(record["relation_path_found"] for record in records),
    }


def build_compare(records: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    if len(modes) < 2:
        return {}
    by_mode = {mode: build_mode_metrics([r for r in records if r["retrieval_mode"] == mode]) for mode in modes}
    left, right = modes[0], modes[1]
    return {
        "baseline_mode": left,
        "candidate_mode": right,
        "source_hit_delta": delta(by_mode[right]["expected_source_hit_rate"], by_mode[left]["expected_source_hit_rate"]),
        "category_hit_delta": delta(by_mode[right]["expected_category_hit_rate"], by_mode[left]["expected_category_hit_rate"]),
        "citation_pass_delta": delta(by_mode[right]["citation_check_pass_rate"], by_mode[left]["citation_check_pass_rate"]),
        "evidence_quality_delta": delta(by_mode[right]["evidence_quality_avg"], by_mode[left]["evidence_quality_avg"]),
        "human_review_delta": int(by_mode[right]["needs_human_review_count"]) - int(by_mode[left]["needs_human_review_count"]),
        "negative_refusal_delta": delta(by_mode[right]["negative_refusal_pass_rate"], by_mode[left]["negative_refusal_pass_rate"]),
        "relation_path_found_delta": delta(by_mode[right]["relation_path_found_rate"], by_mode[left]["relation_path_found_rate"]),
        "metrics_by_mode": by_mode,
    }


def determine_eval_status(records: list[dict[str, Any]], llm_info: dict[str, Any], dry_run: bool) -> str:
    if dry_run:
        return "dry_run_retrieval_only"
    if not llm_info.get("available"):
        return "skipped_llm_unavailable"
    if any(record.get("answer_eval_status") == "skipped_llm_error" for record in records):
        return "completed_with_llm_errors"
    return "completed"


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(resolve_project_path(config_path))
    outputs = config["outputs"]
    eval_json = resolve_project_path(outputs["eval_json"])
    eval_jsonl = resolve_project_path(outputs["eval_jsonl"])
    eval_csv = resolve_project_path(outputs["eval_csv"])
    compare_json = resolve_project_path(outputs["compare_json"])
    compare_csv = resolve_project_path(outputs["compare_csv"])
    report_md = resolve_project_path(outputs["report_md"])
    for path in [eval_json, eval_jsonl, eval_csv, compare_json, compare_csv, report_md]:
        path.parent.mkdir(parents=True, exist_ok=True)

    eval_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with eval_jsonl.open("w", encoding="utf-8", newline="\n") as file:
        for record in result["records"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_eval_csv(eval_csv, result["records"])
    compare_json.write_text(json.dumps(result["mode_compare_summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_compare_csv(compare_csv, result["mode_compare_summary"])
    report_md.write_text(build_markdown_report(result), encoding="utf-8")


def write_eval_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "question",
        "retrieval_mode",
        "answer_eval_status",
        "llm_called",
        "llm_model",
        "answer_generated",
        "evidence_sufficient",
        "insufficient_answer",
        "citation_check_passed",
        "supported_claim_ratio",
        "unsupported_claim_count",
        "weak_claim_count",
        "evidence_quality_score",
        "evidence_quality_level",
        "needs_human_review",
        "review_reasons",
        "expected_source_hit",
        "expected_category_hit",
        "relation_path_found",
        "category_intent",
        "top_sources",
        "answer_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key, "") for key in fieldnames},
                    "review_reasons": "|".join(str(item) for item in record.get("review_reasons", [])),
                    "category_intent": json.dumps(record.get("category_intent", {}), ensure_ascii=False),
                    "top_sources": " | ".join(
                        f"{source.get('rank')}:{source.get('category_dir')}:{source.get('source')}"
                        for source in record.get("top_sources", [])[:5]
                    ),
                }
            )


def write_compare_csv(path: Path, compare: dict[str, Any]) -> None:
    fieldnames = ["metric", "value"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in compare.items():
            if key == "metrics_by_mode":
                continue
            writer.writerow({"metric": key, "value": value})
        for mode, metrics in (compare.get("metrics_by_mode") or {}).items():
            for metric, value in metrics.items():
                writer.writerow({"metric": f"{mode}.{metric}", "value": value})


def build_markdown_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    compare = result["mode_compare_summary"]
    lines = [
        "# V4f Answer Eval Report",
        "",
        f"- status: `{result['answer_eval_status']}`",
        f"- dry_run: `{result['dry_run']}`",
        f"- external_cloud_api_called: `{result['external_cloud_api_called']}`",
        f"- llm_available: `{result['llm_availability'].get('available')}`",
        f"- selected_model: `{result['llm_availability'].get('selected_model', '')}`",
        f"- total_cases: `{metrics['total_cases']}`",
        f"- total_runs: `{metrics['total_runs']}`",
        f"- llm_called_count: `{metrics['llm_called_count']}`",
        f"- answer_generated_count: `{metrics['answer_generated_count']}`",
        f"- negative_refusal_pass_rate: `{metrics['negative_refusal_pass_rate']}`",
        f"- relation_path_found_rate: `{metrics['relation_path_found_rate']}`",
        "",
        "## Mode Compare",
        "",
        f"- baseline_mode: `{compare.get('baseline_mode', '')}`",
        f"- candidate_mode: `{compare.get('candidate_mode', '')}`",
        f"- source_hit_delta: `{compare.get('source_hit_delta', '')}`",
        f"- category_hit_delta: `{compare.get('category_hit_delta', '')}`",
        f"- citation_pass_delta: `{compare.get('citation_pass_delta', '')}`",
        f"- evidence_quality_delta: `{compare.get('evidence_quality_delta', '')}`",
        f"- human_review_delta: `{compare.get('human_review_delta', '')}`",
        "",
        "## Per Run",
        "",
        "| case | mode | status | llm | evidence | quality | review | expected source | expected category | relation path | top source |",
        "|---|---|---|---:|---:|---|---:|---|---|---:|---|",
    ]
    for record in result["records"]:
        top = record.get("top_sources", [{}])[0] if record.get("top_sources") else {}
        lines.append(
            "| {case} | {mode} | {status} | {llm} | {evidence} | {quality} | {review} | {source_hit} | {cat_hit} | {path} | {top_source} |".format(
                case=record["case_id"],
                mode=record["retrieval_mode"],
                status=record["answer_eval_status"],
                llm=record["llm_called"],
                evidence=record["evidence_sufficient"],
                quality=record.get("evidence_quality_level", ""),
                review=record["needs_human_review"],
                source_hit=record["expected_source_hit"],
                cat_hit=record["expected_category_hit"],
                path=record["relation_path_found"],
                top_source=top.get("source", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def print_summary(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print("V4f answer eval complete")
    print(f"answer_eval_status: {result['answer_eval_status']}")
    print(f"llm_available: {result['llm_availability'].get('available')}")
    print(f"selected_model: {result['llm_availability'].get('selected_model', '')}")
    print(f"total_runs: {metrics['total_runs']}")
    print(f"llm_called_count: {metrics['llm_called_count']}")
    print(f"answer_generated_count: {metrics['answer_generated_count']}")
    print(f"negative_refusal_pass_rate: {metrics['negative_refusal_pass_rate']}")
    print(f"needs_human_review_count: {metrics['needs_human_review_count']}")


def summarize_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources[:8]:
        rows.append(
            {
                "rank": source.get("rank"),
                "source": source.get("source", ""),
                "imported_source": source.get("imported_source", ""),
                "category_dir": source.get("category_dir", ""),
                "doc_type": source.get("doc_type", ""),
                "chunk_id": source.get("chunk_id", ""),
                "final_score": source.get("final_score", 0.0),
                "retrieval_channels": source.get("retrieval_channels", []),
                "relation_score": source.get("relation_score", 0.0),
                "relation_score_normalized": source.get("relation_score_normalized", 0.0),
                "relation_path": source.get("relation_path", []),
                "fusion_reasons": source.get("fusion_reasons", []),
            }
        )
    return rows


def strip_thinking(text: str) -> str:
    import re

    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def summarize_text(text: str, limit: int = 360) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def average_bool(values: Any) -> float:
    items = [value for value in values if value is not None]
    if not items:
        return 0.0
    return round(sum(1.0 if bool(value) else 0.0 for value in items) / len(items), 6)


def delta(right: float, left: float) -> float:
    return round(float(right) - float(left), 6)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
