from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v4g.yaml"
V4F_RESULT_PATH = PROJECT_ROOT / "storage" / "logs" / "answer_eval_v4f.json"
INSUFFICIENT_EVIDENCE_ANSWER = (
    "当前知识库没有足够依据回答这个问题。已检索到一些相近资料，但它们不足以支持该结论。"
    " insufficient evidence."
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_answer_eval_v4f import (  # noqa: E402
    answer_should_mention_hit,
    average,
    average_bool,
    build_answer_prompt,
    build_markdown_report,
    category_hit,
    delta,
    prepare_local_llm,
    skipped_citation_result,
    source_hit,
    summarize_sources,
    summarize_text,
)
from src.agent.nodes import build_human_review_decision  # noqa: E402
from src.answering.llm_reliability import call_local_llm_with_reliability  # noqa: E402
from src.evaluation.evidence_quality import score_answer_record  # noqa: E402
from src.generation.citation_checker import check_citations  # noqa: E402
from src.generation.evidence_gate import evaluate_evidence  # noqa: E402
from src.ui.rag_inspection_service import run_retrieval_inspection  # noqa: E402


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Run V4g answer generation reliability eval.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--dry-run", action="store_true", help="Do retrieval/evidence diagnostics without Ollama calls.")
    args = parser.parse_args()
    result = run_answer_eval_v4g(Path(args.config), dry_run=args.dry_run)
    write_outputs(result, Path(args.config))
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_answer_eval_v4g(config_path: Path = CONFIG_PATH, *, dry_run: bool = False) -> dict[str, Any]:
    config = load_merged_config(resolve_project_path(config_path))
    case_path = resolve_project_path(config["cases"]["output_jsonl"])
    if not case_path.exists():
        raise FileNotFoundError(f"Missing V4f/V4g case file: {case_path}")
    cases = load_jsonl(case_path)[: int((config.get("answer_eval") or {}).get("max_cases", 6))]
    modes = [str(mode) for mode in (config.get("answer_eval") or {}).get("modes", [])]
    if not modes:
        raise ValueError("answer_eval.modes is empty")

    llm_info = prepare_local_llm(config, dry_run=dry_run)
    records: list[dict[str, Any]] = []
    for case in cases:
        for mode in modes:
            records.append(run_case_mode_v4g(case, mode, config, llm_info, dry_run=dry_run))

    metrics = build_metrics(records, modes)
    mode_compare = build_mode_compare(records, modes)
    v4f_compare = build_v4f_v4g_compare(metrics, V4F_RESULT_PATH)
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
        "metrics": metrics,
        "mode_compare_summary": mode_compare,
        "v4f_v4g_compare": v4f_compare,
        "records": records,
    }


def run_case_mode_v4g(
    case: dict[str, Any],
    mode: str,
    config: dict[str, Any],
    llm_info: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
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
    prompt = build_answer_prompt(str(case["question"]), final_sources)

    answer = ""
    reliability: dict[str, Any]
    status = ""
    failure_type = ""
    if not bool(gate.get("evidence_sufficient", False)):
        answer = INSUFFICIENT_EVIDENCE_ANSWER
        status = "insufficient_evidence"
        failure_type = "retrieval_insufficient"
        reliability = not_called_reliability(case, mode, failure_type)
    elif dry_run:
        status = "dry_run_retrieval_only"
        failure_type = "llm_unavailable"
        reliability = not_called_reliability(case, mode, failure_type)
    elif not llm_info.get("available"):
        status = "skipped_llm_unavailable"
        failure_type = "llm_unavailable"
        reliability = not_called_reliability(case, mode, failure_type)
    else:
        reliability_config = build_reliability_config(config, llm_info)
        reliability = call_local_llm_with_reliability(
            prompt=prompt,
            config=reliability_config,
            case_id=str(case["id"]),
            retrieval_mode=mode,
        )
        answer = str(reliability.get("answer_text", ""))
        status = str(reliability.get("status") or "")
        failure_type = str(reliability.get("failure_type") or "")

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
        status=status,
        failure_type=failure_type,
        citation=citation,
        reliability=reliability,
        expected_source_hit=expected_source_hit,
        expected_category_hit=expected_category_hit,
        relation_path_found=relation_path_found,
        prompt=prompt,
        config=config,
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
    failure_types = classify_failures(record, citation, quality)
    if failure_types and not record.get("failure_type"):
        record["failure_type"] = failure_types[0]
    record["failure_types"] = failure_types
    decision = build_human_review_decision(
        {
            "run_id": f"v4g::{case['id']}::{mode}",
            "question": case["question"],
            "run_mode": "full_rag" if record["llm_called"] else "retrieval_only_fallback",
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
    return record


def build_record(
    *,
    case: dict[str, Any],
    mode: str,
    retrieval: dict[str, Any],
    gate: dict[str, Any],
    answer: str,
    status: str,
    failure_type: str,
    citation: dict[str, Any],
    reliability: dict[str, Any],
    expected_source_hit: bool | None,
    expected_category_hit: bool | None,
    relation_path_found: bool,
    prompt: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    final_sources = retrieval.get("final_sources", [])
    unsupported_count = len(citation.get("unsupported_claims", []))
    weak_count = int(citation.get("weak_claim_count", 0) or 0)
    supported_ratio = float(citation.get("supported_claim_ratio", 0.0) or 0.0)
    answer_generated = bool(reliability.get("answer_generated", False))
    insufficient_answer = bool(citation.get("insufficient_answer", False) or status == "insufficient_evidence")
    needs_human_review = (
        unsupported_count > 0
        or weak_count > 0
        or (answer_generated and supported_ratio < 0.8)
        or not bool(gate.get("evidence_sufficient", False))
        or status not in {"answered", "answered_after_retry", "insufficient_evidence"}
    )
    logging_config = config.get("logging") or {}
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
        "failure_type": failure_type,
        "llm_called": bool(reliability.get("llm_called", False)),
        "llm_model": str(reliability.get("llm_model", "")),
        "answer_generated": answer_generated,
        "attempt_count": int(reliability.get("attempt_count", 0) or 0),
        "retry_count": int(reliability.get("retry_count", 0) or 0),
        "llm_attempts": reliability.get("attempts", []),
        "raw_response_preview": reliability.get("raw_response_preview", ""),
        "prompt_preview": summarize_text(prompt, int(logging_config.get("max_prompt_preview_chars", 1200)))
        if bool(logging_config.get("save_prompt_preview", True))
        else "",
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
        "negative_refusal_pass": bool(case.get("expected_behavior") == "refuse_or_insufficient" and insufficient_answer and not reliability.get("llm_called", False)),
    }


def build_reliability_config(config: dict[str, Any], llm_info: dict[str, Any]) -> dict[str, Any]:
    reliability_config = dict(config.get("local_llm") or {})
    if llm_info.get("selected_model"):
        reliability_config["model"] = llm_info["selected_model"]
    logging_config = config.get("logging") or {}
    reliability_config.update(logging_config)
    return reliability_config


def classify_failures(record: dict[str, Any], citation: dict[str, Any], quality: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if not bool(record.get("evidence_sufficient", False)):
        labels.append("retrieval_insufficient")
    status = str(record.get("answer_eval_status", ""))
    failure_type = str(record.get("failure_type", ""))
    if failure_type:
        labels.append(failure_type)
    elif status in {"skipped_llm_unavailable", "dry_run_retrieval_only"}:
        labels.append("llm_unavailable")
    if status == "llm_timeout":
        labels.append("llm_timeout")
    if status == "llm_empty_answer":
        labels.append("llm_empty_answer")
    if status == "llm_exception":
        labels.append("llm_exception")
    if record.get("answer_generated") and not bool(citation.get("citation_check_passed", False)):
        labels.append("citation_failure")
    if int(record.get("unsupported_claim_count", 0) or 0) > 0:
        labels.append("unsupported_claims_present")
    if int(record.get("weak_claim_count", 0) or 0) > 0:
        labels.append("weak_claims_present")
    if str(quality.get("evidence_quality_level", "")) in {"medium", "low", "insufficient"}:
        labels.append("medium_or_lower_quality")
    return unique_values(labels)


def not_called_reliability(case: dict[str, Any], mode: str, failure_type: str) -> dict[str, Any]:
    return {
        "case_id": str(case["id"]),
        "retrieval_mode": mode,
        "llm_called": False,
        "llm_model": "",
        "answer_text": "",
        "answer_generated": False,
        "status": failure_type,
        "attempts": [],
        "attempt_count": 0,
        "retry_count": 0,
        "failure_type": failure_type,
        "raw_response_preview": "",
        "raw_response_saved": False,
    }


def build_metrics(records: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    ordinary = [record for record in records if record.get("answer_generated")]
    negative = [record for record in records if record.get("expected_behavior") == "refuse_or_insufficient"]
    failure_counts = Counter(label for record in records for label in record.get("failure_types", []))
    return {
        "total_cases": len({record["case_id"] for record in records}),
        "total_runs": len(records),
        "modes": modes,
        "llm_called_count": sum(1 for record in records if record["llm_called"]),
        "retry_count_total": sum(int(record.get("retry_count", 0) or 0) for record in records),
        "attempt_count_total": sum(int(record.get("attempt_count", 0) or 0) for record in records),
        "answer_generated_count": sum(1 for record in records if record["answer_generated"]),
        "empty_answer_count": sum(1 for record in records if record["answer_eval_status"] == "llm_empty_answer"),
        "expected_source_hit_rate": average_bool(record["expected_source_hit"] for record in records),
        "expected_category_hit_rate": average_bool(record["expected_category_hit"] for record in records),
        "citation_check_pass_rate": average_bool(record["citation_check_passed"] for record in ordinary),
        "avg_supported_claim_ratio": average(record["supported_claim_ratio"] for record in ordinary),
        "unsupported_claim_count_total": sum(int(record["unsupported_claim_count"]) for record in records),
        "weak_claim_count_total": sum(int(record["weak_claim_count"]) for record in records),
        "evidence_quality_avg": average(record["evidence_quality_score"] for record in records),
        "needs_human_review_count": sum(1 for record in records if record["needs_human_review"]),
        "negative_refusal_pass_rate": average_bool(record["negative_refusal_pass"] for record in negative),
        "relation_path_found_rate": average_bool(record["relation_path_found"] for record in records),
        "failure_type_counts": dict(failure_counts),
        "metrics_by_mode": {mode: build_mode_metrics([r for r in records if r["retrieval_mode"] == mode]) for mode in modes},
    }


def build_mode_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary = [record for record in records if record.get("answer_generated")]
    negative = [record for record in records if record.get("expected_behavior") == "refuse_or_insufficient"]
    return {
        "runs": len(records),
        "llm_called_count": sum(1 for record in records if record["llm_called"]),
        "retry_count_total": sum(int(record.get("retry_count", 0) or 0) for record in records),
        "answer_generated_count": sum(1 for record in records if record["answer_generated"]),
        "empty_answer_count": sum(1 for record in records if record["answer_eval_status"] == "llm_empty_answer"),
        "expected_source_hit_rate": average_bool(record["expected_source_hit"] for record in records),
        "expected_category_hit_rate": average_bool(record["expected_category_hit"] for record in records),
        "citation_check_pass_rate": average_bool(record["citation_check_passed"] for record in ordinary),
        "avg_supported_claim_ratio": average(record["supported_claim_ratio"] for record in ordinary),
        "evidence_quality_avg": average(record["evidence_quality_score"] for record in records),
        "needs_human_review_count": sum(1 for record in records if record["needs_human_review"]),
        "negative_refusal_pass_rate": average_bool(record["negative_refusal_pass"] for record in negative),
        "relation_path_found_rate": average_bool(record["relation_path_found"] for record in records),
    }


def build_mode_compare(records: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    if len(modes) < 2:
        return {}
    by_mode = {mode: build_mode_metrics([r for r in records if r["retrieval_mode"] == mode]) for mode in modes}
    left, right = modes[0], modes[1]
    return {
        "baseline_mode": left,
        "candidate_mode": right,
        "empty_answer_delta": int(by_mode[right]["empty_answer_count"]) - int(by_mode[left]["empty_answer_count"]),
        "answer_generated_delta": int(by_mode[right]["answer_generated_count"]) - int(by_mode[left]["answer_generated_count"]),
        "citation_pass_delta": delta(by_mode[right]["citation_check_pass_rate"], by_mode[left]["citation_check_pass_rate"]),
        "evidence_quality_delta": delta(by_mode[right]["evidence_quality_avg"], by_mode[left]["evidence_quality_avg"]),
        "relation_path_found_delta": delta(by_mode[right]["relation_path_found_rate"], by_mode[left]["relation_path_found_rate"]),
        "metrics_by_mode": by_mode,
    }


def build_v4f_v4g_compare(v4g_metrics: dict[str, Any], v4f_path: Path) -> dict[str, Any]:
    if not v4f_path.exists():
        return {"v4f_available": False}
    v4f = json.loads(v4f_path.read_text(encoding="utf-8"))
    v4f_metrics = v4f.get("metrics", {})
    return {
        "v4f_available": True,
        "v4f_empty_answer_count": count_status(v4f.get("records", []), "llm_empty_answer"),
        "v4g_empty_answer_count": v4g_metrics.get("empty_answer_count", 0),
        "empty_answer_delta": int(v4g_metrics.get("empty_answer_count", 0)) - count_status(v4f.get("records", []), "llm_empty_answer"),
        "v4f_answer_generated_count": int(v4f_metrics.get("answer_generated_count", 0)),
        "v4g_answer_generated_count": int(v4g_metrics.get("answer_generated_count", 0)),
        "answer_generated_delta": int(v4g_metrics.get("answer_generated_count", 0)) - int(v4f_metrics.get("answer_generated_count", 0)),
        "v4f_llm_called_count": int(v4f_metrics.get("llm_called_count", 0)),
        "v4g_llm_called_count": int(v4g_metrics.get("llm_called_count", 0)),
        "retry_count_total": int(v4g_metrics.get("retry_count_total", 0)),
        "v4f_evidence_quality_avg": float(v4f_metrics.get("evidence_quality_avg", 0.0)),
        "v4g_evidence_quality_avg": float(v4g_metrics.get("evidence_quality_avg", 0.0)),
        "v4f_citation_check_pass_rate": float(v4f_metrics.get("citation_check_pass_rate", 0.0)),
        "v4g_citation_check_pass_rate": float(v4g_metrics.get("citation_check_pass_rate", 0.0)),
        "v4f_relation_path_found_rate": float(v4f_metrics.get("relation_path_found_rate", 0.0)),
        "v4g_relation_path_found_rate": float(v4g_metrics.get("relation_path_found_rate", 0.0)),
        "negative_refusal_pass_rate": float(v4g_metrics.get("negative_refusal_pass_rate", 0.0)),
    }


def count_status(records: list[dict[str, Any]], status: str) -> int:
    return sum(1 for record in records if record.get("answer_eval_status") == status)


def determine_eval_status(records: list[dict[str, Any]], llm_info: dict[str, Any], dry_run: bool) -> str:
    if dry_run:
        return "dry_run_retrieval_only"
    if not llm_info.get("available"):
        return "skipped_llm_unavailable"
    if any(record.get("answer_eval_status") in {"llm_exception", "llm_timeout"} for record in records):
        return "completed_with_llm_errors"
    return "completed"


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_merged_config(resolve_project_path(config_path))
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
    compare_json.write_text(json.dumps(result["v4f_v4g_compare"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_compare_csv(compare_csv, result["v4f_v4g_compare"])
    report_md.write_text(build_report(result), encoding="utf-8")


def write_eval_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "question",
        "retrieval_mode",
        "answer_eval_status",
        "failure_type",
        "failure_types",
        "llm_called",
        "llm_model",
        "answer_generated",
        "attempt_count",
        "retry_count",
        "evidence_sufficient",
        "insufficient_answer",
        "negative_refusal_pass",
        "citation_check_passed",
        "supported_claim_ratio",
        "unsupported_claim_count",
        "weak_claim_count",
        "evidence_quality_score",
        "evidence_quality_level",
        "needs_human_review",
        "relation_path_found",
        "expected_source_hit",
        "expected_category_hit",
        "raw_response_preview",
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
                    "failure_types": "|".join(str(item) for item in record.get("failure_types", [])),
                    "top_sources": " | ".join(
                        f"{source.get('rank')}:{source.get('category_dir')}:{source.get('source')}"
                        for source in record.get("top_sources", [])[:5]
                    ),
                }
            )


def write_compare_csv(path: Path, compare: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in compare.items():
            writer.writerow({"metric": key, "value": value})


def build_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    compare = result["v4f_v4g_compare"]
    lines = [
        "# V4g Answer Generation Reliability Report",
        "",
        f"- status: `{result['answer_eval_status']}`",
        f"- external_cloud_api_called: `{result['external_cloud_api_called']}`",
        f"- selected_model: `{result['llm_availability'].get('selected_model', '')}`",
        f"- total_runs: `{metrics['total_runs']}`",
        f"- llm_called_count: `{metrics['llm_called_count']}`",
        f"- retry_count_total: `{metrics['retry_count_total']}`",
        f"- empty_answer_count: `{metrics['empty_answer_count']}`",
        f"- answer_generated_count: `{metrics['answer_generated_count']}`",
        f"- negative_refusal_pass_rate: `{metrics['negative_refusal_pass_rate']}`",
        "",
        "## V4f vs V4g",
        "",
    ]
    for key, value in compare.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Per Run",
            "",
            "| case | mode | status | retry | failure_types | quality | review | negative refusal | relation path |",
            "|---|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for record in result["records"]:
        lines.append(
            "| {case} | {mode} | {status} | {retry} | {failures} | {quality} | {review} | {negative} | {path} |".format(
                case=record["case_id"],
                mode=record["retrieval_mode"],
                status=record["answer_eval_status"],
                retry=record["retry_count"],
                failures=";".join(record.get("failure_types", [])),
                quality=record.get("evidence_quality_level", ""),
                review=record["needs_human_review"],
                negative=record["negative_refusal_pass"],
                path=record["relation_path_found"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def print_summary(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    compare = result["v4f_v4g_compare"]
    print("V4g answer eval complete")
    print(f"answer_eval_status: {result['answer_eval_status']}")
    print(f"selected_model: {result['llm_availability'].get('selected_model', '')}")
    print(f"llm_called_count: {metrics['llm_called_count']}")
    print(f"retry_count_total: {metrics['retry_count_total']}")
    print(f"empty_answer_count: {metrics['empty_answer_count']}")
    print(f"answer_generated_count: {metrics['answer_generated_count']}")
    print(f"negative_refusal_pass_rate: {metrics['negative_refusal_pass_rate']}")
    print(f"empty_answer_delta_vs_v4f: {compare.get('empty_answer_delta')}")


def load_merged_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    base_path = config.get("base_config")
    if base_path:
        base = load_merged_config(resolve_project_path(base_path))
        config = deep_merge(base, {key: value for key, value in config.items() if key != "base_config"})
    config["_config_path"] = str(path)
    return config


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


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {key: value for key, value in base.items() if not str(key).startswith("_")}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def unique_values(items: list[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            values.append(item)
    return values


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
