from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "ui_v4h.yaml"


def load_answer_eval_report(config_path: str | Path = DEFAULT_CONFIG_PATH, version: str = "v4g") -> dict[str, Any]:
    config = load_report_config(config_path)
    files = version_files(config, version)
    eval_json = read_json_file(files.get("eval_json", ""))
    records = load_answer_eval_records(config_path, version)
    compare = load_answer_eval_compare(config_path, version)
    report_md = read_text_file(files.get("report_md", ""))
    results_csv = read_csv_file(files.get("results_csv", ""))
    return {
        "version": version,
        "config_path": str(resolve_project_path(config_path)),
        "files": files,
        "eval_json": eval_json.get("data", {}),
        "records": records,
        "compare": compare,
        "report_markdown": report_md.get("text", ""),
        "results_csv_rows": results_csv.get("rows", []),
        "warnings": collect_warnings([eval_json, compare, report_md, results_csv]),
        "metrics": summarize_answer_eval_metrics(eval_json.get("data", {})),
    }


def load_answer_eval_records(config_path: str | Path = DEFAULT_CONFIG_PATH, version: str = "v4g") -> list[dict[str, Any]]:
    config = load_report_config(config_path)
    files = version_files(config, version)
    jsonl_path = resolve_project_path(files.get("eval_jsonl", ""))
    if jsonl_path.exists():
        try:
            rows: list[dict[str, Any]] = []
            with jsonl_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict):
                            rows.append(item)
            return rows
        except Exception:
            return []
    eval_json = read_json_file(files.get("eval_json", "")).get("data", {})
    records = eval_json.get("records", []) if isinstance(eval_json, dict) else []
    return [dict(record) for record in records if isinstance(record, dict)]


def load_answer_eval_compare(config_path: str | Path = DEFAULT_CONFIG_PATH, version: str = "v4g") -> dict[str, Any]:
    config = load_report_config(config_path)
    files = version_files(config, version)
    return read_json_file(files.get("compare_json", "")).get("data", {})


def summarize_answer_eval_metrics(eval_json: dict[str, Any]) -> dict[str, Any]:
    metrics = eval_json.get("metrics", {}) if isinstance(eval_json, dict) else {}
    llm = eval_json.get("llm_availability", {}) if isinstance(eval_json, dict) else {}
    return {
        "status": eval_json.get("answer_eval_status", ""),
        "selected_model": llm.get("selected_model", ""),
        "total_cases": metrics.get("total_cases", 0),
        "total_runs": metrics.get("total_runs", 0),
        "llm_called_count": metrics.get("llm_called_count", 0),
        "answer_generated_count": metrics.get("answer_generated_count", 0),
        "empty_answer_count": metrics.get("empty_answer_count", count_status(eval_json.get("records", []), "llm_empty_answer")),
        "retry_count_total": metrics.get("retry_count_total", 0),
        "citation_check_pass_rate": metrics.get("citation_check_pass_rate", 0.0),
        "evidence_quality_avg": metrics.get("evidence_quality_avg", 0.0),
        "needs_human_review_count": metrics.get("needs_human_review_count", 0),
        "negative_refusal_pass_rate": metrics.get("negative_refusal_pass_rate", 0.0),
        "relation_path_found_rate": metrics.get("relation_path_found_rate", 0.0),
    }


def filter_answer_eval_records(
    records: list[dict[str, Any]],
    case_id: str | None = None,
    retrieval_mode: str | None = None,
    status: str | None = None,
    failure_type: str | None = None,
    quality_level: str | None = None,
    needs_human_review: bool | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        if case_id and str(record.get("case_id") or record.get("id")) != case_id:
            continue
        if retrieval_mode and str(record.get("retrieval_mode", "")) != retrieval_mode:
            continue
        if status and str(record.get("answer_eval_status", "")) != status:
            continue
        if failure_type and failure_type not in failure_labels(record):
            continue
        if quality_level and str(record.get("evidence_quality_level", "")) != quality_level:
            continue
        if needs_human_review is not None and bool(record.get("needs_human_review", False)) is not needs_human_review:
            continue
        filtered.append(record)
    return filtered


def extract_failure_options(records: list[dict[str, Any]]) -> list[str]:
    values = sorted({label for record in records for label in failure_labels(record) if label})
    return values


def extract_case_options(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(record.get("case_id") or record.get("id")) for record in records if record.get("case_id") or record.get("id")})


def extract_mode_options(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(record.get("retrieval_mode", "")) for record in records if record.get("retrieval_mode")})


def extract_status_options(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(record.get("answer_eval_status", "")) for record in records if record.get("answer_eval_status")})


def extract_quality_options(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(record.get("evidence_quality_level", "")) for record in records if record.get("evidence_quality_level")})


def format_run_detail(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id") or record.get("id", ""),
        "question": record.get("question", ""),
        "retrieval_mode": record.get("retrieval_mode", ""),
        "answer_eval_status": record.get("answer_eval_status", ""),
        "failure_types": failure_labels(record),
        "answer_preview": record.get("answer_preview", ""),
        "answer": record.get("answer", ""),
        "llm_attempts": record.get("llm_attempts", []),
        "retry_count": record.get("retry_count", 0),
        "raw_response_preview": record.get("raw_response_preview", ""),
        "prompt_preview": record.get("prompt_preview", ""),
        "citation_summary": {
            "citation_check_passed": record.get("citation_check_passed"),
            "supported_claim_ratio": record.get("supported_claim_ratio"),
            "unsupported_claim_count": record.get("unsupported_claim_count"),
            "weak_claim_count": record.get("weak_claim_count"),
        },
        "unsupported_claims": record.get("unsupported_claims", []),
        "weak_claims": record.get("weak_claims", []),
        "evidence_quality": record.get("evidence_quality", {}),
        "evidence_quality_score": record.get("evidence_quality_score"),
        "evidence_quality_level": record.get("evidence_quality_level"),
        "review_decision": record.get("review_decision", {}),
        "needs_human_review": record.get("needs_human_review", False),
        "review_reasons": record.get("review_reasons", []),
        "expected_source_hit": record.get("expected_source_hit"),
        "expected_category_hit": record.get("expected_category_hit"),
        "negative_refusal_pass": record.get("negative_refusal_pass"),
        "top_sources": record.get("top_sources", []),
        "final_sources": record.get("final_sources", []),
        "relation_paths": extract_relation_paths(record),
    }


def failure_labels(record: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    raw = record.get("failure_types", [])
    if isinstance(raw, list):
        labels.extend(str(item) for item in raw if str(item))
    elif isinstance(raw, str):
        labels.extend(item for item in raw.replace(",", "|").split("|") if item)
    failure_type = str(record.get("failure_type", "") or "")
    if failure_type:
        labels.append(failure_type)
    status = str(record.get("answer_eval_status", "") or "")
    if status in {"llm_empty_answer", "insufficient_evidence", "skipped_llm_unavailable"}:
        if status == "insufficient_evidence":
            labels.append("retrieval_insufficient")
        else:
            labels.append(status)
    if int(record.get("unsupported_claim_count", 0) or 0) > 0:
        labels.append("unsupported_claims_present")
    if int(record.get("weak_claim_count", 0) or 0) > 0:
        labels.append("weak_claims_present")
    return unique_values(labels)


def extract_relation_paths(record: dict[str, Any]) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for source in record.get("final_sources", []) or []:
        if not isinstance(source, dict):
            continue
        relation_path = source.get("relation_path", [])
        if relation_path:
            paths.append(
                {
                    "source": source.get("source", ""),
                    "rank": source.get("rank"),
                    "relation_path": relation_path,
                    "relation_reasons": source.get("relation_reasons", []),
                    "fusion_reasons": source.get("fusion_reasons", []),
                }
            )
    return paths


def load_report_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    config = load_yaml(path)
    base_config = config.get("base_config")
    if base_config:
        base = load_report_config(resolve_project_path(base_config))
        config = deep_merge(base, {key: value for key, value in config.items() if key != "base_config"})
    config["_config_path"] = str(path)
    return config


def version_files(config: dict[str, Any], version: str) -> dict[str, str]:
    browser = config.get("answer_eval_report_browser", {}) if isinstance(config, dict) else {}
    files = (browser.get("files", {}) or {}).get(version, {})
    if not files:
        raise ValueError(f"Unknown answer eval report version: {version}")
    return {str(key): str(value) for key, value in files.items()}


def read_json_file(value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(value)
    if not path.exists():
        return {"path": str(path), "data": {}, "warning": f"Missing file: {path}"}
    try:
        return {"path": str(path), "data": json.loads(path.read_text(encoding="utf-8")), "warning": ""}
    except Exception as exc:  # noqa: BLE001 - UI should show malformed file warnings.
        return {"path": str(path), "data": {}, "warning": str(exc)}


def read_text_file(value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(value)
    if not path.exists():
        return {"path": str(path), "text": "", "warning": f"Missing file: {path}"}
    try:
        return {"path": str(path), "text": path.read_text(encoding="utf-8"), "warning": ""}
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "text": "", "warning": str(exc)}


def read_csv_file(value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(value)
    if not path.exists():
        return {"path": str(path), "rows": [], "warning": f"Missing file: {path}"}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        return {"path": str(path), "rows": rows, "warning": ""}
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "rows": [], "warning": str(exc)}


def collect_warnings(items: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for item in items:
        warning = str(item.get("warning", "") or "")
        if warning:
            warnings.append(warning)
    return warnings


def count_status(records: Any, status: str) -> int:
    if not isinstance(records, list):
        return 0
    return sum(1 for record in records if isinstance(record, dict) and record.get("answer_eval_status") == status)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


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
    values: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            values.append(item)
    return values
