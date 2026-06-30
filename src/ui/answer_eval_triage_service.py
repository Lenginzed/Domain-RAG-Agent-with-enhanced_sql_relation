from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.ui.answer_eval_report_service import failure_labels, format_run_detail


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "review_triage_v4i.yaml"


def load_triage_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid triage config: {path}")
    config["_config_path"] = str(path)
    return config


def load_eval_records_for_triage(config: dict[str, Any], version: str) -> list[dict[str, Any]]:
    jsonl_path = resolve_project_path(config.get("inputs", {}).get("eval_jsonl", {}).get(version, ""))
    if jsonl_path.exists():
        records: list[dict[str, Any]] = []
        try:
            with jsonl_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict):
                            item.setdefault("_triage_version", version)
                            records.append(item)
            return records
        except Exception:
            return []

    json_path = resolve_project_path(config.get("inputs", {}).get("eval_json", {}).get(version, ""))
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_records = data.get("records", []) if isinstance(data, dict) else []
    records = [dict(record) for record in raw_records if isinstance(record, dict)]
    for record in records:
        record.setdefault("_triage_version", version)
    return records


def infer_failure_priority(record: dict[str, Any], config: dict[str, Any]) -> str:
    labels = set(failure_labels(record))
    rules = config.get("priority_rules", {})
    high_labels = set(rules.get("high", []) or [])
    if record.get("negative_refusal_pass") and "retrieval_insufficient" in labels and not labels.intersection(high_labels):
        return "low"
    for priority in ("high", "medium", "low"):
        if labels.intersection(set(rules.get(priority, []) or [])):
            return priority
    if record.get("needs_human_review"):
        return "medium"
    return "low"


def infer_suggested_review_action(record: dict[str, Any], config: dict[str, Any]) -> str:
    labels = failure_labels(record)
    actions = config.get("suggested_actions", {})
    priority = infer_failure_priority(record, config)
    priority_order = config.get("priority_rules", {})
    ordered_labels: list[str] = []
    for group in ("high", "medium", "low"):
        ordered_labels.extend(label for label in priority_order.get(group, []) or [] if label in labels)
    ordered_labels.extend(label for label in labels if label not in ordered_labels)
    for label in ordered_labels:
        if label in actions:
            return str(actions[label])
    if record.get("needs_human_review") and "needs_human_review" in actions:
        return str(actions["needs_human_review"])
    return f"人工复核该 {priority} priority run，并补充 manual fields。"


def build_review_task(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    detail = format_run_detail(record)
    version = str(record.get("_triage_version") or config.get("inputs", {}).get("default_version", "v4g"))
    case_id = str(detail.get("case_id", ""))
    retrieval_mode = str(detail.get("retrieval_mode", ""))
    triage_config = config.get("triage", {})
    manual_fields = config.get("manual_fields", [])
    top_sources = detail.get("top_sources", [])[: int(triage_config.get("max_sources", 5) or 5)]

    task = {
        "task_id": f"review::{version}::{case_id}::{retrieval_mode}",
        "version": version,
        "case_id": case_id,
        "question": detail.get("question", ""),
        "retrieval_mode": retrieval_mode,
        "answer_eval_status": detail.get("answer_eval_status", ""),
        "priority": infer_failure_priority(record, config),
        "failure_types": failure_labels(record),
        "suggested_review_action": infer_suggested_review_action(record, config),
        "evidence_quality_level": detail.get("evidence_quality_level", ""),
        "evidence_quality_score": detail.get("evidence_quality_score"),
        "needs_human_review": bool(detail.get("needs_human_review", False)),
        "review_reasons": normalize_list(detail.get("review_reasons", [])),
        "answer_preview": truncate_text(detail.get("answer_preview", ""), int(triage_config.get("max_answer_preview_chars", 800) or 800)),
        "top_sources_preview": preview_sources(top_sources),
        "unsupported_claims_preview": preview_claims(detail.get("unsupported_claims", [])),
        "weak_claims_preview": preview_claims(detail.get("weak_claims", [])),
        "relation_paths_preview": detail.get("relation_paths", []),
        "retry_count": detail.get("retry_count", 0),
        "llm_attempts_preview": preview_attempts(detail.get("llm_attempts", [])),
        "raw_response_preview": truncate_text(
            detail.get("raw_response_preview", ""),
            int(triage_config.get("max_raw_response_preview_chars", 800) or 800),
        ),
        "prompt_preview": truncate_text(
            detail.get("prompt_preview", ""),
            int(triage_config.get("max_prompt_preview_chars", 800) or 800),
        ),
    }
    for field in manual_fields:
        task[str(field)] = ""
    return task


def build_review_tasks(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    include_failures = set(config.get("triage", {}).get("include_failure_types", []) or [])
    include_review = bool(config.get("triage", {}).get("include_needs_human_review", True))
    exclude_negative_success = bool(config.get("triage", {}).get("exclude_negative_success_refusal", False))
    tasks: list[dict[str, Any]] = []
    for record in records:
        labels = set(failure_labels(record))
        include = bool(labels.intersection(include_failures)) or (include_review and bool(record.get("needs_human_review", False)))
        if not include:
            continue
        if exclude_negative_success and record.get("negative_refusal_pass") and labels == {"retrieval_insufficient"}:
            continue
        tasks.append(build_review_task(record, config))
    return tasks


def summarize_review_tasks(tasks: list[dict[str, Any]], total_records: int | None = None, version: str | None = None) -> dict[str, Any]:
    priority_counts = Counter(str(task.get("priority", "")) for task in tasks)
    failure_counts: Counter[str] = Counter()
    status_counts = Counter(str(task.get("answer_eval_status", "")) for task in tasks)
    mode_counts = Counter(str(task.get("retrieval_mode", "")) for task in tasks)
    for task in tasks:
        for label in task.get("failure_types", []) or []:
            failure_counts[str(label)] += 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": version or infer_task_version(tasks),
        "total_records": total_records if total_records is not None else len(tasks),
        "review_task_count": len(tasks),
        "priority_counts": dict(sorted(priority_counts.items())),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "retrieval_mode_counts": dict(sorted(mode_counts.items())),
    }


def export_review_tasks(tasks: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    outputs = config.get("outputs", {})
    summary = summarize_review_tasks(
        tasks,
        total_records=config.get("_triage_total_records"),
        version=str(config.get("_triage_version") or infer_task_version(tasks)),
    )
    output_paths = {
        "review_tasks_json": resolve_project_path(outputs.get("review_tasks_json", "")),
        "review_tasks_jsonl": resolve_project_path(outputs.get("review_tasks_jsonl", "")),
        "review_tasks_csv": resolve_project_path(outputs.get("review_tasks_csv", "")),
        "review_tasks_md": resolve_project_path(outputs.get("review_tasks_md", "")),
        "summary_json": resolve_project_path(outputs.get("summary_json", "")),
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {"summary": summary, "tasks": tasks}
    output_paths["review_tasks_json"].write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_paths["review_tasks_jsonl"].open("w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(task, ensure_ascii=False) + "\n")
    write_tasks_csv(tasks, output_paths["review_tasks_csv"])
    output_paths["review_tasks_md"].write_text(render_tasks_markdown(tasks, summary), encoding="utf-8")
    output_paths["summary_json"].write_text(
        json.dumps({**summary, "output_paths": stringify_paths(output_paths)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"summary": summary, "output_paths": stringify_paths(output_paths)}


def write_tasks_csv(tasks: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "task_id",
        "version",
        "case_id",
        "question",
        "retrieval_mode",
        "answer_eval_status",
        "priority",
        "failure_types",
        "suggested_review_action",
        "evidence_quality_level",
        "evidence_quality_score",
        "needs_human_review",
        "review_reasons",
        "answer_preview",
        "top_sources_preview",
        "unsupported_claims_preview",
        "weak_claims_preview",
        "relation_paths_preview",
        "retry_count",
        "llm_attempts_preview",
        "raw_response_preview",
        "prompt_preview",
        "manual_judgment",
        "manual_citation_judgment",
        "manual_should_refuse",
        "manual_fix_suggestion",
        "manual_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for task in tasks:
            row = dict(task)
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(row)


def render_tasks_markdown(tasks: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# V4i Answer Eval Review Tasks",
        "",
        "## Summary",
        "",
        f"- version: `{summary.get('version', '')}`",
        f"- total_records: `{summary.get('total_records', 0)}`",
        f"- review_task_count: `{summary.get('review_task_count', 0)}`",
        f"- priority_counts: `{summary.get('priority_counts', {})}`",
        f"- failure_type_counts: `{summary.get('failure_type_counts', {})}`",
        f"- status_counts: `{summary.get('status_counts', {})}`",
        "",
        "## Tasks",
        "",
    ]
    for task in tasks:
        lines.extend(
            [
                f"### {task.get('task_id', '')}",
                "",
                f"- priority: `{task.get('priority', '')}`",
                f"- case_id: `{task.get('case_id', '')}`",
                f"- retrieval_mode: `{task.get('retrieval_mode', '')}`",
                f"- answer_eval_status: `{task.get('answer_eval_status', '')}`",
                f"- failure_types: `{task.get('failure_types', [])}`",
                f"- evidence_quality: `{task.get('evidence_quality_level', '')}` / `{task.get('evidence_quality_score', '')}`",
                f"- suggested_review_action: {task.get('suggested_review_action', '')}",
                "",
                "**Question**",
                "",
                str(task.get("question", "")),
                "",
                "**Answer Preview**",
                "",
                str(task.get("answer_preview", "")),
                "",
                "**Top Sources Preview**",
                "",
                "```json",
                json.dumps(task.get("top_sources_preview", []), ensure_ascii=False, indent=2),
                "```",
                "",
                "**Relation Paths Preview**",
                "",
                "```json",
                json.dumps(task.get("relation_paths_preview", []), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def preview_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preview = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        preview.append(
            {
                "rank": source.get("rank"),
                "source": source.get("source", ""),
                "category_dir": source.get("category_dir", ""),
                "doc_type": source.get("doc_type", ""),
                "retrieval_channels": source.get("retrieval_channels", []),
                "relation_score": source.get("relation_score"),
                "relation_score_normalized": source.get("relation_score_normalized"),
            }
        )
    return preview


def preview_claims(claims: Any) -> list[dict[str, Any]]:
    if not isinstance(claims, list):
        return []
    preview = []
    for claim in claims:
        if isinstance(claim, dict):
            preview.append(
                {
                    "text": truncate_text(claim.get("text") or claim.get("claim", ""), 500),
                    "reason": claim.get("reason") or claim.get("support_reason", ""),
                    "checker_error_hint": claim.get("checker_error_hint", ""),
                }
            )
        else:
            preview.append({"text": truncate_text(str(claim), 500), "reason": "", "checker_error_hint": ""})
    return preview


def preview_attempts(attempts: Any) -> list[dict[str, Any]]:
    if not isinstance(attempts, list):
        return []
    preview = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        preview.append(
            {
                "attempt_index": attempt.get("attempt_index"),
                "model": attempt.get("model", ""),
                "status": attempt.get("status", ""),
                "raw_response_preview": truncate_text(attempt.get("raw_response_preview", ""), 500),
                "error": attempt.get("error", ""),
            }
        )
    return preview


def truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [item for item in value.replace(",", "|").split("|") if item]
    return []


def infer_task_version(tasks: list[dict[str, Any]]) -> str:
    versions = sorted({str(task.get("version", "")) for task in tasks if task.get("version")})
    return versions[0] if len(versions) == 1 else ",".join(versions)


def stringify_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
