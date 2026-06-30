from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "review_adjudication_v4j.yaml"
MANUAL_FIELDS = [
    "manual_judgment",
    "manual_citation_judgment",
    "manual_should_refuse",
    "manual_fix_suggestion",
    "manual_notes",
]
JSONISH_FIELDS = {
    "failure_types",
    "review_reasons",
    "top_sources_preview",
    "unsupported_claims_preview",
    "weak_claims_preview",
    "relation_paths_preview",
    "llm_attempts_preview",
    "fix_categories",
}


def load_adjudication_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid adjudication config: {path}")
    base_path = config.get("base_triage_config")
    if base_path:
        base_resolved = resolve_project_path(base_path)
        if base_resolved.exists():
            with base_resolved.open("r", encoding="utf-8") as file:
                config["_base_triage_config"] = yaml.safe_load(file) or {}
    config["_config_path"] = str(path)
    return config


def load_review_tasks(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return []
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        data = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            tasks = data.get("tasks", [])
        else:
            tasks = data
        return [normalize_task(dict(task)) for task in tasks if isinstance(task, dict)]
    if suffix == ".jsonl":
        tasks = []
        with resolved.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        tasks.append(normalize_task(item))
        return tasks
    if suffix == ".csv":
        with resolved.open("r", encoding="utf-8-sig", newline="") as file:
            return [normalize_task(row) for row in csv.DictReader(file)]
    return []


def load_manual_review_csv(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8-sig", newline="") as file:
        return [normalize_task(row) for row in csv.DictReader(file)]


def validate_manual_fields(tasks: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    manual_config = config.get("manual_fields", {})
    invalid_values: list[dict[str, str]] = []
    missing_required: list[str] = []
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        for field in ("manual_judgment", "manual_citation_judgment", "manual_should_refuse"):
            allowed = set(manual_config.get(field, {}).get("allowed_values", []) or [])
            value = str(task.get(field, "") or "")
            if value not in allowed:
                invalid_values.append({"task_id": task_id, "field": field, "value": value})
        if (
            config.get("adjudication", {}).get("high_priority_requires_manual_judgment", True)
            and task.get("priority") == "high"
            and not str(task.get("manual_judgment", "") or "")
        ):
            missing_required.append(task_id)
    pending_count = sum(1 for task in tasks if infer_adjudication_status(task, config) == "pending")
    return {
        "valid": not invalid_values,
        "invalid_values": invalid_values,
        "missing_required_high_priority_judgments": missing_required,
        "pending_count": pending_count,
        "total_tasks": len(tasks),
    }


def merge_manual_review(tasks: list[dict[str, Any]], manual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manual_by_key = {task_key(row): row for row in manual_rows if task_key(row)}
    merged = []
    for task in tasks:
        item = dict(task)
        row = manual_by_key.get(task_key(item), {})
        if row:
            for field in MANUAL_FIELDS:
                if field in row:
                    item[field] = row.get(field, "")
            if row.get("reviewed_at"):
                item["reviewed_at"] = row.get("reviewed_at")
            if row.get("reviewer"):
                item["reviewer"] = row.get("reviewer")
            item["manual_row_found"] = True
        else:
            item["manual_row_found"] = False
        merged.append(normalize_task(item))
    return merged


def infer_adjudication_status(task: dict[str, Any], config: dict[str, Any]) -> str:
    judgment = str(task.get("manual_judgment", "") or "")
    citation = str(task.get("manual_citation_judgment", "") or "")
    if config.get("adjudication", {}).get("pending_if_manual_judgment_empty", True) and not judgment:
        return "pending"
    if citation == "checker_false_positive" or judgment == "checker_false_positive":
        return "needs_citation_checker_fix"
    if judgment == "accept":
        return "accepted"
    if judgment == "reject_answer":
        return "rejected_answer"
    if judgment == "revise_answer":
        return "needs_revision"
    if judgment == "keep_refusal":
        return "valid_refusal"
    if judgment == "rerun_llm":
        return "needs_prompt_fix"
    if judgment == "expand_evidence":
        return "needs_retrieval_fix"
    if judgment == "needs_discussion":
        return "needs_discussion"
    return "pending"


def infer_fix_categories(task: dict[str, Any], config: dict[str, Any]) -> list[str]:
    judgment = str(task.get("manual_judgment", "") or "")
    citation = str(task.get("manual_citation_judgment", "") or "")
    should_refuse = str(task.get("manual_should_refuse", "") or "")
    failures = set(task.get("failure_types", []) or [])
    categories: list[str] = []
    rules = config.get("fix_categories", {})
    for category, rule in rules.items():
        if judgment in set(rule.get("manual_judgments", []) or []):
            categories.append(category)
        if citation in set(rule.get("manual_citation_judgments", []) or []):
            categories.append(category)
        if should_refuse in set(rule.get("manual_should_refuse", []) or []):
            categories.append(category)
    if judgment == "rerun_llm" or "llm_empty_answer" in failures:
        categories.append("llm_reliability_fix")
    if task.get("relation_paths_preview") and judgment in {"expand_evidence", "revise_answer", "needs_discussion"}:
        categories.append("relation_fusion_review")
    status = infer_adjudication_status(task, config)
    if not categories and status in {"accepted", "valid_refusal"}:
        categories.append("no_action_needed")
    return unique_values(categories)


def enrich_adjudication_tasks(tasks: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    enriched = []
    for task in tasks:
        item = normalize_task(dict(task))
        item["adjudication_status"] = infer_adjudication_status(item, config)
        item["fix_categories"] = infer_fix_categories(item, config)
        enriched.append(item)
    return enriched


def summarize_adjudication(tasks: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    enriched = enrich_adjudication_tasks(tasks, config)
    status_counts = Counter(str(task.get("adjudication_status", "")) for task in enriched)
    judgment_counts = Counter(str(task.get("manual_judgment", "") or "") for task in enriched)
    citation_counts = Counter(str(task.get("manual_citation_judgment", "") or "") for task in enriched)
    should_refuse_counts = Counter(str(task.get("manual_should_refuse", "") or "") for task in enriched)
    fix_counts: Counter[str] = Counter()
    for task in enriched:
        for category in task.get("fix_categories", []) or []:
            fix_counts[str(category)] += 1
    validation = validate_manual_fields(enriched, config)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_tasks": len(enriched),
        "pending_count": status_counts.get("pending", 0),
        "adjudicated_count": len(enriched) - status_counts.get("pending", 0),
        "accepted_tasks": status_counts.get("accepted", 0),
        "rejected_tasks": status_counts.get("rejected_answer", 0),
        "citation_false_positive_count": fix_counts.get("citation_checker_fix", 0),
        "true_unsupported_claims_count": citation_counts.get("unsupported", 0),
        "valid_refusals": status_counts.get("valid_refusal", 0),
        "cases_needing_prompt_fix": fix_counts.get("prompt_fix", 0),
        "cases_needing_retrieval_fix": fix_counts.get("retrieval_fix", 0),
        "cases_needing_fallback_model_or_retry_fix": fix_counts.get("llm_reliability_fix", 0),
        "adjudication_status_counts": dict(sorted(status_counts.items())),
        "manual_judgment_counts": dict(sorted(judgment_counts.items())),
        "manual_citation_judgment_counts": dict(sorted(citation_counts.items())),
        "manual_should_refuse_counts": dict(sorted(should_refuse_counts.items())),
        "fix_category_counts": dict(sorted(fix_counts.items())),
        "validation": validation,
    }


def build_calibration_notes(tasks: list[dict[str, Any]], summary: dict[str, Any], config: dict[str, Any]) -> str:
    enriched = enrich_adjudication_tasks(tasks, config)
    sections = [
        "# V4j Answer Eval Calibration Notes",
        "",
        "## Summary",
        "",
        f"- total_tasks: `{summary.get('total_tasks', 0)}`",
        f"- pending_count: `{summary.get('pending_count', 0)}`",
        f"- adjudicated_count: `{summary.get('adjudicated_count', 0)}`",
        f"- fix_category_counts: `{summary.get('fix_category_counts', {})}`",
        "",
    ]
    sections.extend(render_case_section("Prompt / LLM reliability candidates", enriched, {"prompt_fix", "llm_reliability_fix"}))
    sections.extend(render_case_section("Citation checker false-positive candidates", enriched, {"citation_checker_fix"}))
    sections.extend(render_case_section("Retrieval / fusion candidates", enriched, {"retrieval_fix", "relation_fusion_review"}))
    sections.extend(render_case_section("Negative refusal confirmation candidates", enriched, {"no_action_needed"}, status_filter={"valid_refusal"}))
    if summary.get("pending_count", 0):
        pending = [task for task in enriched if task.get("adjudication_status") == "pending"]
        sections.extend(render_pending_section(pending))
    return "\n".join(sections)


def export_adjudication_outputs(
    tasks: list[dict[str, Any]], summary: dict[str, Any], notes_md: str, config: dict[str, Any]
) -> dict[str, Any]:
    outputs = config.get("outputs", {})
    enriched = enrich_adjudication_tasks(tasks, config)
    paths = {
        "adjudicated_tasks_json": resolve_project_path(outputs.get("adjudicated_tasks_json", "")),
        "adjudicated_tasks_jsonl": resolve_project_path(outputs.get("adjudicated_tasks_jsonl", "")),
        "adjudicated_tasks_csv": resolve_project_path(outputs.get("adjudicated_tasks_csv", "")),
        "adjudication_summary_json": resolve_project_path(outputs.get("adjudication_summary_json", "")),
        "adjudication_report_md": resolve_project_path(outputs.get("adjudication_report_md", "")),
        "calibration_notes_md": resolve_project_path(outputs.get("calibration_notes_md", "")),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"summary": summary, "tasks": enriched}
    paths["adjudicated_tasks_json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with paths["adjudicated_tasks_jsonl"].open("w", encoding="utf-8") as file:
        for task in enriched:
            file.write(json.dumps(task, ensure_ascii=False) + "\n")
    write_adjudicated_csv(enriched, paths["adjudicated_tasks_csv"])
    paths["adjudication_summary_json"].write_text(
        json.dumps({**summary, "output_paths": stringify_paths(paths)}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["adjudication_report_md"].write_text(render_adjudication_report(enriched, summary), encoding="utf-8")
    paths["calibration_notes_md"].write_text(notes_md, encoding="utf-8")
    return {"summary": summary, "output_paths": stringify_paths(paths)}


def write_adjudicated_csv(tasks: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "task_id",
        "case_id",
        "retrieval_mode",
        "priority",
        "failure_types",
        "manual_judgment",
        "manual_citation_judgment",
        "manual_should_refuse",
        "manual_fix_suggestion",
        "manual_notes",
        "adjudication_status",
        "fix_categories",
        "answer_eval_status",
        "evidence_quality_level",
        "suggested_review_action",
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


def render_adjudication_report(tasks: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# V4j Human Review Adjudication",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Tasks",
        "",
    ]
    for task in tasks:
        lines.extend(
            [
                f"### {task.get('task_id', '')}",
                "",
                f"- status: `{task.get('adjudication_status', '')}`",
                f"- fix_categories: `{task.get('fix_categories', [])}`",
                f"- manual_judgment: `{task.get('manual_judgment', '')}`",
                f"- manual_citation_judgment: `{task.get('manual_citation_judgment', '')}`",
                f"- manual_should_refuse: `{task.get('manual_should_refuse', '')}`",
                f"- failure_types: `{task.get('failure_types', [])}`",
                "",
                str(task.get("answer_preview", "")),
                "",
            ]
        )
    return "\n".join(lines)


def render_case_section(
    title: str, tasks: list[dict[str, Any]], categories: set[str], status_filter: set[str] | None = None
) -> list[str]:
    matched = []
    for task in tasks:
        category_hit = bool(set(task.get("fix_categories", []) or []).intersection(categories))
        status_hit = bool(status_filter and task.get("adjudication_status") in status_filter)
        if category_hit or status_hit:
            matched.append(task)
    lines = [f"## {title}", ""]
    if not matched:
        lines.extend(["No adjudicated examples yet.", ""])
        return lines
    for task in matched[:10]:
        lines.extend(
            [
                f"- `{task.get('task_id', '')}`: status=`{task.get('adjudication_status', '')}`, "
                f"fix=`{task.get('fix_categories', [])}`, notes={task.get('manual_notes', '')}",
            ]
        )
    lines.append("")
    return lines


def render_pending_section(tasks: list[dict[str, Any]]) -> list[str]:
    lines = ["## Pending Review Tasks", ""]
    for task in tasks[:10]:
        lines.append(
            f"- `{task.get('task_id', '')}` priority=`{task.get('priority', '')}` failures=`{task.get('failure_types', [])}`"
        )
    lines.append("")
    return lines


def normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    item = dict(task)
    for field in JSONISH_FIELDS:
        if field in item:
            item[field] = parse_jsonish(item.get(field))
    for field in MANUAL_FIELDS:
        item[field] = str(item.get(field, "") or "")
    if "needs_human_review" in item:
        item["needs_human_review"] = parse_bool(item.get("needs_human_review"))
    return item


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return []
    text = str(value)
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        if "|" in text:
            return [item for item in text.split("|") if item]
        return [text]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def task_key(task: dict[str, Any]) -> str:
    task_id = str(task.get("task_id", "") or "")
    if task_id:
        return task_id
    return "::".join([str(task.get("case_id", "") or ""), str(task.get("retrieval_mode", "") or "")])


def unique_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def stringify_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


def resolve_project_path(path: str | Path) -> Path:
    value = Path(str(path))
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value

