from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "manual_review.yaml"

REQUIRED_REVIEW_FIELDS = [
    "id",
    "manual_judgment",
    "manual_citation_judgment",
    "manual_should_refuse",
    "manual_notes",
    "reviewer",
    "reviewed_at",
]

MANUAL_REVIEW_FIELDS = [
    "manual_judgment",
    "manual_citation_judgment",
    "manual_should_refuse",
    "manual_notes",
    "reviewer",
    "reviewed_at",
]

CSV_FIELDNAMES = [
    "id",
    "question",
    "query_type",
    "difficulty",
    "evidence_quality_score",
    "evidence_quality_level",
    "evidence_quality_warnings",
    "needs_human_review",
    "insufficient_answer",
    "evidence_sufficient",
    "supported_claim_ratio",
    "unsupported_claim_count",
    "weak_claim_count",
    "citation_check_passed",
    "final_source_count",
    "manual_judgment",
    "manual_citation_judgment",
    "manual_should_refuse",
    "manual_notes",
    "reviewer",
    "reviewed_at",
    "is_reviewed",
    "manual_review_status",
]


def main() -> int:
    configure_stdout()
    try:
        result = import_answer_review_labels()
    except Exception as exc:  # noqa: BLE001 - this script should fail explicitly.
        print("Manual review label import failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result["summary"])
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def import_answer_review_labels(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    paths = resolve_paths(config)
    ensure_inputs_exist(
        [
            paths["review_template_csv"],
            paths["evidence_quality_json"],
            paths["evidence_quality_jsonl"],
            paths["evidence_quality_csv"],
        ]
    )

    review_rows = read_review_rows(paths["review_template_csv"], config)
    quality_result = load_json(paths["evidence_quality_json"])
    quality_records = quality_result.get("records", [])
    if not isinstance(quality_records, list) or not quality_records:
        raise ValueError(f"No evidence quality records found in {paths['evidence_quality_json']}")

    merged_records = merge_records(quality_records, review_rows)
    summary = build_summary(merged_records, quality_result, paths, config_path, review_rows)
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "inputs": {
            "review_template_csv": str(paths["review_template_csv"]),
            "evidence_quality_json": str(paths["evidence_quality_json"]),
            "evidence_quality_jsonl": str(paths["evidence_quality_jsonl"]),
            "evidence_quality_csv": str(paths["evidence_quality_csv"]),
        },
        "outputs": {
            "merged_json": str(paths["merged_json"]),
            "merged_jsonl": str(paths["merged_jsonl"]),
            "merged_csv": str(paths["merged_csv"]),
            "summary_json": str(paths["summary_json"]),
            "summary_md": str(paths["summary_md"]),
        },
        "summary": summary,
        "records": merged_records,
    }
    write_outputs(result, paths)
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    required_inputs = [
        "review_template_csv",
        "evidence_quality_json",
        "evidence_quality_jsonl",
        "evidence_quality_csv",
    ]
    required_outputs = [
        "merged_json",
        "merged_jsonl",
        "merged_csv",
        "summary_json",
        "summary_md",
    ]
    missing = [key for key in required_inputs if key not in inputs]
    missing.extend(key for key in required_outputs if key not in outputs)
    if missing:
        raise KeyError(f"Missing manual review config keys: {missing}")

    resolved: dict[str, Path] = {}
    for key in required_inputs:
        resolved[key] = resolve_project_path(inputs[key])
    for key in required_outputs:
        resolved[key] = resolve_project_path(outputs[key])
    return resolved


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_inputs_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required V2c.4 input files: {missing}")


def read_review_rows(path: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    allowed_values = config.get("allowed_values", {})
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    text, encoding = read_text_with_encoding_fallback(path, ["utf-8-sig", "gb18030"])
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = set(reader.fieldnames or [])
    missing_fields = [field for field in REQUIRED_REVIEW_FIELDS if field not in fieldnames]
    if missing_fields:
        raise ValueError(
            f"Review template is missing required fields: {missing_fields}; "
            f"detected_encoding={encoding}"
        )

    for line_number, row in enumerate(reader, start=2):
        normalized = normalize_review_row(row)
        row_id = normalized["id"]
        if not row_id:
            errors.append(f"line {line_number}: missing id")
        elif row_id in seen_ids:
            errors.append(f"line {line_number}: duplicate id {row_id}")
        seen_ids.add(row_id)

        errors.extend(validate_allowed_values(normalized, allowed_values, line_number))
        rows.append(normalized)

    if errors:
        preview = "; ".join(errors[:10])
        extra = f" (+{len(errors) - 10} more)" if len(errors) > 10 else ""
        raise ValueError(f"Invalid manual review labels: {preview}{extra}")
    return rows


def read_text_with_encoding_fallback(path: Path, encodings: list[str]) -> tuple[str, str]:
    errors: list[str] = []
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError(
        encodings[0],
        b"",
        0,
        1,
        f"Unable to decode {path} with encodings {encodings}; errors: {' | '.join(errors)}",
    )


def normalize_review_row(row: dict[str, Any]) -> dict[str, str]:
    normalized = {key: strip_cell(row.get(key, "")) for key in row}
    for field in REQUIRED_REVIEW_FIELDS:
        normalized.setdefault(field, "")
    normalized["manual_should_refuse"] = normalized["manual_should_refuse"].lower()
    return normalized


def strip_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validate_allowed_values(
    row: dict[str, str],
    allowed_values: dict[str, list[str]],
    line_number: int,
) -> list[str]:
    errors: list[str] = []
    for field in ["manual_judgment", "manual_citation_judgment", "manual_should_refuse"]:
        allowed = {str(value) for value in allowed_values.get(field, [])}
        value = row.get(field, "")
        if value not in allowed:
            errors.append(
                f"line {line_number}: {field}={value!r} is not in allowed values {sorted(allowed)}"
            )
    return errors


def merge_records(
    quality_records: list[dict[str, Any]],
    review_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    review_by_id = {row["id"]: row for row in review_rows}
    merged: list[dict[str, Any]] = []

    for quality in quality_records:
        record_id = str(quality.get("id", "")).strip()
        review = review_by_id.get(record_id, {})
        manual_fields = {field: review.get(field, "") for field in MANUAL_REVIEW_FIELDS}
        is_reviewed = has_manual_review(manual_fields)
        warnings = normalize_list_field(quality.get("evidence_quality_warnings", []))
        reasons = normalize_list_field(quality.get("evidence_quality_reasons", []))

        merged.append(
            {
                "id": record_id,
                "question": review.get("question") or quality.get("question", ""),
                "query_type": review.get("query_type") or quality.get("query_type", ""),
                "difficulty": review.get("difficulty", ""),
                "evidence_quality_score": quality.get("evidence_quality_score", 0.0),
                "evidence_quality_level": quality.get("evidence_quality_level", ""),
                "retrieval_confidence_score": quality.get("retrieval_confidence_score", 0.0),
                "citation_support_score": quality.get("citation_support_score", 0.0),
                "coverage_score": quality.get("coverage_score", 0.0),
                "source_diversity_score": quality.get("source_diversity_score", 0.0),
                "risk_penalty": quality.get("risk_penalty", 0.0),
                "evidence_quality_reasons": reasons,
                "evidence_quality_warnings": warnings,
                "needs_human_review": as_bool(quality.get("needs_human_review", False)),
                "insufficient_answer": as_bool(quality.get("insufficient_answer", False)),
                "evidence_sufficient": as_bool(quality.get("evidence_sufficient", True)),
                "llm_called": as_bool(quality.get("llm_called", False)),
                "supported_claim_ratio": quality.get("supported_claim_ratio", 0.0),
                "unsupported_claim_count": int_or_zero(quality.get("unsupported_claim_count", 0)),
                "weak_claim_count": int_or_zero(quality.get("weak_claim_count", 0)),
                "citation_check_passed": as_bool(quality.get("citation_check_passed", True)),
                "final_source_count": int_or_zero(quality.get("final_source_count", 0)),
                **manual_fields,
                "is_reviewed": is_reviewed,
                "manual_review_status": "reviewed" if is_reviewed else "unreviewed",
            }
        )
    return merged


def has_manual_review(fields: dict[str, str]) -> bool:
    return any(strip_cell(value) for value in fields.values())


def normalize_list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item for item in value.split("|") if item]
    return []


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_summary(
    merged_records: list[dict[str, Any]],
    quality_result: dict[str, Any],
    paths: dict[str, Path],
    config_path: Path,
    review_rows: list[dict[str, str]],
) -> dict[str, Any]:
    manual_judgments = Counter(
        record["manual_judgment"] for record in merged_records if record.get("manual_judgment")
    )
    manual_citations = Counter(
        record["manual_citation_judgment"]
        for record in merged_records
        if record.get("manual_citation_judgment")
    )
    manual_should_refuse = Counter(
        record["manual_should_refuse"] for record in merged_records if record.get("manual_should_refuse")
    )
    quality_levels = Counter(str(record.get("evidence_quality_level", "")) for record in merged_records)
    needs_review_ids = [
        record["id"] for record in merged_records if record.get("needs_human_review")
    ]
    unreviewed_ids = [record["id"] for record in merged_records if not record.get("is_reviewed")]
    reviewed_ids = [record["id"] for record in merged_records if record.get("is_reviewed")]

    quality_ids = {str(record.get("id", "")).strip() for record in merged_records}
    review_ids = {row["id"] for row in review_rows}
    checker_false_positive_ids = [
        record["id"]
        for record in merged_records
        if "checker_false_positive"
        in {record.get("manual_judgment", ""), record.get("manual_citation_judgment", "")}
    ]
    checker_false_negative_ids = [
        record["id"]
        for record in merged_records
        if "checker_false_negative"
        in {record.get("manual_judgment", ""), record.get("manual_citation_judgment", "")}
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "input_review_template_csv": str(paths["review_template_csv"]),
        "input_evidence_quality_json": str(paths["evidence_quality_json"]),
        "total_records": len(merged_records),
        "reviewed_count": len(reviewed_ids),
        "unreviewed_count": len(unreviewed_ids),
        "needs_human_review_count": len(needs_review_ids),
        "reviewed_needs_human_review_count": sum(
            1 for record in merged_records if record.get("needs_human_review") and record.get("is_reviewed")
        ),
        "manual_judgment_counts": dict(manual_judgments),
        "manual_citation_judgment_counts": dict(manual_citations),
        "manual_should_refuse_counts": dict(manual_should_refuse),
        "checker_false_positive_count": len(checker_false_positive_ids),
        "checker_false_negative_count": len(checker_false_negative_ids),
        "checker_false_positive_ids": checker_false_positive_ids,
        "checker_false_negative_ids": checker_false_negative_ids,
        "quality_level_counts": dict(quality_levels),
        "needs_human_review_ids": needs_review_ids,
        "reviewed_ids": reviewed_ids,
        "unreviewed_ids": unreviewed_ids,
        "missing_review_ids": sorted(quality_ids - review_ids),
        "extra_review_ids": sorted(review_ids - quality_ids),
        "v2c3_summary": quality_result.get("summary", {}),
        "outputs": {
            "merged_json": str(paths["merged_json"]),
            "merged_jsonl": str(paths["merged_jsonl"]),
            "merged_csv": str(paths["merged_csv"]),
            "summary_json": str(paths["summary_json"]),
            "summary_md": str(paths["summary_md"]),
        },
    }


def write_outputs(result: dict[str, Any], paths: dict[str, Path]) -> None:
    for key in ["merged_json", "merged_jsonl", "merged_csv", "summary_json", "summary_md"]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    paths["merged_json"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["merged_jsonl"].open("w", encoding="utf-8") as file:
        for record in result["records"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_merged_csv(paths["merged_csv"], result["records"])
    paths["summary_json"].write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["summary_md"].write_text(build_summary_markdown(result["summary"], result["records"]), encoding="utf-8")


def write_merged_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in CSV_FIELDNAMES}
            row["evidence_quality_warnings"] = "|".join(record.get("evidence_quality_warnings", []))
            writer.writerow(row)


def build_summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Manual Review Import Summary V2c.4",
        "",
        "## 总览",
        "",
        f"- total_records: {summary['total_records']}",
        f"- reviewed_count: {summary['reviewed_count']}",
        f"- unreviewed_count: {summary['unreviewed_count']}",
        f"- needs_human_review_count: {summary['needs_human_review_count']}",
        f"- reviewed_needs_human_review_count: {summary['reviewed_needs_human_review_count']}",
        f"- checker_false_positive_count: {summary['checker_false_positive_count']}",
        f"- checker_false_negative_count: {summary['checker_false_negative_count']}",
        "",
        "## 人工标注覆盖率",
        "",
        f"- reviewed: {summary['reviewed_count']} / {summary['total_records']}",
        f"- unreviewed: {summary['unreviewed_count']} / {summary['total_records']}",
        "",
        "## Needs Human Review 题目",
        "",
    ]
    for record in records:
        if not record.get("needs_human_review"):
            continue
        warnings = "|".join(record.get("evidence_quality_warnings", []))
        lines.append(
            f"- {record['id']}: level={record.get('evidence_quality_level', '')}, "
            f"score={record.get('evidence_quality_score', '')}, warnings={warnings}"
        )
    if not summary["needs_human_review_ids"]:
        lines.append("- None.")

    lines.extend(["", "## Quality Level 分布", ""])
    for level, count in summary["quality_level_counts"].items():
        lines.append(f"- {level}: {count}")

    lines.extend(
        [
            "",
            "## Checker False Positive / False Negative",
            "",
            f"- checker_false_positive_count: {summary['checker_false_positive_count']}",
            f"- checker_false_negative_count: {summary['checker_false_negative_count']}",
            f"- checker_false_positive_ids: {', '.join(summary['checker_false_positive_ids']) or 'None'}",
            f"- checker_false_negative_ids: {', '.join(summary['checker_false_negative_ids']) or 'None'}",
            "",
            "## 未审阅题目",
            "",
        ]
    )
    for record_id in summary["unreviewed_ids"]:
        lines.append(f"- {record_id}")
    if not summary["unreviewed_ids"]:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## 下一步人工审阅建议",
            "",
            "- 优先审阅 q003、q005、q006、q007，因为它们带有 needs_human_review warning。",
            "- 对每题填写 manual_judgment、manual_citation_judgment、manual_should_refuse 和 manual_notes。",
            "- 如果发现规则 checker 误判，使用 checker_false_positive 或 checker_false_negative 标注。",
            "- 本文件只汇总人工标签导入状态，不改变 V2c.2/V2c.3 原始结果。",
            "",
        ]
    )
    return "\n".join(lines)


def print_summary(summary: dict[str, Any]) -> None:
    print("Manual review labels imported.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
