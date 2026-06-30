from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v25d.yaml"


def main() -> int:
    configure_stdout()
    try:
        summary = export_review()
    except Exception as exc:  # noqa: BLE001
        print("V2.5d answer review export failed.")
        print(f"reason: {exc}")
        return 1
    print("V2.5d answer review export completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def export_review(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    outputs = config["outputs"]
    answer_eval_path = resolve_project_path(outputs["answer_eval_json"])
    quality_path = resolve_project_path(outputs["evidence_quality_json"])
    output_md = resolve_project_path(outputs["review_markdown"])
    output_csv = resolve_project_path(outputs["review_template_csv"])
    output_summary = resolve_project_path(outputs["review_summary_json"])
    ensure_inputs_exist([answer_eval_path, quality_path])
    answer_eval = load_json(answer_eval_path)
    quality_eval = load_json(quality_path)
    quality_by_id = {record["id"]: record for record in quality_eval.get("records", [])}
    records = sorted(
        answer_eval.get("records", []),
        key=lambda record: (not bool(record.get("needs_human_review", False)), str(record.get("id", ""))),
    )
    if not records:
        raise ValueError(f"No answer eval records found in {answer_eval_path}")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(answer_eval, quality_by_id, records), encoding="utf-8")
    write_csv_template(output_csv, quality_by_id, records)
    summary = build_summary(answer_eval, quality_eval, records, output_md, output_csv)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_markdown(
    answer_eval: dict[str, Any],
    quality_by_id: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
) -> str:
    metrics = answer_eval.get("metrics", {})
    lines: list[str] = [
        "# V2.5d Answer Review Template",
        "",
        "## Overview",
        "",
    ]
    for key in [
        "total_eval_questions",
        "negative_questions",
        "non_negative_questions",
        "llm_called_count",
        "negative_refusal_rate",
        "citation_check_pass_rate",
        "avg_supported_claim_ratio",
        "unsupported_claim_count",
        "weak_claim_count",
        "needs_human_review_count",
    ]:
        lines.append(f"- `{key}`: {metrics.get(key, '')}")
    lines.extend(
        [
            "",
            "## Manual Review Fields",
            "",
            "- `manual_judgment`: correct | partially_correct | incorrect | insufficient_evidence_correct | should_have_refused | checker_false_positive | checker_false_negative",
            "- `manual_citation_judgment`: fully_supported | partially_supported | unsupported | citation_not_needed | checker_false_positive | checker_false_negative",
            "- `manual_notes`: short explanation.",
            "",
            "## Review Cards",
            "",
        ]
    )
    for record in records:
        quality = quality_by_id.get(str(record.get("id", "")), {})
        lines.extend(build_record_card(record, quality))
    return "\n".join(lines).rstrip() + "\n"


def build_record_card(record: dict[str, Any], quality: dict[str, Any]) -> list[str]:
    marker = "REVIEW" if record.get("needs_human_review") else "OK"
    lines = [
        f"### [{marker}] {record.get('id', '')} - {record.get('query_type', '')}",
        "",
        f"- `question`: {record.get('question', '')}",
        f"- `difficulty`: {record.get('difficulty', '')}",
        f"- `evidence_sufficient`: {record.get('evidence_sufficient', '')}",
        f"- `evidence_gate_reason`: {record.get('evidence_gate_reason', '')}",
        f"- `llm_called`: {record.get('llm_called', '')}",
        f"- `insufficient_answer`: {record.get('insufficient_answer', '')}",
        f"- `needs_human_review`: {record.get('needs_human_review', '')}",
        f"- `evidence_quality_score`: {quality.get('evidence_quality_score', '')}",
        f"- `evidence_quality_level`: {quality.get('evidence_quality_level', '')}",
        f"- `evidence_quality_warnings`: {quality.get('evidence_quality_warnings', [])}",
        "",
        "**Answer**",
        "",
        "```text",
        str(record.get("answer", "")).strip(),
        "```",
        "",
        "**Final Sources (Top 5)**",
        "",
    ]
    for source in record.get("final_sources", [])[:5]:
        lines.append(
            "- "
            f"rank={source.get('rank', '')}; "
            f"category={source.get('category_dir', '')}; "
            f"doc_type={source.get('doc_type', '')}; "
            f"source={source.get('imported_source') or source.get('source', '')}; "
            f"score={source.get('final_score', '')}; "
            f"why={one_line(str(source.get('why_selected', '')), 180)}"
        )
    if not record.get("final_sources"):
        lines.append("- None.")
    lines.extend(
        [
            "",
            "**Citation Summary**",
            "",
            f"- `supported_claim_ratio`: {record.get('supported_claim_ratio', '')}",
            f"- `checked_claim_count`: {record.get('checked_claim_count', '')}",
            f"- `ignored_claim_count`: {record.get('ignored_claim_count', '')}",
            f"- `weak_claim_count`: {record.get('weak_claim_count', '')}",
            f"- `unsupported_claim_count`: {record.get('unsupported_claim_count', '')}",
            "",
            "**Unsupported Claims**",
            "",
        ]
    )
    lines.extend(format_claims(record.get("unsupported_claims", []), "- None."))
    lines.extend(["", "**Weak Claims**", ""])
    lines.extend(format_claims(record.get("weak_claims", []), "- None."))
    lines.extend(["", "**Ignored Claims (first 5)**", ""])
    lines.extend(format_claims(record.get("ignored_claims", [])[:5], "- None."))
    lines.extend(
        [
            "",
            "**Manual Fill Area**",
            "",
            "- `manual_judgment`: ",
            "- `manual_citation_judgment`: ",
            "- `manual_should_refuse`: ",
            "- `manual_notes`: ",
            "",
            "---",
            "",
        ]
    )
    return lines


def format_claims(claims: list[dict[str, Any]], empty_text: str) -> list[str]:
    if not claims:
        return [empty_text]
    return [
        f"- {index}. type={claim.get('claim_type', '')}; hint={claim.get('checker_error_hint', '')}; text={one_line(str(claim.get('text') or claim.get('claim') or ''), 260)}"
        for index, claim in enumerate(claims, start=1)
    ]


def write_csv_template(path: Path, quality_by_id: dict[str, dict[str, Any]], records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question",
        "query_type",
        "difficulty",
        "evidence_sufficient",
        "insufficient_answer",
        "needs_human_review",
        "supported_claim_ratio",
        "unsupported_claim_count",
        "weak_claim_count",
        "evidence_quality_score",
        "evidence_quality_level",
        "evidence_quality_warnings",
        "answer_short",
        "top_sources_short",
        "manual_judgment",
        "manual_citation_judgment",
        "manual_should_refuse",
        "manual_notes",
        "reviewer",
        "reviewed_at",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            quality = quality_by_id.get(str(record.get("id", "")), {})
            writer.writerow(
                {
                    "id": record.get("id", ""),
                    "question": record.get("question", ""),
                    "query_type": record.get("query_type", ""),
                    "difficulty": record.get("difficulty", ""),
                    "evidence_sufficient": record.get("evidence_sufficient", ""),
                    "insufficient_answer": record.get("insufficient_answer", ""),
                    "needs_human_review": record.get("needs_human_review", ""),
                    "supported_claim_ratio": record.get("supported_claim_ratio", ""),
                    "unsupported_claim_count": record.get("unsupported_claim_count", ""),
                    "weak_claim_count": record.get("weak_claim_count", ""),
                    "evidence_quality_score": quality.get("evidence_quality_score", ""),
                    "evidence_quality_level": quality.get("evidence_quality_level", ""),
                    "evidence_quality_warnings": "|".join(quality.get("evidence_quality_warnings", [])),
                    "answer_short": one_line(str(record.get("answer", "")), 320),
                    "top_sources_short": summarize_sources(record.get("final_sources", [])[:5]),
                    "manual_judgment": "",
                    "manual_citation_judgment": "",
                    "manual_should_refuse": "",
                    "manual_notes": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )


def build_summary(
    answer_eval: dict[str, Any],
    quality_eval: dict[str, Any],
    records: list[dict[str, Any]],
    output_md: Path,
    output_csv: Path,
) -> dict[str, Any]:
    metrics = answer_eval.get("metrics", {})
    quality_summary = quality_eval.get("summary", {})
    needs_review = [record for record in records if record.get("needs_human_review")]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_questions": len(records),
        "needs_human_review_count": len(needs_review),
        "needs_human_review_ids": [str(record.get("id", "")) for record in needs_review],
        "negative_questions": int(metrics.get("negative_questions", 0)),
        "insufficient_answer_count": int(metrics.get("insufficient_answer_count", 0)),
        "unsupported_claim_count": int(metrics.get("unsupported_claim_count", 0)),
        "weak_claim_count": int(metrics.get("weak_claim_count", 0)),
        "quality_level_counts": quality_summary.get("level_counts", {}),
        "output_markdown": str(output_md),
        "output_csv_template": str(output_csv),
    }


def ensure_inputs_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required V25d review input files: {missing}")


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


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def summarize_sources(sources: list[dict[str, Any]]) -> str:
    return " | ".join(
        f"{source.get('rank', '')}:{source.get('category_dir', '')}:{source.get('imported_source') or source.get('source', '')}"
        for source in sources
    )


def one_line(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


if __name__ == "__main__":
    sys.exit(main())
