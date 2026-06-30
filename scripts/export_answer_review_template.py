from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = PROJECT_ROOT / "storage" / "logs" / "answer_eval_v2c1.json"
INPUT_JSONL = PROJECT_ROOT / "storage" / "logs" / "answer_eval_v2c1.jsonl"
INPUT_CSV = PROJECT_ROOT / "data" / "eval" / "answer_eval_v2c1_results.csv"
OUTPUT_MD = PROJECT_ROOT / "data" / "eval" / "answer_review_v2c2.md"
OUTPUT_CSV = PROJECT_ROOT / "data" / "eval" / "answer_review_v2c2_template.csv"
OUTPUT_JSON = PROJECT_ROOT / "storage" / "logs" / "answer_review_v2c2_summary.json"

REVIEW_JUDGMENTS = [
    "correct",
    "partially_correct",
    "incorrect",
    "insufficient_evidence_correct",
    "should_have_refused",
    "checker_false_positive",
    "checker_false_negative",
]


def main() -> int:
    configure_stdout()
    try:
        summary = export_review_template()
    except Exception as exc:  # noqa: BLE001 - export should fail explicitly.
        print("Answer review template export failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(summary)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def export_review_template(
    input_json: Path = INPUT_JSON,
    input_jsonl: Path = INPUT_JSONL,
    input_csv: Path = INPUT_CSV,
    output_md: Path = OUTPUT_MD,
    output_csv: Path = OUTPUT_CSV,
    output_json: Path = OUTPUT_JSON,
) -> dict[str, Any]:
    ensure_inputs_exist([input_json, input_jsonl, input_csv])
    result = load_json(input_json)
    records = sorted(
        result.get("records", []),
        key=lambda record: (not bool(record.get("needs_human_review", False)), str(record.get("id", ""))),
    )
    if not records:
        raise ValueError(f"No records found in {input_json}")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_md.write_text(build_markdown(result, records), encoding="utf-8")
    write_csv_template(output_csv, records)
    summary = build_summary(result, records, output_md, output_csv, input_jsonl, input_csv)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def ensure_inputs_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required V2c.1 answer eval files: {missing}")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def build_markdown(result: dict[str, Any], records: list[dict[str, Any]]) -> str:
    metrics = result.get("metrics", {})
    lines: list[str] = [
        "# Answer Evaluation Human Review Template V2c.2",
        "",
        "## 总览",
        "",
    ]
    overview_fields = [
        "total_eval_questions",
        "negative_questions",
        "non_negative_questions",
        "negative_refusal_rate",
        "citation_check_pass_rate",
        "avg_supported_claim_ratio",
        "unsupported_claim_count",
        "weak_claim_count",
        "needs_human_review_count",
    ]
    for field in overview_fields:
        lines.append(f"- `{field}`: {metrics.get(field, '')}")

    lines.extend(
        [
            "",
            "## 人工审阅说明",
            "",
            "建议 `manual_judgment` 取值：",
            "",
        ]
    )
    for value in REVIEW_JUDGMENTS:
        lines.append(f"- `{value}`")
    lines.extend(
        [
            "",
            "建议备注写法：简短说明答案事实是否正确、引用是否支持、是否需要拒答、checker 是否误判。",
            "",
            "## 逐题审阅卡片",
            "",
        ]
    )

    for record in records:
        lines.extend(build_record_card(record))
    return "\n".join(lines).rstrip() + "\n"


def build_record_card(record: dict[str, Any]) -> list[str]:
    review_marker = "REVIEW" if record.get("needs_human_review") else "OK"
    lines = [
        f"### [{review_marker}] {record.get('id', '')} - {record.get('query_type', '')}",
        "",
        f"- `question_id`: {record.get('id', '')}",
        f"- `question`: {record.get('question', '')}",
        f"- `query_type`: {record.get('query_type', '')}",
        f"- `difficulty`: {record.get('difficulty', '')}",
        f"- `evidence_sufficient`: {record.get('evidence_sufficient', '')}",
        f"- `llm_called`: {record.get('llm_called', '')}",
        f"- `insufficient_answer`: {record.get('insufficient_answer', '')}",
        f"- `needs_human_review`: {record.get('needs_human_review', '')}",
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
    sources = record.get("final_sources", [])[:5]
    if not sources:
        lines.append("- No final sources.")
    for source in sources:
        lines.append(
            "- "
            f"rank={source.get('rank', '')}; "
            f"source={source.get('source', '')}; "
            f"category_dir={source.get('category_dir', '')}; "
            f"doc_type={source.get('doc_type', '')}; "
            f"imported_source={source.get('imported_source', '')}"
        )

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
    lines.extend(format_claims(record.get("unsupported_claims", []), empty_text="- None."))
    lines.extend(["", "**Weak Claims**", ""])
    lines.extend(format_claims(record.get("weak_claims", []), empty_text="- None."))
    lines.extend(["", "**Ignored Claims (summary)**", ""])
    lines.extend(format_claims(record.get("ignored_claims", [])[:5], empty_text="- None.", compact=True))
    lines.extend(
        [
            "",
            "**人工填写区**",
            "",
            "- `manual_judgment`: ",
            "- `manual_citation_judgment`: ",
            "- `manual_notes`: ",
            "",
            "---",
            "",
        ]
    )
    return lines


def format_claims(claims: list[dict[str, Any]], *, empty_text: str, compact: bool = False) -> list[str]:
    if not claims:
        return [empty_text]
    lines: list[str] = []
    for index, claim in enumerate(claims, start=1):
        text = one_line(str(claim.get("text") or claim.get("claim") or ""), limit=280)
        if compact:
            lines.append(f"- {index}. `{claim.get('claim_type', '')}`: {text}")
            continue
        lines.append(
            "- "
            f"{index}. type={claim.get('claim_type', '')}; "
            f"hint={claim.get('checker_error_hint', '')}; "
            f"supported={claim.get('supported', '')}; "
            f"text={text}"
        )
        support_reason = one_line(str(claim.get("support_reason", "")), limit=220)
        if support_reason:
            lines.append(f"  - support_reason: {support_reason}")
    return lines


def write_csv_template(path: Path, records: list[dict[str, Any]]) -> None:
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
        "answer_short",
        "top_sources_short",
        "manual_judgment",
        "manual_citation_judgment",
        "manual_should_refuse",
        "manual_notes",
        "reviewer",
        "reviewed_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
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
                    "answer_short": one_line(str(record.get("answer", "")), limit=320),
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
    result: dict[str, Any],
    records: list[dict[str, Any]],
    output_md: Path,
    output_csv: Path,
    input_jsonl: Path,
    input_csv: Path,
) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    needs_human_review = [record for record in records if record.get("needs_human_review")]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_json": str(INPUT_JSON),
        "input_jsonl": str(input_jsonl),
        "input_csv": str(input_csv),
        "total_questions": len(records),
        "needs_human_review_count": len(needs_human_review),
        "no_review_needed_count": len(records) - len(needs_human_review),
        "negative_questions": int(metrics.get("negative_questions", 0)),
        "insufficient_answer_count": int(metrics.get("insufficient_answer_count", 0)),
        "unsupported_claim_count": int(metrics.get("unsupported_claim_count", 0)),
        "weak_claim_count": int(metrics.get("weak_claim_count", 0)),
        "needs_human_review_ids": [str(record.get("id", "")) for record in needs_human_review],
        "output_markdown": str(output_md),
        "output_csv_template": str(output_csv),
    }


def summarize_sources(sources: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for source in sources:
        parts.append(
            f"{source.get('rank', '')}:{source.get('category_dir', '')}:{source.get('imported_source') or source.get('source', '')}"
        )
    return " | ".join(parts)


def one_line(text: str, *, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def print_summary(summary: dict[str, Any]) -> None:
    print("Answer review template export completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
