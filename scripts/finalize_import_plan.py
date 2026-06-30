from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "import_selection_final.yaml"

FINAL_PLAN_FIELDS = [
    "original_path",
    "relative_path",
    "filename",
    "extension",
    "size_bytes",
    "inferred_category",
    "selection_rank",
    "selection_score",
    "selected_reason",
    "destination_category",
    "final_keep_reason",
]

EXACT_SEGMENT_KEYWORDS = {"env", "venv", ".venv", ".git", "cache", "runs", "outputs", "checkpoints", "wandb"}
SNAPSHOT_KEYWORD = "scripts/results/part2/code_snapshots"
BAD_TERM_CHECKS = [
    ".jsbsim_data_tmp",
    "controller_state",
    "tokenizer",
    "stderr",
    "stdout",
    "figures/pdf",
    "figure",
    "state_final",
]
LOW_PRIORITY_PENALTY = {
    ".pdf": 12.0,
    ".tex": 10.0,
    ".json": 8.0,
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid final import selection config: {path}")
    return config


def read_plan(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing curated import plan: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def finalize_import_plan(config: dict[str, Any]) -> dict[str, Any]:
    input_plan = PROJECT_ROOT / str(config["input_plan"])
    output_plan = PROJECT_ROOT / str(config["output_plan"])
    output_report = PROJECT_ROOT / str(config["output_report"])

    rows = read_plan(input_plan)
    quotas = {str(key): int(value) for key, value in dict(config.get("category_quotas", {})).items()}
    max_total = int(config.get("max_total_files", 220))
    min_total = int(config.get("min_total_files", 180))
    mandatory_excludes = [str(item).lower() for item in config.get("mandatory_exclude_path_keywords", [])]
    default_excludes = [str(item).lower() for item in config.get("default_exclude_path_keywords", [])]
    prefer_keywords = [str(item) for item in config.get("prefer_path_keywords", [])]
    preferred_extensions = [str(item).lower() for item in config.get("preferred_extensions", [])]

    filtered_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusion_reasons: Counter[str] = Counter()
    excluded_examples: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        category = str(row.get("destination_category") or row.get("inferred_category") or "unknown_text")
        if quotas.get(category, 0) <= 0:
            record_exclusion(exclusion_reasons, excluded_examples, "category_quota_zero_or_missing", row)
            continue

        relative_path = str(row.get("relative_path") or "")
        mandatory_match = matched_path_keyword(relative_path, mandatory_excludes)
        if mandatory_match:
            record_exclusion(exclusion_reasons, excluded_examples, f"mandatory_keyword:{mandatory_match}", row)
            continue

        default_match = matched_path_keyword(relative_path, default_excludes)
        if default_match:
            record_exclusion(exclusion_reasons, excluded_examples, f"default_keyword:{default_match}", row)
            continue

        enriched = dict(row)
        score, reason = score_final_candidate(
            row=row,
            prefer_keywords=prefer_keywords,
            preferred_extensions=preferred_extensions,
        )
        enriched["_score"] = score
        enriched["_final_keep_reason"] = reason
        filtered_by_category[category].append(enriched)

    selected: list[dict[str, Any]] = []
    quota_filled: dict[str, int] = {}
    for category, quota in quotas.items():
        category_rows = filtered_by_category.get(category, [])
        category_rows.sort(
            key=lambda item: (
                -float(item["_score"]),
                parse_int(item.get("size_bytes")),
                str(item.get("relative_path", "")).lower(),
            )
        )
        chosen = category_rows[:quota]
        selected.extend(chosen)
        quota_filled[category] = len(chosen)

    selected.sort(
        key=lambda item: (
            category_order(str(item.get("destination_category") or item.get("inferred_category"))),
            -float(item["_score"]),
            str(item.get("relative_path", "")).lower(),
        )
    )
    selected = selected[:max_total]

    plan_rows = []
    for rank, row in enumerate(selected, start=1):
        plan_rows.append(
            {
                "original_path": row.get("original_path", ""),
                "relative_path": row.get("relative_path", ""),
                "filename": row.get("filename", ""),
                "extension": str(row.get("extension", "")).lower(),
                "size_bytes": parse_int(row.get("size_bytes")),
                "inferred_category": row.get("inferred_category", "unknown_text"),
                "selection_rank": rank,
                "selection_score": round(float(row["_score"]), 4),
                "selected_reason": row.get("selected_reason", ""),
                "destination_category": row.get("destination_category") or row.get("inferred_category", "unknown_text"),
                "final_keep_reason": row["_final_keep_reason"],
            }
        )

    write_csv(output_plan, plan_rows, FINAL_PLAN_FIELDS)
    summary = build_summary(
        input_rows=rows,
        plan_rows=plan_rows,
        input_plan=input_plan,
        output_plan=output_plan,
        config=config,
        exclusion_reasons=exclusion_reasons,
        excluded_examples=excluded_examples,
        quota_filled=quota_filled,
        min_total=min_total,
        max_total=max_total,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def score_final_candidate(
    *,
    row: dict[str, str],
    prefer_keywords: list[str],
    preferred_extensions: list[str],
) -> tuple[float, str]:
    relative_path = str(row.get("relative_path") or "")
    lowered = relative_path.replace("\\", "/").lower()
    extension = str(row.get("extension") or "").lower()
    previous_score = parse_float(row.get("selection_score"))
    score = previous_score
    matched_prefer = [keyword for keyword in prefer_keywords if keyword.lower() in lowered]

    score += len(matched_prefer) * 4.0
    if extension in preferred_extensions:
        score += max(0.0, 18.0 - preferred_extensions.index(extension) * 2.0)
    score -= LOW_PRIORITY_PENALTY.get(extension, 0.0)

    if any(keyword in lowered for keyword in ["readme", "docs", "report", "summary", "conclusion", "audit"]):
        score += 18.0
    if "train_env0_episodes.csv" in lowered:
        score -= 18.0
    if extension == ".pdf" and any(keyword in lowered for keyword in ["figure", "figures", "plot", "chart"]):
        score -= 30.0
    if extension == ".tex" and any(keyword in lowered for keyword in ["snippet", "section"]):
        score -= 16.0

    reason_parts = ["final_quota", "clean_path", "from_v2a1_plan"]
    if matched_prefer:
        reason_parts.append("matched_keywords:" + "|".join(matched_prefer[:8]))
    if extension:
        reason_parts.append(f"extension:{extension}")
    return score, ";".join(reason_parts)


def matched_path_keyword(relative_path: str, keywords: list[str]) -> str:
    normalized = relative_path.replace("\\", "/")
    lowered = normalized.lower()
    parts = {part.lower() for part in normalized.split("/")}
    for keyword in keywords:
        if keyword in EXACT_SEGMENT_KEYWORDS:
            if keyword in parts:
                return keyword
            continue
        if keyword in lowered:
            return keyword
    return ""


def record_exclusion(
    exclusion_reasons: Counter[str],
    excluded_examples: dict[str, list[str]],
    reason: str,
    row: dict[str, str],
) -> None:
    exclusion_reasons[reason] += 1
    examples = excluded_examples[reason]
    if len(examples) < 10:
        examples.append(str(row.get("relative_path") or ""))


def parse_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def category_order(category: str) -> int:
    order = {
        "code": 0,
        "configs": 1,
        "experiments": 2,
        "notes": 3,
        "papers": 4,
        "thesis_or_reports": 5,
        "unknown_text": 6,
    }
    return order.get(category, 99)


def root_project(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    return normalized.split("/", 1)[0] if normalized else ""


def count_path_hits(rows: list[dict[str, Any]], terms: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for term in terms:
        lowered_term = term.lower()
        counts[term] = sum(
            lowered_term in str(row.get("relative_path", "")).replace("\\", "/").lower()
            for row in rows
        )
    return counts


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    *,
    input_rows: list[dict[str, str]],
    plan_rows: list[dict[str, Any]],
    input_plan: Path,
    output_plan: Path,
    config: dict[str, Any],
    exclusion_reasons: Counter[str],
    excluded_examples: dict[str, list[str]],
    quota_filled: dict[str, int],
    min_total: int,
    max_total: int,
) -> dict[str, Any]:
    selected_size_sorted = sorted(plan_rows, key=lambda item: parse_int(item["size_bytes"]), reverse=True)
    quota_targets = {str(key): int(value) for key, value in dict(config.get("category_quotas", {})).items()}
    quota_shortfalls = {
        category: max(0, quota_targets.get(category, 0) - quota_filled.get(category, 0))
        for category in quota_targets
    }
    snapshot_kept = [
        {
            "relative_path": row["relative_path"],
            "final_keep_reason": row["final_keep_reason"],
        }
        for row in plan_rows
        if SNAPSHOT_KEYWORD in str(row["relative_path"]).replace("\\", "/").lower()
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_environment": "Lenginzed_RAG",
        "input_plan": str(input_plan),
        "output_plan": str(output_plan),
        "original_selected_count": len(input_rows),
        "final_selected_count": len(plan_rows),
        "removed_count": len(input_rows) - len(plan_rows),
        "min_total_files": min_total,
        "max_total_files": max_total,
        "within_target_range": min_total <= len(plan_rows) <= max_total,
        "selected_category_counts": dict(Counter(str(row["destination_category"]) for row in plan_rows)),
        "selected_extension_counts": dict(Counter(str(row["extension"]) or "(none)" for row in plan_rows)),
        "selected_root_project_counts": dict(Counter(root_project(str(row["relative_path"])) for row in plan_rows)),
        "excluded_keyword_counts": dict(exclusion_reasons.most_common()),
        "excluded_keyword_examples": {key: value for key, value in sorted(excluded_examples.items())},
        "quota_targets": quota_targets,
        "quota_filled": quota_filled,
        "quota_shortfalls": quota_shortfalls,
        "mandatory_bad_term_counts_final": count_path_hits(plan_rows, BAD_TERM_CHECKS),
        "code_snapshots_final_count": len(snapshot_kept),
        "code_snapshots_kept_reasons": snapshot_kept,
        "largest_selected_files": [
            {
                "relative_path": row["relative_path"],
                "size_bytes": row["size_bytes"],
                "inferred_category": row["inferred_category"],
                "extension": row["extension"],
                "selection_rank": row["selection_rank"],
                "selection_score": row["selection_score"],
            }
            for row in selected_size_sorted[:20]
        ],
        "real_copy_executed": False,
        "external_source_modified": False,
        "raw_imported_exists": (PROJECT_ROOT / "data" / "raw_imported").exists(),
    }


def main() -> int:
    try:
        config = load_config()
        summary = finalize_import_plan(config)
    except Exception as exc:  # noqa: BLE001 - script should surface data/config failures.
        print("Final import plan generation failed.")
        print(f"reason: {exc}")
        return 1

    print("Final import plan generation completed.")
    print(f"environment: {summary['used_environment']}")
    print(f"input_plan: {summary['input_plan']}")
    print(f"original_selected_count: {summary['original_selected_count']}")
    print(f"final_selected_count: {summary['final_selected_count']}")
    print(f"removed_count: {summary['removed_count']}")
    print(f"within_target_range: {summary['within_target_range']}")
    print("selected_category_counts:")
    for category, count in summary["selected_category_counts"].items():
        print(f"  {category}: {count}")
    print(f"code_snapshots_final_count: {summary['code_snapshots_final_count']}")
    print(f"bad_term_counts_final: {summary['mandatory_bad_term_counts_final']}")
    print(f"output_plan: {summary['output_plan']}")
    print("No files were copied and no external source files were modified.")
    return 0 if summary["within_target_range"] else 2


if __name__ == "__main__":
    sys.exit(main())
