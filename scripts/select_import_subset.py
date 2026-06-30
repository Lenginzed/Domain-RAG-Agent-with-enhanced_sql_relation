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
CONFIG_PATH = PROJECT_ROOT / "config" / "import_selection.yaml"

PLAN_FIELDS = [
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
]

EXACT_SEGMENT_KEYWORDS = {"env", "venv", ".venv", ".git", "cache", "runs", "outputs", "checkpoints", "wandb"}
LOW_PRIORITY_EXTENSION_PENALTY = {
    ".pdf": 20,
    ".tex": 8,
    ".json": 6,
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid import selection config: {path}")
    return config


def read_candidates(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def select_import_subset(config: dict[str, Any]) -> dict[str, Any]:
    input_path = PROJECT_ROOT / str(config["input_candidates"])
    output_plan = PROJECT_ROOT / str(config["output_plan"])
    output_report = PROJECT_ROOT / str(config["output_report"])

    candidates = read_candidates(input_path)
    quotas = {str(key): int(value) for key, value in dict(config.get("category_quotas", {})).items()}
    max_total = int(config.get("max_total_files", 250))
    max_size_by_category = {
        str(key): int(float(value) * 1024 * 1024)
        for key, value in dict(config.get("max_file_size_mb_by_category", {})).items()
    }
    preferred_exts = {
        str(category): [str(ext).lower() for ext in values]
        for category, values in dict(config.get("preferred_extensions_by_category", {})).items()
    }
    exclude_keywords = [str(item).lower() for item in config.get("exclude_path_keywords", [])]
    prefer_keywords = [str(item) for item in config.get("prefer_path_keywords", [])]

    filtered_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusion_reasons: Counter[str] = Counter()
    seen_duplicate_keys: set[tuple[str, int]] = set()

    for row in candidates:
        category = str(row.get("inferred_category") or "unknown_text")
        if quotas.get(category, 0) <= 0:
            exclusion_reasons["category_quota_zero_or_missing"] += 1
            continue

        extension = str(row.get("extension") or "").lower()
        size_bytes = parse_int(row.get("size_bytes"))
        relative_path = str(row.get("relative_path") or "")
        filename = str(row.get("filename") or Path(relative_path).name)

        max_size = max_size_by_category.get(category, 1024 * 1024)
        if size_bytes > max_size:
            exclusion_reasons[f"size_limit_{category}"] += 1
            continue

        excluded_keyword = matched_exclude_keyword(relative_path, exclude_keywords)
        if excluded_keyword:
            exclusion_reasons[f"exclude_keyword:{excluded_keyword}"] += 1
            continue

        allowed_exts = preferred_exts.get(category, [])
        if allowed_exts and extension not in allowed_exts:
            exclusion_reasons[f"extension_not_preferred_{category}"] += 1
            continue

        duplicate_key = (filename.lower(), size_bytes)
        if duplicate_key in seen_duplicate_keys:
            exclusion_reasons["duplicate_filename_and_size"] += 1
            continue
        seen_duplicate_keys.add(duplicate_key)

        score, reason = score_candidate(
            row=row,
            category=category,
            extension=extension,
            size_bytes=size_bytes,
            allowed_exts=allowed_exts,
            prefer_keywords=prefer_keywords,
        )
        enriched = dict(row)
        enriched["_score"] = score
        enriched["_selected_reason"] = reason
        filtered_by_category[category].append(enriched)

    selected: list[dict[str, Any]] = []
    for category, quota in quotas.items():
        category_rows = filtered_by_category.get(category, [])
        category_rows.sort(
            key=lambda item: (
                -float(item["_score"]),
                parse_int(item.get("size_bytes")),
                str(item.get("relative_path", "")).lower(),
            )
        )
        selected.extend(category_rows[:quota])

    selected.sort(
        key=lambda item: (
            category_order(str(item.get("inferred_category"))),
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
                "selected_reason": row["_selected_reason"],
                "destination_category": row.get("inferred_category", "unknown_text"),
            }
        )

    write_csv(output_plan, plan_rows, PLAN_FIELDS)
    summary = build_summary(
        candidates=candidates,
        plan_rows=plan_rows,
        exclusion_reasons=exclusion_reasons,
        output_plan=output_plan,
        config=config,
    )
    output_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def score_candidate(
    *,
    row: dict[str, str],
    category: str,
    extension: str,
    size_bytes: int,
    allowed_exts: list[str],
    prefer_keywords: list[str],
) -> tuple[float, str]:
    relative_path = str(row.get("relative_path") or "")
    lowered = relative_path.lower()
    score = 1000.0
    matched_prefer = [keyword for keyword in prefer_keywords if keyword.lower() in lowered]

    if allowed_exts and extension in allowed_exts:
        score += max(0, 120 - allowed_exts.index(extension) * 20)
    score += len(matched_prefer) * 12
    score += max(0, 80 - min(size_bytes, 8 * 1024 * 1024) / (1024 * 1024) * 8)
    score -= LOW_PRIORITY_EXTENSION_PENALTY.get(extension, 0)

    if category in {"papers", "thesis_or_reports"} and "main.tex" in lowered:
        score += 30
    if category == "experiments" and extension in {".md", ".csv", ".txt"}:
        score += 30
    if category == "code" and extension == ".py":
        score += 25
    if category == "configs" and extension in {".yaml", ".yml"}:
        score += 25

    reason_parts = [
        "category_quota",
        "size_ok",
        f"preferred_extension:{extension or '(none)'}",
    ]
    if matched_prefer:
        reason_parts.append("matched_keywords:" + "|".join(matched_prefer[:8]))
    return score, ";".join(reason_parts)


def matched_exclude_keyword(relative_path: str, exclude_keywords: list[str]) -> str:
    normalized = relative_path.replace("\\", "/")
    lowered = normalized.lower()
    parts = {part.lower() for part in normalized.split("/")}
    for keyword in exclude_keywords:
        if keyword in EXACT_SEGMENT_KEYWORDS:
            if keyword in parts:
                return keyword
            continue
        if keyword in lowered:
            return keyword
    return ""


def parse_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    *,
    candidates: list[dict[str, str]],
    plan_rows: list[dict[str, Any]],
    exclusion_reasons: Counter[str],
    output_plan: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    selected_size_sorted = sorted(plan_rows, key=lambda item: parse_int(item["size_bytes"]), reverse=True)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_environment": "Lenginzed_RAG",
        "input_candidates": str(PROJECT_ROOT / str(config["input_candidates"])),
        "output_plan": str(output_plan),
        "original_candidate_count": len(candidates),
        "selected_count": len(plan_rows),
        "max_total_files": int(config.get("max_total_files", 250)),
        "selected_category_counts": dict(Counter(str(row["inferred_category"]) for row in plan_rows)),
        "selected_extension_counts": dict(Counter(str(row["extension"]) or "(none)" for row in plan_rows)),
        "selected_root_project_counts": dict(Counter(root_project(str(row["relative_path"])) for row in plan_rows)),
        "exclusion_reason_counts": dict(exclusion_reasons.most_common()),
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
    }


def main() -> int:
    try:
        config = load_config()
        summary = select_import_subset(config)
    except Exception as exc:  # noqa: BLE001 - script should surface data/config failures.
        print("Curated import selection failed.")
        print(f"reason: {exc}")
        return 1

    print("Curated import selection completed.")
    print(f"environment: {summary['used_environment']}")
    print(f"input_candidates: {summary['input_candidates']}")
    print(f"original_candidate_count: {summary['original_candidate_count']}")
    print(f"selected_count: {summary['selected_count']}")
    print("selected_category_counts:")
    for category, count in summary["selected_category_counts"].items():
        print(f"  {category}: {count}")
    print(f"output_plan: {summary['output_plan']}")
    print("No files were copied and no external source files were modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
