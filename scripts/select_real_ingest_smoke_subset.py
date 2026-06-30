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
CONFIG_PATH = PROJECT_ROOT / "config" / "real_ingest_smoke.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SELECTED_FIELDS = [
    "selection_rank",
    "relative_path",
    "category_dir",
    "extension",
    "file_size_bytes",
    "document_count",
    "total_text_chars",
    "chunk_count",
    "selection_score",
    "selected_reason",
    "original_source",
]


def main() -> int:
    try:
        config = load_config()
        summary = select_subset(config)
    except Exception as exc:  # noqa: BLE001 - selection failures should be explicit.
        print("Real ingest smoke subset selection failed.")
        print(f"reason: {exc}")
        return 1

    print("Real ingest smoke subset selection completed.")
    print(f"environment: {summary['used_environment']}")
    print(f"candidate_files: {summary['candidate_files']}")
    print(f"selected_files: {summary['selected_files']}")
    print(f"selected_chunks: {summary['selected_chunks']}")
    print("selected_category_counts:")
    for category, count in summary["selected_category_counts"].items():
        print(f"  {category}: {count}")
    print("selected_extension_counts:")
    for extension, count in summary["selected_extension_counts"].items():
        print(f"  {extension}: {count}")
    print(f"excluded_issue_file_count: {summary['excluded_issue_file_count']}")
    print(f"excluded_pattern_file_count: {summary['excluded_pattern_file_count']}")
    print(f"output_selected_files: {summary['output_selected_files']}")
    print("No files were embedded and no Chroma index was written.")
    return 0


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def select_subset(config: dict[str, Any]) -> dict[str, Any]:
    audit_dir = PROJECT_ROOT / str(config["audit_dir"])
    document_rows = read_csv(audit_dir / "document_load_stats.csv")
    chunk_rows = read_csv(audit_dir / "chunk_stats.csv")
    load_error_rows = read_csv(audit_dir / "load_errors.csv")
    mapping = load_original_source_mapping(PROJECT_ROOT / str(config["copy_mapping_csv"]))

    output_selected = PROJECT_ROOT / str(config["output_selected_files"])
    output_summary = PROJECT_ROOT / str(config["output_selection_summary"])
    max_files_total = int(config.get("max_files_total", 30))
    max_chunks_total = int(config.get("max_chunks_total", 1000))
    quotas = {str(key): int(value) for key, value in dict(config.get("category_quotas", {})).items()}
    exclude_issue_types = {str(item) for item in config.get("exclude_issue_types", [])}
    exclude_patterns = [str(item).lower() for item in config.get("exclude_filename_patterns", [])]
    prefer_extensions = [str(item).lower() for item in config.get("prefer_extensions", [])]
    must_include_patterns = [str(item).lower() for item in config.get("must_include_path_patterns", [])]
    keyword_priorities = {
        str(category): [str(keyword).lower() for keyword in keywords]
        for category, keywords in dict(config.get("category_keyword_priorities", {})).items()
    }

    chunk_by_path = {row["relative_path"]: row for row in chunk_rows}
    excluded_by_issue = {
        row["relative_path"]
        for row in load_error_rows
        if str(row.get("issue_type", "")) in exclude_issue_types
    }

    candidates_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusion_counts: Counter[str] = Counter()
    for row in document_rows:
        relative_path = str(row.get("relative_path") or "")
        category = str(row.get("category_dir") or "")
        extension = str(row.get("extension") or "").lower()
        if row.get("load_status") != "loaded":
            exclusion_counts["not_loaded"] += 1
            continue
        if relative_path in excluded_by_issue:
            exclusion_counts["audit_issue"] += 1
            continue
        pattern = matched_pattern(relative_path, exclude_patterns)
        if pattern:
            exclusion_counts[f"filename_pattern:{pattern}"] += 1
            continue
        if quotas.get(category, 0) <= 0:
            exclusion_counts["category_quota_zero_or_missing"] += 1
            continue

        chunk_row = chunk_by_path.get(relative_path, {})
        chunk_count = parse_int(chunk_row.get("chunk_count"))
        score, reason = score_candidate(
            relative_path=relative_path,
            category=category,
            extension=extension,
            chunk_count=chunk_count,
            total_text_chars=parse_int(row.get("total_text_chars")),
            prefer_extensions=prefer_extensions,
            keyword_priorities=keyword_priorities,
        )
        enriched = {
            "relative_path": relative_path,
            "category_dir": category,
            "extension": extension,
            "file_size_bytes": parse_int(row.get("file_size_bytes")),
            "document_count": parse_int(row.get("document_count")),
            "total_text_chars": parse_int(row.get("total_text_chars")),
            "chunk_count": chunk_count,
            "selection_score": round(score, 4),
            "selected_reason": reason,
            "original_source": mapping.get(relative_path, ""),
        }
        candidates_by_category[category].append(enriched)

    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    for pattern in must_include_patterns:
        match = find_must_include_candidate(candidates_by_category, pattern)
        if match is None:
            exclusion_counts[f"must_include_not_found:{pattern}"] += 1
            continue
        category = str(match["category_dir"])
        if len([item for item in selected if item["category_dir"] == category]) >= quotas.get(category, 0):
            exclusion_counts[f"must_include_quota_full:{pattern}"] += 1
            continue
        projected_chunks = sum(int(item["chunk_count"]) for item in selected) + int(match["chunk_count"])
        if projected_chunks > max_chunks_total:
            exclusion_counts[f"must_include_max_chunks_guard:{pattern}"] += 1
            continue
        match = dict(match)
        match["selected_reason"] = str(match["selected_reason"]) + ";must_include"
        selected.append(match)
        selected_paths.add(str(match["relative_path"]))

    for category, quota in quotas.items():
        rows = candidates_by_category.get(category, [])
        rows.sort(
            key=lambda item: (
                -float(item["selection_score"]),
                int(item["chunk_count"]),
                str(item["relative_path"]).lower(),
            )
        )
        for row in rows:
            if str(row["relative_path"]) in selected_paths:
                continue
            if len([item for item in selected if item["category_dir"] == category]) >= quota:
                break
            if len(selected) >= max_files_total:
                break
            projected_chunks = sum(int(item["chunk_count"]) for item in selected) + int(row["chunk_count"])
            if projected_chunks > max_chunks_total:
                exclusion_counts["max_chunks_total_guard"] += 1
                continue
            selected.append(row)
            selected_paths.add(str(row["relative_path"]))

    selected.sort(key=lambda item: (category_order(item["category_dir"]), -float(item["selection_score"])))
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank

    write_csv(output_selected, selected, SELECTED_FIELDS)
    summary = build_summary(
        config=config,
        document_rows=document_rows,
        selected=selected,
        excluded_by_issue=excluded_by_issue,
        exclusion_counts=exclusion_counts,
        output_selected=output_selected,
    )
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def score_candidate(
    *,
    relative_path: str,
    category: str,
    extension: str,
    chunk_count: int,
    total_text_chars: int,
    prefer_extensions: list[str],
    keyword_priorities: dict[str, list[str]],
) -> tuple[float, str]:
    lowered = relative_path.lower()
    filename = Path(relative_path).name.lower()
    score = 1000.0
    matched_keywords = [keyword for keyword in keyword_priorities.get(category, []) if keyword in lowered]
    if extension in prefer_extensions:
        score += max(0, 120 - prefer_extensions.index(extension) * 15)
    score += len(matched_keywords) * 35
    score -= min(chunk_count, 100) * 1.5
    score -= min(total_text_chars, 50000) / 50000 * 20
    if filename.startswith("__init__"):
        score -= 120
    if extension == ".pdf":
        score -= 40
    if extension == ".csv":
        score -= 20

    reason_parts = ["clean_audit", f"extension:{extension or '(none)'}", f"chunks:{chunk_count}"]
    if matched_keywords:
        reason_parts.append("matched_keywords:" + "|".join(matched_keywords[:8]))
    return score, ";".join(reason_parts)


def load_original_source_mapping(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    mapping: dict[str, str] = {}
    for row in rows:
        destination = Path(str(row.get("destination_path") or ""))
        original_path = str(row.get("original_path") or "")
        if not destination.is_absolute():
            destination = PROJECT_ROOT / destination
        relative = relative_to_project(destination)
        mapping[relative] = original_path
    return mapping


def find_must_include_candidate(
    candidates_by_category: dict[str, list[dict[str, Any]]],
    pattern: str,
) -> dict[str, Any] | None:
    for rows in candidates_by_category.values():
        for row in rows:
            if pattern in str(row.get("relative_path", "")).lower():
                return row
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    *,
    config: dict[str, Any],
    document_rows: list[dict[str, str]],
    selected: list[dict[str, Any]],
    excluded_by_issue: set[str],
    exclusion_counts: Counter[str],
    output_selected: Path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_environment": "Lenginzed_RAG",
        "input_dir": str(PROJECT_ROOT / str(config["input_dir"])),
        "audit_dir": str(PROJECT_ROOT / str(config["audit_dir"])),
        "candidate_files": len(document_rows),
        "selected_files": len(selected),
        "selected_chunks": sum(int(row["chunk_count"]) for row in selected),
        "max_files_total": int(config.get("max_files_total", 30)),
        "max_chunks_total": int(config.get("max_chunks_total", 1000)),
        "selected_category_counts": dict(Counter(row["category_dir"] for row in selected)),
        "selected_extension_counts": dict(Counter(row["extension"] for row in selected)),
        "selected_doc_estimate": sum(int(row["document_count"]) for row in selected),
        "excluded_issue_file_count": len(excluded_by_issue),
        "excluded_pattern_file_count": sum(
            count for reason, count in exclusion_counts.items() if reason.startswith("filename_pattern:")
        ),
        "exclusion_counts": dict(exclusion_counts),
        "selected_files_preview": selected[:30],
        "all_selected_have_original_source": all(bool(row.get("original_source")) for row in selected),
        "output_selected_files": str(output_selected),
        "embedding_run": False,
        "chroma_written": False,
    }


def matched_pattern(relative_path: str, patterns: list[str]) -> str:
    lowered = relative_path.lower()
    for pattern in patterns:
        if pattern in lowered:
            return pattern
    return ""


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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
    }
    return order.get(category, 99)


if __name__ == "__main__":
    sys.exit(main())
