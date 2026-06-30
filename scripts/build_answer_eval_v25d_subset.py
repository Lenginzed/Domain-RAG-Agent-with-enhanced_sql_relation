from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v25d.yaml"


def main() -> int:
    configure_stdout()
    try:
        result = build_subset()
    except Exception as exc:  # noqa: BLE001
        print("V2.5d answer eval subset build failed.")
        print(f"reason: {exc}")
        return 1
    print("V2.5d answer eval subset built.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_subset(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    input_path = resolve_project_path(config["input_eval_set"])
    output_jsonl = resolve_project_path(config["output_subset_jsonl"])
    output_manifest = resolve_project_path(config["output_subset_manifest"])
    items = load_jsonl(input_path)
    selection = config.get("selection") or {}
    target_query_types = selection.get("target_query_types") or {}
    include_failure_ids = [str(item) for item in selection.get("include_failure_case_ids", [])]
    target_total = int(selection.get("total_questions", 12))

    by_id = {str(item["id"]): item for item in items}
    selected: list[dict[str, Any]] = []

    for item_id in include_failure_ids:
        if item_id not in by_id:
            raise ValueError(f"Required failure case id not found in eval set: {item_id}")
        add_unique(selected, by_id[item_id])

    for query_type, count in target_query_types.items():
        query_type = str(query_type)
        count = int(count)
        current = sum(1 for item in selected if item.get("query_type") == query_type)
        if current >= count:
            continue
        candidates = [item for item in items if item.get("query_type") == query_type]
        for item in candidates:
            if current >= count:
                break
            if add_unique(selected, item):
                current += 1

    if len(selected) > target_total:
        keep_ids = set(include_failure_ids)
        trimmed: list[dict[str, Any]] = []
        for item in selected:
            if len(trimmed) >= target_total and str(item["id"]) not in keep_ids:
                continue
            add_unique(trimmed, item)
        selected = trimmed[:target_total]

    for item in items:
        if len(selected) >= target_total:
            break
        add_unique(selected, item)

    validate_subset(selected, selection)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, selected)
    manifest = build_manifest(selected, config_path, input_path, output_jsonl, output_manifest, include_failure_ids)
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def add_unique(selected: list[dict[str, Any]], item: dict[str, Any]) -> bool:
    if any(existing["id"] == item["id"] for existing in selected):
        return False
    selected.append(item)
    return True


def validate_subset(items: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    target_total = int(selection.get("total_questions", 12))
    if not 10 <= len(items) <= 14:
        raise ValueError(f"Subset size must be 10-14, got {len(items)}")
    if len(items) != target_total:
        raise ValueError(f"Expected exactly {target_total} questions, got {len(items)}")
    negative_count = sum(1 for item in items if item.get("query_type") == "negative")
    mixed_count = sum(1 for item in items if item.get("query_type") == "mixed")
    if negative_count < int(selection.get("include_negative_questions", 2)):
        raise ValueError(f"Expected at least 2 negative questions, got {negative_count}")
    if mixed_count < int(selection.get("include_mixed_questions", 2)):
        raise ValueError(f"Expected at least 2 mixed questions, got {mixed_count}")
    selected_ids = {str(item["id"]) for item in items}
    for item_id in selection.get("include_failure_case_ids", []):
        if str(item_id) not in selected_ids:
            raise ValueError(f"Required failure case not selected: {item_id}")


def build_manifest(
    items: list[dict[str, Any]],
    config_path: Path,
    input_path: Path,
    output_jsonl: Path,
    output_manifest: Path,
    include_failure_ids: list[str],
) -> dict[str, Any]:
    query_type_counts = dict(Counter(str(item.get("query_type", "")) for item in items))
    difficulty_counts = dict(Counter(str(item.get("difficulty", "")) for item in items))
    category_counts = Counter(
        str(category)
        for item in items
        for category in item.get("expected_categories", [])
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "input_eval_set": str(input_path),
        "output_subset_jsonl": str(output_jsonl),
        "output_subset_manifest": str(output_manifest),
        "total_questions": len(items),
        "negative_questions": query_type_counts.get("negative", 0),
        "mixed_questions": query_type_counts.get("mixed", 0),
        "included_failure_case_ids": include_failure_ids,
        "selected_ids": [str(item["id"]) for item in items],
        "query_type_counts": query_type_counts,
        "difficulty_counts": difficulty_counts,
        "expected_category_counts": dict(sorted(category_counts.items())),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


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


if __name__ == "__main__":
    sys.exit(main())
