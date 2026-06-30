from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "external_sources.yaml"

INVENTORY_FIELDS = [
    "original_path",
    "relative_path",
    "filename",
    "extension",
    "size_bytes",
    "modified_time",
    "inferred_category",
    "is_candidate",
    "skip_reason",
    "safety_flags",
]

CATEGORY_PRIORITY = {
    "papers": 0,
    "thesis_or_reports": 1,
    "code": 2,
    "configs": 3,
    "experiments": 4,
    "notes": 5,
    "unknown_text": 6,
}

PRIVACY_TERMS = {
    ".env",
    "secret",
    "secrets",
    "password",
    "passwd",
    "token",
    "credential",
    "credentials",
    "private_key",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
    "cookies",
    "apikey",
    "api_key",
    "账号",
    "密码",
    "密钥",
    "凭证",
}

PAPER_TERMS = {
    "paper",
    "papers",
    "论文",
    "article",
    "survey",
    "综述",
}

REPORT_TERMS = {
    "report",
    "reports",
    "thesis",
    "dissertation",
    "课程",
    "报告",
    "毕业",
}

EXPERIMENT_TERMS = {
    "experiment",
    "experiments",
    "result",
    "results",
    "eval",
    "evaluation",
    "metrics",
    "log",
    "logs",
    "summary",
    "实验",
    "结果",
    "评估",
    "指标",
}

NOTE_TERMS = {
    "note",
    "notes",
    "readme",
    "doc",
    "docs",
    "intro",
    "summary",
    "阶段",
    "总结",
    "说明",
}


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def inventory_external_sources(config: dict[str, Any]) -> dict[str, Any]:
    manifest_dir = PROJECT_ROOT / str(config["output_manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)

    include_extensions = {str(item).lower() for item in config.get("include_extensions", [])}
    include_filenames = {str(item).lower() for item in config.get("include_filenames", [])}
    exclude_dirs = {str(item).lower() for item in config.get("exclude_dirs", [])}
    exclude_extensions = {str(item).lower() for item in config.get("exclude_extensions", [])}
    max_size_bytes = int(float(config.get("max_file_size_mb", 20)) * 1024 * 1024)

    inventory_rows: list[dict[str, Any]] = []
    skipped_dir_rows: list[dict[str, Any]] = []

    for root_value in config.get("external_roots", []):
        root = Path(root_value)
        if not root.exists():
            skipped_dir_rows.append(
                {
                    "root_path": str(root),
                    "directory_path": str(root),
                    "relative_path": "",
                    "skip_reason": "root_missing",
                }
            )
            continue

        for current_dir, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current_dir)
            allowed_dirs: list[str] = []
            for dirname in dirnames:
                reason = should_skip_dir(dirname, current_path, exclude_dirs)
                if reason:
                    skipped_dir_rows.append(
                        {
                            "root_path": str(root),
                            "directory_path": str(current_path / dirname),
                            "relative_path": safe_relative(current_path / dirname, root),
                            "skip_reason": reason,
                        }
                    )
                else:
                    allowed_dirs.append(dirname)
            dirnames[:] = allowed_dirs

            for filename in filenames:
                path = current_path / filename
                row = build_inventory_row(
                    path=path,
                    root=root,
                    include_extensions=include_extensions,
                    include_filenames=include_filenames,
                    exclude_extensions=exclude_extensions,
                    max_size_bytes=max_size_bytes,
                )
                inventory_rows.append(row)

    inventory_rows.sort(
        key=lambda row: (
            CATEGORY_PRIORITY.get(str(row["inferred_category"]), 99),
            str(row["relative_path"]).lower(),
        )
    )

    candidate_rows = [row for row in inventory_rows if row["is_candidate"]]
    skipped_rows = [row for row in inventory_rows if not row["is_candidate"]]

    write_csv(manifest_dir / "external_inventory.csv", inventory_rows, INVENTORY_FIELDS)
    write_jsonl(manifest_dir / "external_inventory.jsonl", inventory_rows)
    write_csv(manifest_dir / "import_candidates.csv", candidate_rows, INVENTORY_FIELDS)
    write_csv(manifest_dir / "skipped_files.csv", skipped_rows, INVENTORY_FIELDS)
    write_csv(
        manifest_dir / "skipped_dirs.csv",
        skipped_dir_rows,
        ["root_path", "directory_path", "relative_path", "skip_reason"],
    )

    summary = build_summary(
        inventory_rows=inventory_rows,
        candidate_rows=candidate_rows,
        skipped_rows=skipped_rows,
        skipped_dir_rows=skipped_dir_rows,
        max_size_bytes=max_size_bytes,
    )
    (manifest_dir / "external_inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def should_skip_dir(dirname: str, current_path: Path, exclude_dirs: set[str]) -> str:
    lowered = dirname.lower()
    if lowered in exclude_dirs:
        return "excluded_directory"
    if lowered == "runs":
        return "excluded_runs_directory"
    return ""


def build_inventory_row(
    *,
    path: Path,
    root: Path,
    include_extensions: set[str],
    include_filenames: set[str],
    exclude_extensions: set[str],
    max_size_bytes: int,
) -> dict[str, Any]:
    extension = path.suffix.lower()
    filename = path.name
    relative_path = safe_relative(path, root)
    original_path = str(path)
    stat_error = ""
    try:
        stat = path.stat()
        size_bytes = int(stat.st_size)
        modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    except OSError as exc:
        size_bytes = 0
        modified_time = ""
        stat_error = str(exc)

    category = infer_category(path, extension)
    safety_flags = infer_safety_flags(path, extension)

    is_candidate = True
    skip_reason = ""
    if stat_error:
        is_candidate = False
        skip_reason = f"stat_error: {stat_error}"
    elif "privacy_suspect" in safety_flags:
        is_candidate = False
        skip_reason = "privacy_suspect"
    elif extension in exclude_extensions:
        is_candidate = False
        skip_reason = "excluded_extension"
    elif size_bytes > max_size_bytes:
        is_candidate = False
        skip_reason = "file_too_large"
    elif not is_included_file(path, include_extensions, include_filenames):
        is_candidate = False
        skip_reason = "unsupported_extension_or_filename"

    return {
        "original_path": original_path,
        "relative_path": relative_path,
        "filename": filename,
        "extension": extension,
        "size_bytes": size_bytes,
        "modified_time": modified_time,
        "inferred_category": category,
        "is_candidate": is_candidate,
        "skip_reason": skip_reason,
        "safety_flags": ";".join(safety_flags),
    }


def is_included_file(path: Path, include_extensions: set[str], include_filenames: set[str]) -> bool:
    return path.suffix.lower() in include_extensions or path.name.lower() in include_filenames


def infer_category(path: Path, extension: str) -> str:
    text = path.as_posix().lower()
    if extension in {".py", ".sh", ".bat", ".ps1"}:
        return "code"
    if extension in {".yaml", ".yml", ".json", ".toml", ".ini"}:
        if contains_any(text, EXPERIMENT_TERMS):
            return "experiments"
        return "configs"
    if extension == ".csv":
        return "experiments"
    if extension in {".pdf", ".tex"}:
        if contains_any(text, REPORT_TERMS):
            return "thesis_or_reports"
        return "papers"
    if extension in {".md", ".markdown", ".txt"} or path.name.lower().startswith("readme"):
        if contains_any(text, EXPERIMENT_TERMS):
            return "experiments"
        if contains_any(text, PAPER_TERMS):
            return "papers"
        if contains_any(text, REPORT_TERMS):
            return "thesis_or_reports"
        if contains_any(text, NOTE_TERMS):
            return "notes"
        return "unknown_text"
    return "unknown_text"


def infer_safety_flags(path: Path, extension: str) -> list[str]:
    text = path.as_posix().lower()
    flags: list[str] = []
    if contains_any(text, PRIVACY_TERMS):
        flags.append("privacy_suspect")
    if extension in {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".npz", ".npy"}:
        flags.append("model_or_array_file")
    if any(part.lower() in {".venv", "venv", "env", "__pycache__", ".pytest_cache", "cache"} for part in path.parts):
        flags.append("virtualenv_or_cache_path")
    return flags


def contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(
    *,
    inventory_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    skipped_dir_rows: list[dict[str, Any]],
    max_size_bytes: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scanned_files": len(inventory_rows),
        "candidate_files": len(candidate_rows),
        "skipped_files": len(skipped_rows),
        "skipped_dirs": len(skipped_dir_rows),
        "max_file_size_bytes": max_size_bytes,
        "extension_counts": dict(Counter(str(row["extension"]) or "(none)" for row in inventory_rows)),
        "candidate_extension_counts": dict(Counter(str(row["extension"]) or "(none)" for row in candidate_rows)),
        "candidate_category_counts": dict(Counter(str(row["inferred_category"]) for row in candidate_rows)),
        "skipped_reason_counts": dict(Counter(str(row["skip_reason"]) for row in skipped_rows)),
        "skipped_dir_reason_counts": dict(Counter(str(row["skip_reason"]) for row in skipped_dir_rows)),
        "privacy_suspect_files": sum("privacy_suspect" in str(row["safety_flags"]) for row in inventory_rows),
        "model_or_array_files": sum("model_or_array_file" in str(row["safety_flags"]) for row in inventory_rows),
        "virtualenv_or_cache_dirs": sum(
            str(row["skip_reason"]) == "excluded_directory"
            and Path(str(row["directory_path"])).name.lower() in {".venv", "venv", "env", "__pycache__", ".pytest_cache", "cache"}
            for row in skipped_dir_rows
        ),
        "largest_candidates": [
            {
                "relative_path": row["relative_path"],
                "size_bytes": row["size_bytes"],
                "inferred_category": row["inferred_category"],
                "extension": row["extension"],
            }
            for row in sorted(candidate_rows, key=lambda item: int(item["size_bytes"]), reverse=True)[:20]
        ],
    }


def main() -> int:
    try:
        config = load_config()
        summary = inventory_external_sources(config)
    except Exception as exc:  # noqa: BLE001 - script should report environment/path failures.
        print("External inventory failed.")
        print(f"reason: {exc}")
        return 1

    manifest_dir = PROJECT_ROOT / str(config["output_manifest_dir"])
    print("External source inventory completed.")
    print(f"manifest_dir: {manifest_dir}")
    print(f"scanned_files: {summary['scanned_files']}")
    print(f"candidate_files: {summary['candidate_files']}")
    print(f"skipped_files: {summary['skipped_files']}")
    print(f"skipped_dirs: {summary['skipped_dirs']}")
    print("No files under external roots were modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
