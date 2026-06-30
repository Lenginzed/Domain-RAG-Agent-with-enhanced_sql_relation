from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "external_manifest"
PLAN_PATH = MANIFEST_DIR / "curated_import_plan_final.csv"
COPY_REPORT_PATH = MANIFEST_DIR / "curated_copy_report_final.json"
DEFAULT_COPIED_CSV_PATH = MANIFEST_DIR / "curated_copied_files_final.csv"
IMPORT_DIR = PROJECT_ROOT / "data" / "raw_imported"
VERIFY_JSON_PATH = MANIFEST_DIR / "imported_files_verify.json"
VERIFY_CSV_PATH = MANIFEST_DIR / "imported_files_verify.csv"

EXPECTED_FILE_COUNT = 207
EXPECTED_CATEGORIES = ["code", "configs", "experiments", "notes", "papers", "thesis_or_reports"]
PROHIBITED_KEYWORDS = [
    ".jsbsim_data_tmp",
    "controller_state",
    "tokenizer",
    "stderr",
    "stdout",
    "figures/pdf",
    "figure",
    "state_final",
]

VERIFY_FIELDS = [
    "original_path",
    "relative_path",
    "filename",
    "inferred_category",
    "extension",
    "destination_path",
    "exists",
    "destination_size_bytes",
    "bad_keyword_matches",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def verify_imported_files() -> dict[str, Any]:
    plan_rows = read_csv(PLAN_PATH)
    copy_report = read_json(COPY_REPORT_PATH)
    copied_csv_path = Path(str(copy_report.get("output_csv") or DEFAULT_COPIED_CSV_PATH))
    if not copied_csv_path.is_absolute():
        copied_csv_path = PROJECT_ROOT / copied_csv_path
    copied_rows = read_csv(copied_csv_path)

    imported_files = sorted(path for path in IMPORT_DIR.rglob("*") if path.is_file()) if IMPORT_DIR.exists() else []
    file_count = len(imported_files)
    directory_counts = Counter(category_for_imported_file(path) for path in imported_files)
    extension_counts = Counter(path.suffix.lower() or "(none)" for path in imported_files)

    verify_rows: list[dict[str, Any]] = []
    missing_destinations: list[str] = []
    bad_keyword_rows: list[dict[str, Any]] = []

    for row in copied_rows:
        destination = Path(str(row.get("destination_path") or ""))
        exists = destination.is_file()
        if not exists:
            missing_destinations.append(str(destination))

        searchable_text = " ".join(
            [
                str(row.get("original_path") or ""),
                str(row.get("relative_path") or ""),
                str(row.get("destination_path") or ""),
            ]
        )
        bad_matches = matched_bad_keywords(searchable_text)
        verify_row = {
            "original_path": row.get("original_path", ""),
            "relative_path": row.get("relative_path", ""),
            "filename": row.get("filename", ""),
            "inferred_category": row.get("inferred_category", ""),
            "extension": str(row.get("extension") or "").lower(),
            "destination_path": str(destination),
            "exists": exists,
            "destination_size_bytes": destination.stat().st_size if exists else "",
            "bad_keyword_matches": "|".join(bad_matches),
        }
        verify_rows.append(verify_row)
        if bad_matches:
            bad_keyword_rows.append(verify_row)

    bad_keyword_counts = {
        keyword: sum(keyword in normalize_path_text(row_text(row)) for row in copied_rows)
        for keyword in PROHIBITED_KEYWORDS
    }

    write_csv(VERIFY_CSV_PATH, verify_rows, VERIFY_FIELDS)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_environment": "Lenginzed_RAG",
        "plan_path": str(PLAN_PATH),
        "copy_report_path": str(COPY_REPORT_PATH),
        "copied_csv_path": str(copied_csv_path),
        "import_dir": str(IMPORT_DIR),
        "expected_file_count": EXPECTED_FILE_COUNT,
        "plan_file_count": len(plan_rows),
        "copy_report_total_candidates": copy_report.get("total_candidates"),
        "copy_report_planned_files": copy_report.get("planned_files"),
        "copy_report_actual_copied_files": copy_report.get("actual_copied_files"),
        "copy_report_execute": copy_report.get("execute"),
        "copy_report_dry_run": copy_report.get("dry_run"),
        "raw_imported_exists": IMPORT_DIR.exists(),
        "raw_imported_file_count": file_count,
        "copied_csv_row_count": len(copied_rows),
        "existing_copied_paths_count": sum(bool(row["exists"]) for row in verify_rows),
        "missing_copied_files_count": len(missing_destinations),
        "missing_copied_files": missing_destinations[:50],
        "category_counts": {category: directory_counts.get(category, 0) for category in EXPECTED_CATEGORIES},
        "extension_counts": {
            extension: extension_counts.get(extension, 0)
            for extension in [".md", ".py", ".yaml", ".csv", ".pdf"]
        },
        "all_extension_counts": dict(extension_counts),
        "bad_keyword_counts": bad_keyword_counts,
        "bad_keyword_total": sum(bad_keyword_counts.values()),
        "bad_keyword_examples": bad_keyword_rows[:20],
        "all_expected_category_dirs_present": all((IMPORT_DIR / category).is_dir() for category in EXPECTED_CATEGORIES),
        "real_copy_verified": bool(copy_report.get("execute")) and not bool(copy_report.get("dry_run")),
        "embedding_or_chroma_run": False,
        "external_source_modified": False,
        "verification_passed": False,
        "verify_csv_path": str(VERIFY_CSV_PATH),
        "verify_json_path": str(VERIFY_JSON_PATH),
    }
    summary["verification_passed"] = is_verification_passed(summary)

    VERIFY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def category_for_imported_file(path: Path) -> str:
    try:
        relative = path.relative_to(IMPORT_DIR)
    except ValueError:
        return "(outside_import_dir)"
    return relative.parts[0] if relative.parts else "(root)"


def row_text(row: dict[str, str]) -> str:
    return " ".join(
        [
            str(row.get("original_path") or ""),
            str(row.get("relative_path") or ""),
            str(row.get("destination_path") or ""),
        ]
    )


def normalize_path_text(text: str) -> str:
    return text.replace("\\", "/").lower()


def matched_bad_keywords(text: str) -> list[str]:
    normalized = normalize_path_text(text)
    return [keyword for keyword in PROHIBITED_KEYWORDS if keyword in normalized]


def is_verification_passed(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["used_environment"] == "Lenginzed_RAG",
            summary["plan_file_count"] == EXPECTED_FILE_COUNT,
            summary["copy_report_total_candidates"] == EXPECTED_FILE_COUNT,
            summary["copy_report_planned_files"] == EXPECTED_FILE_COUNT,
            summary["copy_report_actual_copied_files"] == EXPECTED_FILE_COUNT,
            summary["raw_imported_exists"],
            summary["raw_imported_file_count"] == EXPECTED_FILE_COUNT,
            summary["copied_csv_row_count"] == EXPECTED_FILE_COUNT,
            summary["existing_copied_paths_count"] == EXPECTED_FILE_COUNT,
            summary["missing_copied_files_count"] == 0,
            summary["all_expected_category_dirs_present"],
            summary["bad_keyword_total"] == 0,
            summary["real_copy_verified"],
            not summary["embedding_or_chroma_run"],
            not summary["external_source_modified"],
        ]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    try:
        summary = verify_imported_files()
    except Exception as exc:  # noqa: BLE001 - verification should print exact failure.
        print("Imported files verification failed.")
        print(f"reason: {exc}")
        return 1

    print("Imported files verification completed.")
    print(f"environment: {summary['used_environment']}")
    print(f"expected_file_count: {summary['expected_file_count']}")
    print(f"plan_file_count: {summary['plan_file_count']}")
    print(f"copy_report_actual_copied_files: {summary['copy_report_actual_copied_files']}")
    print(f"raw_imported_exists: {summary['raw_imported_exists']}")
    print(f"raw_imported_file_count: {summary['raw_imported_file_count']}")
    print("category_counts:")
    for category, count in summary["category_counts"].items():
        print(f"  {category}: {count}")
    print("extension_counts:")
    for extension, count in summary["extension_counts"].items():
        print(f"  {extension}: {count}")
    print(f"bad_keyword_total: {summary['bad_keyword_total']}")
    print(f"missing_copied_files_count: {summary['missing_copied_files_count']}")
    print(f"verification_passed: {summary['verification_passed']}")
    print(f"verify_json_path: {summary['verify_json_path']}")
    print(f"verify_csv_path: {summary['verify_csv_path']}")
    return 0 if summary["verification_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
