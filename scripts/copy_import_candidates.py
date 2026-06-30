from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "external_sources.yaml"

COPY_FIELDS = [
    "original_path",
    "relative_path",
    "filename",
    "extension",
    "size_bytes",
    "inferred_category",
    "destination_path",
    "dry_run",
    "was_copied",
    "status",
    "error",
]


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing import candidates file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def plan_or_copy_candidates(*, dry_run: bool, max_files: int | None) -> dict[str, Any]:
    config = load_config()
    manifest_dir = PROJECT_ROOT / str(config["output_manifest_dir"])
    import_dir = (PROJECT_ROOT / str(config["output_import_dir"])).resolve()
    candidates_path = manifest_dir / "import_candidates.csv"
    return plan_or_copy_from_path(
        plan_path=candidates_path,
        dry_run=dry_run,
        max_files=max_files,
        import_dir=import_dir,
        default_limit=int(config.get("max_copy_files", 200)),
        output_csv=manifest_dir / "copied_files.csv",
        output_report=manifest_dir / "copy_report.json",
    )


def plan_or_copy_from_path(
    *,
    plan_path: Path,
    dry_run: bool,
    max_files: int | None,
    import_dir: Path,
    default_limit: int | None,
    output_csv: Path,
    output_report: Path,
) -> dict[str, Any]:
    candidates = load_candidates(plan_path)
    limit = max_files if max_files is not None else default_limit
    selected_candidates = candidates[:limit] if limit and limit > 0 else candidates

    rows: list[dict[str, Any]] = []
    seen_destinations: set[Path] = set()
    for candidate in selected_candidates:
        row = build_copy_row(
            candidate=candidate,
            import_dir=import_dir,
            dry_run=dry_run,
            seen_destinations=seen_destinations,
        )
        rows.append(row)

    if not dry_run:
        import_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = Path(str(row["original_path"]))
            destination = Path(str(row["destination_path"]))
            try:
                assert_destination_is_safe(destination, import_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                row["was_copied"] = True
                row["status"] = "copied"
            except Exception as exc:  # noqa: BLE001 - keep copy report complete.
                row["was_copied"] = False
                row["status"] = "error"
                row["error"] = str(exc)

    write_csv(output_csv, rows, COPY_FIELDS)
    report = build_report(
        dry_run=dry_run,
        total_candidates=len(candidates),
        planned_files=len(rows),
        max_files=limit,
        import_dir=import_dir,
        plan_path=plan_path,
        output_csv=output_csv,
        rows=rows,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_copy_row(
    *,
    candidate: dict[str, Any],
    import_dir: Path,
    dry_run: bool,
    seen_destinations: set[Path],
) -> dict[str, Any]:
    category = str(candidate.get("destination_category") or candidate.get("inferred_category") or "unknown_text")
    filename = str(candidate.get("filename") or "unnamed")
    relative_path = str(candidate.get("relative_path") or filename)
    destination = unique_destination(
        import_dir=import_dir,
        category=category,
        filename=filename,
        relative_path=relative_path,
        seen_destinations=seen_destinations,
    )
    assert_destination_is_safe(destination, import_dir)
    return {
        "original_path": candidate.get("original_path", ""),
        "relative_path": relative_path,
        "filename": filename,
        "extension": candidate.get("extension", ""),
        "size_bytes": candidate.get("size_bytes", ""),
        "inferred_category": category,
        "destination_path": str(destination),
        "dry_run": dry_run,
        "was_copied": False,
        "status": "dry_run" if dry_run else "pending",
        "error": "",
    }


def unique_destination(
    *,
    import_dir: Path,
    category: str,
    filename: str,
    relative_path: str,
    seen_destinations: set[Path],
) -> Path:
    category_dir = import_dir / safe_category(category)
    candidate = category_dir / filename
    if candidate not in seen_destinations and not candidate.exists():
        seen_destinations.add(candidate)
        return candidate

    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
    path = Path(filename)
    new_name = f"{path.stem}_{digest}{path.suffix}" if path.suffix else f"{filename}_{digest}"
    candidate = category_dir / new_name
    counter = 1
    while candidate in seen_destinations or candidate.exists():
        new_name = f"{path.stem}_{digest}_{counter}{path.suffix}" if path.suffix else f"{filename}_{digest}_{counter}"
        candidate = category_dir / new_name
        counter += 1
    seen_destinations.add(candidate)
    return candidate


def safe_category(category: str) -> str:
    allowed = {
        "papers",
        "thesis_or_reports",
        "code",
        "configs",
        "experiments",
        "notes",
        "unknown_text",
    }
    return category if category in allowed else "unknown_text"


def assert_destination_is_safe(destination: Path, import_dir: Path) -> None:
    resolved_destination = destination.resolve()
    resolved_import_dir = import_dir.resolve()
    if not str(resolved_destination).lower().startswith(str(resolved_import_dir).lower() + "\\"):
        if resolved_destination != resolved_import_dir:
            raise ValueError(f"Refusing to write outside import dir: {resolved_destination}")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(
    *,
    dry_run: bool,
    total_candidates: int,
    planned_files: int,
    max_files: int | None,
    import_dir: Path,
    plan_path: Path,
    output_csv: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    copied_count = sum(str(row["status"]) == "copied" for row in rows)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "execute": not dry_run,
        "total_candidates": total_candidates,
        "planned_files": planned_files,
        "max_files": max_files,
        "actual_copied_files": copied_count,
        "import_dir": str(import_dir),
        "plan_path": str(plan_path),
        "output_csv": str(output_csv),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "planned_category_counts": dict(Counter(str(row["inferred_category"]) for row in rows)),
        "planned_extension_counts": dict(Counter(str(row["extension"]) or "(none)" for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run or execute copying selected external import candidates.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan copy operations without copying files.")
    mode.add_argument("--execute", action="store_true", help="Actually copy files into data/raw_imported.")
    parser.add_argument("--plan", type=Path, default=None, help="CSV plan to copy or dry-run.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit planned/copied candidates.")
    args = parser.parse_args()

    dry_run = not args.execute
    if args.dry_run:
        dry_run = True

    try:
        if args.plan is None:
            report = plan_or_copy_candidates(dry_run=dry_run, max_files=args.max_files)
        else:
            external_config = load_config()
            import_dir = (PROJECT_ROOT / str(external_config["output_import_dir"])).resolve()
            plan_path = args.plan if args.plan.is_absolute() else PROJECT_ROOT / args.plan
            output_csv, output_report = output_paths_for_plan(plan_path)
            report = plan_or_copy_from_path(
                plan_path=plan_path,
                dry_run=dry_run,
                max_files=args.max_files,
                import_dir=import_dir,
                default_limit=None,
                output_csv=output_csv,
                output_report=output_report,
            )
    except Exception as exc:  # noqa: BLE001 - script should report environment/path failures.
        print("Copy import candidates failed.")
        print(f"reason: {exc}")
        return 1

    print("Copy import candidates completed.")
    print(f"mode: {'dry-run' if report['dry_run'] else 'execute'}")
    print(f"total_candidates: {report['total_candidates']}")
    print(f"planned_files: {report['planned_files']}")
    print(f"actual_copied_files: {report['actual_copied_files']}")
    print(f"import_dir: {report['import_dir']}")
    print(f"plan_path: {report['plan_path']}")
    print(f"output_csv: {report['output_csv']}")
    if report["dry_run"]:
        print("No files were copied. Re-run with --execute only after human review.")
    return 0


def output_paths_for_plan(plan_path: Path) -> tuple[Path, Path]:
    manifest_dir = PROJECT_ROOT / "data" / "external_manifest"
    final_selection_path = PROJECT_ROOT / "config" / "import_selection_final.yaml"
    if final_selection_path.exists():
        with final_selection_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        configured_plan = PROJECT_ROOT / str(config.get("output_plan", ""))
        if plan_path.resolve() == configured_plan.resolve():
            output_csv = PROJECT_ROOT / str(
                config.get("output_dry_run", "data/external_manifest/curated_copied_files_final.csv")
            )
            return output_csv, manifest_dir / "curated_copy_report_final.json"

    import_selection_path = PROJECT_ROOT / "config" / "import_selection.yaml"
    if import_selection_path.exists():
        with import_selection_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        configured_plan = PROJECT_ROOT / str(config.get("output_plan", ""))
        if plan_path.resolve() == configured_plan.resolve():
            output_csv = PROJECT_ROOT / str(config.get("output_dry_run", "data/external_manifest/curated_copied_files.csv"))
            return output_csv, manifest_dir / "curated_copy_report.json"
    return manifest_dir / "copied_files.csv", manifest_dir / "copy_report.json"


if __name__ == "__main__":
    sys.exit(main())
