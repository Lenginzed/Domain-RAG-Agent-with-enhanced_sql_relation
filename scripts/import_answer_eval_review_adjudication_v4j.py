from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.answer_eval_adjudication_service import (  # noqa: E402
    build_calibration_notes,
    enrich_adjudication_tasks,
    export_adjudication_outputs,
    load_adjudication_config,
    load_manual_review_csv,
    load_review_tasks,
    merge_manual_review,
    resolve_project_path,
    summarize_adjudication,
    validate_manual_fields,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "review_adjudication_v4j.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import V4j human review adjudication labels.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to adjudication config.")
    parser.add_argument("--manual-csv", default="", help="Optional filled human review CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_adjudication_config(args.config)
    task_path = config.get("inputs", {}).get("default_review_tasks_json", "")
    tasks = load_review_tasks(task_path)

    manual_rows = []
    if args.manual_csv:
        manual_path = resolve_project_path(args.manual_csv)
        if not manual_path.exists():
            print(json.dumps({"error": f"Manual CSV not found: {manual_path}"}, ensure_ascii=False, indent=2))
            return 2
        manual_rows = load_manual_review_csv(manual_path)
        tasks = merge_manual_review(tasks, manual_rows)

    enriched = enrich_adjudication_tasks(tasks, config)
    validation = validate_manual_fields(enriched, config)
    summary = summarize_adjudication(enriched, config)
    summary["manual_csv"] = str(resolve_project_path(args.manual_csv)) if args.manual_csv else ""
    summary["manual_rows_loaded"] = len(manual_rows)
    summary["validation"] = validation
    notes_md = build_calibration_notes(enriched, summary, config)
    result = export_adjudication_outputs(enriched, summary, notes_md, config)
    console_summary = {
        "total_tasks": summary.get("total_tasks", 0),
        "pending_count": summary.get("pending_count", 0),
        "adjudicated_count": summary.get("adjudicated_count", 0),
        "adjudication_status_counts": summary.get("adjudication_status_counts", {}),
        "manual_judgment_counts": summary.get("manual_judgment_counts", {}),
        "manual_citation_judgment_counts": summary.get("manual_citation_judgment_counts", {}),
        "manual_should_refuse_counts": summary.get("manual_should_refuse_counts", {}),
        "fix_category_counts": summary.get("fix_category_counts", {}),
        "output_paths": result.get("output_paths", {}),
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

