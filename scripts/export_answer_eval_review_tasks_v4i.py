from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.answer_eval_triage_service import (  # noqa: E402
    build_review_tasks,
    export_review_tasks,
    load_eval_records_for_triage,
    load_triage_config,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "review_triage_v4i.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V4i answer eval human review tasks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to review triage config.")
    parser.add_argument("--version", default="", help="Answer eval version, for example v4g.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_triage_config(args.config)
    version = args.version or str(config.get("inputs", {}).get("default_version", "v4g"))
    records = load_eval_records_for_triage(config, version)
    config["_triage_total_records"] = len(records)
    config["_triage_version"] = version
    tasks = build_review_tasks(records, config)
    result = export_review_tasks(tasks, config)
    console_summary = {
        "version": version,
        "total_records": len(records),
        "review_task_count": len(tasks),
        "priority_counts": result["summary"].get("priority_counts", {}),
        "failure_type_counts": result["summary"].get("failure_type_counts", {}),
        "status_counts": result["summary"].get("status_counts", {}),
        "output_paths": result.get("output_paths", {}),
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

