from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v4f.yaml"


def main() -> int:
    configure_stdout()
    config = load_yaml(CONFIG_PATH)
    cases = build_cases(config)
    output_path = resolve_project_path(config["cases"]["output_jsonl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote_cases: {len(cases)}")
    print(f"output_jsonl: {output_path}")
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    max_cases = int((config.get("answer_eval") or {}).get("max_cases", 8))
    items = list((config.get("cases") or {}).get("items") or [])
    if not items:
        raise ValueError("config/answer_eval_v4f.yaml has no cases.items")
    cases: list[dict[str, Any]] = []
    for item in items[:max_cases]:
        if not isinstance(item, dict):
            continue
        case = {
            "id": str(item["id"]),
            "question": str(item["question"]),
            "query_type": str(item.get("query_type", "")),
            "difficulty": str(item.get("difficulty", "medium")),
            "expected_focus": str(item.get("expected_focus", "")),
            "expected_behavior": str(item.get("expected_behavior", "answer_with_citations")),
            "expected_sources_contains": [str(value) for value in item.get("expected_sources_contains", [])],
            "expected_categories": [str(value) for value in item.get("expected_categories", [])],
            "required_all_categories": [str(value) for value in item.get("required_all_categories", [])],
            "expected_answer_should_mention": [
                str(value) for value in item.get("expected_answer_should_mention", [])
            ],
            "notes": str(item.get("notes", "V4f small answer eval case")),
        }
        cases.append(case)
    validate_cases(cases)
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> None:
    required_ids = {
        "v4f_q014_missile_engine",
        "v4f_event_driven_reward",
        "v4f_hierarchy_selfplay",
        "v4f_reward_risk",
        "v4f_negative_missing",
    }
    found_ids = {str(case.get("id", "")) for case in cases}
    missing = sorted(required_ids - found_ids)
    if missing:
        raise ValueError(f"V4f cases missing required ids: {missing}")
    if not any(str(case.get("query_type")) == "negative" for case in cases):
        raise ValueError("V4f cases must include at least one negative case")


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
    raise SystemExit(main())
