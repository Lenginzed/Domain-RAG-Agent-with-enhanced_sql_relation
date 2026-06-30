from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_calibration_v26.yaml"


def main() -> int:
    configure_stdout()
    try:
        result = select_recommended_config()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001
        print("V2.6 recommended calibration selection failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def select_recommended_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    sensitivity = load_json(resolve_project_path(config["outputs"]["sensitivity_json"]))
    candidate = sensitivity.get("best_small_boost_combo")
    if not candidate:
        raise ValueError("No accepted calibration combo found; inspect sensitivity output.")

    detail = find_detail_record(sensitivity, str(candidate["combo_id"]), "v25c_30")
    calibration = {
        "enabled": True,
        "exact_filename_boost": float(candidate["exact_filename_boost"]),
        "stem_token_boost": float(candidate["stem_token_boost"]),
        "metadata_title_boost": float(candidate["metadata_title_boost"]),
        "category_match_boost": float(candidate["category_match_boost"]),
        "matched_source_token_min_len": int(config["sensitivity"].get("matched_source_token_min_len", 3)),
        "ignored_source_tokens": list(config["sensitivity"].get("ignored_source_tokens", [])),
        "apply_to_modes": list(config["sensitivity"].get("apply_to_modes", ["enhanced", "enhanced_keyword"])),
    }
    recommended_yaml = {
        "collection": config["collection"],
        "calibration": calibration,
        "retrieval": {
            "modes": ["dense", "enhanced", "keyword", "enhanced_keyword"],
            "dense_top_k": 5,
            "metadata_top_k": 8,
            "keyword_top_k": 8,
            "final_top_k": 8,
            "max_chunks_per_source": 2,
            "enable_source_diversity": True,
            "source_key": "imported_source",
        },
        "selection_notes": {
            "selected_from": str(resolve_project_path(config["outputs"]["sensitivity_json"])),
            "combo_id": candidate["combo_id"],
            "selection_rule": (
                "q014 Hit@5 required; V25c enhanced_keyword Hit@5 must not drop; source diversity must remain 1.0; "
                "MRR drop must be within tolerance; smallest accepted boost is preferred."
            ),
        },
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "recommended_config": recommended_yaml,
        "selected_combo": candidate,
        "selected_detail": detail,
        "baseline_v25c_enhanced_keyword": sensitivity.get("baseline_v25c_enhanced_keyword", {}),
        "v25e_current_enhanced_keyword": sensitivity.get("v25e_current_enhanced_keyword", {}),
        "outputs": {
            "recommended_config_yaml": str(resolve_project_path(config["outputs"]["recommended_config_yaml"])),
            "recommended_config_json": str(resolve_project_path(config["outputs"]["recommended_config_json"])),
        },
    }


def find_detail_record(sensitivity: dict[str, Any], combo_id: str, eval_set: str) -> dict[str, Any]:
    for record in sensitivity.get("detail_records", []):
        if str(record.get("combo_id")) == combo_id and str(record.get("eval_set")) == eval_set:
            return record
    return {}


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    output_yaml = resolve_project_path(config["outputs"]["recommended_config_yaml"])
    output_json = resolve_project_path(config["outputs"]["recommended_config_json"])
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(result["recommended_config"], file, allow_unicode=True, sort_keys=False)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return data


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def print_summary(result: dict[str, Any]) -> None:
    print("V2.6 recommended calibration config selected.")
    print(
        json.dumps(
            {
                "selected_combo": result["selected_combo"],
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
