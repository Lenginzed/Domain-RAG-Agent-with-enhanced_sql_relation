from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_calibration_v26.yaml"
V25E_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_calibration_v25e.yaml"
FIELDS = ["dense_score", "metadata_score", "keyword_score", "final_score", "calibration_score"]


def main() -> int:
    configure_stdout()
    try:
        result = analyze_score_scales()
        write_output(result)
    except Exception as exc:  # noqa: BLE001
        print("V2.6 retrieval score scale analysis failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def analyze_score_scales(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    v25e_config = load_yaml(V25E_CONFIG_PATH)
    v25c = load_json(resolve_project_path(config["inputs"]["v25c_eval_json"]))
    v25e = load_json(resolve_project_path(config["inputs"]["v25e_eval_json"]))
    datasets = {
        "v25c_baseline": v25c,
        "v25e_calibrated": v25e,
    }
    stats_by_dataset: dict[str, Any] = {}
    for dataset_name, dataset in datasets.items():
        stats_by_dataset[dataset_name] = {}
        for mode in ["enhanced", "enhanced_keyword"]:
            sources = collect_sources(dataset, mode)
            stats_by_dataset[dataset_name][mode] = {
                field: describe([to_float(source.get(field)) for source in sources])
                for field in FIELDS
            }
            stats_by_dataset[dataset_name][mode]["source_count"] = len(sources)

    calibration = dict(v25e_config.get("calibration", {}))
    final_score_p95 = stats_by_dataset["v25c_baseline"]["enhanced_keyword"]["final_score"]["p95"]
    keyword_score_p95 = stats_by_dataset["v25c_baseline"]["enhanced_keyword"]["keyword_score"]["p95"]
    exact_boost = float(calibration.get("exact_filename_boost", 0) or 0)
    stem_boost = float(calibration.get("stem_token_boost", 0) or 0)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "v25e_calibration_config": calibration,
        "stats_by_dataset": stats_by_dataset,
        "scale_interpretation": {
            "baseline_enhanced_keyword_final_score_p95": final_score_p95,
            "baseline_enhanced_keyword_keyword_score_p95": keyword_score_p95,
            "exact_filename_boost": exact_boost,
            "stem_token_boost": stem_boost,
            "exact_boost_to_final_p95_ratio": safe_ratio(exact_boost, final_score_p95),
            "two_stem_tokens_to_final_p95_ratio": safe_ratio(2 * stem_boost, final_score_p95),
            "interpretation": (
                "Current fusion final_score often reaches the hundreds because metadata and keyword rank points "
                "plus keyword_score are additive. Filename calibration therefore needs a boost on the same scale "
                "to change the Top-K order, but it should be regression-tested because this is still a rule-based score."
            ),
        },
        "outputs": {
            "score_scale_json": str(resolve_project_path(config["outputs"]["score_scale_json"])),
        },
    }


def collect_sources(dataset: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for record in dataset.get("per_question_results", []):
        if str(record.get("retrieval_mode")) == mode:
            sources.extend(dict(source or {}) for source in record.get("final_sources", []))
    return sources


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def describe(values: list[float | None]) -> dict[str, Any]:
    items = sorted(value for value in values if value is not None)
    if not items:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": len(items),
        "min": round(items[0], 6),
        "max": round(items[-1], 6),
        "mean": round(sum(items) / len(items), 6),
        "median": round(statistics.median(items), 6),
        "p90": round(percentile(items, 0.90), 6),
        "p95": round(percentile(items, 0.95), 6),
    }


def percentile(items: list[float], fraction: float) -> float:
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    index = (len(items) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(items) - 1)
    weight = index - lower
    return items[lower] * (1 - weight) + items[upper] * weight


def safe_ratio(numerator: float, denominator: float | None) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 6)


def write_output(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    output = resolve_project_path(config["outputs"]["score_scale_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    print("V2.6 retrieval score scale analysis completed.")
    print(json.dumps(result["scale_interpretation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
