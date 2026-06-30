from __future__ import annotations

import csv
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_expanded_retrieval_eval_v25b import (  # noqa: E402
    evaluate_record,
    is_negative,
    load_collection_profile,
)
from scripts.run_retrieval_eval_v25c import (  # noqa: E402
    compute_metrics_by_mode,
    expected_presence_strict,
    load_eval_set,
)
from src.retrieval.real_smoke_retriever import retrieve_real_smoke  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_calibration_v26.yaml"
MODE = "enhanced_keyword"
CSV_FIELDS = [
    "combo_id",
    "exact_filename_boost",
    "stem_token_boost",
    "metadata_title_boost",
    "category_match_boost",
    "eval_set",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_8",
    "mrr",
    "source_diversity_pass_rate",
    "q014_rank",
    "q014_hit_at_5",
    "obvious_degradation",
    "accepted_by_v26_rules",
]


def main() -> int:
    configure_stdout()
    try:
        result = run_sensitivity()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001
        print("V2.6 calibration sensitivity analysis failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_sensitivity(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    collection_name = str(config["collection"]["collection_name"])
    persist_directory = resolve_project_path(config["collection"]["persist_directory"])
    base_items = load_eval_set(resolve_project_path(config["inputs"]["base_eval_set_jsonl"]))
    regression_items = load_eval_set(resolve_project_path(config["outputs"]["regression_set_jsonl"]))
    profile = load_collection_profile(persist_directory, collection_name)
    baseline = load_json(resolve_project_path(config["inputs"]["v25c_eval_json"]))
    v25e = load_json(resolve_project_path(config["inputs"]["v25e_eval_json"]))
    baseline_metrics = baseline["metrics_by_mode"][MODE]
    v25e_metrics = v25e["metrics_by_mode"][MODE]

    rows: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []
    combos = build_combos(config)
    for combo_index, calibration in enumerate(combos, start=1):
        combo_id = f"v26_combo_{combo_index:03d}"
        for eval_name, items in [("v25c_30", base_items), ("v26_regression", regression_items)]:
            eval_result = evaluate_items(
                items=items,
                calibration=calibration,
                persist_directory=persist_directory,
                collection_name=collection_name,
                profile=profile,
            )
            metrics = eval_result["metrics_by_mode"][MODE]
            q014 = eval_result["q014_result"]
            accepted = accepted_by_rules(metrics, q014, baseline_metrics, config.get("acceptance", {}))
            row = {
                "combo_id": combo_id,
                "exact_filename_boost": calibration["exact_filename_boost"],
                "stem_token_boost": calibration["stem_token_boost"],
                "metadata_title_boost": calibration["metadata_title_boost"],
                "category_match_boost": calibration["category_match_boost"],
                "eval_set": eval_name,
                "hit_at_1": metrics["hit_at_1"],
                "hit_at_3": metrics["hit_at_3"],
                "hit_at_5": metrics["hit_at_5"],
                "hit_at_8": metrics["hit_at_8"],
                "mrr": metrics["mrr"],
                "source_diversity_pass_rate": metrics["source_diversity_pass_rate"],
                "q014_rank": q014.get("first_source_hit_rank"),
                "q014_hit_at_5": q014.get("source_hit_at_5"),
                "obvious_degradation": obvious_degradation(metrics, baseline_metrics, config.get("acceptance", {})),
                "accepted_by_v26_rules": accepted,
            }
            rows.append(row)
            detail_records.append(
                {
                    **row,
                    "metrics": metrics,
                    "q014_result": q014,
                    "calibration": calibration,
                    "top_sources_q014": q014.get("top_sources", []),
                }
            )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "mode": MODE,
        "combo_count": len(combos),
        "eval_sets": {
            "v25c_30": len(base_items),
            "v26_regression": len(regression_items),
        },
        "baseline_v25c_enhanced_keyword": baseline_metrics,
        "v25e_current_enhanced_keyword": v25e_metrics,
        "rows": rows,
        "detail_records": detail_records,
        "accepted_combo_ids": sorted(
            {row["combo_id"] for row in rows if row["eval_set"] == "v25c_30" and row["accepted_by_v26_rules"]}
        ),
        "best_small_boost_combo": select_smallest_accepted(rows),
        "outputs": {
            "sensitivity_json": str(resolve_project_path(config["outputs"]["sensitivity_json"])),
            "sensitivity_csv": str(resolve_project_path(config["outputs"]["sensitivity_csv"])),
        },
    }


def build_combos(config: dict[str, Any]) -> list[dict[str, Any]]:
    sensitivity = config["sensitivity"]
    combos: list[dict[str, Any]] = []
    for exact, stem, title, category in itertools.product(
        sensitivity["exact_filename_boost_values"],
        sensitivity["stem_token_boost_values"],
        sensitivity["metadata_title_boost_values"],
        sensitivity["category_match_boost_values"],
    ):
        combos.append(
            {
                "enabled": True,
                "exact_filename_boost": float(exact),
                "stem_token_boost": float(stem),
                "metadata_title_boost": float(title),
                "category_match_boost": float(category),
                "matched_source_token_min_len": int(sensitivity.get("matched_source_token_min_len", 3)),
                "ignored_source_tokens": list(sensitivity.get("ignored_source_tokens", [])),
                "apply_to_modes": list(sensitivity.get("apply_to_modes", [MODE])),
            }
        )
    return combos


def evaluate_items(
    *,
    items: list[dict[str, Any]],
    calibration: dict[str, Any],
    persist_directory: Path,
    collection_name: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for item in items:
        presence = expected_presence_strict(item, profile)
        retrieval = retrieve_real_smoke(
            question=str(item["question"]),
            persist_directory=persist_directory,
            collection_name=collection_name,
            retrieval_mode=MODE,
            dense_top_k=5,
            metadata_top_k=8,
            keyword_top_k=8,
            final_top_k=8,
            max_chunks_per_source=2,
            enable_source_diversity=True,
            source_key="imported_source",
            calibration_config=calibration,
        )
        final_sources = retrieval.debug["final_sources"]
        metrics = evaluate_record(
            item=item,
            final_sources=final_sources,
            presence=presence,
            source_key="imported_source",
            max_chunks_per_source=2,
        )
        record = {
            "id": item["id"],
            "retrieval_mode": MODE,
            "is_negative": is_negative(item),
            "final_sources": final_sources,
            **presence,
            **metrics,
        }
        records.append(record)
    metrics_by_mode = compute_metrics_by_mode(records, [MODE])
    q014 = next((record for record in records if str(record.get("id")) == "v25c_q014"), {})
    if q014:
        q014 = {
            "first_source_hit_rank": q014.get("first_source_hit_rank"),
            "source_hit_at_5": q014.get("source_hit_at_5"),
            "source_hit_at_8": q014.get("source_hit_at_8"),
            "top_sources": summarize_top_sources(q014.get("final_sources", [])),
        }
    return {"metrics_by_mode": metrics_by_mode, "q014_result": q014, "records": records}


def summarize_top_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source in sources[:8]:
        rows.append(
            {
                "rank": source.get("rank"),
                "source": source.get("source"),
                "final_score": source.get("final_score"),
                "calibration_score": source.get("calibration_score", 0.0),
                "calibration_reasons": source.get("calibration_reasons", []),
            }
        )
    return rows


def accepted_by_rules(
    metrics: dict[str, Any],
    q014: dict[str, Any],
    baseline_metrics: dict[str, Any],
    acceptance: dict[str, Any],
) -> bool:
    q014_ok = bool(q014.get("source_hit_at_5")) if acceptance.get("q014_required_hit_at_5", True) else True
    hit_drop = float(baseline_metrics["hit_at_5"]) - float(metrics["hit_at_5"])
    mrr_drop = float(baseline_metrics["mrr"]) - float(metrics["mrr"])
    diversity_required = float(acceptance.get("source_diversity_required", 1.0))
    return bool(
        q014_ok
        and hit_drop <= float(acceptance.get("max_hit_at_5_drop_allowed", 0.0)) + 1e-9
        and mrr_drop <= float(acceptance.get("max_mrr_drop_allowed", 0.02)) + 1e-9
        and float(metrics["source_diversity_pass_rate"]) >= diversity_required
    )


def obvious_degradation(metrics: dict[str, Any], baseline: dict[str, Any], acceptance: dict[str, Any]) -> bool:
    hit_drop = float(baseline["hit_at_5"]) - float(metrics["hit_at_5"])
    mrr_drop = float(baseline["mrr"]) - float(metrics["mrr"])
    return bool(
        hit_drop > float(acceptance.get("max_hit_at_5_drop_allowed", 0.0)) + 1e-9
        or mrr_drop > float(acceptance.get("max_mrr_drop_allowed", 0.02)) + 1e-9
    )


def select_smallest_accepted(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    accepted = [row for row in rows if row["eval_set"] == "v25c_30" and row["accepted_by_v26_rules"]]
    if not accepted:
        return None
    return sorted(
        accepted,
        key=lambda row: (
            float(row["exact_filename_boost"]) + 2 * float(row["stem_token_boost"]) + float(row["metadata_title_boost"]),
            -float(row["mrr"]),
            -float(row["hit_at_5"]),
        ),
    )[0]


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    output_json = resolve_project_path(config["outputs"]["sensitivity_json"])
    output_csv = resolve_project_path(config["outputs"]["sensitivity_csv"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


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
    print("V2.6 calibration sensitivity analysis completed.")
    print(
        json.dumps(
            {
                "combo_count": result["combo_count"],
                "eval_sets": result["eval_sets"],
                "accepted_combo_ids": result["accepted_combo_ids"],
                "best_small_boost_combo": result["best_small_boost_combo"],
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
