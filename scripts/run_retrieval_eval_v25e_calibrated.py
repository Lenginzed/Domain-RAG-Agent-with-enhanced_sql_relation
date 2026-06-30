from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_expanded_retrieval_eval_v25b import (  # noqa: E402
    compact_ranking,
    evaluate_record,
    is_negative,
    load_collection_profile,
)
from scripts.run_retrieval_eval_v25c import (  # noqa: E402
    CSV_FIELDS as V25C_CSV_FIELDS,
    compute_metrics_by_mode,
    expected_presence_strict,
    load_eval_set,
)
from src.retrieval.real_smoke_retriever import retrieve_real_smoke  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_calibration_v25e.yaml"

CSV_FIELDS = [
    *V25C_CSV_FIELDS,
    "calibration_applied",
    "expected_source_rank_top8",
]


def main() -> int:
    configure_stdout()
    try:
        result = run_calibrated_eval()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001
        print("V2.5e calibrated retrieval evaluation failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_calibrated_eval(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    collection_name = str(config["collection"]["collection_name"])
    persist_directory = resolve_project_path(config["collection"]["persist_directory"])
    eval_path = resolve_project_path(config["inputs"]["eval_set_jsonl"])
    retrieval_config = dict(config.get("retrieval", {}))
    modes = list(retrieval_config.get("modes", ["dense", "enhanced", "keyword", "enhanced_keyword"]))
    calibration_config = dict(config.get("calibration", {}))
    apply_to_modes = set(str(mode) for mode in calibration_config.get("apply_to_modes", []))
    questions = load_eval_set(eval_path)
    collection_profile = load_collection_profile(persist_directory, collection_name)

    strict_presence_by_id = {
        str(item["id"]): expected_presence_strict(item, collection_profile)
        for item in questions
    }
    non_negative_absent_ids = [
        item_id
        for item_id, presence in strict_presence_by_id.items()
        if presence["expected_source_absent_from_collection"]
        and not is_negative(next(item for item in questions if str(item["id"]) == item_id))
    ]
    if non_negative_absent_ids:
        raise ValueError(f"Non-negative expected sources absent from collection: {non_negative_absent_ids}")

    records: list[dict[str, Any]] = []
    for item in questions:
        presence = strict_presence_by_id[str(item["id"])]
        for mode in modes:
            kwargs = retrieval_kwargs(retrieval_config, mode)
            calibration_applied = mode in apply_to_modes and bool(calibration_config.get("enabled", True))
            retrieval = retrieve_real_smoke(
                question=str(item["question"]),
                persist_directory=persist_directory,
                collection_name=collection_name,
                retrieval_mode=mode,
                calibration_config=calibration_config if calibration_applied else None,
                **kwargs,
            )
            final_sources = retrieval.debug["final_sources"]
            metrics = evaluate_record(
                item=item,
                final_sources=final_sources,
                presence=presence,
                source_key=kwargs["source_key"],
                max_chunks_per_source=kwargs["max_chunks_per_source"],
            )
            records.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "query_type": item["query_type"],
                    "difficulty": item["difficulty"],
                    "retrieval_mode": mode,
                    "is_negative": is_negative(item),
                    "expected_sources_contains": item["expected_sources_contains"],
                    "expected_categories": item["expected_categories"],
                    "required_all_categories": item["required_all_categories"],
                    "notes": item.get("notes", ""),
                    "query_classifier_type": retrieval.debug["query_type"],
                    "expanded_query": retrieval.debug["expanded_query"],
                    "final_sources": final_sources,
                    "final_categories": sorted(
                        {str(src.get("category_dir", "")) for src in final_sources if src.get("category_dir")}
                    ),
                    "calibration_applied": calibration_applied,
                    "expected_source_rank_top8": metrics.get("first_source_hit_rank"),
                    **presence,
                    **metrics,
                }
            )

    metrics_by_mode = compute_metrics_by_mode(records, modes)
    target_case_results = summarize_target_cases(records, config.get("target_cases", []))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_path": str(eval_path),
        "config_path": str(config_path),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "collection_profile": {
            key: value for key, value in collection_profile.items() if key != "source_text_blob"
        },
        "total_questions": len(questions),
        "non_negative_questions": sum(1 for item in questions if not is_negative(item)),
        "negative_questions": sum(1 for item in questions if is_negative(item)),
        "modes": modes,
        "calibration_enabled": bool(calibration_config.get("enabled", True)),
        "calibration_apply_to_modes": sorted(apply_to_modes),
        "calibration_config": calibration_config,
        "metrics_by_mode": metrics_by_mode,
        "compact_ranking": compact_ranking(metrics_by_mode),
        "query_type_distribution": dict(Counter(str(item["query_type"]) for item in questions)),
        "difficulty_distribution": dict(Counter(str(item["difficulty"]) for item in questions)),
        "expected_source_absent_question_ids": sorted(
            item_id
            for item_id, presence in strict_presence_by_id.items()
            if presence["expected_source_absent_from_collection"]
        ),
        "expected_source_absent_count": sum(
            len(presence["missing_expected_sources"]) for presence in strict_presence_by_id.values()
        ),
        "target_case_results": target_case_results,
        "per_question_results": records,
        "outputs": {
            "calibrated_eval_json": str(resolve_project_path(config["outputs"]["calibrated_eval_json"])),
            "calibrated_eval_jsonl": str(resolve_project_path(config["outputs"]["calibrated_eval_jsonl"])),
            "calibrated_eval_csv": str(resolve_project_path(config["outputs"]["calibrated_eval_csv"])),
        },
    }


def retrieval_kwargs(retrieval: dict[str, Any], mode: str) -> dict[str, Any]:
    values = {
        "dense_top_k": int(retrieval.get("dense_top_k", 5)),
        "metadata_top_k": int(retrieval.get("metadata_top_k", 8)),
        "keyword_top_k": int(retrieval.get("keyword_top_k", 8)),
        "final_top_k": int(retrieval.get("final_top_k", 8)),
        "max_chunks_per_source": int(retrieval.get("max_chunks_per_source", 2)),
        "enable_source_diversity": bool(retrieval.get("enable_source_diversity", True)),
        "source_key": str(retrieval.get("source_key", "imported_source")),
    }
    if mode == "dense":
        values["final_top_k"] = min(values["final_top_k"], values["dense_top_k"])
    return values


def summarize_target_cases(records: list[dict[str, Any]], target_cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for case in target_cases:
        case_id = str(case.get("id"))
        rows = [record for record in records if str(record.get("id")) == case_id]
        summary[case_id] = {
            "expected_source": case.get("expected_source", ""),
            "question": case.get("question", ""),
            "by_mode": {
                str(row["retrieval_mode"]): {
                    "first_source_hit_rank": row.get("first_source_hit_rank"),
                    "source_hit_at_1": row.get("source_hit_at_1"),
                    "source_hit_at_3": row.get("source_hit_at_3"),
                    "source_hit_at_5": row.get("source_hit_at_5"),
                    "source_hit_at_8": row.get("source_hit_at_8"),
                    "calibration_applied": row.get("calibration_applied"),
                    "top_sources": [
                        {
                            "rank": source.get("rank"),
                            "source": source.get("source"),
                            "category_dir": source.get("category_dir"),
                            "final_score": source.get("final_score"),
                            "calibration_score": source.get("calibration_score"),
                            "calibration_reasons": source.get("calibration_reasons"),
                        }
                        for source in row.get("final_sources", [])[:8]
                    ],
                }
                for row in rows
            },
        }
    return summary


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    output_json = resolve_project_path(config["outputs"]["calibrated_eval_json"])
    output_jsonl = resolve_project_path(config["outputs"]["calibrated_eval_jsonl"])
    output_csv = resolve_project_path(config["outputs"]["calibrated_eval_csv"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as file:
        for record in result["per_question_results"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(output_csv, result["per_question_results"])


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in CSV_FIELDS}
            row["final_categories"] = "|".join(record.get("final_categories", []))
            row["missing_expected_sources"] = "|".join(record.get("missing_expected_sources", []))
            row["returned_source_count"] = len(record.get("final_sources", []))
            writer.writerow(row)


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


def print_summary(result: dict[str, Any]) -> None:
    print("V2.5e calibrated retrieval evaluation completed.")
    print(
        json.dumps(
            {
                "collection_name": result["collection_name"],
                "total_questions": result["total_questions"],
                "calibration_apply_to_modes": result["calibration_apply_to_modes"],
                "metrics_by_mode": result["metrics_by_mode"],
                "target_case_results": result["target_case_results"],
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
