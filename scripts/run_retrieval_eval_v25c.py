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
    HIT_KS,
    average,
    compact_ranking,
    evaluate_record,
    evaluate_source_diversity,
    first_source_hit_rank,
    is_negative,
    load_collection_profile,
    retrieval_kwargs,
)
from src.retrieval.real_smoke_retriever import retrieve_real_smoke  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_eval_v25c.yaml"

CSV_FIELDS = [
    "id",
    "question",
    "query_type",
    "difficulty",
    "retrieval_mode",
    "is_negative",
    "expected_source_absent_from_collection",
    "missing_expected_sources",
    "source_metric_eligible",
    "category_metric_eligible",
    "source_hit_at_1",
    "source_hit_at_3",
    "source_hit_at_5",
    "source_hit_at_8",
    "first_source_hit_rank",
    "mrr",
    "category_hit",
    "required_all_categories_hit",
    "source_diversity_pass",
    "returned_source_count",
    "final_categories",
]


def main() -> int:
    configure_stdout()
    try:
        result = run_retrieval_eval()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001 - fail explicitly.
        print("V2.5c retrieval evaluation failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_retrieval_eval(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    collection_name = str(config["collection"]["collection_name"])
    persist_directory = resolve_project_path(config["collection"]["persist_directory"])
    eval_path = resolve_project_path(config["outputs"]["eval_set_jsonl"])
    modes = list(config.get("retrieval", {}).get("modes", ["dense", "enhanced", "keyword", "enhanced_keyword"]))
    questions = load_eval_set(eval_path)
    collection_profile = load_collection_profile(persist_directory, collection_name)

    strict_presence_by_id = {
        str(item["id"]): expected_presence_strict(item, collection_profile)
        for item in questions
    }
    expected_source_absent_ids = sorted(
        item_id for item_id, presence in strict_presence_by_id.items() if presence["expected_source_absent_from_collection"]
    )
    non_negative_absent_ids = [
        item_id
        for item_id in expected_source_absent_ids
        if not is_negative(next(item for item in questions if str(item["id"]) == item_id))
    ]
    if non_negative_absent_ids:
        raise ValueError(f"Non-negative expected sources absent from collection: {non_negative_absent_ids}")

    records: list[dict[str, Any]] = []
    for item in questions:
        presence = strict_presence_by_id[str(item["id"])]
        for mode in modes:
            kwargs = retrieval_kwargs({"retrieval": config.get("retrieval", {})}, mode)
            retrieval = retrieve_real_smoke(
                question=str(item["question"]),
                persist_directory=persist_directory,
                collection_name=collection_name,
                retrieval_mode=mode,
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
                    **presence,
                    **metrics,
                }
            )

    metrics_by_mode = compute_metrics_by_mode(records, modes)
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
        "metrics_by_mode": metrics_by_mode,
        "compact_ranking": compact_ranking(metrics_by_mode),
        "query_type_distribution": dict(Counter(str(item["query_type"]) for item in questions)),
        "difficulty_distribution": dict(Counter(str(item["difficulty"]) for item in questions)),
        "expected_source_absent_question_ids": expected_source_absent_ids,
        "expected_source_absent_count": sum(
            len(presence["missing_expected_sources"]) for presence in strict_presence_by_id.values()
        ),
        "expected_category_absent_question_ids": sorted(
            item_id for item_id, presence in strict_presence_by_id.items() if presence["expected_category_absent_from_collection"]
        ),
        "per_question_results": records,
        "outputs": {
            "retrieval_eval_json": str(resolve_project_path(config["outputs"]["eval_results_json"])),
            "retrieval_eval_jsonl": str(resolve_project_path(config["outputs"]["eval_results_jsonl"])),
            "retrieval_eval_csv": str(resolve_project_path(config["outputs"]["eval_results_csv"])),
        },
    }


def expected_presence_strict(item: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    expected_terms = [str(term).lower() for term in item.get("expected_sources_contains", [])]
    expected_categories = [str(category) for category in item.get("expected_categories", [])]
    required_categories = [str(category) for category in item.get("required_all_categories", [])]
    source_blob = str(profile.get("source_text_blob", ""))
    categories = set(profile.get("categories", []))
    missing_sources = [term for term in expected_terms if term not in source_blob]
    missing_expected_categories = [category for category in expected_categories if category not in categories]
    missing_required_categories = [category for category in required_categories if category not in categories]
    return {
        "expected_source_present_in_collection": not missing_sources,
        "expected_source_absent_from_collection": bool(missing_sources),
        "missing_expected_sources": missing_sources,
        "expected_category_present_in_collection": not missing_expected_categories,
        "expected_category_absent_from_collection": bool(missing_expected_categories),
        "missing_expected_categories": missing_expected_categories,
        "required_categories_present_in_collection": not missing_required_categories,
        "required_categories_absent_from_collection": bool(missing_required_categories),
        "missing_required_categories": missing_required_categories,
    }


def compute_metrics_by_mode(records: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for mode in modes:
        mode_records = [record for record in records if record["retrieval_mode"] == mode]
        non_negative = [record for record in mode_records if not record["is_negative"]]
        source_eligible = [record for record in non_negative if record["source_metric_eligible"]]
        category_eligible = [record for record in non_negative if record["category_metric_eligible"]]
        required_eligible = [
            record for record in non_negative if record["required_all_categories_metric_eligible"]
        ]
        metrics[mode] = {
            "hit_at_1": average(record["source_hit_at_1"] for record in source_eligible),
            "hit_at_3": average(record["source_hit_at_3"] for record in source_eligible),
            "hit_at_5": average(record["source_hit_at_5"] for record in source_eligible),
            "hit_at_8": average(record["source_hit_at_8"] for record in source_eligible),
            "mrr": average(record["mrr"] for record in source_eligible),
            "category_hit_rate": average(record["category_hit"] for record in category_eligible),
            "required_all_categories_hit_rate": average(record["required_all_categories_hit"] for record in required_eligible),
            "source_diversity_pass_rate": average(record["source_diversity_pass"] for record in mode_records),
            "negative_cases_returned_nonempty": sum(
                1
                for record in mode_records
                if record["is_negative"] and not record["negative_case_behavior"]["final_sources_empty"]
            ),
            "evaluated_non_negative": len(non_negative),
            "source_metric_eligible_count": len(source_eligible),
            "category_metric_eligible_count": len(category_eligible),
            "required_all_categories_eligible_count": len(required_eligible),
            "expected_source_absent_count": sum(
                len(record.get("missing_expected_sources", [])) for record in non_negative
            ),
            "expected_category_absent_count": sum(record["expected_category_absent_from_collection"] for record in non_negative),
            "evaluated_negative": sum(1 for record in mode_records if record["is_negative"]),
        }
    return metrics


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    output_json = resolve_project_path(config["outputs"]["eval_results_json"])
    output_jsonl = resolve_project_path(config["outputs"]["eval_results_jsonl"])
    output_csv = resolve_project_path(config["outputs"]["eval_results_csv"])
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
    print("V2.5c retrieval evaluation completed.")
    print(
        json.dumps(
            {
                "collection_name": result["collection_name"],
                "persist_directory": result["persist_directory"],
                "total_questions": result["total_questions"],
                "non_negative_questions": result["non_negative_questions"],
                "negative_questions": result["negative_questions"],
                "expected_source_absent_count": result["expected_source_absent_count"],
                "expected_source_absent_question_ids": result["expected_source_absent_question_ids"],
                "metrics_by_mode": result["metrics_by_mode"],
                "compact_ranking": result["compact_ranking"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
