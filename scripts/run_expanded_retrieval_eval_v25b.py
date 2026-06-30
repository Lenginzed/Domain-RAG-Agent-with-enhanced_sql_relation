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

from src.indexes.vector_index import load_index
from src.retrieval.real_smoke_retriever import retrieve_real_smoke


CONFIG_PATH = PROJECT_ROOT / "config" / "expanded_ingest_v25b.yaml"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_v2b5.jsonl"
HIT_KS = [1, 3, 5, 8]

CSV_FIELDS = [
    "id",
    "question",
    "query_type",
    "difficulty",
    "retrieval_mode",
    "is_negative",
    "expected_source_absent_from_collection",
    "expected_category_absent_from_collection",
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
        result = run_expanded_retrieval_eval()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001 - eval should fail explicitly.
        print("V2.5b retrieval evaluation failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_expanded_retrieval_eval(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    outputs = config["outputs"]
    collection_name = str(config["chroma"]["collection_name"])
    persist_directory = resolve_project_path(config["chroma"]["persist_directory"])
    modes = list(config.get("retrieval", {}).get("modes", ["dense", "enhanced", "keyword", "enhanced_keyword"]))
    questions = load_eval_set(EVAL_PATH)
    collection_profile = load_collection_profile(persist_directory, collection_name)

    records: list[dict[str, Any]] = []
    for item in questions:
        presence = expected_presence(item, collection_profile)
        for mode in modes:
            kwargs = retrieval_kwargs(config, mode)
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
        "eval_path": str(EVAL_PATH),
        "config_path": str(config_path),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "collection_profile": collection_profile,
        "total_questions": len(questions),
        "non_negative_questions": sum(1 for item in questions if not is_negative(item)),
        "negative_questions": sum(1 for item in questions if is_negative(item)),
        "modes": modes,
        "metrics_by_mode": metrics_by_mode,
        "compact_ranking": compact_ranking(metrics_by_mode),
        "query_type_distribution": dict(Counter(str(item["query_type"]) for item in questions)),
        "difficulty_distribution": dict(Counter(str(item["difficulty"]) for item in questions)),
        "expected_source_absent_question_ids": sorted(
            {
                record["id"]
                for record in records
                if record["retrieval_mode"] == modes[0] and record["expected_source_absent_from_collection"]
            }
        ),
        "expected_category_absent_question_ids": sorted(
            {
                record["id"]
                for record in records
                if record["retrieval_mode"] == modes[0] and record["expected_category_absent_from_collection"]
            }
        ),
        "per_question_results": records,
        "outputs": {
            "retrieval_eval_json": str(resolve_project_path(outputs["retrieval_eval_json"])),
            "retrieval_eval_jsonl": str(resolve_project_path(outputs["retrieval_eval_jsonl"])),
            "retrieval_eval_csv": str(resolve_project_path(outputs["retrieval_eval_csv"])),
        },
    }


def load_collection_profile(persist_directory: Path, collection_name: str) -> dict[str, Any]:
    vector_store = load_index(persist_directory=persist_directory, collection_name=collection_name)
    raw = vector_store._collection.get(include=["metadatas"])  # noqa: SLF001
    metadatas = [dict(item or {}) for item in raw.get("metadatas", [])]
    source_texts: list[str] = []
    categories: set[str] = set()
    imported_sources: set[str] = set()
    for metadata in metadatas:
        values = [
            str(metadata.get("source", "")),
            str(metadata.get("imported_source", "")),
            str(metadata.get("original_source", "")),
        ]
        source_texts.append(" ".join(values).lower())
        if metadata.get("category_dir"):
            categories.add(str(metadata["category_dir"]))
        if metadata.get("imported_source"):
            imported_sources.add(str(metadata["imported_source"]))
    return {
        "chunk_count": len(metadatas),
        "source_count": len(imported_sources),
        "categories": sorted(categories),
        "source_text_blob": "\n".join(source_texts),
    }


def expected_presence(item: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    expected_terms = [str(term).lower() for term in item.get("expected_sources_contains", [])]
    expected_categories = [str(category) for category in item.get("expected_categories", [])]
    required_categories = [str(category) for category in item.get("required_all_categories", [])]
    source_blob = str(profile.get("source_text_blob", ""))
    categories = set(profile.get("categories", []))
    expected_source_present = True
    if expected_terms:
        expected_source_present = any(term in source_blob for term in expected_terms)
    expected_category_present = True
    if expected_categories:
        expected_category_present = any(category in categories for category in expected_categories)
    required_categories_present = True
    if required_categories:
        required_categories_present = all(category in categories for category in required_categories)
    return {
        "expected_source_present_in_collection": expected_source_present,
        "expected_source_absent_from_collection": not expected_source_present,
        "expected_category_present_in_collection": expected_category_present,
        "expected_category_absent_from_collection": not expected_category_present,
        "required_categories_present_in_collection": required_categories_present,
        "required_categories_absent_from_collection": not required_categories_present,
    }


def retrieval_kwargs(config: dict[str, Any], mode: str) -> dict[str, Any]:
    retrieval = config.get("retrieval", {})
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


def evaluate_record(
    *,
    item: dict[str, Any],
    final_sources: list[dict[str, Any]],
    presence: dict[str, Any],
    source_key: str,
    max_chunks_per_source: int,
) -> dict[str, Any]:
    negative = is_negative(item)
    expected_terms = [str(term).lower() for term in item.get("expected_sources_contains", [])]
    final_categories = [str(src.get("category_dir", "")) for src in final_sources]
    first_hit_rank = first_source_hit_rank(final_sources, expected_terms)
    source_metric_eligible = not negative and bool(expected_terms) and not presence["expected_source_absent_from_collection"]
    category_metric_eligible = (
        not negative
        and bool(item.get("expected_categories"))
        and not presence["expected_category_absent_from_collection"]
    )
    required_metric_eligible = (
        not negative
        and bool(item.get("required_all_categories"))
        and not presence["required_categories_absent_from_collection"]
    )
    source_hit_at = {
        f"source_hit_at_{k}": bool(source_metric_eligible and first_hit_rank and first_hit_rank <= k)
        for k in HIT_KS
    }
    category_hit = bool(category_metric_eligible and set(final_categories).intersection(set(item["expected_categories"])))
    required_all_hit = bool(
        required_metric_eligible
        and all(str(category) in final_categories for category in item.get("required_all_categories", []))
    )
    diversity = evaluate_source_diversity(
        final_sources,
        source_key=source_key,
        max_chunks_per_source=max_chunks_per_source,
    )
    return {
        **source_hit_at,
        "source_metric_eligible": source_metric_eligible,
        "category_metric_eligible": category_metric_eligible,
        "required_all_categories_metric_eligible": required_metric_eligible,
        "first_source_hit_rank": first_hit_rank,
        "mrr": 0.0 if not (source_metric_eligible and first_hit_rank) else 1.0 / first_hit_rank,
        "category_hit": category_hit,
        "required_all_categories_hit": required_all_hit if item.get("required_all_categories") else None,
        "source_diversity_pass": diversity["passed"],
        "source_diversity": diversity,
        "negative_case_behavior": {
            "is_negative": negative,
            "final_sources_empty": len(final_sources) == 0,
            "returned_source_count": len(final_sources),
            "needs_answer_stage_insufficient_evidence_check": negative and len(final_sources) > 0,
        },
    }


def first_source_hit_rank(final_sources: list[dict[str, Any]], expected_terms: list[str]) -> int | None:
    if not expected_terms:
        return None
    for index, source in enumerate(final_sources, start=1):
        text = " ".join(
            [
                str(source.get("source", "")),
                str(source.get("imported_source", "")),
                str(source.get("original_source", "")),
            ]
        ).lower()
        if any(term in text for term in expected_terms):
            return index
    return None


def evaluate_source_diversity(
    sources: list[dict[str, Any]],
    *,
    source_key: str,
    max_chunks_per_source: int,
) -> dict[str, Any]:
    values = [source_identifier(source, source_key) for source in sources]
    counts = dict(Counter(values))
    violations = {source: count for source, count in counts.items() if count > max_chunks_per_source}
    return {
        "passed": not violations,
        "source_key": source_key,
        "max_chunks_per_source": max_chunks_per_source,
        "source_counts": counts,
        "violations": violations,
    }


def source_identifier(source: dict[str, Any], source_key: str) -> str:
    value = str(source.get(source_key, "")).strip()
    if value:
        return value
    for fallback in ["imported_source", "source", "original_source", "chunk_id"]:
        value = str(source.get(fallback, "")).strip()
        if value:
            return value
    return "unknown_source"


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
            "expected_source_absent_count": sum(record["expected_source_absent_from_collection"] for record in non_negative),
            "expected_category_absent_count": sum(record["expected_category_absent_from_collection"] for record in non_negative),
            "evaluated_negative": sum(1 for record in mode_records if record["is_negative"]),
        }
    return metrics


def average(values: Any) -> float:
    items = [float(value) for value in values if value is not None]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def compact_ranking(metrics_by_mode: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, metrics in metrics_by_mode.items():
        score = (
            float(metrics["hit_at_5"]) * 0.5
            + float(metrics["mrr"]) * 0.3
            + float(metrics["required_all_categories_hit_rate"]) * 0.2
        )
        rows.append(
            {
                "mode": mode,
                "score": round(score, 6),
                "hit_at_5": metrics["hit_at_5"],
                "mrr": metrics["mrr"],
                "required_all_categories_hit_rate": metrics["required_all_categories_hit_rate"],
                "source_metric_eligible_count": metrics["source_metric_eligible_count"],
            }
        )
    return sorted(rows, key=lambda row: (-row["score"], row["mode"]))


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))
    return items


def is_negative(item: dict[str, Any]) -> bool:
    return (
        str(item.get("query_type", "")) == "negative"
        or (not item.get("expected_sources_contains") and not item.get("expected_categories"))
    )


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    outputs = config["outputs"]
    output_json = resolve_project_path(outputs["retrieval_eval_json"])
    output_jsonl = resolve_project_path(outputs["retrieval_eval_jsonl"])
    output_csv = resolve_project_path(outputs["retrieval_eval_csv"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_jsonl.open("w", encoding="utf-8") as file:
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
    print("V2.5b retrieval evaluation completed.")
    print(json.dumps({
        "collection_name": result["collection_name"],
        "persist_directory": result["persist_directory"],
        "total_questions": result["total_questions"],
        "non_negative_questions": result["non_negative_questions"],
        "negative_questions": result["negative_questions"],
        "expected_source_absent_question_ids": result["expected_source_absent_question_ids"],
        "metrics_by_mode": result["metrics_by_mode"],
        "compact_ranking": result["compact_ranking"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
