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
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_v2b5.jsonl"
REPAIR_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_repair.yaml"
KEYWORD_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_keyword.yaml"
OUTPUT_JSON = PROJECT_ROOT / "storage" / "logs" / "retrieval_eval_v2b5.json"
OUTPUT_JSONL = PROJECT_ROOT / "storage" / "logs" / "retrieval_eval_v2b5.jsonl"
OUTPUT_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_v2b5_results.csv"
OUTPUT_MD = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_v2b5_summary.md"
MODES = ["dense", "enhanced", "keyword", "enhanced_keyword"]
HIT_KS = [1, 3, 5, 8]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.real_smoke_retriever import retrieve_real_smoke


def main() -> int:
    configure_stdout()
    try:
        result = run_eval()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001 - evaluation should fail loudly.
        print("Retrieval evaluation failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_eval(
    eval_path: Path = EVAL_PATH,
    repair_config_path: Path = REPAIR_CONFIG_PATH,
    keyword_config_path: Path = KEYWORD_CONFIG_PATH,
) -> dict[str, Any]:
    questions = load_eval_set(eval_path)
    config = load_config(repair_config_path, keyword_config_path)
    persist_directory = PROJECT_ROOT / str(config["persist_directory"])
    collection_name = str(config["collection_name"])

    records: list[dict[str, Any]] = []
    for item in questions:
        for mode in MODES:
            kwargs = build_retrieval_kwargs(config, mode)
            retrieval = retrieve_real_smoke(
                question=str(item["question"]),
                persist_directory=persist_directory,
                collection_name=collection_name,
                retrieval_mode=mode,
                **kwargs,
            )
            final_sources = retrieval.debug["final_sources"]
            metrics = evaluate_record(
                item,
                final_sources,
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
                    "final_categories": sorted({str(src.get("category_dir", "")) for src in final_sources if src.get("category_dir")}),
                    **metrics,
                }
            )

    metrics_by_mode = compute_metrics_by_mode(records)
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_path": str(eval_path),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "total_questions": len(questions),
        "non_negative_questions": sum(1 for item in questions if not is_negative(item)),
        "negative_questions": sum(1 for item in questions if is_negative(item)),
        "modes": MODES,
        "metrics_by_mode": metrics_by_mode,
        "compact_ranking": compact_ranking(metrics_by_mode),
        "query_type_distribution": dict(Counter(str(item["query_type"]) for item in questions)),
        "difficulty_distribution": dict(Counter(str(item["difficulty"]) for item in questions)),
        "per_question_results": records,
        "markdown_summary": build_markdown_summary(metrics_by_mode),
    }
    return result


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            validate_eval_item(item, line_number)
            items.append(item)
    return items


def validate_eval_item(item: dict[str, Any], line_number: int) -> None:
    required = [
        "id",
        "question",
        "query_type",
        "difficulty",
        "expected_sources_contains",
        "expected_categories",
        "required_all_categories",
        "notes",
    ]
    missing = [field for field in required if field not in item]
    if missing:
        raise ValueError(f"Eval item line {line_number} missing fields: {missing}")
    for field in ["expected_sources_contains", "expected_categories", "required_all_categories"]:
        if not isinstance(item[field], list):
            raise ValueError(f"Eval item line {line_number} field {field} must be a list")


def load_config(repair_config_path: Path, keyword_config_path: Path) -> dict[str, Any]:
    repair = load_yaml(repair_config_path)
    keyword = load_yaml(keyword_config_path)
    keyword_retrieval = keyword.get("keyword_retrieval") or {}
    fields = keyword_retrieval.get("fields") if isinstance(keyword_retrieval, dict) else {}
    return {
        "collection_name": keyword.get("collection_name") or repair.get("real_smoke_collection_name"),
        "persist_directory": keyword.get("persist_directory") or repair.get("real_smoke_persist_directory"),
        "retrieval": repair.get("retrieval") or {},
        "keyword_retrieval": keyword_retrieval,
        "enhanced_keyword": keyword.get("enhanced_keyword") or {},
        "keyword_field_weights": normalize_keyword_field_weights(fields if isinstance(fields, dict) else {}),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def normalize_keyword_field_weights(fields: dict[str, Any]) -> dict[str, float]:
    mapping = {
        "content_weight": "content",
        "filename_weight": "filename",
        "path_weight": "path",
        "title_section_weight": "title_section",
        "category_doc_type_weight": "category_doc_type",
    }
    weights: dict[str, float] = {}
    for source_key, target_key in mapping.items():
        if source_key in fields:
            weights[target_key] = float(fields[source_key])
    return weights


def build_retrieval_kwargs(config: dict[str, Any], mode: str) -> dict[str, Any]:
    retrieval = config.get("retrieval") or {}
    keyword = config.get("keyword_retrieval") or {}
    enhanced_keyword = config.get("enhanced_keyword") or {}
    values = {
        "dense_top_k": int(retrieval.get("dense_top_k", 5)),
        "metadata_top_k": int(retrieval.get("metadata_top_k", 8)),
        "keyword_top_k": int(keyword.get("top_k", 8)),
        "final_top_k": int(retrieval.get("final_top_k", 8)),
        "max_chunks_per_source": int(retrieval.get("max_chunks_per_source", keyword.get("max_chunks_per_source", 2))),
        "enable_source_diversity": bool(retrieval.get("enable_source_diversity", True)),
        "source_key": str(retrieval.get("source_key", "imported_source")),
        "keyword_lowercase": bool(keyword.get("lowercase", True)),
        "keyword_min_token_len": int(keyword.get("min_token_len", 2)),
        "keyword_field_weights": config.get("keyword_field_weights") or None,
    }
    if mode == "keyword":
        values["final_top_k"] = int(keyword.get("top_k", values["final_top_k"]))
        values["max_chunks_per_source"] = int(keyword.get("max_chunks_per_source", values["max_chunks_per_source"]))
    if mode == "enhanced_keyword" and isinstance(enhanced_keyword, dict):
        for key in ["dense_top_k", "metadata_top_k", "keyword_top_k", "final_top_k", "max_chunks_per_source"]:
            if key in enhanced_keyword:
                values[key] = int(enhanced_keyword[key])
    return values


def evaluate_record(
    item: dict[str, Any],
    final_sources: list[dict[str, Any]],
    *,
    source_key: str,
    max_chunks_per_source: int,
) -> dict[str, Any]:
    negative = is_negative(item)
    expected_terms = [str(term).lower() for term in item["expected_sources_contains"]]
    final_categories = [str(src.get("category_dir", "")) for src in final_sources]
    first_hit_rank = first_source_hit_rank(final_sources, expected_terms)
    source_hit_at = {f"source_hit_at_{k}": bool(first_hit_rank and first_hit_rank <= k) for k in HIT_KS}
    category_hit = bool(set(final_categories).intersection(set(str(cat) for cat in item["expected_categories"])))
    required_all_categories = [str(cat) for cat in item["required_all_categories"]]
    required_all_categories_hit = bool(required_all_categories) and all(category in final_categories for category in required_all_categories)
    diversity = evaluate_source_diversity(final_sources, source_key=source_key, max_chunks_per_source=max_chunks_per_source)
    negative_behavior = {
        "is_negative": negative,
        "final_sources_empty": len(final_sources) == 0,
        "returned_source_count": len(final_sources),
        "needs_answer_stage_insufficient_evidence_check": negative and len(final_sources) > 0,
    }
    return {
        **source_hit_at,
        "first_source_hit_rank": first_hit_rank,
        "mrr": 0.0 if not first_hit_rank else 1.0 / first_hit_rank,
        "category_hit": category_hit,
        "required_all_categories_hit": required_all_categories_hit if required_all_categories else None,
        "source_diversity_pass": diversity["passed"],
        "source_diversity": diversity,
        "negative_case_behavior": negative_behavior,
    }


def is_negative(item: dict[str, Any]) -> bool:
    return (
        str(item.get("query_type", "")) == "negative"
        or (not item.get("expected_sources_contains") and not item.get("expected_categories"))
    )


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
    for fallback_key in ["imported_source", "source", "original_source", "chunk_id"]:
        value = str(source.get(fallback_key, "")).strip()
        if value:
            return value
    return "unknown_source"


def compute_metrics_by_mode(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for mode in MODES:
        mode_records = [record for record in records if record["retrieval_mode"] == mode]
        non_negative = [record for record in mode_records if not record["is_negative"]]
        category_eligible = [record for record in non_negative if record["expected_categories"]]
        required_eligible = [record for record in non_negative if record["required_all_categories"]]
        metrics[mode] = {
            "hit_at_1": average(record["source_hit_at_1"] for record in non_negative),
            "hit_at_3": average(record["source_hit_at_3"] for record in non_negative),
            "hit_at_5": average(record["source_hit_at_5"] for record in non_negative),
            "hit_at_8": average(record["source_hit_at_8"] for record in non_negative),
            "mrr": average(record["mrr"] for record in non_negative),
            "category_hit_rate": average(record["category_hit"] for record in category_eligible),
            "required_all_categories_hit_rate": average(record["required_all_categories_hit"] for record in required_eligible),
            "source_diversity_pass_rate": average(record["source_diversity_pass"] for record in mode_records),
            "negative_cases_returned_nonempty": sum(
                1
                for record in mode_records
                if record["is_negative"] and not record["negative_case_behavior"]["final_sources_empty"]
            ),
            "evaluated_non_negative": len(non_negative),
            "evaluated_negative": sum(1 for record in mode_records if record["is_negative"]),
        }
    return metrics


def average(values: Any) -> float:
    items = [float(value) for value in values if value is not None]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def compact_ranking(metrics_by_mode: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            }
        )
    return sorted(rows, key=lambda row: (-row["score"], row["mode"]))


def build_markdown_summary(metrics_by_mode: dict[str, Any]) -> str:
    headers = [
        "mode",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "Hit@8",
        "MRR",
        "category_hit",
        "required_all",
        "source_diversity",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for mode in MODES:
        metrics = metrics_by_mode[mode]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    fmt(metrics["hit_at_1"]),
                    fmt(metrics["hit_at_3"]),
                    fmt(metrics["hit_at_5"]),
                    fmt(metrics["hit_at_8"]),
                    fmt(metrics["mrr"]),
                    fmt(metrics["category_hit_rate"]),
                    fmt(metrics["required_all_categories_hit_rate"]),
                    fmt(metrics["source_diversity_pass_rate"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def write_outputs(result: dict[str, Any]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for record in result["per_question_results"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(result["per_question_results"])
    OUTPUT_MD.write_text(result["markdown_summary"] + "\n", encoding="utf-8")


def write_csv(records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question",
        "query_type",
        "difficulty",
        "retrieval_mode",
        "is_negative",
        "source_hit_at_1",
        "source_hit_at_3",
        "source_hit_at_5",
        "source_hit_at_8",
        "first_source_hit_rank",
        "mrr",
        "category_hit",
        "required_all_categories_hit",
        "source_diversity_pass",
        "final_categories",
        "top_sources",
        "negative_returned_source_count",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "query_type": record["query_type"],
                    "difficulty": record["difficulty"],
                    "retrieval_mode": record["retrieval_mode"],
                    "is_negative": record["is_negative"],
                    "source_hit_at_1": record["source_hit_at_1"],
                    "source_hit_at_3": record["source_hit_at_3"],
                    "source_hit_at_5": record["source_hit_at_5"],
                    "source_hit_at_8": record["source_hit_at_8"],
                    "first_source_hit_rank": record["first_source_hit_rank"],
                    "mrr": record["mrr"],
                    "category_hit": record["category_hit"],
                    "required_all_categories_hit": record["required_all_categories_hit"],
                    "source_diversity_pass": record["source_diversity_pass"],
                    "final_categories": "|".join(record["final_categories"]),
                    "top_sources": "|".join(str(src.get("imported_source", "")) for src in record["final_sources"][:5]),
                    "negative_returned_source_count": record["negative_case_behavior"]["returned_source_count"],
                }
            )


def print_summary(result: dict[str, Any]) -> None:
    print("Retrieval evaluation completed.")
    print(f"eval_path: {result['eval_path']}")
    print(f"collection_name: {result['collection_name']}")
    print(f"persist_directory: {result['persist_directory']}")
    print(f"total_questions: {result['total_questions']}")
    print(f"non_negative_questions: {result['non_negative_questions']}")
    print(f"negative_questions: {result['negative_questions']}")
    print("metrics_by_mode:")
    for mode in MODES:
        print(f"  {mode}: {json.dumps(result['metrics_by_mode'][mode], ensure_ascii=False)}")
    print("compact_ranking:")
    for row in result["compact_ranking"]:
        print(f"  {row}")
    print("markdown_summary:")
    print(result["markdown_summary"])


if __name__ == "__main__":
    sys.exit(main())
