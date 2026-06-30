from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_repair.yaml"
KEYWORD_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_keyword.yaml"
REGRESSION_MODES = ["enhanced", "enhanced_keyword"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.real_smoke_retriever import retrieve_real_smoke


Condition = Callable[[list[dict[str, Any]]], tuple[bool, str]]


TEST_CASES: list[dict[str, Any]] = [
    {
        "test_id": "test_1_train_entry",
        "question": "多无人机空战环境中的训练入口文件是什么？",
        "conditions": [
            {
                "name": "recall_train_jsbsim",
                "check": lambda sources: contains_any(sources, ["train_jsbsim.py"]),
                "description": "final_sources has at least one imported_source containing train_jsbsim.py",
            }
        ],
    },
    {
        "test_id": "test_2_experiment_or_config",
        "question": "这些资料中有哪些实验结果文件或配置文件？",
        "conditions": [
            {
                "name": "has_experiments",
                "check": lambda sources: has_category(sources, "experiments"),
                "description": "final_sources has category_dir=experiments",
            },
            {
                "name": "has_configs",
                "check": lambda sources: has_category(sources, "configs"),
                "description": "final_sources has category_dir=configs",
            },
        ],
    },
    {
        "test_id": "test_3_reward_or_task_config",
        "question": "JSBSim 环境中有哪些奖励函数或任务配置？",
        "conditions": [
            {
                "name": "has_code",
                "check": lambda sources: has_category(sources, "code"),
                "description": "final_sources has category_dir=code",
            },
            {
                "name": "has_configs",
                "check": lambda sources: has_category(sources, "configs"),
                "description": "final_sources has category_dir=configs",
            },
            {
                "name": "has_reward_or_task_keyword",
                "check": lambda sources: contains_any(
                    sources,
                    ["reward", "task", "multiplecombat", "event_driven_reward", "shoot_penalty_reward"],
                ),
                "description": "final source paths contain reward/task/multiplecombat related keyword",
            },
        ],
    },
    {
        "test_id": "test_4_opd_or_prediction_notes",
        "question": "OPD 或对手预测相关笔记主要讨论了什么？",
        "conditions": [
            {
                "name": "has_notes_or_reports",
                "check": lambda sources: has_any_category(sources, ["notes", "thesis_or_reports"]),
                "description": "final_sources has notes or thesis_or_reports",
            },
            {
                "name": "has_opd_route_risk_keyword",
                "check": lambda sources: contains_any(sources, ["opd", "route", "risk", "stage125", "stage13"]),
                "description": "final source paths contain OPD/route/risk/stage keyword",
            },
        ],
    },
]


def main() -> int:
    configure_stdout()
    try:
        config = load_config()
        result = run_regression(config)
        write_outputs(config, result)
    except Exception as exc:  # noqa: BLE001 - lightweight checks should report explicit failures.
        print("Retrieval regression failed to run.")
        print(f"reason: {exc}")
        return 1

    print_summary(result)
    return 0 if result["overall_passed"] else 1


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    keyword_config = load_keyword_config()
    if keyword_config:
        config.update(keyword_config)
    return config


def load_keyword_config(path: Path = KEYWORD_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {path}")
    keyword = raw.get("keyword_retrieval") or {}
    enhanced_keyword = raw.get("enhanced_keyword") or {}
    fields = keyword.get("fields") if isinstance(keyword, dict) else {}
    return {
        "collection_name": raw.get("collection_name"),
        "persist_directory": raw.get("persist_directory"),
        "keyword_top_k": keyword.get("top_k", 8) if isinstance(keyword, dict) else 8,
        "keyword_lowercase": keyword.get("lowercase", True) if isinstance(keyword, dict) else True,
        "keyword_min_token_len": keyword.get("min_token_len", 2) if isinstance(keyword, dict) else 2,
        "keyword_field_weights": normalize_keyword_field_weights(fields if isinstance(fields, dict) else {}),
        "enhanced_keyword": enhanced_keyword if isinstance(enhanced_keyword, dict) else {},
    }


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


def run_regression(config: dict[str, Any]) -> dict[str, Any]:
    retrieval_config = config.get("retrieval") or {}
    if not isinstance(retrieval_config, dict):
        retrieval_config = {}

    collection_name = str(config.get("collection_name") or config["real_smoke_collection_name"])
    persist_directory = PROJECT_ROOT / str(config.get("persist_directory") or config["real_smoke_persist_directory"])

    records: list[dict[str, Any]] = []
    for mode in REGRESSION_MODES:
        kwargs = build_retrieval_kwargs(config, retrieval_config, mode)
        for case in TEST_CASES:
            retrieval = retrieve_real_smoke(
                question=str(case["question"]),
                persist_directory=persist_directory,
                collection_name=collection_name,
                retrieval_mode=mode,
                **kwargs,
            )
            final_sources = retrieval.debug["final_sources"]
            condition_results = evaluate_conditions(case["conditions"], final_sources)
            diversity = evaluate_source_diversity(
                final_sources,
                source_key=kwargs["source_key"],
                max_chunks_per_source=kwargs["max_chunks_per_source"],
            )
            passed = all(item["passed"] for item in condition_results) and diversity["passed"]
            records.append(
                {
                    "test_id": case["test_id"],
                    "question": case["question"],
                    "retrieval_mode": mode,
                    "passed": passed,
                    "query_type": retrieval.debug["query_type"],
                    "target_categories": retrieval.debug["target_categories"],
                    "expanded_query": retrieval.debug["expanded_query"],
                    "condition_results": condition_results,
                    "source_diversity": diversity,
                    "dense_sources": retrieval.debug["dense_sources"],
                    "metadata_sources": retrieval.debug["metadata_sources"],
                    "keyword_sources": retrieval.debug.get("keyword_sources", []),
                    "metadata_candidate_count": retrieval.debug.get("metadata_candidate_count", 0),
                    "keyword_candidate_count": retrieval.debug.get("keyword_candidate_count", 0),
                    "final_sources": final_sources,
                }
            )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "retrieval_modes": REGRESSION_MODES,
        "test_count": len(records),
        "passed_count": sum(1 for item in records if item["passed"]),
        "failed_count": sum(1 for item in records if not item["passed"]),
        "overall_passed": all(item["passed"] for item in records),
        "tests": records,
    }


def build_retrieval_kwargs(config: dict[str, Any], retrieval_config: dict[str, Any], mode: str) -> dict[str, Any]:
    values = {
        "dense_top_k": int(retrieval_config.get("dense_top_k", 5)),
        "metadata_top_k": int(retrieval_config.get("metadata_top_k", 8)),
        "keyword_top_k": int(config.get("keyword_top_k", 8)),
        "final_top_k": int(retrieval_config.get("final_top_k", 8)),
        "max_chunks_per_source": int(retrieval_config.get("max_chunks_per_source", 2)),
        "enable_source_diversity": bool(retrieval_config.get("enable_source_diversity", True)),
        "source_key": str(retrieval_config.get("source_key", "imported_source")),
        "keyword_lowercase": bool(config.get("keyword_lowercase", True)),
        "keyword_min_token_len": int(config.get("keyword_min_token_len", 2)),
        "keyword_field_weights": config.get("keyword_field_weights") or None,
    }
    if mode == "enhanced_keyword":
        enhanced_keyword = config.get("enhanced_keyword") or {}
        if isinstance(enhanced_keyword, dict):
            for key in ["dense_top_k", "metadata_top_k", "keyword_top_k", "final_top_k", "max_chunks_per_source"]:
                if key in enhanced_keyword:
                    values[key] = int(enhanced_keyword[key])
    return values


def evaluate_conditions(conditions: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for condition in conditions:
        passed, detail = condition["check"](sources)
        results.append(
            {
                "name": condition["name"],
                "description": condition["description"],
                "passed": passed,
                "detail": detail,
            }
        )
    return results


def evaluate_source_diversity(
    sources: list[dict[str, Any]],
    *,
    source_key: str,
    max_chunks_per_source: int,
) -> dict[str, Any]:
    values = [source_identifier(item, source_key) for item in sources]
    counts = dict(Counter(values))
    violations = {
        source: count
        for source, count in counts.items()
        if count > max_chunks_per_source
    }
    return {
        "passed": not violations,
        "source_key": source_key,
        "max_chunks_per_source": max_chunks_per_source,
        "source_counts": counts,
        "violations": violations,
    }


def has_category(sources: list[dict[str, Any]], category: str) -> tuple[bool, str]:
    categories = [str(item.get("category_dir", "")) for item in sources]
    return category in categories, f"categories={sorted(set(categories))}"


def has_any_category(sources: list[dict[str, Any]], categories: list[str]) -> tuple[bool, str]:
    seen = {str(item.get("category_dir", "")) for item in sources}
    return bool(seen.intersection(categories)), f"categories={sorted(seen)}"


def contains_any(sources: list[dict[str, Any]], keywords: list[str]) -> tuple[bool, str]:
    text = " ".join(
        " ".join(
            [
                str(item.get("source", "")),
                str(item.get("imported_source", "")),
                str(item.get("original_source", "")),
            ]
        )
        for item in sources
    ).lower()
    matched = [keyword for keyword in keywords if keyword.lower() in text]
    return bool(matched), f"matched={matched}"


def source_identifier(source: dict[str, Any], source_key: str) -> str:
    value = str(source.get(source_key, "")).strip()
    if value:
        return value
    for fallback_key in ["imported_source", "source", "original_source", "chunk_id"]:
        value = str(source.get(fallback_key, "")).strip()
        if value:
            return value
    return "unknown_source"


def write_outputs(config: dict[str, Any], result: dict[str, Any]) -> None:
    output_config = config.get("regression_tests") or {}
    if not isinstance(output_config, dict):
        output_config = {}
    output_json = PROJECT_ROOT / str(output_config.get("output_json", "storage/logs/retrieval_regression_v2b3.json"))
    output_jsonl = PROJECT_ROOT / str(output_config.get("output_jsonl", "storage/logs/retrieval_regression_v2b3.jsonl"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_jsonl.open("w", encoding="utf-8") as file:
        for record in result["tests"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(result: dict[str, Any]) -> None:
    print("Retrieval regression completed.")
    print(f"collection_name: {result['collection_name']}")
    print(f"persist_directory: {result['persist_directory']}")
    print(f"overall_passed: {result['overall_passed']}")
    print(f"passed_count: {result['passed_count']}")
    print(f"failed_count: {result['failed_count']}")
    for record in result["tests"]:
        print(f"\n{record['test_id']} mode={record['retrieval_mode']}: passed={record['passed']}")
        print(f"question: {record['question']}")
        print(f"query_type: {record['query_type']}")
        print(f"target_categories: {record['target_categories']}")
        print(f"source_diversity_passed: {record['source_diversity']['passed']}")
        for condition in record["condition_results"]:
            print(f"  condition {condition['name']}: passed={condition['passed']} detail={condition['detail']}")
        print("  final_sources:")
        for source in record["final_sources"]:
            print(
                "    "
                f"rank={source.get('rank')} imported_source={source.get('imported_source')} "
                f"category_dir={source.get('category_dir')} chunk_id={source.get('chunk_id')} "
                f"source_rank_within_file={source.get('source_rank_within_file')} "
                f"final_score={source.get('final_score')}"
            )


if __name__ == "__main__":
    sys.exit(main())
