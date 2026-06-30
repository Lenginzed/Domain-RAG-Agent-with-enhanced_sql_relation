from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPAIR_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_repair.yaml"
KEYWORD_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_keyword.yaml"
OUTPUT_JSON = PROJECT_ROOT / "storage" / "logs" / "keyword_retrieval_compare_v2b4.json"
OUTPUT_JSONL = PROJECT_ROOT / "storage" / "logs" / "keyword_retrieval_compare_v2b4.jsonl"
MODES = ["dense", "enhanced", "keyword", "enhanced_keyword"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.real_smoke_retriever import retrieve_real_smoke


QUESTIONS: list[dict[str, Any]] = [
    {
        "question_id": "q1_train_entry",
        "question": "多无人机空战环境中的训练入口文件是什么？",
        "expected_terms": ["train_jsbsim.py"],
        "expected_categories": ["code"],
    },
    {
        "question_id": "q2_experiment_or_config",
        "question": "这些资料中有哪些实验结果文件或配置文件？",
        "expected_terms": [],
        "expected_categories": ["experiments", "configs"],
    },
    {
        "question_id": "q3_reward_or_task",
        "question": "JSBSim 环境中有哪些奖励函数或任务配置？",
        "expected_terms": ["reward", "task", "multiplecombat", "event_driven_reward", "shoot_penalty_reward"],
        "expected_categories": ["code", "configs"],
    },
    {
        "question_id": "q4_opd_notes",
        "question": "OPD 或对手预测相关笔记主要讨论了什么？",
        "expected_terms": ["opd", "route", "risk", "stage125", "stage13"],
        "expected_categories": ["notes"],
        "alternative_categories": ["thesis_or_reports"],
    },
    {
        "question_id": "q5_event_driven_reward",
        "question": "EventDrivenReward 是什么？",
        "expected_terms": ["event_driven_reward.py", "eventdrivenreward", "reward"],
        "expected_categories": ["code"],
    },
    {
        "question_id": "q6_shoot_penalty_reward",
        "question": "shoot penalty reward 在哪个文件里？",
        "expected_terms": ["shoot", "penalty", "reward", "shoot_penalty_reward"],
        "expected_categories": ["code", "configs"],
    },
    {
        "question_id": "q7_offensive_curriculum",
        "question": "offensive curriculum 配置在哪里？",
        "expected_terms": ["offensive_curriculum", "curriculum", ".yaml"],
        "expected_categories": ["configs"],
    },
    {
        "question_id": "q8_stage125_opd_hint",
        "question": "stage125 OPD-HINT 后续路线讨论了什么？",
        "expected_terms": ["stage125", "opd", "hint", "route"],
        "expected_categories": ["notes"],
    },
]


def main() -> int:
    configure_stdout()
    try:
        config = load_config()
        result = run_compare(config)
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001
        print("Keyword retrieval comparison failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config() -> dict[str, Any]:
    repair = load_yaml(REPAIR_CONFIG_PATH)
    keyword = load_yaml(KEYWORD_CONFIG_PATH)
    fields = ((keyword.get("keyword_retrieval") or {}).get("fields") or {})
    return {
        "collection_name": keyword.get("collection_name") or repair.get("real_smoke_collection_name"),
        "persist_directory": keyword.get("persist_directory") or repair.get("real_smoke_persist_directory"),
        "retrieval": repair.get("retrieval") or {},
        "keyword_retrieval": keyword.get("keyword_retrieval") or {},
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


def run_compare(config: dict[str, Any]) -> dict[str, Any]:
    persist_directory = PROJECT_ROOT / str(config["persist_directory"])
    collection_name = str(config["collection_name"])
    records: list[dict[str, Any]] = []
    for question in QUESTIONS:
        for mode in MODES:
            kwargs = build_retrieval_kwargs(config, mode)
            retrieval = retrieve_real_smoke(
                question=str(question["question"]),
                persist_directory=persist_directory,
                collection_name=collection_name,
                retrieval_mode=mode,
                **kwargs,
            )
            final_sources = retrieval.debug["final_sources"]
            final_categories = sorted({str(item.get("category_dir", "")) for item in final_sources if item.get("category_dir")})
            contains_expected_source, source_note = contains_expected_terms(final_sources, question.get("expected_terms", []))
            categories_ok, category_note = contains_expected_categories(
                final_categories,
                question.get("expected_categories", []),
                question.get("alternative_categories", []),
            )
            source_diversity = evaluate_source_diversity(
                final_sources,
                source_key=kwargs["source_key"],
                max_chunks_per_source=kwargs["max_chunks_per_source"],
            )
            notes = [source_note, category_note]
            if not source_diversity["passed"]:
                notes.append(f"source diversity violations={source_diversity['violations']}")
            records.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "retrieval_mode": mode,
                    "query_type": retrieval.debug["query_type"],
                    "expanded_query": retrieval.debug["expanded_query"],
                    "final_sources": final_sources,
                    "final_categories": final_categories,
                    "contains_expected_source": contains_expected_source,
                    "expected_categories_present": categories_ok,
                    "source_diversity_passed": source_diversity["passed"],
                    "source_diversity": source_diversity,
                    "notes": notes,
                }
            )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "modes": MODES,
        "question_count": len(QUESTIONS),
        "record_count": len(records),
        "records": records,
        "summary_by_mode": summarize_by_mode(records),
    }


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


def contains_expected_terms(sources: list[dict[str, Any]], terms: list[str]) -> tuple[bool, str]:
    if not terms:
        return True, "no explicit expected source term"
    text = " ".join(
        " ".join(
            [
                str(item.get("source", "")),
                str(item.get("imported_source", "")),
                str(item.get("original_source", "")),
                " ".join(str(term) for term in item.get("matched_terms", [])),
            ]
        )
        for item in sources
    ).lower()
    matched = [term for term in terms if str(term).lower() in text]
    return bool(matched), f"matched_expected_terms={matched}"


def contains_expected_categories(
    final_categories: list[str],
    required: list[str],
    alternatives: list[str],
) -> tuple[bool, str]:
    if not required:
        return True, "no required categories"
    seen = set(final_categories)
    missing = [category for category in required if category not in seen]
    if missing and alternatives and any(category in seen for category in alternatives):
        return True, f"categories={final_categories}; satisfied_by_alternative={alternatives}"
    return not missing, f"categories={final_categories}; missing={missing}"


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


def source_identifier(source: dict[str, Any], source_key: str) -> str:
    value = str(source.get(source_key, "")).strip()
    if value:
        return value
    for fallback_key in ["imported_source", "source", "original_source", "chunk_id"]:
        value = str(source.get(fallback_key, "")).strip()
        if value:
            return value
    return "unknown_source"


def summarize_by_mode(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in MODES:
        mode_records = [record for record in records if record["retrieval_mode"] == mode]
        summary[mode] = {
            "records": len(mode_records),
            "contains_expected_source_count": sum(1 for record in mode_records if record["contains_expected_source"]),
            "expected_categories_present_count": sum(1 for record in mode_records if record["expected_categories_present"]),
            "source_diversity_passed_count": sum(1 for record in mode_records if record["source_diversity_passed"]),
        }
    return summary


def write_outputs(result: dict[str, Any]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for record in result["records"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(result: dict[str, Any]) -> None:
    print("Keyword retrieval comparison completed.")
    print(f"collection_name: {result['collection_name']}")
    print(f"persist_directory: {result['persist_directory']}")
    print(f"question_count: {result['question_count']}")
    print(f"record_count: {result['record_count']}")
    print(f"summary_by_mode: {json.dumps(result['summary_by_mode'], ensure_ascii=False)}")
    for record in result["records"]:
        print(
            f"{record['question_id']} mode={record['retrieval_mode']} "
            f"contains_expected_source={record['contains_expected_source']} "
            f"categories_ok={record['expected_categories_present']} "
            f"diversity={record['source_diversity_passed']} "
            f"categories={record['final_categories']}"
        )


if __name__ == "__main__":
    sys.exit(main())
