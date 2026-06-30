from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_v2b5.jsonl"
ANSWER_CONFIG_PATH = PROJECT_ROOT / "config" / "answer_verification.yaml"
REPAIR_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_repair.yaml"
KEYWORD_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_keyword.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.citation_checker import check_citations
from src.generation.evidence_gate import evaluate_evidence
from src.generation.llm import OllamaLLM
from src.retrieval.real_smoke_retriever import retrieve_real_smoke

from scripts.run_real_smoke_rag import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    build_empty_answer_fallback,
    build_prompt,
)


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Run V2c answer-stage evaluation on the fixed 10-question sample.")
    parser.add_argument(
        "--output-version",
        default="v2c0",
        choices=["v2c0", "v2c1"],
        help="Output filename version. v2c0 preserves the original default; v2c1 writes answer_eval_v2c1.* files.",
    )
    args = parser.parse_args()
    try:
        result = run_answer_eval(output_version=args.output_version)
        write_outputs(result, output_version=args.output_version)
    except Exception as exc:  # noqa: BLE001 - answer eval should fail explicitly.
        print("Answer evaluation failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_answer_eval(
    eval_path: Path = EVAL_PATH,
    answer_config_path: Path = ANSWER_CONFIG_PATH,
    repair_config_path: Path = REPAIR_CONFIG_PATH,
    keyword_config_path: Path = KEYWORD_CONFIG_PATH,
    output_version: str = "v2c0",
) -> dict[str, Any]:
    answer_config = load_yaml(answer_config_path)
    retrieval_config = load_retrieval_config(answer_config, repair_config_path, keyword_config_path)
    eval_items = select_eval_items(load_eval_set(eval_path), answer_config)
    llm = OllamaLLM()

    records: list[dict[str, Any]] = []
    for item in eval_items:
        mode = str(answer_config.get("retrieval_mode", "enhanced_keyword"))
        retrieval = retrieve_real_smoke(
            question=str(item["question"]),
            persist_directory=PROJECT_ROOT / str(retrieval_config["persist_directory"]),
            collection_name=str(retrieval_config["collection_name"]),
            retrieval_mode=mode,
            **build_retrieval_kwargs(retrieval_config, mode),
        )
        gate = evaluate_evidence(
            question=str(item["question"]),
            retrieval_result=retrieval,
            eval_item=item,
            config=answer_config.get("evidence_gate") or {},
        )
        llm_called = False
        if gate["evidence_sufficient"]:
            llm_called = True
            answer = llm.generate(build_prompt(str(item["question"]), retrieval.documents))
            if not answer.strip():
                answer = build_empty_answer_fallback(retrieval.documents)
        else:
            answer = INSUFFICIENT_EVIDENCE_ANSWER

        citation = check_citations(
            answer=answer,
            sources=retrieval.documents,
            config=answer_config.get("citation_check") or {},
        )
        unsupported_claim_count = len(citation["unsupported_claims"])
        weak_claim_count = int(citation.get("weak_claim_count", 0))
        supported_claim_ratio = float(citation["supported_claim_ratio"])
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "query_type": item["query_type"],
                "difficulty": item["difficulty"],
                "is_negative": is_negative(item),
                "retrieval_mode": mode,
                "evidence_sufficient": gate["evidence_sufficient"],
                "evidence_gate_reason": gate["reason"],
                "evidence_gate_signals": gate["signals"],
                "llm_called": llm_called,
                "answer": answer,
                "answer_summary": summarize_text(answer, 260),
                "insufficient_answer": citation["insufficient_answer"],
                "supported_claim_ratio": supported_claim_ratio,
                "citation_check_passed": citation["citation_check_passed"],
                "supported_claim_count": len(citation["supported_claims"]),
                "checked_claim_count": int(citation.get("checked_claim_count", len(citation.get("checked_claims", [])))),
                "ignored_claim_count": int(citation.get("ignored_claim_count", len(citation.get("ignored_claims", [])))),
                "weak_claim_count": weak_claim_count,
                "unsupported_claim_count": unsupported_claim_count,
                "claim_type_counts": citation.get("claim_type_counts", {}),
                "manual_judgment": "",
                "manual_notes": "",
                "needs_human_review": unsupported_claim_count > 0 or weak_claim_count > 0 or supported_claim_ratio < 0.8,
                "unsupported_claims": citation["unsupported_claims"],
                "ignored_claims": citation.get("ignored_claims", []),
                "weak_claims": citation.get("weak_claims", []),
                "citation_check": citation,
                "final_sources": retrieval.debug["final_sources"],
                "final_categories": sorted(
                    {str(source.get("category_dir", "")) for source in retrieval.debug["final_sources"] if source.get("category_dir")}
                ),
            }
        )

    return build_summary(
        records=records,
        eval_path=eval_path,
        answer_config=answer_config,
        retrieval_config=retrieval_config,
        output_version=output_version,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def load_retrieval_config(answer_config: dict[str, Any], repair_config_path: Path, keyword_config_path: Path) -> dict[str, Any]:
    repair = load_yaml(repair_config_path)
    keyword = load_yaml(keyword_config_path)
    keyword_retrieval = keyword.get("keyword_retrieval") or {}
    fields = keyword_retrieval.get("fields") if isinstance(keyword_retrieval, dict) else {}
    return {
        "collection_name": answer_config.get("collection_name") or keyword.get("collection_name") or repair.get("real_smoke_collection_name"),
        "persist_directory": answer_config.get("persist_directory") or keyword.get("persist_directory") or repair.get("real_smoke_persist_directory"),
        "retrieval": repair.get("retrieval") or {},
        "keyword_retrieval": keyword_retrieval,
        "enhanced_keyword": keyword.get("enhanced_keyword") or {},
        "keyword_field_weights": normalize_keyword_field_weights(fields if isinstance(fields, dict) else {}),
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
    if mode == "enhanced_keyword" and isinstance(enhanced_keyword, dict):
        for key in ["dense_top_k", "metadata_top_k", "keyword_top_k", "final_top_k", "max_chunks_per_source"]:
            if key in enhanced_keyword:
                values[key] = int(enhanced_keyword[key])
    return values


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            for field in ["id", "question", "query_type", "difficulty", "expected_sources_contains", "expected_categories", "required_all_categories"]:
                if field not in item:
                    raise ValueError(f"Eval item line {line_number} missing {field}")
            rows.append(item)
    return rows


def select_eval_items(items: list[dict[str, Any]], answer_config: dict[str, Any]) -> list[dict[str, Any]]:
    eval_config = answer_config.get("answer_eval") or {}
    non_negative_size = int(eval_config.get("non_negative_sample_size", 8))
    include_all_negative = bool(eval_config.get("include_all_negative", True))
    selected_non_negative = [item for item in items if not is_negative(item)][:non_negative_size]
    selected_negative = [item for item in items if is_negative(item)] if include_all_negative else [item for item in items if is_negative(item)][:2]
    selected_ids = {item["id"] for item in selected_non_negative + selected_negative}
    return [item for item in items if item["id"] in selected_ids]


def is_negative(item: dict[str, Any]) -> bool:
    return (
        str(item.get("query_type", "")) == "negative"
        or (not item.get("expected_sources_contains") and not item.get("expected_categories"))
    )


def build_summary(
    *,
    records: list[dict[str, Any]],
    eval_path: Path,
    answer_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    output_version: str,
) -> dict[str, Any]:
    negative_records = [record for record in records if record["is_negative"]]
    non_negative_records = [record for record in records if not record["is_negative"]]
    ordinary_answers = [record for record in records if not record["insufficient_answer"]]
    metrics = {
        "total_eval_questions": len(records),
        "negative_questions": len(negative_records),
        "non_negative_questions": len(non_negative_records),
        "evidence_gate_pass_rate": average(
            (not record["evidence_sufficient"]) if record["is_negative"] else record["evidence_sufficient"]
            for record in records
        ),
        "negative_refusal_rate": average(record["insufficient_answer"] for record in negative_records),
        "non_negative_answered_rate": average(record["evidence_sufficient"] and record["llm_called"] for record in non_negative_records),
        "citation_check_pass_rate": average(record["citation_check_passed"] for record in ordinary_answers),
        "avg_supported_claim_ratio": average(record["supported_claim_ratio"] for record in ordinary_answers),
        "unsupported_claim_count": sum(int(record["unsupported_claim_count"]) for record in ordinary_answers),
        "weak_claim_count": sum(int(record["weak_claim_count"]) for record in ordinary_answers),
        "ignored_claim_count": sum(int(record["ignored_claim_count"]) for record in records),
        "checked_claim_count": sum(int(record["checked_claim_count"]) for record in records),
        "needs_human_review_count": sum(1 for record in records if record["needs_human_review"]),
        "insufficient_answer_count": sum(1 for record in records if record["insufficient_answer"]),
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_version": output_version,
        "eval_path": str(eval_path),
        "retrieval_mode": answer_config.get("retrieval_mode", "enhanced_keyword"),
        "collection_name": retrieval_config["collection_name"],
        "persist_directory": str(PROJECT_ROOT / str(retrieval_config["persist_directory"])),
        "metrics": metrics,
        "records": records,
    }


def average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def write_outputs(
    result: dict[str, Any],
    answer_config_path: Path = ANSWER_CONFIG_PATH,
    output_version: str = "v2c0",
) -> None:
    answer_config = load_yaml(answer_config_path)
    outputs = answer_config.get("outputs") or {}
    output_json, output_jsonl, output_csv = resolve_output_paths(outputs, output_version)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_jsonl.open("w", encoding="utf-8") as file:
        for record in result["records"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(output_csv, result["records"])


def resolve_output_paths(outputs: dict[str, Any], output_version: str) -> tuple[Path, Path, Path]:
    default_json = "storage/logs/answer_eval_v2c0.json"
    default_jsonl = "storage/logs/answer_eval_v2c0.jsonl"
    default_csv = "data/eval/answer_eval_v2c0_results.csv"
    output_json = PROJECT_ROOT / str(outputs.get("answer_eval_json", default_json))
    output_jsonl = PROJECT_ROOT / str(outputs.get("answer_eval_jsonl", default_jsonl))
    output_csv = PROJECT_ROOT / str(outputs.get("answer_eval_csv", default_csv))
    if output_version == "v2c1":
        output_json = output_json.with_name(output_json.name.replace("v2c0", "v2c1"))
        output_jsonl = output_jsonl.with_name(output_jsonl.name.replace("v2c0", "v2c1"))
        output_csv = output_csv.with_name(output_csv.name.replace("v2c0", "v2c1"))
    return output_json, output_jsonl, output_csv


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question",
        "query_type",
        "is_negative",
        "evidence_sufficient",
        "evidence_gate_reason",
        "llm_called",
        "insufficient_answer",
        "supported_claim_ratio",
        "citation_check_passed",
        "supported_claim_count",
        "checked_claim_count",
        "ignored_claim_count",
        "weak_claim_count",
        "unsupported_claim_count",
        "claim_type_counts",
        "manual_judgment",
        "manual_notes",
        "needs_human_review",
        "final_categories",
        "top_sources",
        "answer_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "query_type": record["query_type"],
                    "is_negative": record["is_negative"],
                    "evidence_sufficient": record["evidence_sufficient"],
                    "evidence_gate_reason": record["evidence_gate_reason"],
                    "llm_called": record["llm_called"],
                    "insufficient_answer": record["insufficient_answer"],
                    "supported_claim_ratio": record["supported_claim_ratio"],
                    "citation_check_passed": record["citation_check_passed"],
                    "supported_claim_count": record["supported_claim_count"],
                    "checked_claim_count": record["checked_claim_count"],
                    "ignored_claim_count": record["ignored_claim_count"],
                    "weak_claim_count": record["weak_claim_count"],
                    "unsupported_claim_count": record["unsupported_claim_count"],
                    "claim_type_counts": json.dumps(record["claim_type_counts"], ensure_ascii=False, sort_keys=True),
                    "manual_judgment": record["manual_judgment"],
                    "manual_notes": record["manual_notes"],
                    "needs_human_review": record["needs_human_review"],
                    "final_categories": "|".join(record["final_categories"]),
                    "top_sources": "|".join(str(src.get("imported_source", "")) for src in record["final_sources"][:5]),
                    "answer_summary": record["answer_summary"],
                }
            )


def summarize_text(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def print_summary(result: dict[str, Any]) -> None:
    print("Answer evaluation completed.")
    print(f"output_version: {result['output_version']}")
    print(f"retrieval_mode: {result['retrieval_mode']}")
    print(f"collection_name: {result['collection_name']}")
    print(f"persist_directory: {result['persist_directory']}")
    print("metrics:")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print("records:")
    for record in result["records"]:
        print(
            f"  {record['id']} negative={record['is_negative']} "
            f"evidence_sufficient={record['evidence_sufficient']} "
            f"llm_called={record['llm_called']} "
            f"citation_passed={record['citation_check_passed']} "
            f"supported_claim_ratio={record['supported_claim_ratio']} "
            f"unsupported={record['unsupported_claim_count']} ignored={record['ignored_claim_count']} "
            f"weak={record['weak_claim_count']} review={record['needs_human_review']}"
        )


if __name__ == "__main__":
    sys.exit(main())
