from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "answer_eval_v25d.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.citation_checker import check_citations  # noqa: E402
from src.generation.evidence_gate import evaluate_evidence  # noqa: E402
from src.generation.llm import OllamaLLM  # noqa: E402
from src.retrieval.real_smoke_retriever import retrieve_real_smoke  # noqa: E402


INSUFFICIENT_EVIDENCE_ANSWER = (
    "当前知识库没有足够依据回答这个问题。已检索到一些相近资料，"
    "但它们不足以支持该结论。insufficient evidence."
)


def main() -> int:
    configure_stdout()
    try:
        result = run_answer_eval_v25d()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001
        print("V2.5d answer evaluation failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_answer_eval_v25d(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    subset_path = resolve_project_path(config["output_subset_jsonl"])
    if not subset_path.exists():
        raise FileNotFoundError(f"Missing V25d subset file: {subset_path}")
    items = load_jsonl(subset_path)
    collection = config["collection"]
    retrieval_config = config["retrieval"]
    persist_directory = resolve_project_path(collection["persist_directory"])
    collection_name = str(collection["collection_name"])
    mode = str(retrieval_config.get("mode", "enhanced_keyword"))
    llm = OllamaLLM()

    records: list[dict[str, Any]] = []
    for item in items:
        retrieval = retrieve_real_smoke(
            question=str(item["question"]),
            persist_directory=persist_directory,
            collection_name=collection_name,
            retrieval_mode=mode,
            **build_retrieval_kwargs(retrieval_config, mode),
        )
        gate = evaluate_evidence(
            question=str(item["question"]),
            retrieval_result=retrieval,
            eval_item=item,
            config=config.get("evidence_gate") or {},
        )
        llm_called = False
        if gate["evidence_sufficient"]:
            llm_called = True
            answer = llm.generate(build_answer_prompt(str(item["question"]), retrieval.documents))
            if not answer.strip():
                answer = build_empty_answer_fallback(retrieval.debug["final_sources"])
        else:
            answer = INSUFFICIENT_EVIDENCE_ANSWER

        citation = check_citations(
            answer=answer,
            sources=retrieval.documents,
            config=config.get("citation_check") or {},
        )
        supported_claim_ratio = float(citation.get("supported_claim_ratio", 0.0))
        unsupported_claim_count = len(citation.get("unsupported_claims", []))
        weak_claim_count = int(citation.get("weak_claim_count", 0))
        needs_human_review = (
            unsupported_claim_count > 0
            or weak_claim_count > 0
            or (not citation.get("insufficient_answer", False) and supported_claim_ratio < 0.8)
            or (not gate["evidence_sufficient"] and str(item["query_type"]) != "negative")
        )
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "query_type": item["query_type"],
                "difficulty": item["difficulty"],
                "is_negative": is_negative(item),
                "expected_sources_contains": item.get("expected_sources_contains", []),
                "expected_categories": item.get("expected_categories", []),
                "required_all_categories": item.get("required_all_categories", []),
                "retrieval_mode": mode,
                "collection_name": collection_name,
                "persist_directory": str(persist_directory),
                "query_classifier_type": retrieval.debug.get("query_type"),
                "expanded_query": retrieval.debug.get("expanded_query"),
                "evidence_sufficient": bool(gate["evidence_sufficient"]),
                "evidence_gate_reason": gate["reason"],
                "evidence_gate_signals": gate["signals"],
                "llm_called": llm_called,
                "answer": answer,
                "answer_summary": summarize_text(answer, 320),
                "insufficient_answer": bool(citation.get("insufficient_answer", False)),
                "supported_claim_ratio": supported_claim_ratio,
                "citation_check_passed": bool(citation.get("citation_check_passed", False)),
                "supported_claim_count": len(citation.get("supported_claims", [])),
                "checked_claim_count": int(citation.get("checked_claim_count", 0)),
                "ignored_claim_count": int(citation.get("ignored_claim_count", 0)),
                "weak_claim_count": weak_claim_count,
                "unsupported_claim_count": unsupported_claim_count,
                "claim_type_counts": citation.get("claim_type_counts", {}),
                "manual_judgment": "",
                "manual_notes": "",
                "needs_human_review": needs_human_review,
                "unsupported_claims": citation.get("unsupported_claims", []),
                "ignored_claims": citation.get("ignored_claims", []),
                "weak_claims": citation.get("weak_claims", []),
                "citation_check": citation,
                "source_diversity": retrieval.debug.get("source_diversity", {}),
                "final_sources": retrieval.debug["final_sources"],
                "final_categories": sorted(
                    {str(source.get("category_dir", "")) for source in retrieval.debug["final_sources"] if source.get("category_dir")}
                ),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "subset_path": str(subset_path),
        "retrieval_mode": mode,
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "llm_provider": "ollama_local",
        "external_cloud_api_called": False,
        "metrics": build_metrics(records),
        "records": records,
    }


def build_retrieval_kwargs(config: dict[str, Any], mode: str) -> dict[str, Any]:
    values = {
        "dense_top_k": int(config.get("dense_top_k", 5)),
        "metadata_top_k": int(config.get("metadata_top_k", 8)),
        "keyword_top_k": int(config.get("keyword_top_k", 8)),
        "final_top_k": int(config.get("final_top_k", 8)),
        "max_chunks_per_source": int(config.get("max_chunks_per_source", 2)),
        "enable_source_diversity": bool(config.get("enable_source_diversity", True)),
        "source_key": str(config.get("source_key", "imported_source")),
    }
    if mode == "dense":
        values["final_top_k"] = min(values["final_top_k"], values["dense_top_k"])
    return values


def build_answer_prompt(question: str, documents: list[Any]) -> str:
    evidence = format_evidence(documents)
    return f"""/no_think
你是 Domain-RAG Agent 的回答模块。只能根据下面的检索资料回答问题。
如果资料不足，必须明确说明“当前知识库没有足够依据”。
回答事实、代码位置、配置含义或实验结论时必须给来源编号，例如 [S1]。
不要编造未出现在资料中的文件、指标、实验结果或代码行为。
回答请保持简洁，最多 5 条要点。

检索资料：
{evidence}

用户问题：
{question}

请输出：
1. 回答
2. 来源依据
"""


def format_evidence(documents: list[Any]) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = dict(getattr(document, "metadata", {}) or {})
        content = str(getattr(document, "page_content", "") or "").strip()
        if len(content) > 1400:
            content = f"{content[:1400]}\n...[truncated]"
        blocks.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"source: {metadata.get('source', '')}",
                    f"category_dir: {metadata.get('category_dir', '')}",
                    f"doc_type: {metadata.get('doc_type', '')}",
                    f"chunk_id: {metadata.get('chunk_id', '')}",
                    f"imported_source: {metadata.get('imported_source', '')}",
                    f"section: {metadata.get('section', '')}",
                    f"matched_terms: {metadata.get('matched_terms', [])}",
                    f"matched_fields: {metadata.get('matched_fields', [])}",
                    "content:",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "没有检索到资料。"


def build_empty_answer_fallback(final_sources: list[dict[str, Any]]) -> str:
    if not final_sources:
        return INSUFFICIENT_EVIDENCE_ANSWER
    parts = ["1. 回答", "本地 LLM 未生成有效文本；以下仅列出检索到的来源，不扩展推断。"]
    for source in final_sources[:3]:
        parts.append(f"- {source.get('imported_source') or source.get('source')} ({source.get('category_dir')})")
    parts.append("\n2. 来源依据")
    parts.append("见上述检索来源。")
    return "\n".join(parts)


def build_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    negative = [record for record in records if record["is_negative"]]
    non_negative = [record for record in records if not record["is_negative"]]
    ordinary = [record for record in records if not record["insufficient_answer"]]
    return {
        "total_eval_questions": len(records),
        "negative_questions": len(negative),
        "non_negative_questions": len(non_negative),
        "llm_called_count": sum(1 for record in records if record["llm_called"]),
        "evidence_gate_pass_rate": average(
            (not record["evidence_sufficient"]) if record["is_negative"] else record["evidence_sufficient"]
            for record in records
        ),
        "negative_refusal_rate": average(record["insufficient_answer"] for record in negative),
        "non_negative_answered_rate": average(record["evidence_sufficient"] and record["llm_called"] for record in non_negative),
        "citation_check_pass_rate": average(record["citation_check_passed"] for record in ordinary),
        "avg_supported_claim_ratio": average(record["supported_claim_ratio"] for record in ordinary),
        "unsupported_claim_count": sum(int(record["unsupported_claim_count"]) for record in ordinary),
        "weak_claim_count": sum(int(record["weak_claim_count"]) for record in ordinary),
        "ignored_claim_count": sum(int(record["ignored_claim_count"]) for record in records),
        "insufficient_answer_count": sum(1 for record in records if record["insufficient_answer"]),
        "needs_human_review_count": sum(1 for record in records if record["needs_human_review"]),
        "needs_human_review_ids": [record["id"] for record in records if record["needs_human_review"]],
    }


def average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    outputs = config["outputs"]
    output_json = resolve_project_path(outputs["answer_eval_json"])
    output_jsonl = resolve_project_path(outputs["answer_eval_jsonl"])
    output_csv = resolve_project_path(outputs["answer_eval_csv"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as file:
        for record in result["records"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(output_csv, result["records"])


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
        "checked_claim_count",
        "ignored_claim_count",
        "weak_claim_count",
        "unsupported_claim_count",
        "needs_human_review",
        "final_categories",
        "top_sources",
        "answer_summary",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key, "") for key in fieldnames},
                    "final_categories": "|".join(record.get("final_categories", [])),
                    "top_sources": "|".join(
                        str(source.get("imported_source") or source.get("source", ""))
                        for source in record.get("final_sources", [])[:5]
                    ),
                }
            )


def is_negative(item: dict[str, Any]) -> bool:
    return str(item.get("query_type", "")) == "negative" or (
        not item.get("expected_sources_contains") and not item.get("expected_categories")
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def summarize_text(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def print_summary(result: dict[str, Any]) -> None:
    print("V2.5d answer evaluation completed.")
    print(
        json.dumps(
            {
                "collection_name": result["collection_name"],
                "retrieval_mode": result["retrieval_mode"],
                "external_cloud_api_called": result["external_cloud_api_called"],
                "metrics": result["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
