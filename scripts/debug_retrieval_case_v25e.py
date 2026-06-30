from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexes.vector_index import load_index  # noqa: E402
from src.retrieval.real_smoke_retriever import retrieve_real_smoke  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_calibration_v25e.yaml"


def main() -> int:
    configure_stdout()
    try:
        result = run_debug()
        write_outputs(result)
    except Exception as exc:  # noqa: BLE001
        print("V2.5e retrieval debug failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_debug(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    case = dict(config["target_cases"][0])
    collection_name = str(config["collection"]["collection_name"])
    persist_directory = resolve_project_path(config["collection"]["persist_directory"])
    eval_item = load_eval_item(resolve_project_path(config["inputs"]["eval_set_jsonl"]), str(case["id"]))
    question = str((eval_item or {}).get("question") or case["question"])
    expected_source = str(case["expected_source"])

    collection_rows = load_collection_rows(persist_directory, collection_name)
    target_rows = [
        row
        for row in collection_rows
        if source_contains(row["metadata"], expected_source)
    ]
    source_exists = bool(target_rows)

    modes = ["dense", "enhanced", "keyword", "enhanced_keyword"]
    mode_results: dict[str, Any] = {}
    for mode in modes:
        retrieval = retrieve_real_smoke(
            question=question,
            persist_directory=persist_directory,
            collection_name=collection_name,
            retrieval_mode=mode,
            dense_top_k=20,
            metadata_top_k=20,
            keyword_top_k=20,
            final_top_k=20,
            max_chunks_per_source=20,
            enable_source_diversity=False,
            source_key="imported_source",
        )
        final_sources = retrieval.debug["final_sources"]
        mode_results[mode] = {
            "query_type": retrieval.debug["query_type"],
            "target_categories": retrieval.debug["target_categories"],
            "expanded_query": retrieval.debug["expanded_query"],
            "top20": final_sources,
            "expected_source_in_top20": source_rank(final_sources, expected_source) is not None,
            "expected_source_rank": source_rank(final_sources, expected_source),
            "dense_sources": retrieval.debug.get("dense_sources", []),
            "metadata_sources": retrieval.debug.get("metadata_sources", []),
            "keyword_sources": retrieval.debug.get("keyword_sources", []),
        }

    calibrated_preview: dict[str, Any] = {}
    calibration_config = dict(config.get("calibration", {}))
    for mode in calibration_config.get("apply_to_modes", []):
        retrieval = retrieve_real_smoke(
            question=question,
            persist_directory=persist_directory,
            collection_name=collection_name,
            retrieval_mode=str(mode),
            dense_top_k=20,
            metadata_top_k=20,
            keyword_top_k=20,
            final_top_k=20,
            max_chunks_per_source=20,
            enable_source_diversity=False,
            source_key="imported_source",
            calibration_config=calibration_config,
        )
        final_sources = retrieval.debug["final_sources"]
        calibrated_preview[str(mode)] = {
            "top20": final_sources,
            "expected_source_in_top20": source_rank(final_sources, expected_source) is not None,
            "expected_source_rank": source_rank(final_sources, expected_source),
        }

    failure_analysis = analyze_failure(mode_results, calibrated_preview, expected_source, source_exists)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "target_case": {
            **case,
            "eval_question": question,
            "eval_question_differs_from_config": question != str(case["question"]),
        },
        "collection_chunk_count": len(collection_rows),
        "expected_source_exists": source_exists,
        "expected_source_chunk_count": len(target_rows),
        "expected_source_chunks": summarize_target_rows(target_rows),
        "mode_results": mode_results,
        "calibrated_preview": calibrated_preview,
        "failure_analysis": failure_analysis,
        "outputs": {
            "debug_json": str(resolve_project_path(config["outputs"]["debug_json"])),
            "debug_md": str(resolve_project_path(config["outputs"]["debug_md"])),
        },
    }


def load_collection_rows(persist_directory: Path, collection_name: str) -> list[dict[str, Any]]:
    vector_store = load_index(persist_directory=persist_directory, collection_name=collection_name)
    raw = vector_store._collection.get(include=["documents", "metadatas"])  # noqa: SLF001
    documents = raw.get("documents", []) or []
    metadatas = raw.get("metadatas", []) or []
    ids = raw.get("ids", []) or []
    rows: list[dict[str, Any]] = []
    for index, metadata in enumerate(metadatas):
        rows.append(
            {
                "id": ids[index] if index < len(ids) else "",
                "metadata": dict(metadata or {}),
                "page_content": str(documents[index] if index < len(documents) else ""),
            }
        )
    return rows


def summarize_target_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in rows:
        metadata = row["metadata"]
        summary.append(
            {
                "id": row["id"],
                "source": metadata.get("source", ""),
                "imported_source": metadata.get("imported_source", ""),
                "original_source": metadata.get("original_source", ""),
                "category_dir": metadata.get("category_dir", ""),
                "file_ext": metadata.get("file_ext", ""),
                "doc_type": metadata.get("doc_type", ""),
                "title": metadata.get("title", ""),
                "section": metadata.get("section", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "metadata_keys": sorted(metadata.keys()),
                "page_content_chars": len(row["page_content"]),
                "page_content_preview": compact_text(row["page_content"], 600),
            }
        )
    return summary


def analyze_failure(
    mode_results: dict[str, Any],
    calibrated_preview: dict[str, Any],
    expected_source: str,
    source_exists: bool,
) -> dict[str, Any]:
    ranks = {
        mode: result.get("expected_source_rank")
        for mode, result in mode_results.items()
    }
    calibrated_ranks = {
        mode: result.get("expected_source_rank")
        for mode, result in calibrated_preview.items()
    }
    reasons: list[str] = []
    if not source_exists:
        reasons.append("expected_source_absent_from_collection")
    elif not any(rank for rank in ranks.values()):
        reasons.append("expected_source_not_in_any_top20_candidate_list")
    else:
        if ranks.get("keyword") and not ranks.get("enhanced_keyword"):
            reasons.append("keyword_can_recall_but_fusion_ranking_or_source_selection_drops_it")
        if ranks.get("metadata") and not ranks.get("enhanced_keyword"):
            reasons.append("metadata_can_recall_but_fusion_ranking_or_source_selection_drops_it")
        if not ranks.get("dense"):
            reasons.append("dense_channel_did_not_recall_expected_source_in_top20")
        if not ranks.get("keyword"):
            reasons.append("keyword_channel_did_not_recall_expected_source_in_top20")
        if not ranks.get("metadata"):
            reasons.append("metadata_channel_did_not_recall_expected_source_in_top20")
    if calibrated_ranks:
        reasons.append(f"calibrated_preview_ranks:{calibrated_ranks}")
    return {
        "expected_source": expected_source,
        "pre_calibration_ranks": ranks,
        "calibrated_preview_ranks": calibrated_ranks,
        "reason_summary": reasons,
    }


def source_rank(sources: list[dict[str, Any]], expected_source: str) -> int | None:
    expected = expected_source.lower()
    for index, source in enumerate(sources, start=1):
        if source_contains(source, expected):
            return index
    return None


def source_contains(metadata: dict[str, Any], expected_source: str) -> bool:
    expected = str(expected_source).lower()
    text = " ".join(
        str(metadata.get(key, ""))
        for key in ["source", "imported_source", "original_source", "chunk_id"]
    ).lower()
    return expected in text


def compact_text(text: str, limit: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def load_eval_item(path: Path, item_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("id")) == item_id:
                return item
    return None


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    config = load_yaml(config_path)
    output_json = resolve_project_path(config["outputs"]["debug_json"])
    output_md = resolve_project_path(config["outputs"]["debug_md"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# V2.5e Retrieval Debug: v25c_q014",
        "",
        f"- collection: `{result['collection_name']}`",
        f"- persist_directory: `{result['persist_directory']}`",
        f"- question: {result['target_case']['eval_question']}",
        f"- expected_source: `{result['target_case']['expected_source']}`",
        f"- expected_source_exists: {result['expected_source_exists']}",
        f"- expected_source_chunk_count: {result['expected_source_chunk_count']}",
        "",
        "## Expected Source Chunks",
    ]
    for chunk in result["expected_source_chunks"]:
        lines.extend(
            [
                "",
                f"### {chunk['chunk_id']}",
                f"- source: `{chunk['source']}`",
                f"- category_dir: `{chunk['category_dir']}`",
                f"- doc_type: `{chunk['doc_type']}`",
                f"- title: {chunk['title']}",
                f"- section: {chunk['section']}",
                "",
                chunk["page_content_preview"],
            ]
        )
    lines.extend(["", "## Pre-calibration Top20"])
    for mode, result_for_mode in result["mode_results"].items():
        lines.extend(
            [
                "",
                f"### {mode}",
                f"- expected_source_rank: {result_for_mode['expected_source_rank']}",
                "",
                "| rank | source | category | score | channels | matched_terms |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for source in result_for_mode["top20"]:
            lines.append(
                "| {rank} | `{source}` | {category} | {score} | {channels} | {terms} |".format(
                    rank=source.get("rank", ""),
                    source=source.get("source", ""),
                    category=source.get("category_dir", ""),
                    score=source.get("final_score", source.get("score", "")),
                    channels=",".join(source.get("retrieval_channels", []) or []),
                    terms=",".join(str(term) for term in source.get("matched_terms", [])[:8]),
                )
            )
    lines.extend(["", "## Failure Analysis", ""])
    for reason in result["failure_analysis"]["reason_summary"]:
        lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


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
    print("V2.5e retrieval debug completed.")
    print(
        json.dumps(
            {
                "expected_source_exists": result["expected_source_exists"],
                "expected_source_chunk_count": result["expected_source_chunk_count"],
                "pre_calibration_ranks": result["failure_analysis"]["pre_calibration_ranks"],
                "calibrated_preview_ranks": result["failure_analysis"]["calibrated_preview_ranks"],
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
