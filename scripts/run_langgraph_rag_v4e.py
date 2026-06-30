from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.trace import summarize_node_trace  # noqa: E402
from src.agent.workflow import DEFAULT_V4E_CONFIG_PATH, run_rag_graph_v3b  # noqa: E402


def main() -> None:
    args = parse_args()
    result = run_rag_graph_v3b(
        question=args.question,
        run_mode=args.run_mode,
        retrieval_mode=args.retrieval_mode,
        enable_calibration=not args.no_calibration,
        config_path=args.config,
    )
    print_run_summary(result["summary"], show_trace=args.show_trace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V4e LangGraph retrieval workflow.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--run-mode", choices=["retrieval_only", "full_rag"], default="retrieval_only")
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "enhanced", "keyword", "enhanced_keyword", "sql_relation", "enhanced_sql_relation"],
        default="enhanced_sql_relation",
        help="Retrieval mode.",
    )
    parser.add_argument("--no-calibration", action="store_true", help="Disable calibration where supported.")
    parser.add_argument("--show-trace", action="store_true", help="Print node trace summary.")
    parser.add_argument("--config", default=str(DEFAULT_V4E_CONFIG_PATH), help="Path to LangGraph V4e config.")
    return parser.parse_args()


def print_run_summary(summary: dict[str, Any], *, show_trace: bool) -> None:
    sources = summary.get("top_sources", [])
    relation_path_found_count = sum(1 for row in sources if row.get("relation_path"))
    sql_only_candidate_count = sum(1 for row in sources if row.get("sql_only_candidate"))
    print("=== LangGraph RAG V4e Run ===")
    print(f"run_id: {summary.get('run_id', '')}")
    print(f"question: {summary.get('question', '')}")
    print(f"run_mode: {summary.get('run_mode', '')}")
    print(f"retrieval_mode: {summary.get('retrieval_mode', '')}")
    print(f"llm_called: {summary.get('llm_called')}")
    print(f"evidence_sufficient: {summary.get('evidence_sufficient')}")
    print(f"evidence_quality_level: {summary.get('evidence_quality_level')}")
    print(f"needs_human_review: {summary.get('needs_human_review')}")
    print(f"category_intent: {summary.get('category_intent', {})}")
    print(f"sql_relation_debug: {summary.get('sql_relation_debug', {})}")
    print(f"relation_path_found_count: {relation_path_found_count}")
    print(f"sql_only_candidate_count: {sql_only_candidate_count}")
    print("top_sources:")
    for source in sources:
        print(
            "  - "
            f"rank={source.get('rank')} "
            f"category={source.get('category_dir')} "
            f"source={source.get('source')} "
            f"score={source.get('final_score')} "
            f"relation={source.get('relation_score')} "
            f"channels={source.get('retrieval_channels')}"
        )
        if source.get("relation_path"):
            print(f"    path: {' -> '.join(source.get('relation_path', []))}")
    print(f"trace_path: {summary.get('trace_path', '')}")
    print(f"index_path: {summary.get('trace_index_path', '')}")
    print(f"last_run_path: {summary.get('last_run_path', '')}")
    if summary.get("errors"):
        print("errors:")
        for error in summary.get("errors", []):
            print(f"  - {error}")
    if show_trace:
        print("trace:")
        for item in summarize_node_trace(summary.get("trace", [])):
            print(
                "  - "
                f"step={item.get('step')} "
                f"node={item.get('node')} "
                f"status={item.get('status')} "
                f"output={item.get('output_summary')} "
                f"error={item.get('error')}"
            )


if __name__ == "__main__":
    main()
