from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.workflow import DEFAULT_CONFIG_PATH, run_rag_graph  # noqa: E402


def main() -> None:
    args = parse_args()
    result = run_rag_graph(
        question=args.question,
        run_mode=args.run_mode,
        retrieval_mode=args.retrieval_mode,
        enable_calibration=not args.no_calibration,
        config_path=args.config,
    )
    print_run_summary(result["summary"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled V3a LangGraph RAG workflow.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--run-mode", choices=["retrieval_only", "full_rag"], default="retrieval_only")
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "enhanced", "keyword", "enhanced_keyword"],
        default=None,
        help="Override retrieval mode from config.",
    )
    parser.add_argument("--no-calibration", action="store_true", help="Disable V26 calibration.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to LangGraph V3a config.")
    return parser.parse_args()


def print_run_summary(summary: dict[str, Any]) -> None:
    print("=== LangGraph RAG V3a Run ===")
    print(f"question: {summary.get('question', '')}")
    print(f"run_mode: {summary.get('run_mode', '')}")
    print(f"retrieval_mode: {summary.get('retrieval_mode', '')}")
    print(f"query_type: {summary.get('query_type', '')}")
    print(f"target_categories: {summary.get('target_categories', [])}")
    print(f"evidence_sufficient: {summary.get('evidence_sufficient')}")
    print(f"evidence_gate_reason: {summary.get('evidence_gate_reason', '')}")
    print(f"llm_called: {summary.get('llm_called')}")
    print(f"insufficient_answer: {summary.get('insufficient_answer')}")
    print(f"answer_preview: {summary.get('answer_preview', '')}")
    print(f"evidence_quality_level: {summary.get('evidence_quality_level')}")
    print(f"evidence_quality_score: {summary.get('evidence_quality_score')}")
    print("top_sources:")
    for source in summary.get("top_sources", []):
        print(
            "  - "
            f"rank={source.get('rank')} "
            f"category={source.get('category_dir')} "
            f"source={source.get('source')} "
            f"score={source.get('final_score')} "
            f"calibration={source.get('calibration_score')}"
        )
    print(f"trace_path: {summary.get('trace_path', '')}")
    print(f"last_run_path: {summary.get('last_run_path', '')}")
    if summary.get("errors"):
        print("errors:")
        for error in summary.get("errors", []):
            print(f"  - {error}")


if __name__ == "__main__":
    main()
