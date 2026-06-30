from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sql_sag.fusion_retriever import load_v4d_config, retrieve_enhanced_sql_relation, write_json  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4d.yaml"
DEFAULT_QUESTION = "reward 和 risk 相关的实验或报告有哪些？"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V4d enhanced SQL relation fusion retrieval.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    config = load_v4d_config(args.config)
    result = retrieve_enhanced_sql_relation(args.question, config)
    outputs = config["outputs"]
    write_json(outputs["single_debug_json"], result)
    write_debug_markdown(PROJECT_ROOT / outputs["single_debug_md"], result)

    print("=== Enhanced SQL Relation V4d ===")
    print(f"question: {result['question']}")
    print(f"category_intent: {result['category_intent']}")
    print("enhanced_keyword top sources:")
    for row in result["enhanced_keyword_candidates"][:5]:
        print(f"  score={row.get('final_score')} source={row.get('source')} category={row.get('category_dir')}")
    print("sql_relation top sources:")
    for row in result.get("sql_relation_result", {}).get("candidates", [])[:5]:
        print(f"  score={row.get('relation_score')} source={row.get('source')} category={row.get('category_dir')}")
    print("final fused top sources:")
    for row in result["final_sources"][:8]:
        print(
            f"  rank={row['rank']} final={row['final_score']} relation={row.get('relation_score')} "
            f"category_adj={row.get('category_intent_adjustment')} hf_penalty={row.get('high_frequency_entity_penalty')} "
            f"source={row.get('source')} category={row.get('category_dir')} channels={row.get('retrieval_channels')}"
        )
        if row.get("relation_path"):
            print(f"    path: {' -> '.join(row['relation_path'])}")
        print(f"    fusion: {'; '.join(row.get('fusion_reasons', []))}")
    print(f"debug_json: {PROJECT_ROOT / outputs['single_debug_json']}")
    print(f"debug_md: {PROJECT_ROOT / outputs['single_debug_md']}")


def write_debug_markdown(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Enhanced SQL Relation V4d Single Debug",
        "",
        f"- question: {result['question']}",
        f"- category_intent: `{result['category_intent']}`",
        f"- enhanced_keyword_candidates: {len(result['enhanced_keyword_candidates'])}",
        f"- sql_relation_candidates: {len(result.get('sql_relation_result', {}).get('candidates', []))}",
        f"- final_sources: {len(result['final_sources'])}",
        "",
        "## Final Fused Sources",
        "",
    ]
    for row in result["final_sources"]:
        lines.extend(
            [
                f"### Rank {row['rank']} | final_score={row['final_score']}",
                "",
                f"- source: `{row.get('source', '')}`",
                f"- category_dir: `{row.get('category_dir', '')}`",
                f"- doc_type: `{row.get('doc_type', '')}`",
                f"- channels: `{row.get('retrieval_channels', [])}`",
                f"- enhanced_keyword_score: {row.get('enhanced_keyword_score', 0)}",
                f"- relation_score_raw: {row.get('relation_score_raw', 0)}",
                f"- relation_score_normalized: {row.get('relation_score_normalized', 0)}",
                f"- category_intent_adjustment: {row.get('category_intent_adjustment', 0)}",
                f"- high_frequency_entity_penalty: {row.get('high_frequency_entity_penalty', 0)}",
                f"- relation_reasons: {'; '.join(row.get('relation_reasons', []))}",
                f"- fusion_reasons: {'; '.join(row.get('fusion_reasons', []))}",
                f"- relation_path: {' -> '.join(row.get('relation_path', []))}",
                f"- preview: {str(row.get('content_preview', ''))[:500]}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
