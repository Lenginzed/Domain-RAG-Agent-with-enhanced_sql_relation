from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sql_sag.index_builder import build_sql_sag_index  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4a.yaml"


def main() -> None:
    summary = build_sql_sag_index(CONFIG_PATH)
    print("=== SQL SAG-lite V4a Build ===")
    print(f"db_path: {summary['db_path']}")
    print(f"documents: {summary['document_count']}")
    print(f"chunks: {summary['chunk_count']}")
    print(f"events: {summary['event_count']}")
    print(f"entities: {summary['entity_count']}")
    print(f"event_entities: {summary['event_entity_count']}")
    print(f"llm_called: {summary['llm_called']}")
    print(f"chroma_written: {summary['chroma_written']}")


if __name__ == "__main__":
    main()
