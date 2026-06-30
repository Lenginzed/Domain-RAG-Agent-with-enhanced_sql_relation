from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sql_sag.audit import audit_sql_sag_index  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "sql_sag_v4b.yaml"


def main() -> None:
    summary = audit_sql_sag_index(CONFIG_PATH)
    print("=== SQL SAG-lite V4b Audit ===")
    print(f"db_path: {summary['db_path']}")
    print(f"documents: {summary['document_count']}")
    print(f"chunks: {summary['chunk_count']}")
    print(f"events: {summary['event_count']}")
    print(f"entities: {summary['entity_count']}")
    print(f"event_entities: {summary['event_entity_count']}")
    print(f"avg_entities_per_event: {summary['avg_entities_per_event']}")
    print(f"orphan_chunks: {summary['orphan_chunk_count']}")
    print(f"orphan_events: {summary['orphan_event_count']}")
    print(f"removed_entities: {summary.get('removed_entity_count', 0)}")
    print(f"entity_type_merges: {summary.get('entity_type_merge_count', 0)}")
    print(f"key_entity_hits: {summary['key_entity_hits']}/{summary['key_entity_total']}")
    print(f"audit_passed: {summary['audit_passed']}")


if __name__ == "__main__":
    main()
