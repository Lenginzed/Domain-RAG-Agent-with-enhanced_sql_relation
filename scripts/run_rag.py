from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer_generator import generate_answer  # noqa: E402
from src.indexes.vector_index import DEFAULT_COLLECTION_NAME, collection_count, load_index  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V1 dense RAG over the local Chroma index.")
    parser.add_argument("--question", required=True, help="User question to answer.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    args = parser.parse_args()

    try:
        vector_store = load_index(collection_name=args.collection_name)
        if collection_count(vector_store) == 0:
            print("No Chroma index data found. Please run:")
            print("  python scripts/ingest_documents.py")
            return 1

        retriever = DenseRetriever(collection_name=args.collection_name)
        documents = retriever.retrieve(args.question)
        result = generate_answer(args.question, documents)
    except Exception as exc:  # noqa: BLE001 - script should report clear failures.
        print("RAG query failed.")
        print(f"reason: {exc}")
        return 1

    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        score = source.get("score")
        score_text = f", score={float(score):.6f}" if score is not None else ""
        print(
            f"- [{source['label']}] source={source['source']}, "
            f"chunk_id={source['chunk_id']}, section={source['section']}{score_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
