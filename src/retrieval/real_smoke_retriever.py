from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.indexes.vector_index import load_index
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.query_classifier import classify_query
from src.retrieval.query_expansion import expand_query
from src.retrieval.retrieval_calibration import apply_retrieval_calibration


@dataclass
class RetrievalResult:
    documents: list[Document]
    debug: dict[str, Any]


def retrieve_real_smoke(
    *,
    question: str,
    persist_directory: Path,
    collection_name: str,
    retrieval_mode: str = "enhanced",
    dense_top_k: int = 5,
    metadata_top_k: int = 5,
    keyword_top_k: int = 8,
    final_top_k: int = 8,
    max_chunks_per_source: int = 2,
    enable_source_diversity: bool = True,
    source_key: str = "imported_source",
    keyword_lowercase: bool = True,
    keyword_min_token_len: int = 2,
    keyword_field_weights: dict[str, float] | None = None,
    calibration_config: dict[str, Any] | None = None,
) -> RetrievalResult:
    validate_retrieval_mode(retrieval_mode)
    classification = classify_query(question)
    expanded_query = expand_query(question, classification)
    dense_query = question if retrieval_mode == "dense" else expanded_query

    dense_docs: list[Document] = []
    if retrieval_mode in {"dense", "enhanced", "enhanced_keyword"}:
        vector_store = load_index(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        dense_docs = run_dense(vector_store, dense_query, dense_top_k)

    metadata_docs: list[Document] = []
    if retrieval_mode in {"enhanced", "enhanced_keyword"}:
        metadata_fetch_k = max(metadata_top_k, final_top_k * max(1, max_chunks_per_source) * 2)
        metadata_docs = MetadataRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
        ).retrieve(question, classification, top_k=metadata_fetch_k)

    keyword_docs: list[Document] = []
    if retrieval_mode in {"keyword", "enhanced_keyword"}:
        keyword_fetch_k = max(keyword_top_k, final_top_k * max(1, max_chunks_per_source) * 2)
        keyword_docs = KeywordRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            lowercase=keyword_lowercase,
            min_token_len=keyword_min_token_len,
            field_weights=keyword_field_weights,
        ).retrieve(
            question=question,
            expanded_query=expanded_query,
            classification=classification,
            top_k=keyword_fetch_k,
        )

    if retrieval_mode == "dense":
        final_docs = select_final_documents(
            prepare_dense_only(dense_docs),
            target_categories=set(),
            final_top_k=min(final_top_k, dense_top_k),
            max_chunks_per_source=max_chunks_per_source,
            enable_source_diversity=enable_source_diversity,
            source_key=source_key,
        )
    else:
        final_docs = fuse_documents(
            question=question,
            dense_docs=dense_docs,
            metadata_docs=metadata_docs,
            keyword_docs=keyword_docs,
            classification=classification,
            final_top_k=final_top_k,
            max_chunks_per_source=max_chunks_per_source,
            enable_source_diversity=enable_source_diversity,
            source_key=source_key,
            calibration_config=calibration_config,
        )

    debug = {
        "retrieval_mode": retrieval_mode,
        "query_type": classification["query_type"],
        "target_categories": classification["target_categories"],
        "keywords": classification["keywords"],
        "expanded_query": expanded_query,
        "dense_query": dense_query,
        "dense_sources": summarize_documents(dense_docs),
        "metadata_sources": summarize_documents(metadata_docs[:metadata_top_k]),
        "metadata_candidate_count": len(metadata_docs),
        "keyword_sources": summarize_documents(keyword_docs[:keyword_top_k]),
        "keyword_candidate_count": len(keyword_docs),
        "final_sources": summarize_documents(final_docs),
        "source_diversity": summarize_source_diversity(
            final_docs,
            source_key=source_key,
            max_chunks_per_source=max_chunks_per_source,
            enabled=enable_source_diversity,
        ),
    }
    return RetrievalResult(documents=final_docs, debug=debug)


def validate_retrieval_mode(retrieval_mode: str) -> None:
    allowed = {"dense", "enhanced", "keyword", "enhanced_keyword"}
    if retrieval_mode not in allowed:
        raise ValueError(f"Unsupported retrieval_mode={retrieval_mode!r}; expected one of {sorted(allowed)}")


def run_dense(vector_store: Any, query: str, top_k: int) -> list[Document]:
    results = vector_store.similarity_search_with_score(query, k=top_k)
    documents: list[Document] = []
    for rank, (document, score) in enumerate(results, start=1):
        metadata = dict(document.metadata or {})
        metadata["dense_score"] = float(score)
        metadata["dense_rank"] = rank
        metadata["retrieval_source"] = "dense"
        metadata["why_selected"] = f"dense_rank:{rank};dense_distance:{float(score):.6f}"
        document.metadata = metadata
        documents.append(document)
    return documents


def prepare_dense_only(dense_docs: list[Document]) -> list[Document]:
    final_docs: list[Document] = []
    for rank, document in enumerate(dense_docs, start=1):
        metadata = dict(document.metadata or {})
        metadata["final_score"] = float(len(dense_docs) - rank + 1)
        metadata["why_selected"] = metadata.get("why_selected") or f"dense_rank:{rank}"
        document.metadata = metadata
        final_docs.append(document)
    return final_docs


def fuse_documents(
    *,
    question: str,
    dense_docs: list[Document],
    metadata_docs: list[Document],
    keyword_docs: list[Document],
    classification: dict[str, Any],
    final_top_k: int,
    max_chunks_per_source: int,
    enable_source_diversity: bool,
    source_key: str,
    calibration_config: dict[str, Any] | None = None,
) -> list[Document]:
    candidates: dict[str, Document] = {}
    dense_total = len(dense_docs)
    target_categories = set(str(item) for item in classification.get("target_categories", []))

    for rank, document in enumerate(dense_docs, start=1):
        key = document_key(document)
        metadata = dict(document.metadata or {})
        metadata["dense_rank_points"] = float((dense_total - rank + 1) * 2)
        metadata["metadata_rank_points"] = 0.0
        metadata["keyword_rank_points"] = 0.0
        metadata["metadata_score"] = float(metadata.get("metadata_score", 0) or 0)
        metadata["keyword_score"] = float(metadata.get("keyword_score", 0) or 0)
        metadata["retrieval_source"] = "dense"
        metadata["retrieval_channels"] = ["dense"]
        metadata["match_reason"] = metadata.get("match_reason", "")
        metadata["why_selected"] = f"dense_rank:{rank}"
        document.metadata = metadata
        candidates[key] = document

    metadata_total = len(metadata_docs)
    for rank, document in enumerate(metadata_docs, start=1):
        key = document_key(document)
        existing = candidates.get(key)
        target = existing or document
        metadata = dict(target.metadata or {})
        source_metadata = dict(document.metadata or {})
        metadata["metadata_score"] = max(
            float(metadata.get("metadata_score", 0) or 0),
            float(source_metadata.get("metadata_score", 0) or 0),
        )
        metadata["metadata_rank_points"] = float((metadata_total - rank + 1) * 1.5)
        metadata["match_reason"] = source_metadata.get("match_reason", metadata.get("match_reason", ""))
        add_channel(metadata, "metadata")
        previous_why = str(metadata.get("why_selected", ""))
        metadata["why_selected"] = ";".join(filter(None, [previous_why, f"metadata_rank:{rank}", metadata["match_reason"]]))
        target.metadata = metadata
        if existing is None:
            candidates[key] = target

    keyword_total = len(keyword_docs)
    for rank, document in enumerate(keyword_docs, start=1):
        key = document_key(document)
        existing = candidates.get(key)
        target = existing or document
        metadata = dict(target.metadata or {})
        source_metadata = dict(document.metadata or {})
        metadata["keyword_score"] = max(
            float(metadata.get("keyword_score", 0) or 0),
            float(source_metadata.get("keyword_score", 0) or 0),
        )
        metadata["keyword_rank_points"] = float((keyword_total - rank + 1) * 1.25)
        metadata["matched_terms"] = merge_list_values(metadata.get("matched_terms"), source_metadata.get("matched_terms"))
        metadata["matched_fields"] = merge_list_values(metadata.get("matched_fields"), source_metadata.get("matched_fields"))
        add_channel(metadata, "keyword")
        previous_why = str(metadata.get("why_selected", ""))
        keyword_why = str(source_metadata.get("why_selected", f"keyword_rank:{rank}"))
        metadata["why_selected"] = ";".join(filter(None, [previous_why, f"keyword_rank:{rank}", keyword_why]))
        target.metadata = metadata
        if existing is None:
            candidates[key] = target

    for document in candidates.values():
        metadata = dict(document.metadata or {})
        category = str(metadata.get("category_dir", ""))
        category_bonus = 3.0 if category in target_categories else 0.0
        metadata["category_match_bonus"] = category_bonus
        metadata["final_score"] = round(
            float(metadata.get("dense_rank_points", 0) or 0)
            + float(metadata.get("metadata_rank_points", 0) or 0)
            + float(metadata.get("metadata_score", 0) or 0)
            + float(metadata.get("keyword_rank_points", 0) or 0)
            + float(metadata.get("keyword_score", 0) or 0)
            + category_bonus,
            4,
        )
        if category_bonus:
            metadata["why_selected"] = ";".join(filter(None, [str(metadata.get("why_selected", "")), f"category_bonus:{category}"]))
        document.metadata = metadata
        if calibration_config and bool(calibration_config.get("enabled", True)):
            apply_retrieval_calibration(
                document,
                question=question,
                classification=classification,
                config=calibration_config,
            )

    sorted_docs = sorted(
        candidates.values(),
        key=lambda doc: (
            -float(doc.metadata.get("final_score", 0) or 0),
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("chunk_id", "")),
        ),
    )
    return select_final_documents(
        sorted_docs,
        target_categories=target_categories,
        final_top_k=final_top_k,
        max_chunks_per_source=max_chunks_per_source,
        enable_source_diversity=enable_source_diversity,
        source_key=source_key,
    )


def add_channel(metadata: dict[str, Any], channel: str) -> None:
    channels = metadata.get("retrieval_channels")
    if isinstance(channels, str):
        values = [channels]
    elif isinstance(channels, list):
        values = [str(item) for item in channels]
    else:
        values = []
    if channel not in values:
        values.append(channel)
    metadata["retrieval_channels"] = values
    metadata["retrieval_source"] = "+".join(values)


def merge_list_values(left: Any, right: Any) -> list[str]:
    values: list[str] = []
    for item in to_list(left) + to_list(right):
        text = str(item)
        if text and text not in values:
            values.append(text)
    return values


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def select_final_documents(
    sorted_docs: list[Document],
    *,
    target_categories: set[str],
    final_top_k: int,
    max_chunks_per_source: int,
    enable_source_diversity: bool,
    source_key: str,
) -> list[Document]:
    if not enable_source_diversity:
        return annotate_source_ranks(
            diversify_by_target_categories(sorted_docs, target_categories, final_top_k),
            source_key=source_key,
            max_chunks_per_source=max_chunks_per_source,
            relaxed=False,
        )

    selected = pick_with_source_cap(
        sorted_docs,
        target_categories=target_categories,
        final_top_k=final_top_k,
        max_chunks_per_source=max_chunks_per_source,
        source_key=source_key,
    )
    return annotate_source_ranks(
        selected[:final_top_k],
        source_key=source_key,
        max_chunks_per_source=max_chunks_per_source,
        relaxed=False,
    )


def pick_with_source_cap(
    sorted_docs: list[Document],
    *,
    target_categories: set[str],
    final_top_k: int,
    max_chunks_per_source: int,
    source_key: str,
) -> list[Document]:
    selected: list[Document] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}
    for category in target_categories:
        for document in sorted_docs:
            key = document_key(document)
            if key in seen:
                continue
            source = source_identifier(document, source_key)
            if source_counts.get(source, 0) >= max_chunks_per_source:
                continue
            if str(document.metadata.get("category_dir", "")) == category:
                selected.append(document)
                seen.add(key)
                source_counts[source] = source_counts.get(source, 0) + 1
                break

    for document in sorted_docs:
        if len(selected) >= final_top_k:
            break
        key = document_key(document)
        if key in seen:
            continue
        source = source_identifier(document, source_key)
        if source_counts.get(source, 0) >= max_chunks_per_source:
            continue
        selected.append(document)
        seen.add(key)
        source_counts[source] = source_counts.get(source, 0) + 1
    return selected[:final_top_k]


def fill_with_relaxed_source_cap(
    selected: list[Document],
    sorted_docs: list[Document],
    *,
    final_top_k: int,
    source_key: str,
) -> list[Document]:
    selected_keys = {document_key(document) for document in selected}
    filled = list(selected)
    for document in sorted_docs:
        if len(filled) >= final_top_k:
            break
        key = document_key(document)
        if key in selected_keys:
            continue
        metadata = dict(document.metadata or {})
        metadata["source_diversity_relaxed"] = True
        metadata["why_selected"] = ";".join(
            filter(None, [str(metadata.get("why_selected", "")), f"source_diversity_relaxed:{source_identifier(document, source_key)}"])
        )
        document.metadata = metadata
        filled.append(document)
        selected_keys.add(key)
    return filled


def diversify_by_target_categories(
    sorted_docs: list[Document],
    target_categories: set[str],
    final_top_k: int,
) -> list[Document]:
    selected: list[Document] = []
    seen: set[str] = set()
    for category in target_categories:
        for document in sorted_docs:
            if document_key(document) in seen:
                continue
            if str(document.metadata.get("category_dir", "")) == category:
                selected.append(document)
                seen.add(document_key(document))
                break

    for document in sorted_docs:
        if len(selected) >= final_top_k:
            break
        key = document_key(document)
        if key in seen:
            continue
        selected.append(document)
        seen.add(key)
    return selected[:final_top_k]


def annotate_source_ranks(
    documents: list[Document],
    *,
    source_key: str,
    max_chunks_per_source: int,
    relaxed: bool,
) -> list[Document]:
    source_counts: dict[str, int] = {}
    for document in documents:
        metadata = dict(document.metadata or {})
        source = source_identifier(document, source_key)
        source_counts[source] = source_counts.get(source, 0) + 1
        rank_within_file = source_counts[source]
        metadata["source_rank_within_file"] = rank_within_file
        metadata["source_diversity_key"] = source
        metadata["source_diversity_max_chunks"] = max_chunks_per_source
        metadata["source_diversity_relaxed"] = bool(metadata.get("source_diversity_relaxed", relaxed))
        metadata["why_selected"] = ";".join(
            filter(None, [str(metadata.get("why_selected", "")), f"source_rank_within_file:{rank_within_file}"])
        )
        document.metadata = metadata
    return documents


def document_key(document: Document) -> str:
    metadata = document.metadata or {}
    return str(metadata.get("chunk_id") or f"{metadata.get('source')}:{metadata.get('section')}")


def source_identifier(document: Document, source_key: str) -> str:
    metadata = document.metadata or {}
    value = str(metadata.get(source_key) or "").strip()
    if value:
        return value
    for fallback_key in ["imported_source", "source", "original_source"]:
        value = str(metadata.get(fallback_key) or "").strip()
        if value:
            return value
    return document_key(document)


def summarize_source_diversity(
    documents: list[Document],
    *,
    source_key: str,
    max_chunks_per_source: int,
    enabled: bool,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    relaxed_sources: list[str] = []
    for document in documents:
        source = source_identifier(document, source_key)
        counts[source] = counts.get(source, 0) + 1
        if document.metadata.get("source_diversity_relaxed") and source not in relaxed_sources:
            relaxed_sources.append(source)
    violations = {
        source: count
        for source, count in counts.items()
        if enabled and count > max_chunks_per_source
    }
    return {
        "enabled": enabled,
        "source_key": source_key,
        "max_chunks_per_source": max_chunks_per_source,
        "counts": counts,
        "violations": violations,
        "relaxed_sources": relaxed_sources,
        "relaxed": bool(relaxed_sources),
    }


def summarize_documents(documents: list[Document]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        rows.append(
            {
                "rank": rank,
                "source": metadata.get("source", ""),
                "category_dir": metadata.get("category_dir", ""),
                "doc_type": metadata.get("doc_type", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "imported_source": metadata.get("imported_source", ""),
                "original_source": metadata.get("original_source", ""),
                "dense_score": metadata.get("dense_score"),
                "score": metadata.get("final_score"),
                "metadata_score": metadata.get("metadata_score"),
                "keyword_score": metadata.get("keyword_score"),
                "calibration_score": metadata.get("calibration_score", 0.0),
                "calibration_reasons": metadata.get("calibration_reasons", []),
                "calibration_breakdown": metadata.get("calibration_breakdown", []),
                "matched_terms": metadata.get("matched_terms", []),
                "matched_fields": metadata.get("matched_fields", []),
                "final_score": metadata.get("final_score"),
                "retrieval_source": metadata.get("retrieval_source", ""),
                "retrieval_channels": metadata.get("retrieval_channels", []),
                "match_reason": metadata.get("match_reason", ""),
                "why_selected": metadata.get("why_selected", ""),
                "source_rank_within_file": metadata.get("source_rank_within_file"),
                "source_diversity_key": metadata.get("source_diversity_key", ""),
                "source_diversity_relaxed": metadata.get("source_diversity_relaxed", False),
            }
        )
    return rows
