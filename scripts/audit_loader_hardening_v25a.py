from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunkers.text_chunker import make_chunk_id
from src.loaders.base import load_file
from src.loaders.csv_summary_loader import load_csv_summary_file
from src.loaders.pdf_quality_filter import analyze_pdf_quality
from src.loaders.utils import base_metadata, detect_language, relative_source
from src.metadata.validators import normalize_metadata


CONFIG_PATH = PROJECT_ROOT / "config" / "loader_hardening.yaml"
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".py", ".yaml", ".yml", ".csv"}

DOCUMENT_STATS_FIELDS = [
    "source",
    "category_dir",
    "file_ext",
    "file_size_bytes",
    "loader_type",
    "load_status",
    "document_count",
    "total_text_chars",
    "min_doc_chars",
    "max_doc_chars",
    "should_index",
    "error_message",
]

CHUNK_STATS_FIELDS = [
    "source",
    "category_dir",
    "file_ext",
    "loader_type",
    "document_count",
    "chunk_count",
    "min_chunk_chars",
    "max_chunk_chars",
    "mean_chunk_chars",
    "empty_chunk_count",
    "oversized_chunk_count",
    "too_many_chunks",
]

CSV_AUDIT_FIELDS = [
    "source",
    "category_dir",
    "filename",
    "file_size_bytes",
    "row_count",
    "column_count",
    "large_csv",
    "csv_encoding",
    "document_count",
    "chunk_count",
    "truncated",
    "should_index",
    "reason",
]

PDF_AUDIT_FIELDS = [
    "source",
    "category_dir",
    "filename",
    "file_size_bytes",
    "pdf_quality",
    "should_index",
    "text_char_count",
    "page_count",
    "uni_marker_count",
    "uni_marker_ratio",
    "low_text_page_ratio",
    "document_count",
    "chunk_count",
    "reason",
]

LOAD_ERROR_FIELDS = [
    "source",
    "category_dir",
    "file_ext",
    "file_size_bytes",
    "issue_type",
    "message",
]

CANDIDATE_FIELDS = [
    "source",
    "category_dir",
    "file_ext",
    "doc_type",
    "loader_type",
    "estimated_chunks",
    "pdf_quality",
    "csv_large",
    "should_index",
    "recommended_for_v25b",
    "reason",
]

CATEGORY_DOC_TYPES = {
    "code": "code",
    "configs": "config",
    "experiments": "experiment",
    "notes": "note",
    "papers": "paper",
    "thesis_or_reports": "report",
}

CATEGORY_QUOTAS_V25B = {
    "code": 15,
    "configs": 15,
    "experiments": 15,
    "notes": 15,
    "papers": 10,
    "thesis_or_reports": 10,
}


@dataclass
class HardenedFileAudit:
    path: Path
    source: str
    category_dir: str
    file_ext: str
    file_size_bytes: int
    loader_type: str
    load_status: str
    documents: list[Document]
    chunks: list[Document]
    should_index: bool
    error_message: str = ""
    pdf_quality: dict[str, Any] = field(default_factory=dict)


def main() -> int:
    configure_stdout()
    try:
        result = audit_loader_hardening()
    except Exception as exc:  # noqa: BLE001 - audit should fail explicitly.
        print("V2.5a loader hardening audit failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result["summary"])
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def audit_loader_hardening(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    audit_config = config.get("audit", {})
    source_dir = resolve_project_path(audit_config.get("source_dir", "data/raw_imported"))
    output_dir = resolve_project_path(audit_config.get("output_dir", "data/processed/loader_hardening_v25a"))
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing audit source_dir: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = int(audit_config.get("chunk_size", 800))
    chunk_overlap = int(audit_config.get("chunk_overlap", 120))
    too_many_threshold = int(audit_config.get("too_many_chunks_threshold", 100))
    target_count = int(audit_config.get("candidate_target_count", 80))

    all_files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    supported_files = [path for path in all_files if path.suffix.lower() in SUPPORTED_EXTENSIONS]

    audits: list[HardenedFileAudit] = []
    document_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    pdf_rows: list[dict[str, Any]] = []
    load_error_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for path in supported_files:
        audit = audit_one_file(
            path=path,
            config=config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            too_many_threshold=too_many_threshold,
        )
        audits.append(audit)
        document_rows.append(build_document_row(audit))
        chunk_row = build_chunk_row(audit, chunk_size, too_many_threshold)
        chunk_rows.append(chunk_row)
        collect_issues(audit, chunk_row, load_error_rows)
        if audit.file_ext == ".csv":
            csv_rows.append(build_csv_audit_row(audit, chunk_row))
        if audit.file_ext == ".pdf":
            pdf_rows.append(build_pdf_audit_row(audit, chunk_row))
        candidate_rows.append(build_candidate_row(audit, chunk_row, too_many_threshold))

    mark_recommended_candidates(candidate_rows, target_count)

    paths = {
        "summary": output_dir / "loader_hardening_summary.json",
        "document_stats": output_dir / "document_stats.csv",
        "chunk_stats": output_dir / "chunk_stats.csv",
        "csv_summary_audit": output_dir / "csv_summary_audit.csv",
        "pdf_quality_audit": output_dir / "pdf_quality_audit.csv",
        "load_errors": output_dir / "load_errors.csv",
        "candidate_expansion_plan": output_dir / "candidate_expansion_plan_v25b.csv",
    }

    write_csv(paths["document_stats"], document_rows, DOCUMENT_STATS_FIELDS)
    write_csv(paths["chunk_stats"], chunk_rows, CHUNK_STATS_FIELDS)
    write_csv(paths["csv_summary_audit"], csv_rows, CSV_AUDIT_FIELDS)
    write_csv(paths["pdf_quality_audit"], pdf_rows, PDF_AUDIT_FIELDS)
    write_csv(paths["load_errors"], load_error_rows, LOAD_ERROR_FIELDS)
    write_csv(paths["candidate_expansion_plan"], candidate_rows, CANDIDATE_FIELDS)

    summary = build_summary(
        all_files=all_files,
        supported_files=supported_files,
        audits=audits,
        document_rows=document_rows,
        chunk_rows=chunk_rows,
        csv_rows=csv_rows,
        pdf_rows=pdf_rows,
        load_error_rows=load_error_rows,
        candidate_rows=candidate_rows,
        paths=paths,
        config_path=config_path,
        source_dir=source_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        too_many_threshold=too_many_threshold,
    )
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "audits": audits}


def audit_one_file(
    *,
    path: Path,
    config: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    too_many_threshold: int,
) -> HardenedFileAudit:
    source = relative_source(path)
    category_dir = category_for_imported_file(path)
    file_ext = path.suffix.lower()
    file_size_bytes = path.stat().st_size
    documents: list[Document] = []
    chunks: list[Document] = []
    loader_type = "unsupported"
    load_status = "loaded"
    error_message = ""
    pdf_quality: dict[str, Any] = {}
    should_index = True

    try:
        if file_ext == ".csv":
            loader_type = "csv_summary"
            documents = load_csv_summary_file(path, config.get("csv", {}))
        elif file_ext == ".pdf":
            loader_type = "pdf_quality_filtered"
            documents, pdf_quality = load_pdf_with_quality(path, config.get("pdf", {}))
            should_index = bool(pdf_quality.get("should_index", False))
        elif file_ext in SUPPORTED_EXTENSIONS:
            loader_type = "standard_loader"
            documents = load_file(path)
        else:
            load_status = "unsupported"

        documents = enrich_documents(documents, path, loader_type, pdf_quality)
        if documents:
            chunks = chunk_hardened_documents(documents, chunk_size, chunk_overlap)
        elif load_status == "loaded":
            load_status = "loaded_no_documents"
            should_index = False
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop audit.
        load_status = "failed"
        error_message = str(exc)
        should_index = False
        if file_ext == ".pdf":
            pdf_quality = analyze_pdf_quality(error=str(exc), config=config.get("pdf", {}))

    if file_ext == ".csv" and documents:
        metadata = documents[0].metadata
        should_index = not bool(metadata.get("large_csv", False))
    if chunks and len(chunks) > too_many_threshold:
        should_index = False

    return HardenedFileAudit(
        path=path,
        source=source,
        category_dir=category_dir,
        file_ext=file_ext,
        file_size_bytes=file_size_bytes,
        loader_type=loader_type,
        load_status=load_status,
        documents=documents,
        chunks=chunks,
        should_index=should_index,
        error_message=error_message,
        pdf_quality=pdf_quality,
    )


def load_pdf_with_quality(path: Path, pdf_config: dict[str, Any]) -> tuple[list[Document], dict[str, Any]]:
    try:
        reader = PdfReader(str(path))
        page_texts = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # noqa: BLE001
        quality = analyze_pdf_quality(error=str(exc), config=pdf_config)
        return [], quality

    quality = analyze_pdf_quality(page_texts, config=pdf_config)
    text = "\n\n".join(page_texts).strip()
    if not text:
        return [], quality

    limit = int(pdf_config.get("text_length_limit", 20000))
    truncated = len(text) > limit
    if truncated:
        text = text[:limit].rstrip() + "\n[TRUNCATED: PDF text exceeded text_length_limit]\n"

    category_dir = category_for_imported_file(path)
    metadata = base_metadata(
        path,
        doc_type=CATEGORY_DOC_TYPES.get(category_dir, "paper"),
        title=pdf_title(reader) or path.stem,
        section="pdf_extracted_text",
        language=detect_language(text),
        is_citable=bool(quality.get("should_index", False)),
        imported_source=relative_source(path),
        file_ext=".pdf",
        category_dir=category_dir,
        loader_type="pdf_quality_filtered",
        truncated=truncated,
        **quality,
    )
    return [Document(page_content=text, metadata=metadata)], quality


def pdf_title(reader: PdfReader) -> str | None:
    metadata = getattr(reader, "metadata", None)
    title = getattr(metadata, "title", None) if metadata else None
    return str(title).strip() if title else None


def enrich_documents(
    documents: list[Document],
    path: Path,
    loader_type: str,
    pdf_quality: dict[str, Any],
) -> list[Document]:
    category_dir = category_for_imported_file(path)
    doc_type = CATEGORY_DOC_TYPES.get(category_dir)
    enriched: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata or {})
        metadata["source"] = relative_source(path)
        metadata["imported_source"] = relative_source(path)
        metadata["category_dir"] = category_dir
        metadata["file_ext"] = path.suffix.lower()
        metadata["loader_type"] = metadata.get("loader_type", loader_type)
        if doc_type:
            metadata["doc_type"] = doc_type
        for key, value in pdf_quality.items():
            metadata.setdefault(key, value)
        enriched.append(Document(page_content=document.page_content or "", metadata=normalize_metadata(metadata)))
    return enriched


def chunk_hardened_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for document in documents:
        base = normalize_metadata(document.metadata)
        for chunk_index, text in enumerate(splitter.split_text(document.page_content or "")):
            metadata = dict(base)
            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = make_chunk_id(metadata, chunk_index)
            chunks.append(Document(page_content=text, metadata=normalize_metadata(metadata)))
    return chunks


def build_document_row(audit: HardenedFileAudit) -> dict[str, Any]:
    lengths = [len(document.page_content or "") for document in audit.documents]
    return {
        "source": audit.source,
        "category_dir": audit.category_dir,
        "file_ext": audit.file_ext,
        "file_size_bytes": audit.file_size_bytes,
        "loader_type": audit.loader_type,
        "load_status": audit.load_status,
        "document_count": len(audit.documents),
        "total_text_chars": sum(lengths),
        "min_doc_chars": min(lengths) if lengths else 0,
        "max_doc_chars": max(lengths) if lengths else 0,
        "should_index": audit.should_index,
        "error_message": audit.error_message,
    }


def build_chunk_row(
    audit: HardenedFileAudit,
    chunk_size: int,
    too_many_threshold: int,
) -> dict[str, Any]:
    lengths = [len(chunk.page_content or "") for chunk in audit.chunks]
    oversized_threshold = int(chunk_size * 1.5)
    return {
        "source": audit.source,
        "category_dir": audit.category_dir,
        "file_ext": audit.file_ext,
        "loader_type": audit.loader_type,
        "document_count": len(audit.documents),
        "chunk_count": len(audit.chunks),
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "mean_chunk_chars": round(mean(lengths), 2) if lengths else 0,
        "empty_chunk_count": sum(length == 0 for length in lengths),
        "oversized_chunk_count": sum(length > oversized_threshold for length in lengths),
        "too_many_chunks": len(audit.chunks) > too_many_threshold,
    }


def build_csv_audit_row(audit: HardenedFileAudit, chunk_row: dict[str, Any]) -> dict[str, Any]:
    metadata = audit.documents[0].metadata if audit.documents else {}
    large_csv = bool(metadata.get("large_csv", False))
    return {
        "source": audit.source,
        "category_dir": audit.category_dir,
        "filename": audit.path.name,
        "file_size_bytes": audit.file_size_bytes,
        "row_count": metadata.get("row_count", 0),
        "column_count": metadata.get("column_count", 0),
        "large_csv": large_csv,
        "csv_encoding": metadata.get("csv_encoding", ""),
        "document_count": len(audit.documents),
        "chunk_count": chunk_row["chunk_count"],
        "truncated": metadata.get("truncated", False),
        "should_index": audit.should_index,
        "reason": "large_csv_excluded" if large_csv else "summary_aware_csv_ok",
    }


def build_pdf_audit_row(audit: HardenedFileAudit, chunk_row: dict[str, Any]) -> dict[str, Any]:
    quality = audit.pdf_quality or {}
    return {
        "source": audit.source,
        "category_dir": audit.category_dir,
        "filename": audit.path.name,
        "file_size_bytes": audit.file_size_bytes,
        "pdf_quality": quality.get("pdf_quality", ""),
        "should_index": quality.get("should_index", False),
        "text_char_count": quality.get("text_char_count", 0),
        "page_count": quality.get("page_count", 0),
        "uni_marker_count": quality.get("uni_marker_count", 0),
        "uni_marker_ratio": quality.get("uni_marker_ratio", 0),
        "low_text_page_ratio": quality.get("low_text_page_ratio", 0),
        "document_count": len(audit.documents),
        "chunk_count": chunk_row["chunk_count"],
        "reason": quality.get("reason", ""),
    }


def build_candidate_row(
    audit: HardenedFileAudit,
    chunk_row: dict[str, Any],
    too_many_threshold: int,
) -> dict[str, Any]:
    metadata = audit.documents[0].metadata if audit.documents else {}
    csv_large = bool(metadata.get("large_csv", False)) if audit.file_ext == ".csv" else False
    pdf_quality = str(audit.pdf_quality.get("pdf_quality", "")) if audit.file_ext == ".pdf" else ""
    reasons: list[str] = []
    should_index = bool(audit.should_index)

    if audit.load_status == "failed":
        reasons.append(f"load_failed:{audit.error_message}")
        should_index = False
    if not audit.documents or not audit.chunks:
        reasons.append("no_indexable_text")
        should_index = False
    if audit.file_ext == ".csv" and csv_large:
        reasons.append("large_csv_excluded_for_v25b")
        should_index = False
    if audit.file_ext == ".csv" and "train_env0_episodes" in audit.path.name:
        reasons.append("train_env0_episodes_excluded")
        should_index = False
    if audit.file_ext == ".pdf" and pdf_quality != "ok":
        reasons.append(f"pdf_quality_{pdf_quality}_excluded")
        should_index = False
    if int(chunk_row["chunk_count"]) > too_many_threshold:
        reasons.append("too_many_chunks_excluded")
        should_index = False
    if should_index:
        reasons.append("eligible_for_v25b")

    return {
        "source": audit.source,
        "category_dir": audit.category_dir,
        "file_ext": audit.file_ext,
        "doc_type": metadata.get("doc_type", CATEGORY_DOC_TYPES.get(audit.category_dir, "unknown")),
        "loader_type": audit.loader_type,
        "estimated_chunks": chunk_row["chunk_count"],
        "pdf_quality": pdf_quality,
        "csv_large": csv_large,
        "should_index": should_index,
        "recommended_for_v25b": False,
        "reason": ";".join(reasons),
    }


def collect_issues(
    audit: HardenedFileAudit,
    chunk_row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if audit.load_status == "failed":
        add_issue(rows, audit, "load_failed", audit.error_message)
    if audit.load_status == "loaded_no_documents":
        add_issue(rows, audit, "loaded_no_documents", "loader returned no indexable documents")
    if audit.file_ext == ".csv" and audit.documents:
        metadata = audit.documents[0].metadata
        if bool(metadata.get("large_csv", False)):
            add_issue(rows, audit, "large_csv", "CSV marked large; excluded from V2.5b recommendation")
        if "train_env0_episodes" in audit.path.name:
            add_issue(rows, audit, "train_env0_episodes", "raw episode CSV excluded from V2.5b recommendation")
    if audit.file_ext == ".pdf":
        quality = audit.pdf_quality or {}
        if not bool(quality.get("should_index", False)):
            add_issue(rows, audit, f"pdf_{quality.get('pdf_quality', 'unknown')}", quality.get("reason", ""))
    if bool(chunk_row.get("too_many_chunks", False)):
        add_issue(rows, audit, "too_many_chunks", f"chunk_count={chunk_row.get('chunk_count')}")


def add_issue(rows: list[dict[str, Any]], audit: HardenedFileAudit, issue_type: str, message: str) -> None:
    rows.append(
        {
            "source": audit.source,
            "category_dir": audit.category_dir,
            "file_ext": audit.file_ext,
            "file_size_bytes": audit.file_size_bytes,
            "issue_type": issue_type,
            "message": message,
        }
    )


def mark_recommended_candidates(rows: list[dict[str, Any]], target_count: int) -> None:
    eligible = [row for row in rows if bool(row["should_index"]) and int(row["estimated_chunks"]) > 0]
    selected: set[str] = set()

    for category, quota in CATEGORY_QUOTAS_V25B.items():
        category_rows = sorted(
            [row for row in eligible if row["category_dir"] == category],
            key=candidate_sort_key,
        )
        for row in category_rows[:quota]:
            row["recommended_for_v25b"] = True
            row["reason"] += ";selected_for_v25b_category_quota"
            selected.add(str(row["source"]))

    if len(selected) < target_count:
        remaining = sorted(
            [row for row in eligible if str(row["source"]) not in selected],
            key=candidate_sort_key,
        )
        for row in remaining[: target_count - len(selected)]:
            row["recommended_for_v25b"] = True
            row["reason"] += ";selected_for_v25b_fill"
            selected.add(str(row["source"]))

    for row in eligible:
        if not row["recommended_for_v25b"]:
            row["reason"] += ";eligible_not_selected_target_limit"


def candidate_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    ext_priority = {
        ".md": 0,
        ".py": 1,
        ".yaml": 2,
        ".yml": 2,
        ".csv": 3,
        ".txt": 4,
        ".pdf": 5,
    }.get(str(row["file_ext"]), 9)
    return ext_priority, int(row["estimated_chunks"]), str(row["source"])


def build_summary(
    *,
    all_files: list[Path],
    supported_files: list[Path],
    audits: list[HardenedFileAudit],
    document_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    csv_rows: list[dict[str, Any]],
    pdf_rows: list[dict[str, Any]],
    load_error_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    paths: dict[str, Path],
    config_path: Path,
    source_dir: Path,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    too_many_threshold: int,
) -> dict[str, Any]:
    failed_files = sum(row["load_status"] == "failed" for row in document_rows)
    loaded_files = len(document_rows) - failed_files
    total_documents = sum(int(row["document_count"]) for row in document_rows)
    total_chunks = sum(int(row["chunk_count"]) for row in chunk_rows)
    csv_total_chunks = sum(int(row["chunk_count"]) for row in csv_rows)
    csv_total_documents = sum(int(row["document_count"]) for row in csv_rows)
    pdf_quality_counts = Counter(str(row["pdf_quality"]) for row in pdf_rows)
    recommended_count = sum(bool(row["recommended_for_v25b"]) for row in candidate_rows)
    issue_counts = Counter(str(row["issue_type"]) for row in load_error_rows)
    chunks_by_category = Counter()
    chunks_by_extension = Counter()
    for row in chunk_rows:
        chunks_by_category[str(row["category_dir"])] += int(row["chunk_count"])
        chunks_by_extension[str(row["file_ext"])] += int(row["chunk_count"])

    train_episode_rows = [
        row for row in csv_rows if "train_env0_episodes" in str(row["source"])
    ]
    previous_summary = load_previous_imported_audit_summary()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_environment": "Lenginzed_RAG",
        "config_path": str(config_path),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "external_source_modified": False,
        "raw_imported_modified": False,
        "chroma_written": False,
        "embedding_run": False,
        "llm_called": False,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "too_many_chunks_threshold": too_many_threshold,
        "total_files": len(all_files),
        "supported_files": len(supported_files),
        "loaded_files": loaded_files,
        "failed_files": failed_files,
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "chunks_by_category": dict(chunks_by_category),
        "chunks_by_extension": dict(chunks_by_extension),
        "csv_files": len(csv_rows),
        "csv_total_documents": csv_total_documents,
        "csv_total_chunks": csv_total_chunks,
        "large_csv_files": sum(as_bool(row["large_csv"]) for row in csv_rows),
        "train_env0_episodes_files": len(train_episode_rows),
        "train_env0_episodes_total_chunks": sum(int(row["chunk_count"]) for row in train_episode_rows),
        "pdf_files": len(pdf_rows),
        "pdf_quality_counts": dict(pdf_quality_counts),
        "pdf_ok_files": pdf_quality_counts.get("ok", 0),
        "pdf_low_quality_files": len(pdf_rows) - pdf_quality_counts.get("ok", 0),
        "pdf_should_index_false": sum(not as_bool(row["should_index"]) for row in pdf_rows),
        "too_many_chunks_files": sum(as_bool(row["too_many_chunks"]) for row in chunk_rows),
        "recommended_for_v25b_count": recommended_count,
        "recommended_by_category": dict(
            Counter(row["category_dir"] for row in candidate_rows if as_bool(row["recommended_for_v25b"]))
        ),
        "recommended_by_extension": dict(
            Counter(row["file_ext"] for row in candidate_rows if as_bool(row["recommended_for_v25b"]))
        ),
        "load_issue_counts": dict(issue_counts),
        "previous_v2b0_csv_total_chunks": previous_summary.get("chunks_by_extension", {}).get(".csv"),
        "csv_chunk_reduction_note": (
            f"CSV chunks now {csv_total_chunks}; previous V2b.0 CSV chunks "
            f"{previous_summary.get('chunks_by_extension', {}).get('.csv', 'unknown')}"
        ),
        "top_20_files_by_chunk_count": top_rows(chunk_rows, "chunk_count"),
        "top_20_files_by_text_chars": top_rows(document_rows, "total_text_chars"),
        "outputs": {key: str(value) for key, value in paths.items()},
    }


def top_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: int(float(row.get(key, 0) or 0)), reverse=True)
    return [
        {
            "source": row["source"],
            "category_dir": row["category_dir"],
            "file_ext": row.get("file_ext", ""),
            key: int(float(row.get(key, 0) or 0)),
        }
        for row in sorted_rows[:20]
    ]


def load_previous_imported_audit_summary() -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "processed" / "imported_audit" / "audit_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - previous summary is optional context.
        return {}


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


def category_for_imported_file(path: Path) -> str:
    parts = path.as_posix().split("/")
    if "raw_imported" in parts:
        index = parts.index("raw_imported")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(summary: dict[str, Any]) -> None:
    print("V2.5a loader hardening audit completed.")
    print(json.dumps({
        "total_files": summary["total_files"],
        "loaded_files": summary["loaded_files"],
        "failed_files": summary["failed_files"],
        "total_documents": summary["total_documents"],
        "total_chunks": summary["total_chunks"],
        "csv_files": summary["csv_files"],
        "csv_total_documents": summary["csv_total_documents"],
        "csv_total_chunks": summary["csv_total_chunks"],
        "large_csv_files": summary["large_csv_files"],
        "pdf_files": summary["pdf_files"],
        "pdf_quality_counts": summary["pdf_quality_counts"],
        "pdf_should_index_false": summary["pdf_should_index_false"],
        "too_many_chunks_files": summary["too_many_chunks_files"],
        "recommended_for_v25b_count": summary["recommended_for_v25b_count"],
        "output_dir": summary["output_dir"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
