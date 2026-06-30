from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.indexes.vector_index import CHROMA_DIR, DEFAULT_COLLECTION_NAME, load_index


DEFAULT_FIELD_WEIGHTS = {
    "content": 1.0,
    "filename": 2.0,
    "path": 1.5,
    "title_section": 1.5,
    "category_doc_type": 1.0,
}

STOPWORDS = {
    "what",
    "where",
    "which",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "about",
    "是什么",
    "什么",
    "哪个",
    "哪些",
    "资料",
    "文件",
    "里面",
}


@dataclass
class KeywordRecord:
    content: str
    metadata: dict[str, Any]
    field_counters: dict[str, Counter[str]]
    field_texts: dict[str, str]
    length: int


class KeywordRetriever:
    def __init__(
        self,
        *,
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        lowercase: bool = True,
        min_token_len: int = 2,
        field_weights: dict[str, float] | None = None,
    ) -> None:
        self.lowercase = lowercase
        self.min_token_len = min_token_len
        self.field_weights = dict(DEFAULT_FIELD_WEIGHTS)
        if field_weights:
            self.field_weights.update(field_weights)
        self.vector_store = load_index(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        self.records = self._load_records()
        self.document_frequency = self._document_frequency()
        self.average_length = sum(record.length for record in self.records) / max(len(self.records), 1)

    def retrieve(
        self,
        *,
        question: str,
        expanded_query: str = "",
        classification: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> list[Document]:
        query_terms = self.query_terms(question, expanded_query, classification or {})
        scored: list[Document] = []
        for record in self.records:
            score, matched_terms, matched_fields = self.score_record(record, query_terms)
            if score <= 0:
                continue
            metadata = dict(record.metadata)
            metadata["keyword_score"] = round(score, 4)
            metadata["matched_terms"] = matched_terms
            metadata["matched_fields"] = matched_fields
            metadata["retrieval_source"] = "keyword"
            metadata["retrieval_channels"] = ["keyword"]
            metadata["why_selected"] = ";".join(
                [
                    f"keyword_score:{score:.4f}",
                    f"matched_terms:{','.join(matched_terms[:12])}",
                    f"matched_fields:{','.join(matched_fields[:12])}",
                ]
            )
            scored.append(Document(page_content=record.content, metadata=metadata))

        scored.sort(
            key=lambda doc: (
                -float(doc.metadata.get("keyword_score", 0) or 0),
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("chunk_id", "")),
            )
        )
        for rank, document in enumerate(scored[:top_k], start=1):
            metadata = dict(document.metadata or {})
            metadata["keyword_rank"] = rank
            metadata["why_selected"] = ";".join(filter(None, [str(metadata.get("why_selected", "")), f"keyword_rank:{rank}"]))
            document.metadata = metadata
        return scored[:top_k]

    def query_terms(
        self,
        question: str,
        expanded_query: str,
        classification: dict[str, Any],
    ) -> list[str]:
        values: list[str] = [question, expanded_query]
        values.extend(str(item) for item in classification.get("keywords", []))
        values.extend(str(item) for item in classification.get("target_categories", []))
        terms: list[str] = []
        for value in values:
            for term in tokenize(value, lowercase=self.lowercase, min_token_len=self.min_token_len):
                if term not in STOPWORDS and term not in terms:
                    terms.append(term)
        return terms

    def score_record(self, record: KeywordRecord, query_terms: list[str]) -> tuple[float, list[str], list[str]]:
        score = 0.0
        matched_terms: list[str] = []
        matched_fields: list[str] = []
        for term in query_terms:
            term_score = 0.0
            term_fields: list[str] = []
            idf = self.idf(term)
            for field_name, counter in record.field_counters.items():
                tf = counter.get(term, 0)
                if tf <= 0 and len(term) >= 4 and term in record.field_texts[field_name]:
                    tf = 1
                if tf <= 0:
                    continue
                weight = self.field_weights.get(field_name, 1.0)
                norm = 1.0
                if field_name == "content":
                    norm = 0.75 + 0.25 * (record.length / max(self.average_length, 1.0))
                term_score += (1.0 + math.log(tf)) * idf * weight / norm
                term_fields.append(field_name)
            if term_score > 0:
                score += term_score
                matched_terms.append(term)
                matched_fields.extend(field for field in term_fields if field not in matched_fields)
        return score, matched_terms, matched_fields

    def idf(self, term: str) -> float:
        document_count = max(len(self.records), 1)
        df = self.document_frequency.get(term, 0)
        return math.log((document_count + 1) / (df + 1)) + 1.0

    def _load_records(self) -> list[KeywordRecord]:
        raw = self.vector_store._collection.get(include=["documents", "metadatas"])  # noqa: SLF001
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        records: list[KeywordRecord] = []
        for content, metadata in zip(documents, metadatas, strict=False):
            metadata = dict(metadata or {})
            content_text = str(content or "")
            field_texts = build_field_texts(content_text, metadata, lowercase=self.lowercase)
            field_counters = {
                field: Counter(tokenize(text, lowercase=False, min_token_len=self.min_token_len))
                for field, text in field_texts.items()
            }
            records.append(
                KeywordRecord(
                    content=content_text,
                    metadata=metadata,
                    field_counters=field_counters,
                    field_texts=field_texts,
                    length=sum(field_counters["content"].values()),
                )
            )
        return records

    def _document_frequency(self) -> Counter[str]:
        df: Counter[str] = Counter()
        for record in self.records:
            terms = set()
            for counter in record.field_counters.values():
                terms.update(counter)
            df.update(terms)
        return df


def build_field_texts(content: str, metadata: dict[str, Any], *, lowercase: bool) -> dict[str, str]:
    source = str(metadata.get("source") or "")
    imported_source = str(metadata.get("imported_source") or "")
    original_source = str(metadata.get("original_source") or "")
    title = str(metadata.get("title") or "")
    section = str(metadata.get("section") or "")
    doc_type = str(metadata.get("doc_type") or "")
    category_dir = str(metadata.get("category_dir") or "")
    filename_text = " ".join([Path(source).name, Path(imported_source).name, Path(original_source).name])
    path_text = " ".join([source, imported_source, original_source])
    values = {
        "content": content,
        "filename": filename_text,
        "path": path_text,
        "title_section": " ".join([title, section]),
        "category_doc_type": " ".join([category_dir, doc_type]),
    }
    if lowercase:
        values = {key: value.lower() for key, value in values.items()}
    return values


def tokenize(text: str, *, lowercase: bool = True, min_token_len: int = 2) -> list[str]:
    normalized = text
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_./-]*|[0-9]+|[\u4e00-\u9fff]{2,}", normalized):
        parts = split_ascii_token(raw) if re.match(r"[A-Za-z0-9_./-]+$", raw) else [raw]
        for part in parts:
            part = part.lower() if lowercase else part
            if len(part) >= min_token_len:
                tokens.append(part)
    return tokens


def split_ascii_token(raw: str) -> list[str]:
    value = raw.replace("\\", "/")
    pieces = re.split(r"[^A-Za-z0-9]+", value)
    tokens: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        tokens.append(piece.lower())
        tokens.extend(split_camel_case(piece))
    joined = "".join(piece.lower() for piece in pieces if piece)
    if joined:
        tokens.append(joined)
    return list(dict.fromkeys(tokens))


def split_camel_case(value: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return [item.lower() for item in spaced.split() if item]
