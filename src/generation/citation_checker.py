from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.documents import Document


DEFAULT_CONFIG = {
    "min_supported_claim_ratio": 0.6,
    "claim_split_delimiters": ["。", "；", ";"],
    "max_claims_to_check": 8,
    "min_keyword_overlap": 1,
    "ignore_claim_types": ["heading", "list_intro", "citation_marker_only", "too_short"],
    "negative_claim_patterns": ["没有明确", "未提到", "未提及", "没有足够依据", "无法确定", "不能确定", "不足以支持"],
    "heading_patterns": ["实验结果文件", "配置文件", "奖励函数", "任务配置", "主要讨论", "相关资料"],
}

INSUFFICIENT_PATTERNS = [
    "当前知识库没有足够依据",
    "没有足够依据",
    "insufficient evidence",
    "not enough evidence",
]

DOMAIN_TERMS = [
    "训练",
    "入口",
    "启动",
    "脚本",
    "文件",
    "代码",
    "配置",
    "任务",
    "奖励",
    "函数",
    "环境",
    "实验",
    "结果",
    "指标",
    "报告",
    "笔记",
    "论文",
    "路线",
    "对手",
    "预测",
    "风险",
    "空战",
    "无人机",
    "课程",
    "仿真",
    "阶段",
]

STOPWORDS = {
    "回答",
    "来源",
    "依据",
    "这个",
    "这些",
    "资料",
    "当前",
    "知识库",
    "可以",
    "主要",
    "包括",
    "说明",
    "因此",
    "如下",
    "以下",
    "文件",
}


@dataclass
class ClaimCheck:
    text: str
    claim: str
    claim_type: str
    checked: bool
    supported: bool
    support_reason: str
    reason: str
    keywords: list[str]
    matched_keywords: list[str]
    matched_terms: list[str]
    cited_sources: list[str]
    checker_error_hint: str


def check_citations(
    *,
    answer: str,
    sources: list[Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checker_config = {**DEFAULT_CONFIG, **(config or {})}
    if is_insufficient_answer(answer):
        claim = make_claim_check(
            text=answer.strip(),
            claim_type="insufficient_answer",
            checked=False,
            supported=True,
            support_reason="insufficient-evidence template answer; ordinary citation support is not required.",
            keywords=[],
            matched_keywords=[],
            cited_sources=[],
            checker_error_hint="",
        )
        return build_result(
            checked_claims=[],
            supported_claims=[],
            unsupported_claims=[],
            ignored_claims=[claim],
            weak_claims=[],
            threshold=float(checker_config.get("min_supported_claim_ratio", 0.6)),
            insufficient_answer=True,
            reason="insufficient-evidence answer; ordinary claim citation support is not required.",
        )

    source_records = normalize_sources(sources)
    raw_claims = split_claims(
        answer,
        delimiters=[str(item) for item in checker_config.get("claim_split_delimiters", ["。", "；", ";"])],
        max_claims=int(checker_config.get("max_claims_to_check", 8)),
    )

    checked_claims: list[dict[str, Any]] = []
    supported_claims: list[dict[str, Any]] = []
    unsupported_claims: list[dict[str, Any]] = []
    ignored_claims: list[dict[str, Any]] = []
    weak_claims: list[dict[str, Any]] = []
    min_overlap = int(checker_config.get("min_keyword_overlap", 1))
    ignore_claim_types = {str(item) for item in checker_config.get("ignore_claim_types", [])}

    for claim in raw_claims:
        claim_type = classify_claim(claim, checker_config)
        keywords = extract_keywords(claim)
        cited_labels = extract_citation_labels(claim)
        candidate_sources = select_candidate_sources(source_records, cited_labels)
        matched_keywords = sorted({term for term in keywords if term_in_sources(term, candidate_sources)})

        if claim_type in ignore_claim_types:
            ignored_claims.append(
                make_claim_check(
                    text=claim,
                    claim_type=claim_type,
                    checked=False,
                    supported=True,
                    support_reason=f"{claim_type} is ignored for citation scoring.",
                    keywords=keywords,
                    matched_keywords=matched_keywords,
                    cited_sources=cited_labels,
                    checker_error_hint="ignored_non_factual_claim",
                )
            )
            continue

        if claim_type == "negative_statement":
            negative_claim = evaluate_negative_statement(claim, keywords, matched_keywords, cited_labels, min_overlap)
            if negative_claim["supported"]:
                supported_claims.append(negative_claim)
                checked_claims.append(negative_claim)
            else:
                weak_claims.append(negative_claim)
            continue

        important_terms = important_exact_terms(keywords)
        important_matches = [term for term in important_terms if term in matched_keywords]
        enough_overlap = len(matched_keywords) >= min_overlap
        important_term_ok = not important_terms or bool(important_matches)
        supported = enough_overlap and important_term_ok
        hint = ""
        if not enough_overlap:
            hint = "keyword_overlap_below_threshold"
        elif not important_term_ok:
            hint = "important_identifier_or_filename_not_found"
        check = make_claim_check(
            text=claim,
            claim_type="factual_claim",
            checked=True,
            supported=supported,
            support_reason=(
                f"matched_keywords={matched_keywords}"
                if supported
                else f"matched_keywords={matched_keywords}; important_terms={important_terms}"
            ),
            keywords=keywords,
            matched_keywords=matched_keywords,
            cited_sources=cited_labels,
            checker_error_hint=hint,
        )
        checked_claims.append(check)
        if supported:
            supported_claims.append(check)
        else:
            unsupported_claims.append(check)

    return build_result(
        checked_claims=checked_claims,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        ignored_claims=ignored_claims,
        weak_claims=weak_claims,
        threshold=float(checker_config.get("min_supported_claim_ratio", 0.6)),
        insufficient_answer=False,
        reason="rule-based claim citation check completed.",
    )


def build_result(
    *,
    checked_claims: list[dict[str, Any]],
    supported_claims: list[dict[str, Any]],
    unsupported_claims: list[dict[str, Any]],
    ignored_claims: list[dict[str, Any]],
    weak_claims: list[dict[str, Any]],
    threshold: float,
    insufficient_answer: bool,
    reason: str,
) -> dict[str, Any]:
    scoreable_count = len(supported_claims) + len(unsupported_claims)
    ratio = 1.0 if scoreable_count == 0 else round(len(supported_claims) / scoreable_count, 6)
    claim_type_counts = Counter(
        str(item["claim_type"])
        for group in [checked_claims, ignored_claims, weak_claims]
        for item in group
    )
    return {
        "insufficient_answer": insufficient_answer,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "ignored_claims": ignored_claims,
        "weak_claims": weak_claims,
        "checked_claims": checked_claims,
        "supported_claim_ratio": ratio,
        "checked_claim_count": len(checked_claims),
        "ignored_claim_count": len(ignored_claims),
        "weak_claim_count": len(weak_claims),
        "claim_type_counts": dict(claim_type_counts),
        "citation_check_passed": ratio >= threshold,
        "reason": f"{reason} supported_claim_ratio={ratio} threshold={threshold}",
    }


def make_claim_check(
    *,
    text: str,
    claim_type: str,
    checked: bool,
    supported: bool,
    support_reason: str,
    keywords: list[str],
    matched_keywords: list[str],
    cited_sources: list[str],
    checker_error_hint: str,
) -> dict[str, Any]:
    item = ClaimCheck(
        text=text,
        claim=text,
        claim_type=claim_type,
        checked=checked,
        supported=supported,
        support_reason=support_reason,
        reason=support_reason,
        keywords=keywords,
        matched_keywords=matched_keywords,
        matched_terms=matched_keywords,
        cited_sources=cited_sources,
        checker_error_hint=checker_error_hint,
    )
    return asdict(item)


def evaluate_negative_statement(
    claim: str,
    keywords: list[str],
    matched_keywords: list[str],
    cited_labels: list[str],
    min_overlap: int,
) -> dict[str, Any]:
    object_keywords = [
        keyword
        for keyword in keywords
        if keyword not in {"没有", "明确", "未提到", "无法", "不能", "确定", "不足以支持"}
    ]
    object_matches = [term for term in object_keywords if term in matched_keywords]
    if len(object_matches) >= min_overlap:
        return make_claim_check(
            text=claim,
            claim_type="negative_statement",
            checked=True,
            supported=True,
            support_reason=f"negative statement object appears in retrieved evidence: {object_matches}",
            keywords=keywords,
            matched_keywords=matched_keywords,
            cited_sources=cited_labels,
            checker_error_hint="",
        )
    return make_claim_check(
        text=claim,
        claim_type="negative_statement",
        checked=True,
        supported=False,
        support_reason="negative statement object is weakly supported or absent in retrieved evidence.",
        keywords=keywords,
        matched_keywords=matched_keywords,
        cited_sources=cited_labels,
        checker_error_hint="weak_negative_claim",
    )


def is_insufficient_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(pattern.lower() in lowered for pattern in INSUFFICIENT_PATTERNS)


def split_claims(answer: str, *, delimiters: list[str], max_claims: int) -> list[str]:
    normalized = answer.replace("\r\n", "\n")
    for delimiter in delimiters:
        normalized = normalized.replace(delimiter, "\n")
    raw_parts = re.split(r"\n+|(?<=[.!?])\s+", normalized)
    claims: list[str] = []
    for part in raw_parts:
        cleaned = re.sub(r"^\s*[-*0-9.、)）]+\s*", "", part).strip()
        if not cleaned:
            continue
        if normalize_for_type(cleaned) in {"回答", "来源依据", "来源"}:
            continue
        claims.append(cleaned)
        if len(claims) >= max_claims:
            break
    return claims


def classify_claim(claim: str, config: dict[str, Any]) -> str:
    normalized = normalize_for_type(claim)
    without_citations = normalize_for_type(re.sub(r"\[S\d+\]", "", claim))
    if not without_citations:
        return "citation_marker_only"
    if len(without_citations) < 4 and not re.search(r"[A-Za-z0-9_./-]", without_citations):
        return "too_short"

    negative_patterns = [str(item) for item in config.get("negative_claim_patterns", [])]
    if any(pattern and pattern in normalized for pattern in negative_patterns):
        return "negative_statement"

    if is_list_intro(normalized):
        return "list_intro"

    heading_patterns = [str(item) for item in config.get("heading_patterns", [])]
    if is_heading(normalized, heading_patterns):
        return "heading"

    return "factual_claim"


def normalize_for_type(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^\s*#+\s*", "", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" ：:;；。,.，")
    return value


def is_list_intro(normalized: str) -> bool:
    intro_markers = ["以下", "如下", "包含以下", "包括以下", "列出", "有以下"]
    subject_markers = ["这些资料", "资料中", "检索资料", "来源中", "文档中"]
    if any(marker in normalized for marker in intro_markers) and (
        any(marker in normalized for marker in subject_markers) or "：" in normalized or ":" in normalized
    ):
        return True
    if normalized.endswith(("如下", "如下所示")):
        return True
    return False


def is_heading(normalized: str, heading_patterns: list[str]) -> bool:
    compact = normalized.replace(" ", "")
    if compact in {pattern.replace(" ", "") for pattern in heading_patterns}:
        return True
    if len(compact) <= 18 and any(pattern and pattern.replace(" ", "") in compact for pattern in heading_patterns):
        return True
    if len(compact) <= 24 and not any(verb in compact for verb in ["是", "为", "定义", "包含", "记录", "讨论", "位于"]):
        if any(term in compact for term in ["文件", "函数", "配置", "资料", "结论", "结果", "相关"]):
            return True
    return False


def extract_keywords(claim: str) -> list[str]:
    keywords: list[str] = []
    text_without_citations = re.sub(r"\[S\d+\]", " ", claim)
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*|stage\d+[A-Za-z0-9_.-]*|\d+v\d+|\d+", text_without_citations):
        add_keyword(keywords, token.lower())
        if "." in token or "_" in token or "-" in token:
            for part in re.split(r"[^A-Za-z0-9]+", token):
                add_keyword(keywords, part.lower())
    for term in DOMAIN_TERMS:
        if term in claim:
            add_keyword(keywords, term)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,8}", claim):
        if chunk not in STOPWORDS and not any(stop in chunk for stop in STOPWORDS):
            add_keyword(keywords, chunk)
    return keywords


def add_keyword(keywords: list[str], token: str) -> None:
    cleaned = token.strip().lower()
    if not cleaned or cleaned in STOPWORDS:
        return
    if len(cleaned) < 2:
        return
    if cleaned not in keywords:
        keywords.append(cleaned)


def important_exact_terms(keywords: list[str]) -> list[str]:
    important: list[str] = []
    for keyword in keywords:
        if re.search(r"\.(py|yaml|yml|csv|md|pdf|json|toml|ini)$", keyword):
            important.append(keyword)
        elif "_" in keyword and len(keyword) >= 6:
            important.append(keyword)
        elif re.match(r"stage\d+[a-z0-9_.-]*", keyword) and len(keyword) >= 7:
            important.append(keyword)
    return list(dict.fromkeys(important))


def extract_citation_labels(claim: str) -> list[str]:
    labels: list[str] = []
    for match in re.findall(r"\[S(\d+)\]", claim):
        label = f"S{match}"
        if label not in labels:
            labels.append(label)
    return labels


def select_candidate_sources(source_records: list[dict[str, Any]], cited_labels: list[str]) -> list[dict[str, Any]]:
    if not cited_labels:
        return source_records
    selected = [record for record in source_records if record["label"] in cited_labels]
    return selected or source_records


def term_in_sources(term: str, source_records: list[dict[str, Any]]) -> bool:
    lowered = term.lower()
    return any(lowered in record["text"] for record in source_records)


def normalize_sources(sources: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        label = f"S{index}"
        if isinstance(source, Document):
            metadata = dict(source.metadata or {})
            content = source.page_content or ""
        elif isinstance(source, dict):
            metadata = dict(source)
            content = str(source.get("page_content") or source.get("content") or source.get("text") or "")
            label = str(source.get("label") or label)
        else:
            metadata = {}
            content = str(source)
        metadata_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        records.append(
            {
                "label": label,
                "metadata": metadata,
                "content": content,
                "text": f"{content}\n{metadata_text}".lower(),
            }
        )
    return records
