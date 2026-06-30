from __future__ import annotations

from typing import Any


EXPANSION_BY_QUERY_TYPE = {
    "code_entry": ["train", "train_jsbsim.py", "train_jsbsim", "runner", "main", "script", "entry", "训练入口", "启动脚本"],
    "code_general": ["code", "function", "class", "reward", "task", "env", "runner"],
    "config": ["config", "yaml", "configuration", "task", "reward", "scenario", "curriculum"],
    "experiment": ["experiment", "result", "results", "summary", "report", "csv", "metric", "eval"],
    "note_or_report": ["note", "report", "opd", "opponent prediction", "world model", "route", "risk"],
    "paper": ["paper", "table", "manuscript", "mechanism", "repositioning"],
    "mixed": [],
}

EXPANSION_BY_CATEGORY = {
    "code": ["code", "py", "script", "function", "class"],
    "configs": ["config", "yaml", "configuration"],
    "experiments": ["experiment", "result", "summary", "report", "csv"],
    "notes": ["note", "design", "readme"],
    "papers": ["paper", "table"],
    "thesis_or_reports": ["report", "route", "summary"],
}


def expand_query(question: str, classification: dict[str, Any]) -> str:
    terms: list[str] = [question]
    query_type = str(classification.get("query_type") or "general")
    keywords = [str(item) for item in classification.get("keywords", [])]
    target_categories = [str(item) for item in classification.get("target_categories", [])]

    add_unique(terms, keywords)
    add_unique(terms, EXPANSION_BY_QUERY_TYPE.get(query_type, []))
    for category in target_categories:
        add_unique(terms, EXPANSION_BY_CATEGORY.get(category, []))

    return " ".join(term for term in terms if term)


def add_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)
