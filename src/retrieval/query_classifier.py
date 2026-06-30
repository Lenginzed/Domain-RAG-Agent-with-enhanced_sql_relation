from __future__ import annotations

import re
from typing import Any


CATEGORY_BY_INTENT = {
    "code": "code",
    "configs": "configs",
    "experiments": "experiments",
    "notes": "notes",
    "papers": "papers",
    "thesis_or_reports": "thesis_or_reports",
}


def classify_query(question: str) -> dict[str, Any]:
    lowered = question.lower()
    target_categories: list[str] = []
    keywords: list[str] = []
    intent_flags: set[str] = set()

    if has_any(lowered, ["入口", "训练入口", "启动", "运行脚本", "train", "runner", "main"]):
        intent_flags.add("code_entry")
        add_unique(target_categories, ["code"])
        add_unique(
            keywords,
            [
                "train",
                "train_jsbsim",
                "train_jsbsim.py",
                "scripts/train",
                "runner",
                "main",
                "script",
                "entry",
                "入口",
                "训练入口",
                "启动脚本",
                "code",
            ],
        )

    if has_any(lowered, ["代码", "函数", "源码", "class", "function", "reward", "奖励函数", "任务"]):
        intent_flags.add("code_general")
        add_unique(target_categories, ["code"])
        add_unique(keywords, ["code", "function", "class", "reward", "task", "runner", "env"])

    if has_any(lowered, ["配置", "yaml", "任务配置", "reward config", "scenario", "config", "configuration"]):
        intent_flags.add("config")
        add_unique(target_categories, ["configs"])
        add_unique(keywords, ["config", "configs", "configuration", "yaml", "task", "reward", "scenario", "curriculum"])

    if has_any(lowered, ["实验", "结果", "指标", "report", "summary", "csv", "eval", "评估"]):
        intent_flags.add("experiment")
        add_unique(target_categories, ["experiments", "thesis_or_reports"])
        add_unique(keywords, ["experiment", "experiments", "result", "results", "summary", "report", "csv", "eval", "metric"])

    if has_any(lowered, ["opd", "对手预测", "预测", "笔记", "note", "路线", "route", "报告"]):
        intent_flags.add("note_or_report")
        add_unique(target_categories, ["notes", "thesis_or_reports"])
        add_unique(keywords, ["opd", "opponent", "prediction", "predictor", "note", "report", "route", "risk", "wm"])

    if has_any(lowered, ["论文", "paper", "表格", "table", "manuscript"]):
        intent_flags.add("paper")
        add_unique(target_categories, ["papers"])
        add_unique(keywords, ["paper", "table", "manuscript", "repositioning", "mechanism"])

    add_unique(keywords, extract_ascii_terms(question))
    query_type = choose_query_type(intent_flags, target_categories)
    if not target_categories:
        target_categories = []
    if not keywords:
        keywords = extract_loose_terms(question)

    return {
        "query_type": query_type,
        "target_categories": target_categories,
        "keywords": keywords,
    }


def choose_query_type(intent_flags: set[str], target_categories: list[str]) -> str:
    if len(set(target_categories)) > 1 and ("config" in intent_flags or "experiment" in intent_flags):
        return "mixed"
    if "code_entry" in intent_flags:
        return "code_entry"
    for query_type in ["code_general", "config", "experiment", "note_or_report", "paper"]:
        if query_type in intent_flags:
            return query_type
    if len(set(target_categories)) > 1:
        return "mixed"
    return "general"


def has_any(text: str, needles: list[str]) -> bool:
    return any(needle.lower() in text for needle in needles)


def add_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def extract_ascii_terms(text: str) -> list[str]:
    return [item.lower() for item in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", text)]


def extract_loose_terms(text: str) -> list[str]:
    terms = extract_ascii_terms(text)
    for token in ["训练", "入口", "配置", "实验", "结果", "笔记", "论文"]:
        if token in text:
            terms.append(token)
    return list(dict.fromkeys(terms))
