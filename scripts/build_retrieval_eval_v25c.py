from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_eval_v25c.yaml"


def main() -> int:
    configure_stdout()
    try:
        result = build_eval_set()
    except Exception as exc:  # noqa: BLE001 - fail loudly for eval-set integrity.
        print("V2.5c retrieval eval set build failed.")
        print(f"reason: {exc}")
        return 1
    print("V2.5c retrieval eval set built.")
    print(
        json.dumps(
            {
                "question_count": result["question_count"],
                "negative_questions": result["negative_questions"],
                "mixed_questions": result["mixed_questions"],
                "expected_source_absent_count": result["expected_source_absent_count"],
                "category_coverage": result["category_coverage"],
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_eval_set(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    expanded_files = read_csv(resolve_project_path(config["inputs"]["expanded_files_csv"]))
    old_items = load_jsonl(resolve_project_path(config["inputs"]["old_eval_jsonl"]))
    source_blob = "\n".join(row.get("source", "").lower() for row in expanded_files)
    source_categories = {row.get("source", ""): row.get("category_dir", "") for row in expanded_files}

    items = build_blueprint()
    missing = validate_eval_items(items, source_blob)
    if missing:
        raise ValueError(f"Expected sources are absent from expanded collection: {missing}")

    target = config.get("target", {})
    expected_total = int(target.get("total_questions", 30))
    if len(items) != expected_total:
        raise ValueError(f"Expected {expected_total} questions, got {len(items)}")

    output_jsonl = resolve_project_path(config["outputs"]["eval_set_jsonl"])
    output_manifest = resolve_project_path(config["outputs"]["eval_manifest_json"])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, items)

    manifest = build_manifest(
        items=items,
        expanded_files=expanded_files,
        old_items=old_items,
        source_blob=source_blob,
        source_categories=source_categories,
        config_path=config_path,
        output_jsonl=output_jsonl,
        output_manifest=output_manifest,
    )
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_blueprint() -> list[dict[str, Any]]:
    return [
        eval_item(
            "v25c_q001",
            "EventDrivenReward 奖励逻辑在哪个代码文件中？",
            "code_general",
            "easy",
            ["event_driven_reward.py"],
            ["code"],
            [],
            "Adapted from V2b.5 q005; exact code filename exists in the 80-file collection.",
        ),
        eval_item(
            "v25c_q002",
            "shoot penalty reward 在哪个文件里？",
            "code_general",
            "easy",
            ["shoot_penalty_reward.py"],
            ["code"],
            [],
            "Adapted from V2b.5 q006; filename/path query.",
        ),
        eval_item(
            "v25c_q003",
            "相对高度奖励 relative altitude reward 的实现文件是什么？",
            "code_general",
            "easy",
            ["relative_altitude_reward.py"],
            ["code"],
            [],
            "Code query for a reward module in the expanded collection.",
        ),
        eval_item(
            "v25c_q004",
            "timeout 终止条件对应哪个代码文件？",
            "code_general",
            "medium",
            ["timeout.py"],
            ["code"],
            [],
            "Code query for a termination condition file.",
        ),
        eval_item(
            "v25c_q005",
            "HierarchySelfplay 的 YAML 配置来自哪里？",
            "config",
            "easy",
            ["HierarchySelfplay.yaml"],
            ["configs"],
            [],
            "Adapted from V2b.5 q011; config filename query.",
        ),
        eval_item(
            "v25c_q006",
            "offensive curriculum 相关配置文件在哪里？",
            "config",
            "easy",
            ["offensive_curriculum_original_1v1.yaml"],
            ["configs"],
            [],
            "Adapted from V2b.5 q007; config path query.",
        ),
        eval_item(
            "v25c_q007",
            "physical_1v1 物理配置文件是哪一个？",
            "config",
            "easy",
            ["physical_1v1.yaml"],
            ["configs"],
            [],
            "Adapted from V2b.5 q012; exact config filename query.",
        ),
        eval_item(
            "v25c_q008",
            "HumanFreeFly 场景配置在哪个 YAML 文件中？",
            "config",
            "medium",
            ["HumanFreeFly.yaml"],
            ["configs"],
            [],
            "Config query for a scenario file in the expanded collection.",
        ),
        eval_item(
            "v25c_q009",
            "stage11b reward audit report 是哪份实验报告？",
            "experiment",
            "easy",
            ["stage11b_reward_audit_report.md"],
            ["experiments"],
            [],
            "Adapted from V2b.5 q014; experiment report filename query.",
        ),
        eval_item(
            "v25c_q010",
            "ablation stage1 stage2 table 记录在哪个实验文件中？",
            "experiment",
            "medium",
            ["ablation_stage1_stage2_table.md"],
            ["experiments"],
            [],
            "Experiment table query for ablation material.",
        ),
        eval_item(
            "v25c_q011",
            "curriculum training summary 是哪份实验总结？",
            "experiment",
            "medium",
            ["curriculum_training_summary.md"],
            ["experiments"],
            [],
            "Experiment summary query.",
        ),
        eval_item(
            "v25c_q012",
            "stage13 risk report 在实验资料中对应哪个文件？",
            "experiment",
            "medium",
            ["stage13_risk_report.md"],
            ["experiments"],
            [],
            "Experiment risk report query.",
        ),
        eval_item(
            "v25c_q013",
            "stage125 OPD-HINT 后续路线讨论在哪份笔记里？",
            "note_or_report",
            "easy",
            ["stage125_opd_hint_future_route_note.md"],
            ["notes"],
            [],
            "Adapted from V2b.5 q008; OPD note query.",
        ),
        eval_item(
            "v25c_q014",
            "missile engine 相关说明在哪份 notes 文件中？",
            "note_or_report",
            "medium",
            ["missile_engine.md"],
            ["notes"],
            [],
            "Notes query for missile engine material.",
        ),
        eval_item(
            "v25c_q015",
            "parameterized shooting 的说明文件是哪一个？",
            "note_or_report",
            "medium",
            ["parameterized_shooting.md"],
            ["notes"],
            [],
            "Notes query for parameterized shooting material.",
        ),
        eval_item(
            "v25c_q016",
            "research ablation design 讨论在哪份笔记中？",
            "note_or_report",
            "medium",
            ["research_ablation_design.md"],
            ["notes"],
            [],
            "Research note query.",
        ),
        eval_item(
            "v25c_q017",
            "stage72 paper table draft 是哪份论文表格草稿？",
            "paper",
            "easy",
            ["stage72_paper_table_draft.md"],
            ["papers"],
            [],
            "Adapted from V2b.5 q016; paper table draft query.",
        ),
        eval_item(
            "v25c_q018",
            "stage15 paper mechanism positioning update 在哪里？",
            "paper",
            "medium",
            ["stage15_7c_paper_mechanism_positioning_update.md"],
            ["papers"],
            [],
            "Adapted from V2b.5 q017; paper positioning note query.",
        ),
        eval_item(
            "v25c_q019",
            "stage10 paper repositioning note 是哪份论文相关资料？",
            "paper",
            "medium",
            ["stage10_paper_repositioning_note.md"],
            ["papers"],
            [],
            "Paper repositioning note query.",
        ),
        eval_item(
            "v25c_q020",
            "stage93 paper table draft 对应哪个 paper 文件？",
            "paper",
            "medium",
            ["stage93_paper_table_draft.md"],
            ["papers"],
            [],
            "Paper table draft query.",
        ),
        eval_item(
            "v25c_q021",
            "stage133 LAG benchmark thesis route update 是哪份报告？",
            "thesis_or_reports",
            "medium",
            ["stage133_lag_benchmark_thesis_route_update.md"],
            ["thesis_or_reports"],
            [],
            "Thesis/report route update query.",
        ),
        eval_item(
            "v25c_q022",
            "stage80 option predictor report 在哪里？",
            "thesis_or_reports",
            "medium",
            ["stage80_option_predictor_report.md"],
            ["thesis_or_reports"],
            [],
            "Thesis/report query for option predictor material.",
        ),
        eval_item(
            "v25c_q023",
            "stage4c dataset audit report 是哪份报告？",
            "thesis_or_reports",
            "medium",
            ["stage4c_dataset_audit_report.md"],
            ["thesis_or_reports"],
            [],
            "Dataset audit report query.",
        ),
        eval_item(
            "v25c_q024",
            "JSBSim 奖励函数和 HierarchySelfplay 配置分别有哪些资料？",
            "mixed",
            "medium",
            ["event_driven_reward.py", "HierarchySelfplay.yaml"],
            ["code", "configs"],
            ["code", "configs"],
            "Mixed code/config query; both expected sources exist in the expanded collection.",
        ),
        eval_item(
            "v25c_q025",
            "实验 reward audit 和 offensive curriculum 配置分别对应哪些文件？",
            "mixed",
            "medium",
            ["stage11b_reward_audit_report.md", "offensive_curriculum_original_1v1.yaml"],
            ["experiments", "configs"],
            ["experiments", "configs"],
            "Mixed experiment/config query.",
        ),
        eval_item(
            "v25c_q026",
            "OPD-HINT 笔记和 LAG benchmark thesis route report 分别在哪里？",
            "mixed",
            "hard",
            ["stage125_opd_hint_future_route_note.md", "stage133_lag_benchmark_thesis_route_update.md"],
            ["notes", "thesis_or_reports"],
            ["notes", "thesis_or_reports"],
            "Mixed note/report query.",
        ),
        eval_item(
            "v25c_q027",
            "paper mechanism positioning update 和 option predictor report 分别是哪两个文件？",
            "mixed",
            "hard",
            ["stage15_7c_paper_mechanism_positioning_update.md", "stage80_option_predictor_report.md"],
            ["papers", "thesis_or_reports"],
            ["papers", "thesis_or_reports"],
            "Mixed paper/report query.",
        ),
        eval_item(
            "v25c_q028",
            "当前 80 文件知识库里有没有完整的 3v3 空战训练结果？",
            "negative",
            "medium",
            [],
            [],
            [],
            "Negative case; retrieval may return neighbors, answer stage should refuse without evidence.",
        ),
        eval_item(
            "v25c_q029",
            "当前资料中是否包含真实飞行试验的视频数据？",
            "negative",
            "medium",
            [],
            [],
            [],
            "Negative case for video / real flight test evidence.",
        ),
        eval_item(
            "v25c_q030",
            "这个 collection 是否包含可直接加载的 .pth 模型权重？",
            "negative",
            "easy",
            [],
            [],
            [],
            "Negative case for excluded model checkpoint artifacts.",
        ),
    ]


def eval_item(
    item_id: str,
    question: str,
    query_type: str,
    difficulty: str,
    expected_sources: list[str],
    expected_categories: list[str],
    required_categories: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "question": question,
        "query_type": query_type,
        "difficulty": difficulty,
        "expected_sources_contains": expected_sources,
        "expected_categories": expected_categories,
        "required_all_categories": required_categories,
        "notes": notes,
    }


def validate_eval_items(items: list[dict[str, Any]], source_blob: str) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for item in items:
        item_id = str(item["id"])
        if item_id in seen_ids:
            raise ValueError(f"Duplicate question id: {item_id}")
        seen_ids.add(item_id)
        negative = str(item.get("query_type")) == "negative"
        expected = [str(term) for term in item.get("expected_sources_contains", [])]
        if not negative and not expected:
            raise ValueError(f"Non-negative item has empty expected_sources_contains: {item_id}")
        absent = [term for term in expected if term.lower() not in source_blob]
        if absent:
            missing[item_id] = absent
    return missing


def build_manifest(
    *,
    items: list[dict[str, Any]],
    expanded_files: list[dict[str, str]],
    old_items: list[dict[str, Any]],
    source_blob: str,
    source_categories: dict[str, str],
    config_path: Path,
    output_jsonl: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    non_negative = [item for item in items if item.get("query_type") != "negative"]
    negative = [item for item in items if item.get("query_type") == "negative"]
    mixed = [item for item in items if item.get("query_type") == "mixed"]
    category_counter: Counter[str] = Counter()
    for item in non_negative:
        for category in item.get("expected_categories", []):
            category_counter[str(category)] += 1
    missing = validate_eval_items(items, source_blob)
    old_present = count_old_expected_source_present(old_items, source_blob)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "eval_set_jsonl": str(output_jsonl),
        "manifest_path": str(output_manifest),
        "question_count": len(items),
        "non_negative_questions": len(non_negative),
        "negative_questions": len(negative),
        "mixed_questions": len(mixed),
        "query_type_distribution": dict(Counter(str(item["query_type"]) for item in items)),
        "difficulty_distribution": dict(Counter(str(item["difficulty"]) for item in items)),
        "category_coverage": dict(sorted(category_counter.items())),
        "expanded_source_count": len({row.get("source", "") for row in expanded_files}),
        "expanded_category_counts": dict(Counter(source_categories.values())),
        "old_eval_question_count": len(old_items),
        "old_eval_expected_source_present_count": old_present,
        "adapted_from_old_eval_count": sum("Adapted from V2b.5" in str(item.get("notes", "")) for item in items),
        "expected_source_absent_count": sum(len(values) for values in missing.values()),
        "expected_source_absent_by_question": missing,
        "expected_source_policy": "Every non-negative expected_sources_contains term must appear in the expanded 80-file source list.",
        "negative_policy": "Negative questions intentionally have no expected source and are excluded from source Hit@K metrics.",
        "outputs": {
            "eval_set_jsonl": str(output_jsonl),
            "eval_manifest_json": str(output_manifest),
        },
    }


def count_old_expected_source_present(old_items: list[dict[str, Any]], source_blob: str) -> int:
    count = 0
    for item in old_items:
        terms = [str(term).lower() for term in item.get("expected_sources_contains", [])]
        if terms and any(term in source_blob for term in terms):
            count += 1
    return count


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))
    return items


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


if __name__ == "__main__":
    sys.exit(main())
