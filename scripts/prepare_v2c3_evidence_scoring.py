from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KEY_FILES = {
    "retrieval_eval": PROJECT_ROOT / "storage" / "logs" / "retrieval_eval_v2b5.json",
    "answer_eval": PROJECT_ROOT / "storage" / "logs" / "answer_eval_v2c1.json",
    "answer_review_summary": PROJECT_ROOT / "storage" / "logs" / "answer_review_v2c2_summary.json",
    "answer_review_template": PROJECT_ROOT / "data" / "eval" / "answer_review_v2c2_template.csv",
    "real_smoke_retriever": PROJECT_ROOT / "src" / "retrieval" / "real_smoke_retriever.py",
    "keyword_retriever": PROJECT_ROOT / "src" / "retrieval" / "keyword_retriever.py",
    "evidence_gate": PROJECT_ROOT / "src" / "generation" / "evidence_gate.py",
    "citation_checker": PROJECT_ROOT / "src" / "generation" / "citation_checker.py",
}

OUTPUT_INVENTORY_JSON = PROJECT_ROOT / "data" / "eval" / "evidence_signal_inventory_v2c3_prep.json"
OUTPUT_MILESTONE_MD = PROJECT_ROOT / "docs" / "v2_milestone_summary.md"
OUTPUT_INVENTORY_MD = PROJECT_ROOT / "docs" / "evidence_signal_inventory_v2c3_prep.md"
OUTPUT_DESIGN_MD = PROJECT_ROOT / "docs" / "design_v2c3_evidence_quality_scoring.md"
OUTPUT_REPORT_MD = PROJECT_ROOT / "docs" / "dev_report_v2c3_prep.md"


def main() -> int:
    configure_stdout()
    try:
        result = prepare_v2c3_outputs()
    except Exception as exc:  # noqa: BLE001 - prep should fail explicitly.
        print("V2c.3-prep generation failed.")
        print(f"reason: {exc}")
        return 1
    print_summary(result)
    return 0


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def prepare_v2c3_outputs() -> dict[str, Any]:
    file_checks = check_key_files()
    retrieval_eval = load_json(KEY_FILES["retrieval_eval"])
    answer_eval = load_json(KEY_FILES["answer_eval"])
    review_summary = load_json(KEY_FILES["answer_review_summary"])
    metrics = extract_metrics(retrieval_eval, answer_eval, review_summary)
    signals = build_signal_inventory()
    inventory = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "scope": "V2c.3-prep only; no final evidence quality scoring implementation.",
        "file_checks": file_checks,
        "summary_metrics": metrics,
        "signals": signals,
        "signal_counts": signal_counts(signals),
    }

    OUTPUT_INVENTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MILESTONE_MD.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_INVENTORY_JSON.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MILESTONE_MD.write_text(build_milestone_summary(metrics), encoding="utf-8")
    OUTPUT_INVENTORY_MD.write_text(build_signal_inventory_md(inventory), encoding="utf-8")
    OUTPUT_DESIGN_MD.write_text(build_design_doc(metrics), encoding="utf-8")
    OUTPUT_REPORT_MD.write_text(build_dev_report(inventory), encoding="utf-8")
    return {
        "outputs": [
            str(OUTPUT_MILESTONE_MD),
            str(OUTPUT_INVENTORY_MD),
            str(OUTPUT_DESIGN_MD),
            str(OUTPUT_REPORT_MD),
            str(OUTPUT_INVENTORY_JSON),
        ],
        "signal_counts": inventory["signal_counts"],
        "summary_metrics": metrics,
        "file_checks": file_checks,
    }


def check_key_files() -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, path in KEY_FILES.items():
        exists = path.exists()
        checks[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
        }
        if not exists:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing required V2c.3-prep input files: {missing}")
    return checks


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def extract_metrics(
    retrieval_eval: dict[str, Any],
    answer_eval: dict[str, Any],
    review_summary: dict[str, Any],
) -> dict[str, Any]:
    retrieval_metrics = retrieval_eval.get("metrics_by_mode", {}).get("enhanced_keyword", {})
    answer_metrics = answer_eval.get("metrics", {})
    return {
        "retrieval_eval_total_questions": retrieval_eval.get("total_questions"),
        "retrieval_eval_non_negative_questions": retrieval_eval.get("non_negative_questions"),
        "retrieval_eval_negative_questions": retrieval_eval.get("negative_questions"),
        "enhanced_keyword_hit_at_5": retrieval_metrics.get("hit_at_5"),
        "enhanced_keyword_mrr": retrieval_metrics.get("mrr"),
        "enhanced_keyword_category_hit_rate": retrieval_metrics.get("category_hit_rate"),
        "enhanced_keyword_required_all_categories_hit_rate": retrieval_metrics.get("required_all_categories_hit_rate"),
        "enhanced_keyword_source_diversity_pass_rate": retrieval_metrics.get("source_diversity_pass_rate"),
        "answer_eval_total_questions": answer_metrics.get("total_eval_questions"),
        "negative_refusal_rate": answer_metrics.get("negative_refusal_rate"),
        "citation_check_pass_rate": answer_metrics.get("citation_check_pass_rate"),
        "avg_supported_claim_ratio": answer_metrics.get("avg_supported_claim_ratio"),
        "unsupported_claim_count": answer_metrics.get("unsupported_claim_count"),
        "weak_claim_count": answer_metrics.get("weak_claim_count"),
        "ignored_claim_count": answer_metrics.get("ignored_claim_count"),
        "checked_claim_count": answer_metrics.get("checked_claim_count"),
        "needs_human_review_count": answer_metrics.get("needs_human_review_count"),
        "review_template_total_questions": review_summary.get("total_questions"),
        "review_template_needs_human_review_ids": review_summary.get("needs_human_review_ids", []),
    }


def build_signal_inventory() -> list[dict[str, Any]]:
    signal = make_signal
    return [
        signal("dense_rank", "src/retrieval/real_smoke_retriever.py", True, "Dense 向量检索返回顺序。", "能反映语义近邻排序，便于定位 Top-K 证据。", "不同查询/模型距离不可直接横向比较，中文到英文代码名可能不稳。", True, "positive", "retrieval"),
        signal("dense_score", "src/retrieval/real_smoke_retriever.py", True, "Chroma similarity_search_with_score 返回的距离或分数。", "来自向量库原始检索结果，可辅助判断 dense 证据强弱。", "当前更像距离值，数值方向需标准化；不宜直接跨问题比较。", True, "diagnostic only", "retrieval"),
        signal("metadata_score", "src/retrieval/metadata_retriever.py", True, "metadata/file/path/title/section 命中的加权分数。", "对文件名、路径型问题很有效，能修补 dense 漏召回。", "规则匹配可能偏向路径关键词，不能代表正文语义支持。", True, "positive", "retrieval"),
        signal("metadata_match_reason", "src/retrieval/metadata_retriever.py", True, "metadata 命中的字段与关键词原因。", "解释性强，适合作为 evidence_quality_reasons。", "文本较长且规则化，只适合解释，不宜直接打分。", False, "diagnostic only", "retrieval"),
        signal("keyword_score", "src/retrieval/keyword_retriever.py", True, "BM25-lite / 词频字段权重计算出的关键词召回分。", "能捕获类名、文件名、配置项等精确词。", "没有复杂中文分词；泛词可能抬高无关文档。", True, "positive", "retrieval"),
        signal("matched_terms", "src/retrieval/keyword_retriever.py", True, "关键词检索命中的查询词列表。", "可解释具体为何命中，也可估计 claim/source 词面覆盖。", "命中词多不等于证据充分，可能包含泛词。", True, "positive", "retrieval"),
        signal("matched_fields", "src/retrieval/keyword_retriever.py", True, "关键词命中的字段，如 content、filename、path、title_section。", "正文命中通常比路径命中更可靠，可做字段加权。", "字段粒度粗，content 命中也不代表完整事实支持。", True, "positive", "retrieval"),
        signal("retrieval_channels", "src/retrieval/real_smoke_retriever.py", True, "证据来自 dense / metadata / keyword 的通道集合。", "多通道一致命中通常更可信。", "通道数量不等于相关性；metadata/keyword 可能同源同偏。", True, "positive", "retrieval"),
        signal("category_match", "src/retrieval/real_smoke_retriever.py", True, "检索结果类别是否匹配 query classifier target_categories。", "对 mixed/config/experiment/code 查询的覆盖很关键。", "query classifier 是规则版，分类错误会传导。", True, "positive", "retrieval"),
        signal("source_diversity", "src/retrieval/real_smoke_retriever.py", True, "同一 imported_source 在 final_sources 中是否超过上限。", "防止单文件多 chunk 占满 Top-K，提升证据覆盖。", "过强 diversity 可能压低同一长文件的关键连续证据。", True, "positive", "retrieval"),
        signal("source_rank_within_file", "src/retrieval/real_smoke_retriever.py", True, "同一来源文件内第几个 chunk 被保留。", "可检测单一文件占比，辅助 source diversity。", "单独不能代表质量。", False, "diagnostic only", "retrieval"),
        signal("final_rank", "src/retrieval/real_smoke_retriever.py", True, "融合检索后 final_sources 排名。", "最终面向回答的证据排序，适合给 Top ranks 加权。", "融合公式是启发式，仍需 eval 校准。", True, "positive", "retrieval"),
        signal("evidence_sufficient", "src/generation/evidence_gate.py", True, "回答前证据门的通过/拒答布尔值。", "能把负例转为拒答，防止近邻硬答。", "依赖 eval label 或弱关键词，交互式无标签时较保守。", True, "positive", "evidence_gate"),
        signal("evidence_gate_reason", "src/generation/evidence_gate.py", True, "证据门给出的拒答或通过原因。", "对审计和报告很有用。", "自然语言 reason 不适合直接数值化。", False, "diagnostic only", "evidence_gate"),
        signal("expected_source_hit", "src/generation/evidence_gate.py / scripts/run_retrieval_eval.py", True, "eval expected_sources_contains 是否被 final_sources 命中。", "在评测集中是强监督信号。", "只在有标注的 eval item 中可用，线上查询没有。", True, "positive", "evidence_gate"),
        signal("expected_category_hit", "src/generation/evidence_gate.py / scripts/run_retrieval_eval.py", True, "expected_categories 是否被 final_sources 覆盖。", "能衡量查询意图类别覆盖。", "依赖人工标注；类别粒度较粗。", True, "positive", "evidence_gate"),
        signal("required_all_categories_hit", "scripts/run_retrieval_eval.py", True, "多意图查询要求的类别是否全部出现。", "对 mixed 查询很关键，如 experiments + configs。", "只适用于有 required_all_categories 的评测项。", True, "positive", "evidence_gate"),
        signal("negative_query_detection", "src/generation/evidence_gate.py", True, "query_type=negative 或弱证据关键词触发。", "支持 insufficient evidence 策略。", "弱关键词可能误杀真实可回答问题。", True, "negative", "evidence_gate"),
        signal("weak_evidence_keyword_hit", "src/generation/evidence_gate.py", True, "有没有、完整、3v3、视频数据等弱证据风险词命中。", "适合触发保守拒答或风险提示。", "只看问题表面词，不理解真实证据内容。", True, "negative", "evidence_gate"),
        signal("supported_claim_ratio", "src/generation/citation_checker.py", True, "被规则 checker 支持的 claim 占比。", "直接反映回答与检索证据的一致性。", "关键词规则会误判复杂语义。", True, "positive", "citation"),
        signal("checked_claim_count", "src/generation/citation_checker.py", True, "参与 citation scoring 的 claim 数量。", "防止单 claim 答案和多 claim 答案同等解释。", "多不代表好，主要用于置信度校准。", True, "diagnostic only", "citation"),
        signal("ignored_claim_count", "src/generation/citation_checker.py", True, "heading/list_intro/insufficient 等被忽略 claim 数。", "可降低标题误判，辅助审计答案结构。", "过多 ignored 可能掩盖回答实体不足。", False, "diagnostic only", "citation"),
        signal("weak_claim_count", "src/generation/citation_checker.py", True, "弱支持 claim 数，尤其 negative_statement。", "提示需要人工审阅但不直接判死刑。", "弱支持定义保守，需人工校准。", True, "negative", "citation"),
        signal("unsupported_claim_count", "src/generation/citation_checker.py", True, "未被来源支持的 claim 数。", "回答忠实度的核心风险信号。", "规则误判仍存在，需要人工审阅闭环。", True, "negative", "citation"),
        signal("claim_type_counts", "src/generation/citation_checker.py", True, "factual/heading/list_intro/negative_statement 等数量分布。", "可区分事实句和结构性文本。", "分类规则启发式，不宜直接过度加权。", False, "diagnostic only", "citation"),
        signal("unsupported_claims", "src/generation/citation_checker.py", True, "具体 unsupported claim 列表。", "可用于人工审阅和错误分析。", "内容级列表不适合直接进入通用数值公式。", False, "diagnostic only", "citation"),
        signal("weak_claims", "src/generation/citation_checker.py", True, "具体 weak claim 列表。", "适合人工复核和 warning 输出。", "不等价于错误。", False, "diagnostic only", "citation"),
        signal("citation_check_passed", "src/generation/citation_checker.py", True, "supported_claim_ratio 是否达到阈值。", "简单可解释，适合 level gating。", "二值阈值损失细节。", True, "positive", "citation"),
        signal("needs_human_review", "scripts/run_answer_eval.py / scripts/export_answer_review_template.py", True, "answer eval 标记是否需要人工审阅。", "聚合 unsupported/weak/low ratio 的风险提示。", "不是自动质量分，只是审阅队列标记。", True, "negative", "answer_review"),
        signal("manual_judgment", "data/eval/answer_review_v2c2_template.csv", True, "人工对答案整体正确性的标注。", "最可信的后验质量信号，可用于校准规则分。", "当前默认空，需要人工填写。", True, "positive", "answer_review"),
        signal("manual_citation_judgment", "data/eval/answer_review_v2c2_template.csv", True, "人工对引用支持程度的标注。", "可纠正规则 checker false positive/negative。", "当前默认空，需要审阅流程。", True, "positive", "answer_review"),
        signal("manual_should_refuse", "data/eval/answer_review_v2c2_template.csv", True, "人工判断该问题是否应拒答。", "可校准 evidence gate 与 insufficient 策略。", "当前默认空。", True, "negative", "answer_review"),
        signal("manual_notes", "data/eval/answer_review_v2c2_template.csv", True, "人工审阅备注。", "保留错误类型和改进建议。", "自由文本，不直接入分。", False, "diagnostic only", "answer_review"),
    ]


def make_signal(
    name: str,
    source_module: str,
    available_in_outputs: bool,
    meaning: str,
    strength: str,
    limitation: str,
    use_in_v2c3: bool,
    weight_direction: str,
    group: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "group": group,
        "source_module": source_module,
        "available_in_outputs": available_in_outputs,
        "meaning": meaning,
        "strength": strength,
        "limitation": limitation,
        "use_in_v2c3": use_in_v2c3,
        "weight_direction": weight_direction,
    }


def signal_counts(signals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(signals),
        "use_in_v2c3": sum(1 for signal in signals if signal["use_in_v2c3"]),
        "diagnostic_only": sum(1 for signal in signals if not signal["use_in_v2c3"]),
        "by_group": count_by(signals, "group"),
        "by_weight_direction": count_by(signals, "weight_direction"),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_milestone_summary(metrics: dict[str, Any]) -> str:
    return f"""# V2 Milestone Summary

## 项目目标

Domain-RAG Agent 以多无人机空战资料为案例，构建一个可复现、可审计、可评测的垂直领域 RAG + Agent 工程项目。当前重点不是论文写作助手，而是展示真实资料治理、检索增强、引用约束回答、评测与人工审阅闭环。

## V1 到 V2c.2 阶段路线

| 阶段 | 解决的问题 | 主要产物 |
| --- | --- | --- |
| V1 demo RAG | 打通多格式 demo 文档读取、chunk、embedding、Chroma、dense retrieval、回答生成 | `scripts/ingest_documents.py`, `scripts/run_rag.py` |
| V2a | 真实外部资料只读扫描、候选筛选、平衡计划、安全复制 | external manifest, curated import plan, 207 个真实导入文件 |
| V2b.0 | loader/chunk/metadata 入库前审计 | imported corpus audit CSV/JSON |
| V2b.1 | 小规模真实资料 Chroma smoke 入库 | 30 文件 / 130 chunks, `domain_rag_real_smoke_v2b1` |
| V2b.2 | 修补文件名/路径/多意图查询召回 | query classifier, query expansion, metadata retriever |
| V2b.3 | source diversity + retrieval regression | regression JSON/JSONL, source cap |
| V2b.4 | keyword-lite 检索增强 | keyword / enhanced_keyword modes |
| V2b.5 | 小型 retrieval evaluation | 20 题 retrieval eval, Hit@K/MRR/category/source diversity 指标 |
| V2c.0 | evidence gate + citation checker | 负例拒答、规则版引用支持检查 |
| V2c.1 | citation checker 误判修正 | claim_type, weak/ignored claims, v2c1 answer eval |
| V2c.2 | human review template | Markdown 审阅卡片、人工标注 CSV 模板 |

## 当前系统能力

- 真实资料治理：从 `E:\\Lenginzed` 只读 inventory、清洗、导入 207 个候选资料。
- 小规模真实资料入库：30 个真实文件、130 个 chunks，独立 Chroma collection。
- 检索模式：dense、enhanced、keyword、enhanced_keyword。
- 检索可靠性：source diversity、regression tests、retrieval eval。
- 回答安全：evidence gate 能让负例输出“当前知识库没有足够依据”。
- 引用审计：citation checker 输出 claim_type、supported/unsupported/weak/ignored claims。
- 人工审阅：CSV/Markdown 模板可用于人工校准。

## 当前关键指标

- enhanced_keyword Hit@5：{metrics.get("enhanced_keyword_hit_at_5")}
- enhanced_keyword MRR：{metrics.get("enhanced_keyword_mrr")}
- source diversity pass rate：{metrics.get("enhanced_keyword_source_diversity_pass_rate")}
- negative_refusal_rate：{metrics.get("negative_refusal_rate")}
- avg_supported_claim_ratio：{metrics.get("avg_supported_claim_ratio")}
- unsupported_claim_count：{metrics.get("unsupported_claim_count")}
- needs_human_review_count：{metrics.get("needs_human_review_count")}

## 当前局限

- 小规模真实入库仅覆盖 30 文件 / 130 chunks，尚非全量知识库。
- CSV 和 PDF 的 loader/chunk 策略仍需按 V2b.0 审计结果优化。
- keyword-lite 不等于正式 BM25；没有 rerank。
- citation checker 是规则版，不做语义蕴含。
- 人工审阅字段尚未回写到统一评价闭环。

## 可写入 README / 简历的项目亮点

- 完成真实资料治理与安全导入，避免直接对大目录盲目入库。
- 实现 metadata-aware + keyword-lite + dense 的多通道检索，并通过 regression/evaluation 量化提升。
- 构建 answer-stage evidence gate，能对负例进行明确拒答。
- 实现 citation-grounded answer verification，并把自动检查结果转成可人工审阅模板。
- 项目强调可复现脚本、机器可读日志、阶段报告和验收标准。

## 下一阶段路线

V2c.3 建议实现 Evidence Quality Scoring：把 retrieval、evidence gate、citation checker、manual review 信号合成为可解释质量分，不做复杂 rerank 或 LLM judge。
"""


def build_signal_inventory_md(inventory: dict[str, Any]) -> str:
    signals = inventory["signals"]
    lines = [
        "# Evidence Signal Inventory V2c.3-prep",
        "",
        "## 总览",
        "",
        f"- signals total: {inventory['signal_counts']['total']}",
        f"- use_in_v2c3: {inventory['signal_counts']['use_in_v2c3']}",
        f"- diagnostic_only: {inventory['signal_counts']['diagnostic_only']}",
        f"- by_group: `{inventory['signal_counts']['by_group']}`",
        "",
        "## 信号清单",
        "",
        "| signal name | group | source module | output? | meaning | strength | limitation | V2c.3 | weight |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in signals:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["name"],
                    item["group"],
                    item["source_module"],
                    "yes" if item["available_in_outputs"] else "no",
                    escape_table(item["meaning"]),
                    escape_table(item["strength"]),
                    escape_table(item["limitation"]),
                    "yes" if item["use_in_v2c3"] else "no",
                    item["weight_direction"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 建议进入 V2c.3 score 的信号",
            "",
        ]
    )
    for item in [signal for signal in signals if signal["use_in_v2c3"]]:
        lines.append(f"- `{item['name']}` ({item['weight_direction']}): {item['meaning']}")
    lines.extend(["", "## Diagnostic-only 信号", ""])
    for item in [signal for signal in signals if not signal["use_in_v2c3"]]:
        lines.append(f"- `{item['name']}`: {item['limitation']}")
    return "\n".join(lines) + "\n"


def escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build_design_doc(metrics: dict[str, Any]) -> str:
    return f"""# Design V2c.3 Evidence Quality Scoring

## 1. V2c.3 目标

V2c.3 的目标是实现一个轻量、可审计、规则加权版 evidence quality score，用于判断一次 RAG 回答的证据质量，而不是替代 retrieval eval、answer eval 或人工审阅。

## 2. 为什么需要 Evidence Quality Scoring

当前系统已经能检索、回答、拒答、检查引用，并导出人工审阅模板。但这些信号分散在 retrieval、evidence gate、citation checker 和 review template 中。Evidence quality scoring 可以：

- 给调用方一个统一质量分和等级。
- 解释回答是否可靠、是否需要人工审阅。
- 将负例拒答、引用支持率、检索覆盖和 source diversity 放到同一报告里。
- 为后续 V2c.4 的人工校准或更细评测打基础。

## 3. 输入数据来源

- `retrieve_real_smoke(...).debug.final_sources`
- `storage/logs/retrieval_eval_v2b5.json`
- `storage/logs/answer_eval_v2c1.json`
- `storage/logs/answer_review_v2c2_summary.json`
- `data/eval/answer_review_v2c2_template.csv`
- citation checker 输出：supported/unsupported/weak/ignored claims
- evidence gate 输出：evidence_sufficient、reason、signals

## 4. 输出字段设计

```json
{{
  "evidence_quality_score": 0.0,
  "evidence_quality_level": "high | medium | low | insufficient",
  "retrieval_confidence_score": 0.0,
  "citation_support_score": 0.0,
  "coverage_score": 0.0,
  "source_diversity_score": 0.0,
  "risk_penalty": 0.0,
  "evidence_quality_reasons": [],
  "evidence_quality_warnings": []
}}
```

## 5. 建议评分公式

V2c.3 可先采用规则加权版：

```text
evidence_quality_score =
  0.30 * retrieval_confidence_score
+ 0.30 * citation_support_score
+ 0.20 * coverage_score
+ 0.10 * source_diversity_score
- 0.10 * risk_penalty
```

分数建议裁剪到 `[0.0, 1.0]`。

## 6. 子分数计算建议

### retrieval_confidence_score

建议使用：

- final Top-K 中是否多通道命中：`retrieval_channels`
- Top ranks 中 `metadata_score` / `keyword_score` 是否有明显命中
- `matched_terms` / `matched_fields`，其中 content/filename 命中高于 path-only 命中
- final rank，前 3 个来源权重更高

初版可设计：

```text
retrieval_confidence_score = clamp(
  0.25 * has_dense_channel
+ 0.25 * has_keyword_channel
+ 0.20 * has_metadata_channel
+ 0.20 * top_sources_have_matched_terms
+ 0.10 * top_sources_include_content_or_filename_match
)
```

### citation_support_score

建议使用：

- `supported_claim_ratio`
- `citation_check_passed`
- `unsupported_claim_count`
- `weak_claim_count`

初版：

```text
citation_support_score = clamp(
  supported_claim_ratio
- 0.08 * unsupported_claim_count
- 0.04 * weak_claim_count
)
```

### coverage_score

建议使用：

- eval 场景：expected source/category/required_all_categories hit
- 线上场景：query classifier target_categories 是否被 final_sources 覆盖
- mixed query 必须奖励多类别覆盖

初版：

```text
coverage_score =
  1.0 if required_all_categories_hit
  0.75 if any expected category hit
  0.5 if only general category match
  0.0 if no relevant category signal
```

### source_diversity_score

建议使用：

- `source_diversity.violations` 是否为空
- `source_rank_within_file` 是否超过 cap

初版：

```text
source_diversity_score = 1.0 if no violations else 0.5
```

当前 enhanced_keyword source diversity pass rate：{metrics.get("enhanced_keyword_source_diversity_pass_rate")}。

### risk_penalty

建议使用：

- evidence_sufficient=false
- negative query detection
- weak evidence keyword hit
- unsupported_claim_count
- weak_claim_count
- needs_human_review

初版：

```text
risk_penalty =
  1.0 if evidence_sufficient is false
  else clamp(0.25 * needs_human_review + 0.10 * unsupported_claim_count + 0.05 * weak_claim_count)
```

## 7. 负例 / Insufficient Evidence 处理

- 如果 `evidence_sufficient=false` 且回答是 insufficient evidence 模板：
  - `evidence_quality_level = insufficient`
  - `evidence_quality_score` 不表示答案错误，而表示“不足以支持实质回答”
  - `citation_support_score = 1.0` 可保留，因为拒答模板本身无需普通 claim support
- 如果负例没有拒答，应给高 risk_penalty。

## 8. Needs Human Review 处理

`needs_human_review=True` 不应直接判定答案错误，但应：

- 增加 `risk_penalty`
- 在 `evidence_quality_warnings` 中输出原因
- 保留 unsupported/weak claim 摘要

## 9. 与 Manual Review 结合

人工审阅字段可作为后验校准：

- `manual_judgment=correct` 可用于确认规则分。
- `checker_false_positive` 可用于调整 citation checker。
- `checker_false_negative` 可用于提高风险惩罚。
- `manual_should_refuse=true` 可反向校准 evidence gate。

V2c.3 最小实现只读取人工字段，不要求已有人工标注。

## 10. V2c.3 最小实现范围

建议只实现：

- `src/evaluation/evidence_quality.py` 或 `src/generation/evidence_quality.py`
- 输入一个 answer eval record 或 RAG result
- 输出 evidence quality 字段
- 脚本 `scripts/run_evidence_quality_eval.py`
- 输出 `storage/logs/evidence_quality_eval_v2c3.json/jsonl` 和 CSV
- pytest 检查字段存在、分数范围、负例 level=insufficient

## 11. V2c.3 不做什么

- 不实现 Agent / MCP。
- 不实现 rerank。
- 不扩大语料或重建 Chroma。
- 不引入 LLM judge / NLI。
- 不把分数当作真实性最终裁决。
- 不修改 retrieval evaluation 指标。

## 12. 验收标准草案

- 使用 `Lenginzed_RAG` 环境。
- 不调用 LLM。
- 不扩大入库规模。
- 能读取 V2c.1 answer eval 输出。
- 每条记录生成 `evidence_quality_score` 和 level。
- 分数在 `[0.0, 1.0]`。
- 2 个 negative insufficient answer 输出 `level=insufficient`。
- `needs_human_review=True` 的题目带 warning。
- pytest 通过。

## 13. 后续 V2c.4 可能方向

- 使用人工审阅 CSV 校准权重。
- 引入更细的 claim-level evidence mapping。
- 对 source chunk 做 evidence span 标注。
- 在不引入复杂模型的前提下改进 negative statement 检查。
"""


def build_dev_report(inventory: dict[str, Any]) -> str:
    metrics = inventory["summary_metrics"]
    signals = inventory["signals"]
    use_signals = [item["name"] for item in signals if item["use_in_v2c3"]]
    diagnostic_signals = [item["name"] for item in signals if not item["use_in_v2c3"]]
    file_check_lines = "\n".join(
        f"- `{name}`: exists={check['exists']}, size_bytes={check['size_bytes']}, path={check['path']}"
        for name, check in inventory["file_checks"].items()
    )
    return f"""# Dev Report V2c.3-prep

## 1. 本轮目标

为 V2c.3 Evidence Quality Scoring 做准备：总结 V2 里程碑、盘点现有证据质量信号、设计评分字段/公式/输出格式，并明确最小实现边界。本轮不实现最终 scoring 模块。

## 2. 环境与范围

- 使用环境：`conda run -n Lenginzed_RAG python ...`
- 是否扩大入库规模：没有。
- 是否修改 `E:\\Lenginzed`：没有。
- 是否修改 `data/raw_imported/`：没有。
- 是否调用 LLM：没有。
- 是否重建 Chroma：没有。
- 是否实现 Agent / MCP / rerank：没有。
- 是否实现最终 evidence quality scoring 模块：没有。

## 3. 新增文件列表

- `scripts/prepare_v2c3_evidence_scoring.py`
- `docs/v2_milestone_summary.md`
- `docs/evidence_signal_inventory_v2c3_prep.md`
- `docs/design_v2c3_evidence_quality_scoring.md`
- `docs/dev_report_v2c3_prep.md`
- `data/eval/evidence_signal_inventory_v2c3_prep.json`
- `tests/test_v2c3_prep_outputs.py`

## 4. 实际检查的关键文件

{file_check_lines}

## 5. 已盘点 Evidence Signals

- signals total: {inventory['signal_counts']['total']}
- use_in_v2c3: {inventory['signal_counts']['use_in_v2c3']}
- diagnostic_only: {inventory['signal_counts']['diagnostic_only']}
- by_group: `{inventory['signal_counts']['by_group']}`
- by_weight_direction: `{inventory['signal_counts']['by_weight_direction']}`

## 6. 建议进入 V2c.3 Score 的信号

{format_name_list(use_signals)}

## 7. Diagnostic-only 信号

{format_name_list(diagnostic_signals)}

## 8. 当前关键指标

- enhanced_keyword Hit@5: {metrics.get('enhanced_keyword_hit_at_5')}
- enhanced_keyword MRR: {metrics.get('enhanced_keyword_mrr')}
- source diversity pass rate: {metrics.get('enhanced_keyword_source_diversity_pass_rate')}
- negative_refusal_rate: {metrics.get('negative_refusal_rate')}
- avg_supported_claim_ratio: {metrics.get('avg_supported_claim_ratio')}
- unsupported_claim_count: {metrics.get('unsupported_claim_count')}
- weak_claim_count: {metrics.get('weak_claim_count')}
- needs_human_review_count: {metrics.get('needs_human_review_count')}

## 9. V2c.3 评分设计摘要

建议输出：

```json
{{
  "evidence_quality_score": 0.0,
  "evidence_quality_level": "high | medium | low | insufficient",
  "retrieval_confidence_score": 0.0,
  "citation_support_score": 0.0,
  "coverage_score": 0.0,
  "source_diversity_score": 0.0,
  "risk_penalty": 0.0,
  "evidence_quality_reasons": [],
  "evidence_quality_warnings": []
}}
```

建议公式：

```text
score =
  0.30 * retrieval_confidence_score
+ 0.30 * citation_support_score
+ 0.20 * coverage_score
+ 0.10 * source_diversity_score
- 0.10 * risk_penalty
```

## 10. V2c.3 最小实现边界

- 只读取已有 answer eval / retrieval debug / citation checker 输出。
- 不调用 LLM。
- 不重建 Chroma。
- 不改检索逻辑。
- 生成 evidence quality JSON/JSONL/CSV。
- pytest 覆盖分数范围、负例 insufficient、needs_human_review warning。

## 11. 当前风险

- dense_score 数值方向需要标准化，不能直接当相似度使用。
- citation checker 仍是关键词规则，存在 false positive / false negative。
- manual review 字段目前为空，无法真正校准权重。
- mixed query 的 coverage score 需要小心处理，否则容易被单类证据误判为高质量。

## 12. 是否建议进入正式 V2c.3

建议进入。当前信号足够支撑一个可审计的规则加权版 evidence quality scoring，但应明确它是工程质量提示，不是真实性最终裁决。

## 13. 下一步建议

1. 实现最小 `evidence_quality_score` 模块。
2. 对 V2c.1 的 10 题 answer eval 生成 evidence quality 结果。
3. 检查 q003/q005/q006/q007 是否因为 human review 风险被降级。
4. 保留人工审阅接口，为后续权重校准做准备。
"""


def format_name_list(names: list[str]) -> str:
    return "\n".join(f"- `{name}`" for name in names)


def print_summary(result: dict[str, Any]) -> None:
    print("V2c.3-prep outputs generated.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
