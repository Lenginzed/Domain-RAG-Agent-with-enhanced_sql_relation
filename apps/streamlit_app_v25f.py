from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.answer_eval_report_service import (  # noqa: E402
    extract_case_options,
    extract_failure_options,
    extract_mode_options,
    extract_quality_options,
    extract_status_options,
    filter_answer_eval_records,
    format_run_detail,
    load_answer_eval_compare,
    load_answer_eval_records,
    load_answer_eval_report,
)
from src.ui.answer_eval_adjudication_service import (  # noqa: E402
    build_calibration_notes as build_adjudication_calibration_notes,
    enrich_adjudication_tasks,
    load_adjudication_config,
    load_review_tasks,
    merge_manual_review,
    summarize_adjudication,
    validate_manual_fields,
)
from src.ui.answer_eval_triage_service import (  # noqa: E402
    build_review_tasks,
    load_triage_config,
    render_tasks_markdown,
    summarize_review_tasks,
)
from src.ui.rag_inspection_service import (  # noqa: E402
    append_manual_review_export,
    get_collection_info,
    load_demo_cases,
    load_interaction_logs,
    load_manual_review_exports,
    load_ui_config,
    run_full_rag_inspection,
    run_retrieval_inspection,
    summarize_interaction_record,
)
from src.agent.trace import summarize_node_trace  # noqa: E402
from src.agent.workflow import (  # noqa: E402
    DEFAULT_V4E_CONFIG_PATH,
    load_graph_trace,
    load_recent_graph_runs,
    run_rag_graph_v3b,
)


CONFIG_PATH = PROJECT_ROOT / "config" / "ui_v4h.yaml"
GRAPH_CONFIG_PATH = DEFAULT_V4E_CONFIG_PATH
DEFAULT_QUESTION = "missile engine related notes file?"


def main() -> None:
    st.set_page_config(page_title="Domain-RAG Evidence Inspector", layout="wide")
    config = load_ui_config(CONFIG_PATH)
    apply_pending_question()

    st.title(str(config.get("ui", {}).get("title", "Domain-RAG Agent Local Evidence Inspector")))
    st.caption(
        "Local demo UI for retrieval evidence, citation checks, evidence quality, calibration reasons, "
        "and manual review export. Retrieval-only mode does not call an LLM; full RAG mode calls local Ollama only after a click."
    )

    controls = render_sidebar(config)
    ask_tab, graph_tab, answer_eval_tab, history_tab, review_tab = st.tabs(
        ["Ask / Demo Cases", "LangGraph Trace", "Answer Eval Reports", "Recent Interactions", "Manual Review Exports"]
    )

    with ask_tab:
        render_ask_demo_tab(config, controls)
    with graph_tab:
        render_langgraph_trace_tab(controls)
    with answer_eval_tab:
        render_answer_eval_reports_tab(config)
    with history_tab:
        render_interaction_log_browser(config)
    with review_tab:
        render_manual_review_exports_browser(config)


def apply_pending_question() -> None:
    if "v25f_question" not in st.session_state:
        st.session_state["v25f_question"] = DEFAULT_QUESTION
    pending = st.session_state.pop("v25f_pending_question", None)
    if pending:
        st.session_state["v25f_question"] = str(pending)


def render_ask_demo_tab(config: dict[str, Any], controls: dict[str, Any]) -> None:
    render_demo_questions(config, controls)
    st.divider()
    st.subheader("Question")
    question = st.text_area(
        "Question",
        key="v25f_question",
        height=90,
        placeholder="Enter a domain question. Run retrieval-only first; generate an answer only when you want to call the local LLM.",
    )
    left, right = st.columns([1, 1])
    run_retrieval = left.button("Run Retrieval", type="primary", use_container_width=True)
    run_answer = right.button("Generate Answer (local LLM)", use_container_width=True)

    if run_retrieval and question.strip():
        run_retrieval_for_question(question.strip(), controls)
    if run_answer and question.strip():
        with st.spinner("Running retrieval and local Ollama answer generation..."):
            st.session_state["v25f_result"] = run_full_rag_inspection(
                question=question.strip(),
                retrieval_mode=controls["retrieval_mode"],
                enable_calibration=controls["enable_calibration"],
                final_top_k=controls["final_top_k"],
                config_path=CONFIG_PATH,
            )

    result = st.session_state.get("v25f_result")
    if result:
        render_result(result, config, controls)
    else:
        st.info("Run retrieval to inspect evidence. Generate Answer calls the local Ollama LLM only after you click the button.")


def render_sidebar(config: dict[str, Any]) -> dict[str, Any]:
    st.sidebar.header("Collection")
    collection = get_collection_info(config)
    st.sidebar.code(collection["collection_name"])
    st.sidebar.caption(collection["persist_directory"])
    if collection.get("chunk_count") is not None:
        st.sidebar.metric("Chunks", collection["chunk_count"])
    if collection.get("error"):
        st.sidebar.warning(collection["error"])

    st.sidebar.header("Retrieval")
    default_retrieval = config.get("default_retrieval", {})
    modes = list(config.get("retrieval_modes", ["dense", "enhanced", "keyword", "enhanced_keyword"]))
    default_mode = str(config.get("default_retrieval_mode") or default_retrieval.get("mode", "enhanced_keyword"))
    retrieval_mode = st.sidebar.selectbox(
        "Retrieval mode",
        modes,
        index=modes.index(default_mode) if default_mode in modes else modes.index("enhanced_keyword"),
    )
    final_top_k = st.sidebar.number_input(
        "Final Top-K",
        min_value=1,
        max_value=20,
        value=int(default_retrieval.get("final_top_k", 8)),
        step=1,
    )
    enable_calibration = st.sidebar.checkbox(
        "Enable V26 calibration",
        value=bool(default_retrieval.get("enable_calibration", True)),
        help="Applies source/path filename calibration only to configured fusion modes.",
    )
    show_debug_fields = st.sidebar.checkbox(
        "Show debug fields",
        value=bool(config.get("ui", {}).get("show_debug_fields", True)),
    )
    st.sidebar.header("Run Mode")
    st.sidebar.caption("Retrieval-only never calls the LLM. Generate Answer calls the local Ollama LLM.")
    st.sidebar.header("Local Models")
    model = config.get("model", {})
    st.sidebar.write(f"LLM: `{model.get('llm_model', '')}`")
    st.sidebar.write(f"Embedding: `{model.get('embedding_model', '')}`")
    return {
        "retrieval_mode": retrieval_mode,
        "final_top_k": int(final_top_k),
        "enable_calibration": bool(enable_calibration),
        "show_debug_fields": bool(show_debug_fields),
    }


def render_demo_questions(config: dict[str, Any], controls: dict[str, Any]) -> None:
    st.subheader("Demo Questions")
    demo = load_demo_cases(config)
    source_rows = {
        name: payload for name, payload in demo.get("sources", {}).items() if payload.get("cases")
    }
    warning_rows = [
        f"{name}: {payload.get('warning')}"
        for name, payload in demo.get("sources", {}).items()
        if payload.get("warning")
    ]
    for warning in warning_rows:
        st.warning(warning)

    render_highlighted_cases(demo, controls)
    if not source_rows:
        st.info("No demo cases loaded yet. Check V25c/V25d/V26 eval files.")
        return

    left, right = st.columns([1, 2])
    source_name = left.selectbox("Case source", list(source_rows.keys()), format_func=format_source_name)
    cases = source_rows[source_name]["cases"]
    selected_index = right.selectbox(
        "Case",
        list(range(len(cases))),
        format_func=lambda index: format_case_label(cases[index]),
        key=f"demo_case_select_{source_name}",
    )
    selected_case = cases[int(selected_index)]
    render_case_details(selected_case)
    if st.button("Load question", key="load_demo_case", use_container_width=True):
        queue_question(selected_case.get("question", ""))


def render_highlighted_cases(demo: dict[str, Any], controls: dict[str, Any]) -> None:
    highlighted = demo.get("highlighted_cases", [])
    if not highlighted:
        return
    st.markdown("**Highlighted Cases**")
    labels = [
        f"{item.get('id', '')} | {item.get('label', '')}" + ("" if item.get("found") else " (missing)")
        for item in highlighted
    ]
    selected = st.selectbox("Highlighted case", list(range(len(highlighted))), format_func=lambda index: labels[index])
    item = highlighted[int(selected)]
    case = item.get("case") or {}
    if not item.get("found"):
        st.warning(f"Configured highlighted case not found in loaded demo files: {item.get('id')}")
        return
    render_case_details(case)
    left, right = st.columns([1, 1])
    if left.button("Load highlighted question", key="load_highlighted_question", use_container_width=True):
        queue_question(case.get("question", ""))
    if right.button("Run highlighted retrieval-only", key="run_highlighted_retrieval", use_container_width=True):
        question = str(case.get("question", "")).strip()
        if question:
            run_retrieval_for_question(question, controls)


def render_case_details(case: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("ID", str(case.get("id", "")))
    cols[1].metric("Type", str(case.get("query_type", "")))
    cols[2].metric("Difficulty", str(case.get("difficulty", "")))
    cols[3].metric("Required categories", len(case.get("required_all_categories", []) or []))
    st.write("Question:", case.get("question", ""))
    st.write("Expected sources:", case.get("expected_sources_contains", []))
    st.write("Expected categories:", case.get("expected_categories", []))
    if case.get("notes"):
        st.caption(str(case.get("notes")))


def run_retrieval_for_question(question: str, controls: dict[str, Any]) -> None:
    with st.spinner("Running retrieval-only inspection..."):
        st.session_state["v25f_result"] = run_retrieval_inspection(
            question=question,
            retrieval_mode=controls["retrieval_mode"],
            enable_calibration=controls["enable_calibration"],
            final_top_k=controls["final_top_k"],
            config_path=CONFIG_PATH,
        )


def queue_question(question: Any) -> None:
    value = str(question or "").strip()
    if not value:
        st.warning("Selected case has an empty question.")
        return
    st.session_state["v25f_pending_question"] = value
    st.success("Question loaded. The page will refresh with the selected question.")
    st.rerun()


def render_result(result: dict[str, Any], config: dict[str, Any], controls: dict[str, Any]) -> None:
    st.divider()
    st.subheader("Run Summary")
    cols = st.columns(5)
    cols[0].metric("Mode", result.get("retrieval_mode", ""))
    cols[1].metric("Run", result.get("run_mode", ""))
    cols[2].metric("LLM called", str(result.get("llm_called", False)))
    cols[3].metric("Calibration", str(result.get("calibration_enabled", False)))
    cols[4].metric("Query type", str(result.get("query_type", "")))

    st.write("Target categories:", result.get("target_categories", []))
    if controls["show_debug_fields"]:
        st.caption(f"Expanded query: {result.get('expanded_query', '')}")

    render_evidence_gate(result.get("evidence_gate", {}))
    render_sql_relation_summary(result, controls)
    render_sources(result.get("final_sources", []), controls)
    if result.get("run_mode") == "full_rag_answer":
        render_answer(result)
        render_citation(result.get("citation_check", {}))
        render_quality(result.get("evidence_quality", {}))
    render_manual_review_export(result, config)


def render_evidence_gate(gate: dict[str, Any]) -> None:
    st.subheader("Evidence Gate")
    sufficient = bool(gate.get("evidence_sufficient", False))
    if sufficient:
        st.success(f"Evidence sufficient: {gate.get('reason', '')}")
    else:
        st.warning(f"Insufficient evidence: {gate.get('reason', '')}")
    with st.expander("Evidence gate signals", expanded=False):
        st.json(gate.get("signals", {}))


def render_sources(sources: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    st.subheader("Retrieval Results")
    if not sources:
        st.info("No sources returned.")
        return
    table_rows = []
    for source in sources:
        table_rows.append(
            {
                "rank": source.get("rank"),
                "source": source.get("source"),
                "category": source.get("category_dir"),
                "doc_type": source.get("doc_type"),
                "final_score": source.get("final_score"),
                "dense_score": source.get("dense_score"),
                "metadata_score": source.get("metadata_score"),
                "keyword_score": source.get("keyword_score"),
                "calibration_score": source.get("calibration_score"),
                "retrieval_channels": ", ".join(source.get("retrieval_channels", []) or []),
                "relation_score": source.get("relation_score"),
                "relation_score_normalized": source.get("relation_score_normalized"),
                "category_intent_adjustment": source.get("category_intent_adjustment"),
                "high_frequency_entity_penalty": source.get("high_frequency_entity_penalty"),
                "sql_only": source.get("sql_only_candidate", False),
                "matched_terms": ", ".join(str(item) for item in (source.get("matched_terms", []) or [])[:12]),
                "calibration_reasons": "; ".join(str(item) for item in (source.get("calibration_reasons", []) or [])),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    for source in sources:
        title = f"#{source.get('rank')} {source.get('category_dir', '')} - {source.get('source', '')}"
        with st.expander(title, expanded=False):
            st.write("Imported source:", source.get("imported_source", ""))
            st.write("Original source:", source.get("original_source", ""))
            st.write("Chunk:", source.get("chunk_id", ""))
            st.write("Section:", source.get("section", ""))
            st.write("Why selected:", source.get("why_selected", ""))
            if source.get("relation_path") or source.get("fusion_reasons") or source.get("relation_reasons"):
                st.markdown("**SQL relation / fusion details**")
                st.write("retrieval_channels:", source.get("retrieval_channels", []))
                st.write("relation_score_raw:", source.get("relation_score_raw", source.get("relation_score", 0)))
                st.write("relation_score_normalized:", source.get("relation_score_normalized", 0))
                st.write("category_intent_adjustment:", source.get("category_intent_adjustment", 0))
                st.write("high_frequency_entity_penalty:", source.get("high_frequency_entity_penalty", 0))
                st.write("sql_only_candidate:", source.get("sql_only_candidate", False))
                st.write("relation_reasons:", source.get("relation_reasons", []))
                st.write("fusion_reasons:", source.get("fusion_reasons", []))
                st.write("relation_path:", source.get("relation_path", []))
            if controls["show_debug_fields"]:
                st.json(
                    {
                        "matched_fields": source.get("matched_fields", []),
                        "match_reason": source.get("match_reason", ""),
                        "calibration_breakdown": source.get("calibration_breakdown", []),
                    }
                )
            preview_key = f"source_preview_{source.get('rank', '')}_{source.get('chunk_id', '')}"
            st.text_area(
                "Preview",
                value=str(source.get("content_preview", "")),
                height=160,
                disabled=True,
                key=preview_key,
            )


def render_sql_relation_summary(result: dict[str, Any], controls: dict[str, Any]) -> None:
    mode = str(result.get("retrieval_mode", ""))
    if mode not in {"sql_relation", "enhanced_sql_relation"}:
        return
    st.subheader("SQL Relation / Fusion Summary")
    if result.get("category_intent"):
        st.markdown("**Category Intent**")
        st.json(result.get("category_intent", {}))
    sql_summary = result.get("sql_relation_summary", {})
    cols = st.columns(4)
    cols[0].metric("SQL candidates", sql_summary.get("candidate_count", 0))
    cols[1].metric("Seed entities", sql_summary.get("seed_entity_count", 0))
    cols[2].metric("Seed events", sql_summary.get("seed_event_count", 0))
    cols[3].metric("Relation paths", sql_summary.get("relation_path_found_count", 0))
    if controls.get("show_debug_fields", False):
        with st.expander("SQL / fusion debug", expanded=False):
            st.json(
                {
                    "sql_relation_summary": result.get("sql_relation_summary", {}),
                    "fusion_debug": result.get("fusion_debug", {}),
                }
            )


def render_answer(result: dict[str, Any]) -> None:
    st.subheader("Answer")
    if result.get("llm_called"):
        st.info("Local LLM call: Ollama was called for this answer.")
    else:
        st.info("No LLM call: answer came from the insufficient-evidence template.")
    st.markdown(str(result.get("answer", "")))


def render_citation(citation: dict[str, Any]) -> None:
    st.subheader("Citation Checker")
    cols = st.columns(5)
    cols[0].metric("Passed", str(citation.get("citation_check_passed", "")))
    cols[1].metric("Supported ratio", citation.get("supported_claim_ratio", ""))
    cols[2].metric("Unsupported", len(citation.get("unsupported_claims", [])))
    cols[3].metric("Weak", citation.get("weak_claim_count", 0))
    cols[4].metric("Ignored", citation.get("ignored_claim_count", 0))
    with st.expander("Unsupported claims", expanded=bool(citation.get("unsupported_claims"))):
        st.json(citation.get("unsupported_claims", []))
    with st.expander("Weak claims", expanded=bool(citation.get("weak_claims"))):
        st.json(citation.get("weak_claims", []))
    with st.expander("Ignored claims", expanded=False):
        st.json(citation.get("ignored_claims", []))


def render_quality(quality: dict[str, Any]) -> None:
    st.subheader("Evidence Quality")
    cols = st.columns(6)
    cols[0].metric("Score", quality.get("evidence_quality_score", ""))
    cols[1].metric("Level", quality.get("evidence_quality_level", ""))
    cols[2].metric("Retrieval", quality.get("retrieval_confidence_score", ""))
    cols[3].metric("Citation", quality.get("citation_support_score", ""))
    cols[4].metric("Coverage", quality.get("coverage_score", ""))
    cols[5].metric("Risk", quality.get("risk_penalty", ""))
    st.write("Reasons:", quality.get("evidence_quality_reasons", []))
    st.write("Warnings:", quality.get("evidence_quality_warnings", []))


def render_manual_review_export(result: dict[str, Any], config: dict[str, Any]) -> None:
    st.subheader("Manual Review Export")
    with st.form("manual_review_export"):
        manual_judgment = st.selectbox(
            "manual_judgment",
            ["", "correct", "partially_correct", "incorrect", "insufficient_evidence_correct", "should_have_refused"],
        )
        manual_citation_judgment = st.selectbox(
            "manual_citation_judgment",
            ["", "fully_supported", "partially_supported", "unsupported", "citation_not_needed"],
        )
        manual_should_refuse = st.selectbox("manual_should_refuse", ["", "true", "false"])
        manual_notes = st.text_area("manual_notes", height=90)
        submitted = st.form_submit_button("Append Manual Review Record")
    if submitted:
        export_record = {
            **result,
            "manual_judgment": manual_judgment,
            "manual_citation_judgment": manual_citation_judgment,
            "manual_should_refuse": manual_should_refuse,
            "manual_notes": manual_notes,
        }
        output = append_manual_review_export(export_record, config["outputs"]["review_export_csv"])
        st.success(f"Appended manual review record to {output}")


def render_interaction_log_browser(config: dict[str, Any]) -> None:
    st.subheader("Recent Interactions")
    log_path = config.get("outputs", {}).get("interaction_log_jsonl", "")
    logs = load_interaction_logs(log_path, limit=100)
    if not logs:
        st.info("No UI interaction logs found yet.")
        return

    summaries = [summarize_interaction_record(row) for row in logs]
    st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)
    selected_index = st.selectbox(
        "Select an interaction",
        list(range(len(summaries))),
        format_func=lambda index: format_history_label(summaries[index]),
    )
    selected_summary = summaries[int(selected_index)]
    left, right = st.columns([1, 1])
    if left.button("Load question from selected interaction", use_container_width=True):
        queue_question(selected_summary.get("question", ""))
    if right.button("Clear current displayed result", use_container_width=True):
        st.session_state.pop("v25f_result", None)
        st.rerun()
    with st.expander("Selected interaction summary", expanded=True):
        st.json(selected_summary)
    with st.expander("Raw interaction detail", expanded=False):
        st.json(logs[int(selected_index)])


def render_manual_review_exports_browser(config: dict[str, Any]) -> None:
    st.subheader("Manual Review Exports")
    export_path = config.get("outputs", {}).get("review_export_csv", "")
    rows = load_manual_review_exports(export_path, limit=100)
    if not rows:
        st.info("No manual review export CSV found yet.")
        return
    st.caption(f"Showing the most recent {len(rows)} exported review rows.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_answer_eval_reports_tab(config: dict[str, Any]) -> None:
    st.subheader("Answer Eval Reports")
    browser = config.get("answer_eval_report_browser", {}) or {}
    if not browser.get("enabled", False):
        st.info("Answer eval report browser is disabled in UI config.")
        return

    versions = list(browser.get("available_versions", ["v4f", "v4g"]))
    default_version = str(browser.get("default_version", versions[-1] if versions else "v4g"))
    version = st.selectbox(
        "Report version",
        versions,
        index=versions.index(default_version) if default_version in versions else 0,
        key="answer_eval_report_version",
    )
    report = load_answer_eval_report(CONFIG_PATH, version)
    records = load_answer_eval_records(CONFIG_PATH, version)
    compare = load_answer_eval_compare(CONFIG_PATH, version)
    for warning in report.get("warnings", []):
        st.warning(warning)
    if not records:
        st.info("No answer eval records found for the selected version.")
        return

    render_answer_eval_metrics(report.get("metrics", {}))
    with st.expander("V4f / V4g compare", expanded=version == "v4g"):
        if compare:
            render_answer_eval_compare(compare)
        else:
            st.info("No compare JSON was loaded for this version.")

    st.markdown("**Filters**")
    filter_cols = st.columns(6)
    case_options = [""] + extract_case_options(records)
    mode_options = [""] + extract_mode_options(records)
    status_options = [""] + extract_status_options(records)
    failure_options = [""] + extract_failure_options(records)
    quality_options = [""] + extract_quality_options(records)
    selected_case = filter_cols[0].selectbox("case_id", case_options, key=f"{version}_case_filter")
    selected_mode = filter_cols[1].selectbox("retrieval_mode", mode_options, key=f"{version}_mode_filter")
    selected_status = filter_cols[2].selectbox("status", status_options, key=f"{version}_status_filter")
    selected_failure = filter_cols[3].selectbox("failure_type", failure_options, key=f"{version}_failure_filter")
    selected_quality = filter_cols[4].selectbox("quality", quality_options, key=f"{version}_quality_filter")
    review_choice = filter_cols[5].selectbox(
        "human_review",
        ["", "true", "false"],
        key=f"{version}_review_filter",
    )
    review_value = None if not review_choice else review_choice == "true"
    filtered = filter_answer_eval_records(
        records,
        case_id=selected_case or None,
        retrieval_mode=selected_mode or None,
        status=selected_status or None,
        failure_type=selected_failure or None,
        quality_level=selected_quality or None,
        needs_human_review=review_value,
    )
    st.caption(f"Showing {len(filtered)} / {len(records)} runs.")
    if not filtered:
        st.info("No runs match the current filters.")
        return

    table_rows = build_answer_eval_table_rows(filtered)
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download filtered records as JSON",
        data=json.dumps(filtered, ensure_ascii=False, indent=2),
        file_name=f"answer_eval_{version}_filtered.json",
        mime="application/json",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download filtered table as CSV",
        data=pd.DataFrame(table_rows).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"answer_eval_{version}_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    selected_index = st.selectbox(
        "Run detail",
        list(range(len(filtered))),
        format_func=lambda index: format_answer_eval_run_label(filtered[index]),
        key=f"{version}_answer_eval_detail",
    )
    render_answer_eval_run_detail(filtered[int(selected_index)], browser.get("display", {}))
    st.divider()
    render_human_review_queue_section(config, version, records)
    st.divider()
    render_human_review_adjudication_section(config)


def render_answer_eval_metrics(metrics: dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("total_cases", metrics.get("total_cases", 0))
    cols[1].metric("total_runs", metrics.get("total_runs", 0))
    cols[2].metric("LLM calls", metrics.get("llm_called_count", 0))
    cols[3].metric("answers", metrics.get("answer_generated_count", 0))
    cols[4].metric("empty", metrics.get("empty_answer_count", 0))
    cols = st.columns(5)
    cols[0].metric("retries", metrics.get("retry_count_total", 0))
    cols[1].metric("citation pass", metrics.get("citation_check_pass_rate", 0.0))
    cols[2].metric("quality avg", metrics.get("evidence_quality_avg", 0.0))
    cols[3].metric("human review", metrics.get("needs_human_review_count", 0))
    cols[4].metric("negative refusal", metrics.get("negative_refusal_pass_rate", 0.0))
    st.metric("relation_path_found_rate", metrics.get("relation_path_found_rate", 0.0))


def render_answer_eval_compare(compare: dict[str, Any]) -> None:
    if "metrics_by_mode" in compare:
        st.json({key: value for key, value in compare.items() if key != "metrics_by_mode"})
        with st.expander("metrics_by_mode", expanded=False):
            st.json(compare.get("metrics_by_mode", {}))
    else:
        st.json(compare)


def build_answer_eval_table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        detail = format_run_detail(record)
        top = detail.get("top_sources", [{}])[0] if detail.get("top_sources") else {}
        rows.append(
            {
                "case_id": detail.get("case_id", ""),
                "retrieval_mode": detail.get("retrieval_mode", ""),
                "status": detail.get("answer_eval_status", ""),
                "quality": detail.get("evidence_quality_level", ""),
                "needs_human_review": detail.get("needs_human_review", False),
                "failure_types": "|".join(detail.get("failure_types", [])),
                "retry_count": detail.get("retry_count", 0),
                "relation_path_found": bool(detail.get("relation_paths")),
                "expected_source_hit": detail.get("expected_source_hit"),
                "expected_category_hit": detail.get("expected_category_hit"),
                "negative_refusal_pass": detail.get("negative_refusal_pass"),
                "top_source": top.get("source", ""),
            }
        )
    return rows


def render_answer_eval_run_detail(record: dict[str, Any], display_config: dict[str, Any]) -> None:
    detail = format_run_detail(record)
    status = str(detail.get("answer_eval_status", ""))
    failures = detail.get("failure_types", [])
    if status == "llm_empty_answer":
        st.error("LLM empty answer")
    if "retrieval_insufficient" in failures:
        st.warning("Retrieval insufficient / refusal path")
    if "unsupported_claims_present" in failures:
        st.warning("Unsupported claims present")
    if "weak_claims_present" in failures:
        st.warning("Weak claims present")
    if detail.get("needs_human_review"):
        st.info("Needs human review")
    if detail.get("negative_refusal_pass"):
        st.success("Negative case refusal passed")

    cols = st.columns(5)
    cols[0].metric("status", status)
    cols[1].metric("quality", detail.get("evidence_quality_level", ""))
    cols[2].metric("retry", detail.get("retry_count", 0))
    cols[3].metric("source hit", detail.get("expected_source_hit"))
    cols[4].metric("category hit", detail.get("expected_category_hit"))
    st.markdown("**Question**")
    st.write(detail.get("question", ""))
    st.markdown("**Failure taxonomy**")
    st.write(failures)
    st.markdown("**Answer preview**")
    st.write(detail.get("answer_preview", ""))

    with st.expander("Final answer", expanded=False):
        st.markdown(str(detail.get("answer", "")))
    if display_config.get("show_llm_attempts", True):
        with st.expander("LLM attempts / retry records", expanded=bool(detail.get("llm_attempts"))):
            st.json(detail.get("llm_attempts", []))
    if display_config.get("show_raw_response_preview", True):
        with st.expander("raw_response_preview", expanded=False):
            st.text(detail.get("raw_response_preview", ""))
    if display_config.get("show_prompt_preview", True):
        with st.expander("prompt_preview", expanded=False):
            st.text(detail.get("prompt_preview", ""))
    if display_config.get("show_citation_details", True):
        with st.expander("Citation checker", expanded=bool(detail.get("unsupported_claims") or detail.get("weak_claims"))):
            st.json(
                {
                    "summary": detail.get("citation_summary", {}),
                    "unsupported_claims": detail.get("unsupported_claims", []),
                    "weak_claims": detail.get("weak_claims", []),
                }
            )
    with st.expander("Evidence quality / human review", expanded=bool(detail.get("needs_human_review"))):
        st.json(
            {
                "evidence_quality": detail.get("evidence_quality", {}),
                "review_decision": detail.get("review_decision", {}),
                "review_reasons": detail.get("review_reasons", []),
            }
        )
    with st.expander("Top sources", expanded=True):
        st.dataframe(pd.DataFrame(detail.get("top_sources", [])), use_container_width=True, hide_index=True)
    if display_config.get("show_relation_path", True):
        with st.expander("Relation path / relation reasons / fusion reasons", expanded=bool(detail.get("relation_paths"))):
            st.json(detail.get("relation_paths", []))


def render_human_review_queue_section(config: dict[str, Any], version: str, records: list[dict[str, Any]]) -> None:
    triage_ui = config.get("answer_eval_review_triage", {}) or {}
    if not triage_ui.get("enabled", False):
        return
    st.subheader("Human Review Queue / Failure Triage")
    config_path = triage_ui.get("config_path", "config/review_triage_v4i.yaml")
    try:
        triage_config = load_triage_config(PROJECT_ROOT / str(config_path))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load triage config: {exc}")
        return

    triage_records = []
    for record in records:
        item = dict(record)
        item["_triage_version"] = version
        triage_records.append(item)
    tasks = build_review_tasks(triage_records, triage_config)
    failure_options = sorted({label for task in tasks for label in task.get("failure_types", [])})

    left, middle, right = st.columns([1, 2, 1])
    only_review = left.checkbox("Only needs_human_review", value=False, key=f"{version}_triage_only_review")
    selected_failures = middle.multiselect(
        "Failure types",
        failure_options,
        default=[],
        key=f"{version}_triage_failure_types",
    )
    priority_filter = right.selectbox("Priority", ["", "high", "medium", "low"], key=f"{version}_triage_priority")

    filtered_tasks = []
    for task in tasks:
        failures = set(task.get("failure_types", []) or [])
        if only_review and not task.get("needs_human_review", False):
            continue
        if selected_failures and not failures.intersection(set(selected_failures)):
            continue
        if priority_filter and task.get("priority") != priority_filter:
            continue
        filtered_tasks.append(task)

    summary = summarize_review_tasks(filtered_tasks, total_records=len(records), version=version)
    render_review_task_summary(summary)
    if not filtered_tasks:
        st.info("No review tasks match the current triage filters.")
        return

    task_rows = build_review_task_table_rows(filtered_tasks)
    st.dataframe(pd.DataFrame(task_rows), use_container_width=True, hide_index=True)

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download review tasks JSON",
        data=json.dumps({"summary": summary, "tasks": filtered_tasks}, ensure_ascii=False, indent=2),
        file_name=f"answer_eval_review_tasks_{version}.json",
        mime="application/json",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download review tasks CSV",
        data=pd.DataFrame(task_rows).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"answer_eval_review_tasks_{version}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[2].download_button(
        "Download review tasks Markdown",
        data=render_tasks_markdown(filtered_tasks, summary),
        file_name=f"answer_eval_review_tasks_{version}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    selected_task_index = st.selectbox(
        "Review task detail",
        list(range(len(filtered_tasks))),
        format_func=lambda index: f"{filtered_tasks[index].get('priority')} | {filtered_tasks[index].get('case_id')} | {filtered_tasks[index].get('retrieval_mode')}",
        key=f"{version}_review_task_detail",
    )
    render_review_task_detail(filtered_tasks[int(selected_task_index)])


def render_review_task_summary(summary: dict[str, Any]) -> None:
    failure_counts = summary.get("failure_type_counts", {}) or {}
    priority_counts = summary.get("priority_counts", {}) or {}
    cols = st.columns(6)
    cols[0].metric("tasks", summary.get("review_task_count", 0))
    cols[1].metric("high", priority_counts.get("high", 0))
    cols[2].metric("medium", priority_counts.get("medium", 0))
    cols[3].metric("low", priority_counts.get("low", 0))
    cols[4].metric("empty", failure_counts.get("llm_empty_answer", 0))
    cols[5].metric("unsupported", failure_counts.get("unsupported_claims_present", 0))
    more_cols = st.columns(2)
    more_cols[0].metric("retrieval_insufficient", failure_counts.get("retrieval_insufficient", 0))
    more_cols[1].metric("weak_claims_present", failure_counts.get("weak_claims_present", 0))


def build_review_task_table_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        rows.append(
            {
                "task_id": task.get("task_id", ""),
                "case_id": task.get("case_id", ""),
                "retrieval_mode": task.get("retrieval_mode", ""),
                "priority": task.get("priority", ""),
                "failure_types": "|".join(task.get("failure_types", []) or []),
                "answer_eval_status": task.get("answer_eval_status", ""),
                "evidence_quality_level": task.get("evidence_quality_level", ""),
                "needs_human_review": task.get("needs_human_review", False),
                "suggested_review_action": task.get("suggested_review_action", ""),
            }
        )
    return rows


def render_review_task_detail(task: dict[str, Any]) -> None:
    with st.expander("Review task detail", expanded=True):
        st.markdown("**Question**")
        st.write(task.get("question", ""))
        st.markdown("**Suggested review action**")
        st.write(task.get("suggested_review_action", ""))
        st.markdown("**Answer preview**")
        st.write(task.get("answer_preview", ""))
        st.markdown("**Manual fields**")
        st.json(
            {
                "manual_judgment": task.get("manual_judgment", ""),
                "manual_citation_judgment": task.get("manual_citation_judgment", ""),
                "manual_should_refuse": task.get("manual_should_refuse", ""),
                "manual_fix_suggestion": task.get("manual_fix_suggestion", ""),
                "manual_notes": task.get("manual_notes", ""),
            }
        )
    with st.expander("Top sources preview", expanded=False):
        st.json(task.get("top_sources_preview", []))
    with st.expander("Unsupported / weak claims", expanded=bool(task.get("unsupported_claims_preview") or task.get("weak_claims_preview"))):
        st.json(
            {
                "unsupported_claims_preview": task.get("unsupported_claims_preview", []),
                "weak_claims_preview": task.get("weak_claims_preview", []),
            }
        )
    with st.expander("Relation paths preview", expanded=bool(task.get("relation_paths_preview"))):
        st.json(task.get("relation_paths_preview", []))
    with st.expander("LLM attempts / raw response / prompt preview", expanded=False):
        st.json(
            {
                "retry_count": task.get("retry_count", 0),
                "llm_attempts_preview": task.get("llm_attempts_preview", []),
                "raw_response_preview": task.get("raw_response_preview", ""),
                "prompt_preview": task.get("prompt_preview", ""),
            }
        )


def render_human_review_adjudication_section(config: dict[str, Any]) -> None:
    adjudication_ui = config.get("answer_eval_review_adjudication", {}) or {}
    if not adjudication_ui.get("enabled", False):
        return
    st.subheader("Human Review Adjudication")
    config_path = adjudication_ui.get("config_path", "config/review_adjudication_v4j.yaml")
    try:
        adjudication_config = load_adjudication_config(PROJECT_ROOT / str(config_path))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load adjudication config: {exc}")
        return

    default_task_path = adjudication_config.get("inputs", {}).get("default_review_tasks_json", "")
    tasks = load_review_tasks(default_task_path)
    if not tasks:
        st.info("No V4i review tasks found for adjudication.")
        return

    st.caption("Default mode reads V4i review tasks. Uploading a filled CSV merges manual fields in memory only.")
    uploaded = st.file_uploader(
        "Upload filled human review CSV",
        type=["csv"],
        key="v4j_manual_csv_upload",
    )
    manual_rows: list[dict[str, Any]] = []
    if uploaded is not None:
        try:
            manual_rows = pd.read_csv(uploaded).fillna("").to_dict("records")
            tasks = merge_manual_review(tasks, manual_rows)
            st.success(f"Loaded {len(manual_rows)} manual rows from uploaded CSV.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not parse uploaded CSV: {exc}")

    enriched = enrich_adjudication_tasks(tasks, adjudication_config)
    validation = validate_manual_fields(enriched, adjudication_config)
    summary = summarize_adjudication(enriched, adjudication_config)
    notes_md = build_adjudication_calibration_notes(enriched, summary, adjudication_config)
    render_adjudication_summary(summary)
    render_adjudication_validation(validation)

    rows = build_adjudication_table_rows(enriched)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download adjudicated tasks JSON",
        data=json.dumps({"summary": summary, "tasks": enriched}, ensure_ascii=False, indent=2),
        file_name="answer_eval_review_adjudicated_v4j.json",
        mime="application/json",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download adjudicated tasks CSV",
        data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
        file_name="answer_eval_review_adjudicated_v4j.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[2].download_button(
        "Download calibration notes Markdown",
        data=notes_md,
        file_name="answer_eval_calibration_notes_v4j.md",
        mime="text/markdown",
        use_container_width=True,
    )

    selected = st.selectbox(
        "Adjudication task detail",
        list(range(len(enriched))),
        format_func=lambda index: f"{enriched[index].get('adjudication_status')} | {enriched[index].get('case_id')} | {enriched[index].get('retrieval_mode')}",
        key="v4j_adjudication_detail",
    )
    render_adjudication_task_detail(enriched[int(selected)])
    with st.expander("Calibration notes preview", expanded=False):
        st.markdown(notes_md)


def render_adjudication_summary(summary: dict[str, Any]) -> None:
    status_counts = summary.get("adjudication_status_counts", {}) or {}
    fix_counts = summary.get("fix_category_counts", {}) or {}
    cols = st.columns(7)
    cols[0].metric("tasks", summary.get("total_tasks", 0))
    cols[1].metric("pending", summary.get("pending_count", 0))
    cols[2].metric("adjudicated", summary.get("adjudicated_count", 0))
    cols[3].metric("accepted", status_counts.get("accepted", 0))
    cols[4].metric("revision", status_counts.get("needs_revision", 0))
    cols[5].metric("valid refusal", status_counts.get("valid_refusal", 0))
    cols[6].metric("discussion", status_counts.get("needs_discussion", 0))
    more_cols = st.columns(3)
    more_cols[0].metric("citation checker fix", fix_counts.get("citation_checker_fix", 0))
    more_cols[1].metric("retrieval fix", fix_counts.get("retrieval_fix", 0))
    more_cols[2].metric("prompt fix", fix_counts.get("prompt_fix", 0))


def render_adjudication_validation(validation: dict[str, Any]) -> None:
    if validation.get("invalid_values"):
        st.error("Invalid manual field values found.")
        st.json(validation.get("invalid_values", []))
    if validation.get("missing_required_high_priority_judgments"):
        st.warning("High-priority tasks still require manual_judgment.")
        st.json(validation.get("missing_required_high_priority_judgments", []))
    if not validation.get("invalid_values") and not validation.get("missing_required_high_priority_judgments"):
        st.success("Manual field validation passed.")


def build_adjudication_table_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        rows.append(
            {
                "task_id": task.get("task_id", ""),
                "case_id": task.get("case_id", ""),
                "retrieval_mode": task.get("retrieval_mode", ""),
                "priority": task.get("priority", ""),
                "manual_judgment": task.get("manual_judgment", ""),
                "manual_citation_judgment": task.get("manual_citation_judgment", ""),
                "manual_should_refuse": task.get("manual_should_refuse", ""),
                "adjudication_status": task.get("adjudication_status", ""),
                "fix_categories": "|".join(task.get("fix_categories", []) or []),
            }
        )
    return rows


def render_adjudication_task_detail(task: dict[str, Any]) -> None:
    with st.expander("Adjudicated task detail", expanded=True):
        st.json(
            {
                "task_id": task.get("task_id", ""),
                "failure_types": task.get("failure_types", []),
                "suggested_review_action": task.get("suggested_review_action", ""),
                "manual_fields": {
                    "manual_judgment": task.get("manual_judgment", ""),
                    "manual_citation_judgment": task.get("manual_citation_judgment", ""),
                    "manual_should_refuse": task.get("manual_should_refuse", ""),
                    "manual_fix_suggestion": task.get("manual_fix_suggestion", ""),
                    "manual_notes": task.get("manual_notes", ""),
                },
                "adjudication_status": task.get("adjudication_status", ""),
                "fix_categories": task.get("fix_categories", []),
            }
        )
        st.markdown("**Answer preview**")
        st.write(task.get("answer_preview", ""))
    with st.expander("Evidence details", expanded=False):
        st.json(
            {
                "unsupported_claims_preview": task.get("unsupported_claims_preview", []),
                "weak_claims_preview": task.get("weak_claims_preview", []),
                "relation_paths_preview": task.get("relation_paths_preview", []),
            }
        )


def format_answer_eval_run_label(record: dict[str, Any]) -> str:
    case_id = str(record.get("case_id") or record.get("id", ""))
    mode = str(record.get("retrieval_mode", ""))
    status = str(record.get("answer_eval_status", ""))
    return f"{case_id} | {mode} | {status}"


def render_langgraph_trace_tab(controls: dict[str, Any]) -> None:
    st.subheader("LangGraph Trace")
    st.caption(
        "Runs the controlled V3b LangGraph workflow. Retrieval-only does not call an LLM; full RAG may call local Ollama after you click run."
    )
    if "v3b_graph_question" not in st.session_state:
        st.session_state["v3b_graph_question"] = st.session_state.get("v25f_question", DEFAULT_QUESTION)

    if st.button("Use current Ask question", key="v3b_use_current_question"):
        st.session_state["v3b_graph_question"] = st.session_state.get("v25f_question", DEFAULT_QUESTION)
        st.rerun()

    question = st.text_area("LangGraph question", key="v3b_graph_question", height=90)
    left, middle, right = st.columns([1, 1, 1])
    run_mode = left.selectbox("Graph run mode", ["retrieval_only", "full_rag"], index=0, key="v3b_run_mode")
    modes = ["dense", "enhanced", "keyword", "enhanced_keyword", "sql_relation", "enhanced_sql_relation"]
    default_mode = controls.get("retrieval_mode", "enhanced_keyword")
    retrieval_mode = middle.selectbox(
        "Graph retrieval mode",
        modes,
        index=modes.index(default_mode) if default_mode in modes else modes.index("enhanced_keyword"),
        key="v3b_retrieval_mode",
    )
    enable_calibration = right.checkbox(
        "Enable calibration",
        value=bool(controls.get("enable_calibration", True)),
        key="v3b_enable_calibration",
    )
    run_graph = st.button("Run LangGraph Workflow", type="primary", key="v3b_run_graph", use_container_width=True)
    if run_graph and question.strip():
        if run_mode == "full_rag":
            st.warning("Full RAG mode may call the local Ollama LLM after evidence gate passes.")
        with st.spinner("Running controlled LangGraph workflow..."):
            st.session_state["v3b_graph_result"] = run_rag_graph_v3b(
                question=question.strip(),
                run_mode=run_mode,
                retrieval_mode=retrieval_mode,
                enable_calibration=enable_calibration,
                config_path=GRAPH_CONFIG_PATH,
            )

    result = st.session_state.get("v3b_graph_result")
    if result:
        render_langgraph_result(result)

    st.divider()
    render_recent_langgraph_runs()


def render_langgraph_result(result: dict[str, Any]) -> None:
    summary = result.get("summary", {})
    decision = summary.get("review_decision") if isinstance(summary.get("review_decision"), dict) else {}
    st.markdown("**Current Graph Run**")
    cols = st.columns(6)
    cols[0].metric("run_id", str(summary.get("run_id", ""))[-12:])
    cols[1].metric("mode", summary.get("run_mode", ""))
    cols[2].metric("LLM called", str(summary.get("llm_called", False)))
    cols[3].metric("Evidence", str(summary.get("evidence_sufficient")))
    cols[4].metric("Quality", summary.get("evidence_quality_level", ""))
    cols[5].metric("Review", str(summary.get("needs_human_review", False)))
    st.write("run_id:", summary.get("run_id", ""))
    st.write("review_reasons:", decision.get("review_reasons", []))
    st.write("recommended_action:", decision.get("recommended_action", ""))
    if summary.get("category_intent") or summary.get("sql_relation_debug"):
        with st.expander("Enhanced SQL relation debug", expanded=True):
            st.json(
                {
                    "category_intent": summary.get("category_intent", {}),
                    "sql_relation_debug": summary.get("sql_relation_debug", {}),
                    "retrieval_debug": summary.get("retrieval_debug", {}),
                }
            )

    top_sources = summary.get("top_sources", [])
    if top_sources:
        st.dataframe(pd.DataFrame(top_sources), use_container_width=True, hide_index=True)
    trace_rows = summarize_node_trace(summary.get("trace", []))
    if trace_rows:
        st.markdown("**Node Trace**")
        st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)
        selected = st.selectbox(
            "Trace node detail",
            list(range(len(trace_rows))),
            format_func=lambda index: f"{trace_rows[index].get('step')} | {trace_rows[index].get('node')} | {trace_rows[index].get('status')}",
            key="v3b_trace_node_detail",
        )
        st.json(trace_rows[int(selected)])


def render_recent_langgraph_runs() -> None:
    st.markdown("**Recent Graph Runs**")
    try:
        runs = load_recent_graph_runs(GRAPH_CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001 - UI should show trace read failures gently.
        st.warning(f"Could not load LangGraph trace index: {exc}")
        return
    if not runs:
        st.info("No LangGraph trace index entries found yet.")
        return
    runs = list(reversed(runs))
    st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)
    selected = st.selectbox(
        "Load trace by run_id",
        list(range(len(runs))),
        format_func=lambda index: f"{runs[index].get('run_id', '')} | {runs[index].get('run_mode', '')} | {runs[index].get('question', '')}",
        key="v3b_recent_run_select",
    )
    if st.button("Load selected graph trace", key="v3b_load_trace"):
        run_id = str(runs[int(selected)].get("run_id", ""))
        record = load_graph_trace(run_id, GRAPH_CONFIG_PATH)
        if record:
            st.session_state["v3b_graph_result"] = record
            st.success(f"Loaded trace for {run_id}")
            st.rerun()
        else:
            st.warning(f"Trace not found for run_id={run_id}")


def format_source_name(name: str) -> str:
    labels = {
        "retrieval_eval_v25c": "V25c retrieval eval",
        "answer_eval_v25d": "V25d answer eval subset",
        "calibration_regression_v26": "V26 calibration regression",
        "v4e_demo_cases": "V4e SQL relation demo cases",
    }
    return labels.get(name, name)


def format_case_label(case: dict[str, Any]) -> str:
    case_id = str(case.get("id", ""))
    query_type = str(case.get("query_type", ""))
    question = str(case.get("question", "")).replace("\n", " ")
    if len(question) > 80:
        question = question[:77] + "..."
    return f"{case_id} | {query_type} | {question}"


def format_history_label(summary: dict[str, Any]) -> str:
    question = str(summary.get("question", "")).replace("\n", " ")
    if len(question) > 70:
        question = question[:67] + "..."
    return f"{summary.get('timestamp', '')} | {summary.get('run_mode', '')} | {question}"


if __name__ == "__main__":
    main()

