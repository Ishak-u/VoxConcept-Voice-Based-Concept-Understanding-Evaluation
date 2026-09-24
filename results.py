"""
History page - browse, inspect, export and delete previously saved evaluations.

All data is read from the SQLite ``results`` table written by the analysis
pipeline. Reports for historical rows are regenerated on demand rather than
stored, which keeps the database small.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List

import streamlit as st

import ui
from analysis import result_from_record
from config import Settings
from database import Database
from report import generate_pdf_report


def render(settings: Settings, database: Database) -> None:
    ui.hero(
        "Evaluation history",
        "Every completed analysis is stored locally so past performance can be reviewed.",
    )

    try:
        total = database.count_results()
    except Exception as exc:  # noqa: BLE001
        st.error(f"The history database could not be read: {exc}")
        return

    if total == 0:
        ui.empty_state(
            "No evaluations saved yet",
            "Run an analysis and the result will be stored here automatically.",
        )
        return

    page_size = max(5, settings.history_page_size)
    page_count = max(1, (total + page_size - 1) // page_size)

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.caption(f"{total} saved evaluation(s) across {page_count} page(s).")
    with top_right:
        page = st.number_input(
            "Page", min_value=1, max_value=page_count, value=1, step=1, label_visibility="collapsed"
        )

    try:
        records = database.fetch_results(limit=page_size, offset=(int(page) - 1) * page_size)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load this page of results: {exc}")
        return

    _render_table(records)
    st.divider()
    _render_details(records)
    st.divider()
    _render_tools(records, database)


def _render_table(records: List[Dict[str, Any]]) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "ID": record.get("id"),
                "Date": (record.get("created_at") or "").replace("T", " ")[:16] or "-",
                "Audio": record.get("audio_name") or "-",
                "Score": record.get("score"),
                "Similarity": record.get("similarity"),
                "Grade": record.get("grade") or "-",
                "Engine": record.get("engine") or "offline",
            }
            for record in records
        ]
    )
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f"
            ),
            "Similarity": st.column_config.NumberColumn("Similarity", format="%.0f%%"),
        },
    )


def _render_details(records: List[Dict[str, Any]]) -> None:
    st.markdown("### Inspect an evaluation")

    options = {
        f"#{record.get('id')} - {(record.get('created_at') or '')[:16] or 'unknown date'} "
        f"- {record.get('grade') or 'N/A'}": record
        for record in records
    }
    if not options:
        return

    choice = st.selectbox("Select a record", list(options.keys()))
    record = options[choice]
    result = result_from_record(record)

    ui.score_summary(result)
    ui.bullet_columns(result)

    with st.expander("Transcript", expanded=False):
        st.write(record.get("transcript") or "No transcript stored.")

    reference = record.get("reference_answer")
    if reference:
        with st.expander("Expected answer", expanded=False):
            st.write(reference)

    if not (result.strengths or result.weaknesses or result.suggestions) and record.get("feedback"):
        with st.expander("Feedback", expanded=True):
            st.write(record["feedback"])

    metrics = record.get("audio_metrics") or {}
    if metrics:
        with st.expander("Delivery metrics", expanded=False):
            st.json(metrics)

    try:
        pdf_bytes = generate_pdf_report(result)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The PDF report could not be generated for this record: {exc}")
        return

    st.download_button(
        "Download this report (PDF)",
        data=pdf_bytes,
        file_name=f"VBCUA_report_{record.get('id')}.pdf",
        mime="application/pdf",
        width="stretch",
    )


def _render_tools(records: List[Dict[str, Any]], database: Database) -> None:
    st.markdown("### Manage history")
    col_export, col_delete, col_clear = st.columns(3)

    with col_export:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["id", "created_at", "audio_name", "score", "similarity", "grade", "engine", "transcript"])
        for record in records:
            writer.writerow(
                [
                    record.get("id"),
                    record.get("created_at"),
                    record.get("audio_name"),
                    record.get("score"),
                    record.get("similarity"),
                    record.get("grade"),
                    record.get("engine"),
                    (record.get("transcript") or "").replace("\n", " "),
                ]
            )
        st.download_button(
            "Export page as CSV",
            data=buffer.getvalue().encode("utf-8"),
            file_name="vbcua_history.csv",
            mime="text/csv",
            width="stretch",
        )

    with col_delete:
        ids = [record.get("id") for record in records if record.get("id") is not None]
        target = st.selectbox("Delete record", ids, key="delete_target", label_visibility="collapsed")
        if st.button("Delete selected", width="stretch"):
            try:
                if database.delete_result(int(target)):
                    st.success(f"Record #{target} deleted.")
                    st.rerun()
                else:
                    st.warning(f"Record #{target} no longer exists.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not delete the record: {exc}")

    with col_clear:
        confirm = st.checkbox("Confirm", key="confirm_clear")
        if st.button("Clear all history", width="stretch", disabled=not confirm):
            try:
                removed = database.clear_results()
                st.success(f"Removed {removed} record(s).")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not clear the history: {exc}")


__all__ = ["render"]
