"""
Dashboard page - visualises the current evaluation and the stored history.

Every number and every chart on this page comes from a real analysis run; there
is no placeholder or synthetic data.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

import ui
from config import Settings
from database import Database


def render(settings: Settings, database: Database) -> None:
    ui.hero(
        "Performance dashboard",
        "Score breakdown, delivery metrics and progress across all saved evaluations.",
    )

    result = st.session_state.get("result")

    if result is None:
        ui.empty_state(
            "No current evaluation",
            "Run an analysis from the AI Analysis page to populate this dashboard.",
        )
    else:
        _render_current(result)

    st.divider()
    _render_history(settings, database)


def _render_current(result: Any) -> None:
    ui.score_summary(result)

    if getattr(result, "summary", ""):
        st.markdown("")
        st.info(result.summary)

    st.markdown("### Audio waveform")
    waveform = getattr(result, "waveform", None) or []
    duration = (getattr(result, "audio_metrics", {}) or {}).get("duration")
    if waveform:
        ui.waveform_chart(waveform, duration)
    else:
        st.caption(
            "Waveform could not be rendered for this recording - install ffmpeg to enable "
            "decoding of compressed formats."
        )

    st.markdown("### Score components")
    components: Dict[str, float] = {"Semantic similarity": float(getattr(result, "semantic_similarity", 0) or 0)}
    detail = getattr(result, "fluency_detail", {}) or {}
    label_map = {
        "fillers": "Filler control",
        "pace": "Speaking pace",
        "pauses": "Pause control",
        "energy": "Energy consistency",
        "overall": "Fluency overall",
    }
    for key, label in label_map.items():
        if key in detail:
            components[label] = float(detail[key])
    ui.score_breakdown_chart(components)

    metrics = getattr(result, "audio_metrics", {}) or {}
    rows = []
    if metrics.get("duration") is not None:
        rows.append(("Duration", f"{float(metrics['duration']):.1f} s"))
    if detail.get("words_per_minute"):
        rows.append(("Speaking rate", f"{float(detail['words_per_minute']):.0f} wpm"))
    if "filler_count" in detail:
        rows.append(("Filler words", str(int(detail["filler_count"]))))
    if metrics.get("pause_ratio") is not None:
        rows.append(("Silence ratio", f"{float(metrics['pause_ratio']) * 100:.0f}%"))
    if metrics.get("silence_segments"):
        rows.append(("Long pauses", str(metrics["silence_segments"])))
    if metrics.get("longest_pause") is not None:
        rows.append(("Longest pause", f"{float(metrics['longest_pause']):.1f} s"))

    if rows:
        st.markdown("### Delivery metrics")
        columns = st.columns(min(3, len(rows)))
        for index, (label, value) in enumerate(rows):
            with columns[index % len(columns)]:
                ui.metric_card(label, value)

    st.markdown("")
    ui.bullet_columns(result)

    with st.expander("Transcript"):
        st.write(getattr(result, "transcript", "") or "No transcript available.")


def _render_history(settings: Settings, database: Database) -> None:
    st.markdown("## History analytics")

    try:
        stats = database.summary_stats()
        trend = database.score_trend(limit=20)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The evaluation history could not be loaded: {exc}")
        return

    if stats["total"] == 0:
        ui.empty_state(
            "No saved evaluations yet",
            "Completed analyses are stored automatically and will appear here.",
        )
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.metric_card("Evaluations", str(stats["total"]))
    with col2:
        ui.metric_card(
            "Average score",
            f"{stats['avg_score']:.0f}/100" if stats["avg_score"] is not None else "-",
        )
    with col3:
        ui.metric_card(
            "Average similarity",
            f"{stats['avg_similarity']:.0f}%" if stats["avg_similarity"] is not None else "-",
        )
    with col4:
        ui.metric_card(
            "Best score",
            f"{stats['best_score']:.0f}/100" if stats["best_score"] is not None else "-",
        )

    st.markdown("### Score trend")
    ui.trend_chart(trend)

    if stats["grades"]:
        st.markdown("### Grade distribution")
        pills = "".join(
            f"<span class='vbcua-pill'>{grade}: {count}</span>" for grade, count in stats["grades"]
        )
        st.markdown(pills, unsafe_allow_html=True)


__all__ = ["render"]
