"""
PDF report generation with ReportLab.

The report is built from a real :class:`analysis.AnalysisResult` (or the dict
form of a stored row) - there is no sample data anywhere in this module.
"""

from __future__ import annotations

import html
import io
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

ACCENT = colors.HexColor("#c2185b")
ACCENT_LIGHT = colors.HexColor("#fce4ec")
INK = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")

MAX_TEXT_CHARS = 12000


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "VBCUATitle", parent=base["Title"], fontSize=19, leading=23,
            textColor=ACCENT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "VBCUASubtitle", parent=base["Normal"], fontSize=9.5, leading=13,
            alignment=TA_CENTER, textColor=MUTED,
        ),
        "heading": ParagraphStyle(
            "VBCUAHeading", parent=base["Heading2"], fontSize=13, leading=16,
            textColor=ACCENT, spaceBefore=12, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "VBCUABody", parent=base["BodyText"], fontSize=10, leading=14.5,
            textColor=INK, spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "VBCUABullet", parent=base["BodyText"], fontSize=10, leading=14,
            textColor=INK, spaceAfter=2,
        ),
        "muted": ParagraphStyle(
            "VBCUAMuted", parent=base["BodyText"], fontSize=8.5, leading=11, textColor=MUTED,
        ),
    }


def _clean(text: Any, limit: int = MAX_TEXT_CHARS) -> str:
    """Escape user text for ReportLab's mini-HTML and preserve line breaks."""
    value = "" if text is None else str(text)
    value = value.strip()
    if len(value) > limit:
        value = value[:limit] + " [...truncated]"
    return html.escape(value).replace("\n", "<br/>")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip("-* ").strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bullets(items: List[str], styles) -> List[Any]:
    if not items:
        return [Paragraph("<i>None recorded.</i>", styles["body"])]
    return [
        ListFlowable(
            [ListItem(Paragraph(_clean(item), styles["bullet"]), leftIndent=10) for item in items],
            bulletType="bullet",
            bulletColor=ACCENT,
            start="circle",
            leftIndent=14,
        )
    ]


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        18 * mm, 12 * mm,
        "Voice Based Concept Understanding Analyser - automated evaluation report",
    )
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(result: Any) -> bytes:
    """Render an evaluation into PDF bytes. Accepts a dict or an AnalysisResult."""
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="VBCUA Evaluation Report",
        author="Voice Based Concept Understanding Analyser",
    )

    score = _number(_get(result, "score"))
    similarity = _number(_get(result, "similarity"))
    fluency = _number(_get(result, "fluency_score"))
    grade = str(_get(result, "grade") or "N/A")
    engine = str(_get(result, "engine") or "offline")
    audio_name = str(_get(result, "audio_name") or "-")
    metrics: Dict[str, Any] = _get(result, "audio_metrics") or {}

    story: List[Any] = [
        Paragraph("Voice Based Concept Understanding Analyser", styles["title"]),
        Paragraph("Automated Conceptual Understanding Evaluation Report", styles["subtitle"]),
        Spacer(1, 4),
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} &nbsp;|&nbsp; "
            f"Audio: {_clean(audio_name, 120)} &nbsp;|&nbsp; Engine: {_clean(engine, 60)}",
            styles["subtitle"],
        ),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=ACCENT),
        Spacer(1, 10),
    ]

    summary_table = Table(
        [
            ["Overall Score", "Similarity", "Fluency", "Grade"],
            [f"{score:.0f}/100", f"{similarity:.0f}%", f"{fluency:.0f}/100", grade],
        ],
        colWidths=[doc.width / 4.0] * 4,
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), ACCENT_LIGHT),
                ("TEXTCOLOR", (0, 1), (-1, 1), INK),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, 1), 15),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
            ]
        )
    )
    story.append(summary_table)

    summary_text = _get(result, "summary") or ""
    if summary_text:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Examiner Summary", styles["heading"]))
        story.append(Paragraph(_clean(summary_text), styles["body"]))

    metric_rows: List[List[str]] = []
    for key, label in (
        ("duration", "Duration (s)"),
        ("pause_ratio", "Silence ratio"),
        ("silence_segments", "Long pauses"),
        ("energy_consistency", "Energy consistency"),
        ("longest_pause", "Longest pause (s)"),
        ("sample_rate", "Sample rate (Hz)"),
    ):
        value = metrics.get(key)
        if value in (None, ""):
            continue
        if key in ("pause_ratio", "energy_consistency"):
            metric_rows.append([label, f"{_number(value) * 100:.0f}%"])
        else:
            metric_rows.append([label, str(value)])

    fluency_detail = _get(result, "fluency_detail") or {}
    if isinstance(fluency_detail, dict):
        if fluency_detail.get("words_per_minute"):
            metric_rows.append(
                ["Speaking rate", f"{_number(fluency_detail['words_per_minute']):.0f} wpm"]
            )
        if "filler_count" in fluency_detail:
            metric_rows.append(["Filler words", f"{int(_number(fluency_detail['filler_count']))}"])

    if metric_rows:
        story.append(Paragraph("Delivery Metrics", styles["heading"]))
        table = Table(
            [["Metric", "Value"]] + metric_rows,
            colWidths=[doc.width * 0.55, doc.width * 0.45],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), ACCENT_LIGHT),
                    ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(table)

    for title, key in (
        ("Strengths", "strengths"),
        ("Weaknesses", "weaknesses"),
        ("Suggestions for Improvement", "suggestions"),
    ):
        story.append(Paragraph(title, styles["heading"]))
        story.extend(_bullets(_as_list(_get(result, key)), styles))

    story.append(PageBreak())
    story.append(Paragraph("Student Transcript", styles["heading"]))
    story.append(
        Paragraph(
            _clean(_get(result, "transcript") or "") or "<i>No transcript available.</i>",
            styles["body"],
        )
    )

    reference = _get(result, "reference_answer") or ""
    if reference:
        story.append(Paragraph("Expected Answer", styles["heading"]))
        story.append(Paragraph(_clean(reference), styles["body"]))

    raw_feedback = _get(result, "feedback") or ""
    if raw_feedback and not (
        _as_list(_get(result, "strengths")) or _as_list(_get(result, "suggestions"))
    ):
        story.append(Paragraph("Feedback", styles["heading"]))
        story.append(Paragraph(_clean(raw_feedback), styles["body"]))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "This report was produced automatically from a speech recording. Scores combine a "
            "semantic comparison against the expected answer with measured delivery fluency and "
            "are intended to support, not replace, human assessment.",
            styles["muted"],
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buffer.seek(0)
    return buffer.getvalue()


def generate_pdf(result: Any) -> bytes:
    """Backwards-compatible alias kept from the original implementation."""
    return generate_pdf_report(result)


def display_results(result: Any) -> None:
    """Render an evaluation inside Streamlit (imported lazily on purpose)."""
    from ui import render_result_details

    render_result_details(result)


__all__ = ["generate_pdf_report", "generate_pdf", "display_results"]
