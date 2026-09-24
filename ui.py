"""
Shared Streamlit presentation helpers.

Keeping the styling and the reusable widgets here means the page modules stay
short and every page looks consistent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import streamlit as st


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------

THEME_CSS = """
<style>
:root{
  --vbcua-bg-1:#0b1220;
  --vbcua-bg-2:#141d33;
  --vbcua-panel:rgba(255,255,255,.05);
  --vbcua-border:rgba(255,255,255,.10);
  --vbcua-accent:#ff2e88;
  --vbcua-accent-soft:#ff7ab8;
  --vbcua-text:#e8ecf5;
  --vbcua-muted:#9aa7bd;
}

.stApp{
  background:
    radial-gradient(
      1100px 600px at 12% -12%,
      rgba(255,46,136,.16),
      transparent 60%
    ),
    radial-gradient(
      900px 520px at 92% 4%,
      rgba(56,189,248,.12),
      transparent 62%
    ),
    linear-gradient(
      160deg,
      var(--vbcua-bg-1) 0%,
      var(--vbcua-bg-2) 100%
    );
  color:var(--vbcua-text);
}

section[data-testid="stSidebar"]{
  background:rgba(8,13,25,.94);
  border-right:1px solid var(--vbcua-border);
}

section[data-testid="stSidebar"] *{
  color:var(--vbcua-text);
}

.block-container{
  padding-top:2.2rem;
  padding-bottom:3.5rem;
  max-width:1200px;
}

h1,h2,h3,h4{
  color:var(--vbcua-text) !important;
  letter-spacing:-.01em;
}

h1{
  font-weight:800;
  font-size:clamp(1.6rem,3.4vw,2.4rem);
}

h2{
  font-weight:700;
  font-size:clamp(1.25rem,2.4vw,1.6rem);
}

h3{
  font-weight:650;
  font-size:clamp(1.05rem,2vw,1.28rem);
}

p,li,label,span,
div[data-testid="stMarkdownContainer"] p{
  color:var(--vbcua-text);
  font-size:1rem;
  line-height:1.65;
}

a{
  color:var(--vbcua-accent-soft) !important;
}


/* -----------------------------------------------------------------------
   Hero
   ----------------------------------------------------------------------- */

.vbcua-hero{
  border:1px solid var(--vbcua-border);
  background:
    linear-gradient(
      135deg,
      rgba(255,46,136,.16),
      rgba(56,189,248,.08)
    );
  border-radius:18px;
  padding:1.5rem 1.7rem;
  margin-bottom:1.4rem;
}

.vbcua-hero h1{
  margin:0 0 .35rem 0;
}

.vbcua-hero p{
  margin:0;
  color:var(--vbcua-muted);
  font-size:1.02rem;
}


/* -----------------------------------------------------------------------
   Cards
   ----------------------------------------------------------------------- */

.vbcua-card{
  border:1px solid var(--vbcua-border);
  background:var(--vbcua-panel);
  border-radius:14px;
  padding:1.05rem 1.2rem;
  height:100%;
}

.vbcua-card h4{
  margin:.1rem 0 .45rem 0;
  font-size:1.02rem;
}

.vbcua-card p{
  margin:0;
  color:var(--vbcua-muted);
  font-size:.95rem;
}


/* -----------------------------------------------------------------------
   Custom metric card styling
   ----------------------------------------------------------------------- */

.vbcua-metric{
  border:1px solid var(--vbcua-border);
  background:
    linear-gradient(
      135deg,
      rgba(255,46,136,.14),
      rgba(255,255,255,.04)
    );
  border-radius:14px;
  padding:.95rem 1.1rem;
  text-align:center;
}

.vbcua-metric .label{
  color:var(--vbcua-muted);
  font-size:.8rem;
  text-transform:uppercase;
  letter-spacing:.09em;
  margin-bottom:.3rem;
}

.vbcua-metric .value{
  font-size:1.85rem;
  font-weight:800;
  line-height:1.15;
}

.vbcua-metric .hint{
  color:var(--vbcua-muted);
  font-size:.78rem;
  margin-top:.2rem;
}


/* -----------------------------------------------------------------------
   Pills
   ----------------------------------------------------------------------- */

.vbcua-pill{
  display:inline-block;
  padding:.22rem .7rem;
  border-radius:999px;
  font-size:.78rem;
  font-weight:600;
  border:1px solid var(--vbcua-border);
  background:rgba(255,255,255,.06);
  color:var(--vbcua-muted);
  margin:0 .3rem .35rem 0;
}

.vbcua-pill.ok{
  background:rgba(34,197,94,.16);
  color:#86efac;
  border-color:rgba(34,197,94,.3);
}

.vbcua-pill.warn{
  background:rgba(245,158,11,.16);
  color:#fcd34d;
  border-color:rgba(245,158,11,.3);
}

.vbcua-pill.bad{
  background:rgba(239,68,68,.16);
  color:#fca5a5;
  border-color:rgba(239,68,68,.3);
}


/* -----------------------------------------------------------------------
   Empty state
   ----------------------------------------------------------------------- */

.vbcua-empty{
  border:1px dashed var(--vbcua-border);
  border-radius:14px;
  padding:2.1rem 1.4rem;
  text-align:center;
  background:rgba(255,255,255,.03);
}

.vbcua-empty h3{
  margin:.2rem 0 .4rem 0;
}

.vbcua-empty p{
  color:var(--vbcua-muted);
  margin:0;
}


/* -----------------------------------------------------------------------
   Buttons
   ----------------------------------------------------------------------- */

.stButton>button,
.stDownloadButton>button{
  width:100%;
  border:none;
  border-radius:11px;
  font-weight:700;
  padding:.62rem 1rem;
  background:
    linear-gradient(
      135deg,
      #ff2e88,
      #ff6fae
    );
  color:#fff;
  transition:
    transform .12s ease,
    box-shadow .12s ease,
    filter .12s ease;
}

.stButton>button:hover,
.stDownloadButton>button:hover{
  filter:brightness(1.06);
  transform:translateY(-1px);
  box-shadow:0 8px 22px rgba(255,46,136,.28);
  color:#fff;
}

.stButton>button:focus-visible,
.stDownloadButton>button:focus-visible{
  outline:3px solid var(--vbcua-accent-soft);
  outline-offset:2px;
}

.stButton>button:disabled{
  filter:grayscale(.6) brightness(.8);
  cursor:not-allowed;
}


/* -----------------------------------------------------------------------
   Native Streamlit metrics
   ----------------------------------------------------------------------- */

div[data-testid="stMetric"],
div[data-testid="metric-container"]{
  background:var(--vbcua-panel);
  border:1px solid var(--vbcua-border);
  border-radius:13px;
  padding:.8rem .9rem;
}


/* -----------------------------------------------------------------------
   Text inputs
   ----------------------------------------------------------------------- */

.stTextArea textarea,
.stTextInput input{
  background:rgba(255,255,255,.05) !important;
  color:var(--vbcua-text) !important;
  border:1px solid var(--vbcua-border) !important;
  border-radius:11px !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus{
  border-color:var(--vbcua-accent) !important;
  box-shadow:
    0 0 0 2px rgba(255,46,136,.22) !important;
}


/* -----------------------------------------------------------------------
   File uploader
   ----------------------------------------------------------------------- */

section[data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.04);
  border:1.5px dashed var(--vbcua-border);
  border-radius:13px;
}


/* -----------------------------------------------------------------------
   Expander
   ----------------------------------------------------------------------- */

div[data-testid="stExpander"]{
  border:1px solid var(--vbcua-border);
  border-radius:12px;
  background:var(--vbcua-panel);
}


/* -----------------------------------------------------------------------
   Progress
   ----------------------------------------------------------------------- */

.stProgress > div > div > div > div{
  background:
    linear-gradient(
      90deg,
      #ff2e88,
      #ff9ac8
    );
}

hr{
  border-color:var(--vbcua-border);
}


/* -----------------------------------------------------------------------
   Responsive
   ----------------------------------------------------------------------- */

@media (max-width:640px){

  .block-container{
    padding-left:1rem;
    padding-right:1rem;
  }

  .vbcua-metric .value{
    font-size:1.45rem;
  }

  .vbcua-hero{
    padding:1.15rem 1.15rem;
  }

}

@media (prefers-reduced-motion: reduce){

  .stButton>button,
  .stDownloadButton>button{
    transition:none;
  }

  .stButton>button:hover,
  .stDownloadButton>button:hover{
    transform:none;
  }

}
</style>
"""


# --------------------------------------------------------------------------
# General helpers
# --------------------------------------------------------------------------

def inject_theme() -> None:
    """Apply the application stylesheet once per script run."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def _escape(text: Any) -> str:
    """Safely escape text before inserting it into HTML."""
    import html

    return html.escape("" if text is None else str(text))


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------

def hero(title: str, subtitle: str) -> None:
    """Render the page hero section."""
    st.markdown(
        f"""
        <div class="vbcua-hero">
            <h1>{_escape(title)}</h1>
            <p>{_escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metric_card(
    label: str,
    value: str,
    hint: str = "",
) -> None:
    """
    Render a metric using native Streamlit components.

    This intentionally avoids raw closing </div> tags so that Streamlit
    cannot display HTML markup as visible page content.
    """

    if hint:
        st.metric(
            label=label,
            value=value,
            help=hint,
        )
    else:
        st.metric(
            label=label,
            value=value,
        )


# --------------------------------------------------------------------------
# Information cards
# --------------------------------------------------------------------------

def info_card(title: str, body: str) -> None:
    """Render an informational card."""
    st.markdown(
        f"""
        <div class="vbcua-card">
            <h4>{_escape(title)}</h4>
            <p>{_escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str) -> None:
    """Render an empty-state message."""
    st.markdown(
        f"""
        <div class="vbcua-empty">
            <h3>{_escape(title)}</h3>
            <p>{_escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Grade
# --------------------------------------------------------------------------

def grade_pill(grade: str) -> str:
    """Return HTML for a grade pill."""
    tone = (
        "ok"
        if grade in ("A+", "A")
        else "warn"
        if grade in ("B", "C")
        else "bad"
    )

    return (
        f"<span class='vbcua-pill {tone}'>"
        f"Grade {_escape(grade)}"
        f"</span>"
    )


# --------------------------------------------------------------------------
# Notices
# --------------------------------------------------------------------------

def notices(messages: Sequence[str]) -> None:
    """Render pipeline notices compactly."""
    messages = [str(message) for message in messages if message]

    if not messages:
        return

    with st.expander(
        f"Analysis notes ({len(messages)})",
        expanded=False,
    ):
        for message in messages:
            st.markdown(f"- {message}")


# ==========================================================================
# Charts
# ==========================================================================

def _figure(width: float, height: float):
    """Create a transparent Matplotlib figure matching the application theme."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(width, height)
    )

    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#3b4761")

    ax.tick_params(
        colors="#9aa7bd",
        labelsize=8,
    )

    ax.yaxis.label.set_color("#9aa7bd")
    ax.xaxis.label.set_color("#9aa7bd")
    ax.title.set_color("#e8ecf5")

    return fig, ax


def waveform_chart(
    waveform: Sequence[float],
    duration: Optional[float] = None,
) -> None:
    """Plot the real recorded waveform."""

    import matplotlib.pyplot as plt
    import numpy as np

    if not waveform:
        st.info(
            "Waveform is unavailable for this recording."
        )
        return

    values = np.asarray(
        waveform,
        dtype=float,
    )

    if duration and duration > 0:
        x_axis = np.linspace(
            0,
            duration,
            values.size,
        )
    else:
        x_axis = np.arange(
            values.size
        )

    fig, ax = _figure(
        10,
        2.2,
    )

    ax.fill_between(
        x_axis,
        values,
        0,
        color="#ff2e88",
        alpha=0.35,
        linewidth=0,
    )

    ax.plot(
        x_axis,
        values,
        color="#ff7ab8",
        linewidth=0.7,
    )

    ax.set_xlabel(
        "Seconds"
        if duration
        else "Sample window"
    )

    ax.set_ylabel("Amplitude")

    ax.set_ylim(
        -1.05,
        1.05,
    )

    ax.margins(x=0)

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)


def score_breakdown_chart(
    components: Dict[str, float],
) -> None:
    """Horizontal bar chart of individual score components."""

    import matplotlib.pyplot as plt

    filtered = {
        key: float(value)
        for key, value in components.items()
        if isinstance(value, (int, float))
        and value is not None
    }

    if not filtered:
        st.info(
            "No component scores are available for this evaluation."
        )
        return

    labels = list(filtered.keys())
    values = [
        filtered[label]
        for label in labels
    ]

    fig, ax = _figure(
        8,
        max(
            1.6,
            0.52 * len(labels) + 0.8,
        ),
    )

    bars = ax.barh(
        labels,
        values,
        color="#ff2e88",
        alpha=0.85,
        height=0.55,
    )

    ax.set_xlim(
        0,
        100,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Score (0-100)"
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            min(value + 2, 96),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.0f}",
            va="center",
            fontsize=8,
            color="#e8ecf5",
        )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)


def trend_chart(
    rows: Sequence[Dict[str, Any]],
) -> None:
    """Score / similarity trend over stored evaluation history."""

    import matplotlib.pyplot as plt

    points = [
        row
        for row in rows
        if row.get("score") is not None
    ]

    if len(points) < 2:
        st.info(
            "At least two saved evaluations are needed "
            "to plot a trend."
        )
        return

    x_axis = list(
        range(
            1,
            len(points) + 1,
        )
    )

    scores = [
        float(row["score"])
        for row in points
    ]

    similarities = [
        (
            float(row["similarity"])
            if row.get("similarity") is not None
            else float(row["score"])
        )
        for row in points
    ]

    fig, ax = _figure(
        9,
        2.8,
    )

    ax.plot(
        x_axis,
        scores,
        marker="o",
        markersize=4,
        color="#ff2e88",
        linewidth=1.8,
        label="Score",
    )

    ax.plot(
        x_axis,
        similarities,
        marker="s",
        markersize=3.5,
        color="#38bdf8",
        linewidth=1.4,
        linestyle="--",
        label="Similarity",
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xlabel(
        "Evaluation (oldest to newest)"
    )

    ax.set_ylabel("Value")

    legend = ax.legend(
        frameon=False,
        fontsize=8,
    )

    for text in legend.get_texts():
        text.set_color("#e8ecf5")

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)


# ==========================================================================
# Result rendering
# ==========================================================================

def _value(
    result: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a value from either a dictionary or an object."""
    if isinstance(result, dict):
        return result.get(
            key,
            default,
        )

    return getattr(
        result,
        key,
        default,
    )


def score_summary(result: Any) -> None:
    """Render the four main evaluation metrics."""

    score = float(
        _value(
            result,
            "score",
            0,
        )
        or 0
    )

    similarity = float(
        _value(
            result,
            "similarity",
            0,
        )
        or 0
    )

    fluency = float(
        _value(
            result,
            "fluency_score",
            0,
        )
        or 0
    )

    grade = str(
        _value(
            result,
            "grade",
            "N/A",
        )
        or "N/A"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Overall score",
            f"{score:.0f}/100",
            "Similarity + fluency",
        )

    with col2:
        metric_card(
            "Similarity",
            f"{similarity:.0f}%",
            "Against expected answer",
        )

    with col3:
        metric_card(
            "Fluency",
            f"{fluency:.0f}/100",
            "Delivery quality",
        )

    with col4:
        metric_card(
            "Grade",
            grade,
            _grade_hint(grade),
        )


def _grade_hint(grade: str) -> str:
    """Return explanatory text for a grade."""

    return {
        "A+": "Excellent understanding",
        "A": "Strong understanding",
        "B": "Good, with gaps",
        "C": "Partial understanding",
        "D": "Weak understanding",
        "F": "Concept needs revision",
    }.get(
        grade,
        "",
    )


def bullet_columns(result: Any) -> None:
    """Render strengths, weaknesses and suggestions."""

    strengths = list(
        _value(
            result,
            "strengths",
            [],
        )
        or []
    )

    weaknesses = list(
        _value(
            result,
            "weaknesses",
            [],
        )
        or []
    )

    suggestions = list(
        _value(
            result,
            "suggestions",
            [],
        )
        or []
    )

    if not (
        strengths
        or weaknesses
        or suggestions
    ):
        return

    col1, col2, col3 = st.columns(3)

    for column, title, items in (
        (
            col1,
            "Strengths",
            strengths,
        ),
        (
            col2,
            "Weaknesses",
            weaknesses,
        ),
        (
            col3,
            "Suggestions",
            suggestions,
        ),
    ):
        with column:

            st.markdown(
                f"#### {title}"
            )

            if items:

                for item in items:
                    st.markdown(
                        f"- {item}"
                    )

            else:
                st.caption(
                    "Nothing recorded."
                )


def render_result_details(
    result: Any,
    show_transcript: bool = True,
) -> None:
    """
    Full read-only rendering of an evaluation,
    shared by several pages.
    """

    score_summary(result)

    summary = _value(
        result,
        "summary",
        "",
    )

    if summary:
        st.markdown("")

        st.info(summary)

    bullet_columns(result)

    if show_transcript:

        st.markdown(
            "### Transcript"
        )

        transcript = (
            _value(
                result,
                "transcript",
                "",
            )
            or "No transcript available."
        )

        st.markdown(
            f"""
            <div class="vbcua-card">
                <p>{_escape(transcript)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        reference = _value(
            result,
            "reference_answer",
            "",
        )

        if reference:

            with st.expander(
                "Expected answer used for comparison"
            ):
                st.write(reference)

    feedback = _value(
        result,
        "feedback",
        "",
    )

    if feedback and not (
        _value(result, "strengths")
        or _value(result, "weaknesses")
        or _value(result, "suggestions")
    ):

        st.markdown(
            "### Feedback"
        )

        st.write(feedback)


# ==========================================================================
# Public API
# ==========================================================================

__all__ = [
    "inject_theme",
    "hero",
    "metric_card",
    "info_card",
    "empty_state",
    "grade_pill",
    "notices",
    "waveform_chart",
    "score_breakdown_chart",
    "trend_chart",
    "score_summary",
    "bullet_columns",
    "render_result_details",
]