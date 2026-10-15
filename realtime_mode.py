
"""
VBCUA Phase 5 — Real-time Live Coach
====================================

Architecture
------------

PDF / RAG context
        +
Real-time browser coach
        |
        |-- live speech preview
        |-- word count
        |-- approximate WPM
        |-- filler count
        |-- elapsed time
        |
        v
Authoritative recording
        |
        v
Whisper
        |
        v
Existing analyse()
        |
        +-------------------+
        |                   |
        v                   v
       RAG                Gemini
        |                   |
        +---------+---------+
                  |
                  v
             Final Result


IMPORTANT
---------

Browser SpeechRecognition is NEVER authoritative.

The browser transcript is only a live coaching preview.

The actual submitted audio is processed by Whisper, and Whisper's
transcript is used by the existing VBCUA analyse() pipeline.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


# ============================================================================
# Configuration
# ============================================================================

FILLERS = [
    "um",
    "uh",
    "erm",
    "hmm",
    "like",
    "you know",
]


# ============================================================================
# Session helpers
# ============================================================================

def _session_value(key: str, default: Any = None) -> Any:
    """Safely read a value from Streamlit session state."""
    return st.session_state.get(key, default)


def _reset_realtime_state() -> None:
    """
    Remove the previous Phase 5 result.

    This does NOT delete the normal VBCUA result.
    """
    keys = (
        "realtime_result",
        "realtime_transcript",
        "realtime_audio",
        "realtime_submitted",
    )

    for key in keys:
        st.session_state.pop(key, None)


def _get_app_dependencies():
    """
    Get the same Settings and Database objects used by app.py.

    We import app lazily to avoid circular-import problems during startup.
    """

    settings = _session_value("_vbcua_settings")
    database = _session_value("_vbcua_database")

    if settings is not None and database is not None:
        return settings, database

    try:
        import app

        return app.settings, app.database

    except Exception as exc:
        raise RuntimeError(
            "Could not access the VBCUA application settings/database. "
            f"Details: {exc}"
        ) from exc


# ============================================================================
# Browser live coach
# ============================================================================

def _render_browser_live_coach() -> None:
    """
    Render the browser-only live speaking coach.

    Features
    --------
    - live browser speech preview
    - word count
    - approximate WPM
    - filler count
    - elapsed time
    - start/stop preview

    IMPORTANT
    ---------
    This transcript is NOT sent to analyse().

    Whisper remains the authoritative final transcription.
    """

    fillers_json = json.dumps(FILLERS)

    components.html(
        f"""
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<style>

body {{
    margin: 0;
    padding: 0;

    font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background: transparent;
    color: #e8ecf5;
}}

.panel {{
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 14px;

    padding: 16px;

    background: rgba(255,255,255,.045);
}}

.status {{
    display: inline-block;

    padding: 6px 10px;

    border-radius: 999px;

    background: rgba(34,197,94,.14);
    color: #86efac;

    font-size: 13px;
    font-weight: 700;

    margin-bottom: 12px;
}}

.transcript {{
    min-height: 90px;

    border: 1px solid rgba(255,255,255,.10);
    border-radius: 11px;

    padding: 13px;

    background: rgba(0,0,0,.14);

    line-height: 1.55;

    color: #e8ecf5;

    overflow-wrap: anywhere;
}}

.metrics {{
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 10px;

    margin-top: 12px;
}}

.metric {{
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 10px;

    padding: 10px;

    text-align: center;

    background: rgba(255,255,255,.035);
}}

.metric .number {{
    display: block;

    font-size: 20px;
    font-weight: 800;
}}

.metric .label {{
    display: block;

    margin-top: 3px;

    font-size: 11px;

    opacity: .65;
}}

.controls {{
    display: flex;

    gap: 9px;

    margin-top: 13px;
}}

button {{
    border: 0;

    border-radius: 9px;

    padding: 9px 14px;

    font-weight: 700;

    cursor: pointer;
}}

.start {{
    background: #ff2e88;
    color: white;
}}

.stop {{
    background: #374151;
    color: white;
}}

.reset {{
    background: #4b5563;
    color: white;
}}

button:disabled {{
    opacity: .45;
    cursor: not-allowed;
}}

.note {{
    margin-top: 10px;

    font-size: 12px;

    opacity: .62;
}}

.warning {{
    margin-top: 10px;

    padding: 9px 11px;

    border-radius: 9px;

    background: rgba(245,158,11,.10);

    color: #fcd34d;

    font-size: 12px;
}}

@media(max-width:650px) {{

    .metrics {{
        grid-template-columns:
            repeat(2, 1fr);
    }}

    .controls {{
        flex-wrap: wrap;
    }}
}}

</style>

</head>


<body>

<div class="panel">

    <div id="status" class="status">
        READY
    </div>


    <div id="transcript" class="transcript">

        Press <b>Start live coach</b>
        and speak normally.

        <br><br>

        Browser speech is only a live preview.

    </div>


    <div class="metrics">

        <div class="metric">

            <span
                id="words"
                class="number"
            >
                0
            </span>

            <span class="label">
                WORDS
            </span>

        </div>


        <div class="metric">

            <span
                id="wpm"
                class="number"
            >
                0
            </span>

            <span class="label">
                WPM
            </span>

        </div>


        <div class="metric">

            <span
                id="fillers"
                class="number"
            >
                0
            </span>

            <span class="label">
                FILLERS
            </span>

        </div>


        <div class="metric">

            <span
                id="timer"
                class="number"
            >
                0s
            </span>

            <span class="label">
                TIME
            </span>

        </div>

    </div>


    <div class="controls">

        <button
            id="start"
            class="start"
        >
            Start live coach
        </button>


        <button
            id="stop"
            class="stop"
            disabled
        >
            Stop preview
        </button>


        <button
            id="reset"
            class="reset"
        >
            Reset preview
        </button>

    </div>


    <div class="note">

        Live preview:
        browser SpeechRecognition

        ·

        Final answer:
        Whisper

    </div>


    <div class="warning">

        ⚠️ The browser transcript is for coaching only.
        It is never used as the final VBCUA answer.

    </div>

</div>


<script>

const FILLERS = {fillers_json};


let recognition = null;

let running = false;

let startTime = 0;

let transcript = "";


const $ = id =>
    document.getElementById(id);


// -------------------------------------------------------------------------
// Filler counting
// -------------------------------------------------------------------------

function countFillers(text) {{

    const words =
        text
            .toLowerCase()
            .match(/[a-z']+/g) || [];

    return words.filter(
        word => FILLERS.includes(word)
    ).length;
}}


// -------------------------------------------------------------------------
// Metrics
// -------------------------------------------------------------------------

function updateMetrics() {{

    const seconds =
        startTime
            ? (Date.now() - startTime) / 1000
            : 0;


    const words =
        (transcript.match(/\\S+/g) || [])
            .length;


    const wpm =
        seconds > 2
            ? Math.round(
                words / (seconds / 60)
            )
            : 0;


    $("words").textContent =
        words;


    $("wpm").textContent =
        wpm;


    $("fillers").textContent =
        countFillers(transcript);


    $("timer").textContent =
        Math.round(seconds) + "s";
}}


// -------------------------------------------------------------------------
// Status
// -------------------------------------------------------------------------

function setStatus(
    text,
    background,
    color
) {{

    $("status").textContent =
        text;

    $("status").style.background =
        background;

    $("status").style.color =
        color;
}}


// -------------------------------------------------------------------------
// Browser SpeechRecognition
// -------------------------------------------------------------------------

const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;


if (!SpeechRecognition) {{

    setStatus(
        "PREVIEW UNAVAILABLE",
        "rgba(245,158,11,.14)",
        "#fcd34d"
    );


    $("transcript").innerHTML =
        "This browser does not provide " +
        "SpeechRecognition.<br><br>" +

        "The live preview is unavailable, " +

        "but <b>Whisper final transcription " +
        "still works</b>.";


    $("start").disabled = true;

}} else {{

    recognition =
        new SpeechRecognition();


    recognition.continuous =
        true;


    recognition.interimResults =
        true;


    recognition.lang =
        "en-US";


    recognition.onstart = () => {{

        setStatus(
            "LISTENING",
            "rgba(34,197,94,.14)",
            "#86efac"
        );

    }};


    recognition.onresult =
        event => {{

            let interim = "";


            for (
                let i = event.resultIndex;
                i < event.results.length;
                i++
            ) {{

                const text =
                    event
                        .results[i][0]
                        .transcript;


                if (
                    event
                        .results[i]
                        .isFinal
                ) {{

                    transcript +=
                        " " + text;

                }} else {{

                    interim +=
                        " " + text;
                }}
            }}


            $("transcript").textContent =
                (
                    transcript +
                    " " +
                    interim
                ).trim()
                || "Listening...";


            updateMetrics();
        }};


    recognition.onerror =
        event => {{

            setStatus(
                "PREVIEW ERROR",
                "rgba(245,158,11,.14)",
                "#fcd34d"
            );


            $("transcript").textContent =
                "Browser preview error: " +
                event.error +
                ". Whisper final transcription " +
                "is unaffected.";
        }};


    recognition.onend = () => {{

        if (running) {{

            try {{

                recognition.start();

            }} catch (error) {{

                // Browser may already be restarting.

            }}

        }} else {{

            setStatus(
                "PREVIEW STOPPED",
                "rgba(107,114,128,.18)",
                "#d1d5db"
            );
        }}
    }};
}}


// -------------------------------------------------------------------------
// Start
// -------------------------------------------------------------------------

$("start").onclick = () => {{

    if (!recognition) {{
        return;
    }}


    transcript = "";

    startTime =
        Date.now();

    running = true;


    $("start").disabled =
        true;


    $("stop").disabled =
        false;


    $("transcript").textContent =
        "Listening...";


    updateMetrics();


    try {{

        recognition.start();

    }} catch (error) {{

        // Recognition may already be running.

    }}
}};


// -------------------------------------------------------------------------
// Stop
// -------------------------------------------------------------------------

$("stop").onclick = () => {{

    running = false;


    $("stop").disabled =
        true;


    $("start").disabled =
        false;


    if (recognition) {{

        try {{

            recognition.stop();

        }} catch (error) {{

            // Already stopped.

        }}
    }}


    setStatus(
        "PREVIEW STOPPED",
        "rgba(107,114,128,.18)",
        "#d1d5db"
    );


    $("transcript").textContent =
        transcript.trim()
        || "No browser preview transcript captured.";
}};


// -------------------------------------------------------------------------
// Reset
// -------------------------------------------------------------------------

$("reset").onclick = () => {{

    running = false;

    transcript = "";

    startTime = 0;


    if (recognition) {{

        try {{
            recognition.stop();
        }} catch (error) {{}}
    }}


    $("start").disabled =
        false;


    $("stop").disabled =
        true;


    setStatus(
        "READY",
        "rgba(34,197,94,.14)",
        "#86efac"
    );


    $("transcript").innerHTML =
        "Press <b>Start live coach</b> " +
        "and speak normally." +
        "<br><br>" +
        "Browser speech is only a live preview.";


    updateMetrics();
}};


// -------------------------------------------------------------------------
// Update timer
// -------------------------------------------------------------------------

setInterval(
    updateMetrics,
    500
);

</script>

</body>

</html>
        """,
        height=365,
        scrolling=False,
    )


# ============================================================================
# RAG context
# ============================================================================

def _show_rag_context() -> None:
    """
    Display the PDF/RAG context already loaded by earlier VBCUA phases.
    """

    rag_context = (
        _session_value(
            "rag_context",
            "",
        )
        or ""
    ).strip()


    if not rag_context:

        st.info(
            "No PDF/RAG context is currently loaded. "
            "You can still record and analyse against "
            "an expected answer."
        )

        return


    document_name = (
        _session_value(
            "rag_document_name",
            "",
        )
        or "Uploaded study material"
    )


    rag_query = (
        _session_value(
            "rag_query",
            "",
        )
        or "Retrieved concept"
    )


    st.success(
        f"RAG grounding active: "
        f"{document_name} → {rag_query}"
    )


    with st.expander(
        "View PDF/RAG context used for final analysis",
        expanded=False,
    ):

        st.write(
            rag_context
        )


# ============================================================================
# Authoritative Whisper + RAG + Gemini analysis
# ============================================================================

def _run_authoritative_analysis(
    audio_bytes: bytes,
    audio_name: str,
    settings: Any,
    database: Any,
) -> None:
    """
    Run the existing VBCUA analyse() pipeline.

    This function deliberately does NOT implement another scoring system.

    Existing pipeline:

        audio
          ↓
        Whisper
          ↓
        transcript
          ↓
        semantic scoring
          ↓
        RAG concept extraction
          ↓
        Gemini feedback
          ↓
        database
          ↓
        final VBCUA result
    """

    from analysis import (
        AnalysisError,
        analyse,
    )


    rag_context = (
        _session_value(
            "rag_context",
            "",
        )
        or ""
    )


    rag_query = (
        _session_value(
            "rag_query",
            "",
        )
        or ""
    )


    reference_answer = (
        _session_value(
            "reference_answer",
            "",
        )
        or ""
    )


    status = st.status(
        "Starting authoritative analysis...",
        expanded=True,
    )


    progress_bar = st.progress(0)


    progress_steps = {

        "Analysing audio signal...":
            15,

        "Converting speech to text with Whisper...":
            35,

        "Measuring semantic similarity...":
            55,

        "Building RAG-grounded concept map with Gemini...":
            72,

        "Requesting AI feedback from Gemini...":
            88,

        "Saving results...":
            96,

        "Done.":
            100,
    }


    def progress(message: str) -> None:

        status.update(
            label=message
        )

        status.write(
            message
        )

        progress_bar.progress(
            progress_steps.get(
                message,
                50,
            )
        )


    try:

        result = analyse(

            audio_bytes,

            audio_name,

            reference_answer,

            rag_context=rag_context,

            rag_query=rag_query,

            settings=settings,

            database=database,

            progress=progress,
        )


    except AnalysisError as exc:

        progress_bar.empty()

        status.update(
            label="Final analysis stopped",
            state="error",
        )

        st.error(
            str(exc)
        )

        return


    except Exception as exc:

        progress_bar.empty()

        status.update(
            label="Final analysis failed",
            state="error",
        )

        st.error(
            "The final Whisper/RAG/Gemini "
            "pipeline failed: "
            f"{exc}"
        )

        return


    progress_bar.progress(
        100
    )


    status.update(
        label="Final analysis complete",
        state="complete",
        expanded=False,
    )


    # ----------------------------------------------------------------------
    # Store Phase 5 result
    # ----------------------------------------------------------------------

    st.session_state.realtime_result = result

    st.session_state.result = result


    # ----------------------------------------------------------------------
    # Store authoritative Whisper transcript
    # ----------------------------------------------------------------------

    transcript = getattr(
        result,
        "transcript",
        "",
    )


    st.session_state.realtime_transcript = (
        transcript
        or ""
    )


    st.success(
        "Final recording processed successfully. "
        "Whisper is the authoritative transcript."
    )


# ============================================================================
# Final result
# ============================================================================

def _render_final_result() -> None:
    """
    Display the final authoritative VBCUA result.
    """

    result = st.session_state.get(
        "realtime_result"
    )


    if result is None:
        return


    st.divider()


    st.markdown(
        "## 4. Final authoritative result"
    )


    st.success(
        "This result uses the Whisper transcription, "
        "not the browser preview transcript."
    )


    # ----------------------------------------------------------------------
    # Whisper transcript
    # ----------------------------------------------------------------------

    transcript = (
        getattr(
            result,
            "transcript",
            "",
        )
        or ""
    ).strip()


    if transcript:

        with st.expander(
            "🎙 Authoritative Whisper transcript",
            expanded=True,
        ):

            st.write(
                transcript
            )


    # ----------------------------------------------------------------------
    # Existing VBCUA renderer
    # ----------------------------------------------------------------------

    try:

        ui = __import__(
            "ui"
        )


        ui.render_result_details(
            result
        )

    except Exception as exc:

        st.error(
            "Could not render the final VBCUA result: "
            f"{exc}"
        )


    # ----------------------------------------------------------------------
    # RAG reminder
    # ----------------------------------------------------------------------

    rag_context = (
        _session_value(
            "rag_context",
            "",
        )
        or ""
    ).strip()


    if rag_context:

        st.markdown(
            "### 📚 RAG grounding"
        )


        document_name = (
            _session_value(
                "rag_document_name",
                "",
            )
            or "Study material"
        )


        rag_query = (
            _session_value(
                "rag_query",
                "",
            )
            or "Retrieved concept"
        )


        st.info(
            "Final evaluation grounded in: "
            f"{document_name} · {rag_query}"
        )


        with st.expander(
            "View RAG context used",
            expanded=False,
        ):

            st.write(
                rag_context
            )


    # ----------------------------------------------------------------------
    # Phase summary
    # ----------------------------------------------------------------------

    st.divider()


    st.markdown(
        "### What happened?"
    )


    st.markdown(
        """
**1. Live Coach**

Browser speech recognition provided a
real-time preview of your speech.

↓

**2. Authoritative Recording**

The complete explanation was captured using
Streamlit's audio recorder.

↓

**3. Whisper**

The submitted audio was transcribed by Whisper.

↓

**4. RAG / PDF**

The previously retrieved study material remained
the grounding context.

↓

**5. Gemini**

The existing Gemini pipeline generated the
concept map and qualitative feedback when Gemini
was available.

↓

**6. VBCUA**

The existing scoring, feedback, concepts and
results were displayed.
        """
    )


# ============================================================================
# Main Phase 5 page
# ============================================================================

def render_realtime_panel() -> None:
    """
    Render the complete Phase 5 Real-time Live Coach.
    """

    import ui


    # ----------------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------------

    ui.hero(
        "Real-time Live Coach",
        (
            "Speak naturally while VBCUA previews your delivery. "
            "Submit the completed recording when you are ready. "
            "Whisper then performs the authoritative transcription "
            "and the existing VBCUA pipeline performs the final analysis."
        ),
    )


    # ----------------------------------------------------------------------
    # Architecture
    # ----------------------------------------------------------------------

    st.markdown(
        "### Phase 5 pipeline"
    )


    c1, c2, c3, c4, c5 = st.columns(
        5
    )


    with c1:

        ui.info_card(
            "📚 PDF / RAG",
            "Uses the study material already loaded from Phase 1/2.",
        )


    with c2:

        ui.info_card(
            "🔴 Live Coach",
            "Browser preview with WPM, fillers and timer.",
        )


    with c3:

        ui.info_card(
            "🎙 Whisper",
            "Final submitted audio is transcribed authoritatively.",
        )


    with c4:

        ui.info_card(
            "🧠 RAG + Gemini",
            "Existing concept mapping and evaluation pipeline.",
        )


    with c5:

        ui.info_card(
            "📊 Final Result",
            "Existing VBCUA score, feedback and concepts.",
        )


    st.divider()


    # ----------------------------------------------------------------------
    # Study context
    # ----------------------------------------------------------------------

    st.markdown(
        "### 1. Study context"
    )


    _show_rag_context()


    # ----------------------------------------------------------------------
    # Live coach
    # ----------------------------------------------------------------------

    st.markdown(
        "### 2. Live speaking coach"
    )


    st.caption(
        "The browser transcript is only a coaching preview. "
        "It is never used as the final answer."
    )


    _render_browser_live_coach()


    # ----------------------------------------------------------------------
    # Authoritative recording
    # ----------------------------------------------------------------------

    st.markdown(
        "### 3. Authoritative recording"
    )


    st.info(
        "Use the recorder below for the actual answer. "
        "After recording, submit it for Whisper + RAG + Gemini analysis."
    )


    recording = st.audio_input(
        "Record your complete explanation",
        key="phase5_authoritative_recording",
    )


    if recording is None:

        st.caption(
            "Waiting for your authoritative recording..."
        )


    else:

        audio_bytes = recording.getvalue()


        st.success(
            "Recording captured · "
            f"{len(audio_bytes) / 1024:.1f} KB"
        )


        st.audio(
            audio_bytes,
            format="audio/wav",
        )


        col_submit, col_clear = st.columns(
            [3, 1]
        )


        with col_submit:

            submit = st.button(
                "🚀 Submit recording for Whisper + RAG + Gemini analysis",
                type="primary",
                width="stretch",
            )


        with col_clear:

            clear = st.button(
                "Clear result",
                width="stretch",
            )


        if clear:

            _reset_realtime_state()

            st.rerun()


        if submit:

            # Clear any previous Phase 5 result before processing.
            _reset_realtime_state()


            st.session_state.realtime_audio = (
                audio_bytes
            )


            st.session_state.realtime_submitted = (
                True
            )


            try:

                settings, database = (
                    _get_app_dependencies()
                )

            except Exception as exc:

                st.error(
                    str(exc)
                )

                return


            _run_authoritative_analysis(

                audio_bytes=audio_bytes,

                audio_name=(
                    "phase5_live_recording.wav"
                ),

                settings=settings,

                database=database,
            )


    # ----------------------------------------------------------------------
    # Final result
    # ----------------------------------------------------------------------

    _render_final_result()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "render_realtime_panel",
]
