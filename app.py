"""
Voice Based Concept Understanding Analyser
==========================================

Streamlit entry point.

Developed by:
    1. Ishak Baba Shaik
    2. Vijay Kumar Nangana
    3. Pavani Lakshmi Gonthina

Developed as part of the APSCHE Internship 2026 - Google Cloud Generative AI.

Run with:  streamlit run "Source Code/app.py"
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

# Allow ``streamlit run`` from any working directory.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st  # noqa: E402

import dashboard  # noqa: E402
import results as results_page  # noqa: E402
import transcription  # noqa: E402
import ui  # noqa: E402
import upload as upload_page  # noqa: E402
import tutor  # noqa: E402
import realtime_mode
from analysis import AnalysisError, analyse  # noqa: E402
from config import load_settings  # noqa: E402
from database import get_database  # noqa: E402
from report import generate_pdf_report  # noqa: E402

# --------------------------------------------------------------------------
# Logging (useful developer output; users never see raw tracebacks)
# --------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("vbcua")

# --------------------------------------------------------------------------
# Page config must be the first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Voice Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Voice Based Concept Understanding Analyser - evaluates spoken explanations "
            "using Whisper, sentence embeddings and optional Gemini feedback. "
            "Runs entirely on free and open-source components."
        )
    },
)

ui.inject_theme()


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _settings():
    return load_settings()


@st.cache_resource(show_spinner=False)
def _database(db_path: str):
    return get_database(db_path)


settings = _settings()
try:
    database = _database(str(settings.database_path))
    database_error = None
except Exception as exc:  # noqa: BLE001
    logger.exception("Database initialisation failed")
    database = None
    database_error = str(exc)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
DEFAULT_STATE = {
    "audio": None,
    "audio_name": "",
    "audio_hash": "",
    "reference_answer": "",
    "rag_document_id": None,
    "rag_document_name": "",
    "rag_query": "",
    "rag_context": "",
    "rag_sources": [],
    "result": None,
    "tutor_mode": False,
    "tutor_concepts": [],
    "tutor_scores": [],
    "tutor_current": None,
    "tutor_question": "",
    "tutor_round": 0,
    "tutor_history": [],
    "tutor_answer_audio": None,
    "tutor_answer_hash": "",
    "tutor_answer_text": "",
}
for _key, _value in DEFAULT_STATE.items():
    st.session_state.setdefault(_key, _value)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def _status_pill(label: str, ok: bool, detail: str = "") -> str:
    tone = "ok" if ok else "warn"
    text = f"{label}: {detail}" if detail else label
    return f"<span class='vbcua-pill {tone}'>{text}</span>"


def render_sidebar() -> str:
    st.sidebar.markdown("## 🎙 VBCUA")
    st.sidebar.caption("Voice Based Concept Understanding Analyser")

    page = st.sidebar.radio(
        "Navigation",
        [
    "🏠 Home",
    "🎤 Upload Audio",
    "🧠 AI Analysis",
    "🧑‍🏫 Tutor Mode",
    "🔴 Real-time Mode",
    "📊 Dashboard",
    "📄 Report",
    "🗂 History",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("#### System status")

    whisper_ok = transcription.whisper_available()
    ffmpeg_ok = transcription.ffmpeg_available()

    st.sidebar.markdown(
        "".join(
            [
                _status_pill("Whisper", whisper_ok, settings.whisper_model if whisper_ok else "not installed"),
                _status_pill("ffmpeg", ffmpeg_ok, "installed" if ffmpeg_ok else "missing"),
                _status_pill(
                    "Gemini",
                    settings.gemini_enabled,
                    "connected" if settings.gemini_enabled else "offline evaluator",
                ),
                _status_pill("Database", database is not None, "ready" if database else "error"),
            ]
        ),
        unsafe_allow_html=True,
    )

    if not whisper_ok:
        st.sidebar.error(
            "openai-whisper is not installed. Run `pip install -r requirements.txt` before analysing audio."
        )
    if not ffmpeg_ok:
        st.sidebar.warning(
            "ffmpeg is not on PATH. Only .wav uploads can be decoded until it is installed."
        )
    if not settings.gemini_enabled:
        st.sidebar.info(
            "Running without Gemini. Scoring and feedback are produced locally at zero cost."
        )
    elif database is not None:
        try:
            used_today = database.count_api_calls("gemini", within_seconds=86400)
            st.sidebar.caption(
                f"Gemini calls today: {used_today}/{settings.gemini_calls_per_day} (self-imposed cap)"
            )
        except Exception:  # noqa: BLE001
            logger.debug("Could not read API usage", exc_info=True)

    st.sidebar.divider()
    tutor_enabled = st.sidebar.toggle(
        "🧑‍🏫 Tutor Mode",
        value=bool(st.session_state.get("tutor_mode", False)),
        help="Turn the Socratic follow-up tutor on or off. Your normal evaluator remains unchanged when off.",
    )
    if tutor_enabled != st.session_state.get("tutor_mode", False):
        st.session_state.tutor_mode = tutor_enabled
        if not tutor_enabled:
            st.session_state.tutor_question = ""
            st.session_state.tutor_current = None
        st.rerun()
    st.sidebar.caption("APSCHE Internship 2026 · Google Cloud Generative AI")
    return page


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home() -> None:
    ui.hero(
        "Voice Based Concept Understanding Analyser",
        "Assess how well a student understands a concept from the way they explain it out loud.",
    )

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        ui.info_card(
            "Speech to text",
            "OpenAI Whisper transcribes the recording locally. No audio ever leaves the machine.",
        )
    with col2:
        ui.info_card(
            "Semantic scoring",
            "Sentence embeddings compare meaning, not keywords, against the expected answer.",
        )
    with col3:
        ui.info_card(
            "Delivery analysis",
            "Librosa measures pauses, energy and pace; filler words are counted from the transcript.",
        )

    st.markdown("")
    col4, col5, col6 = st.columns(3, gap="medium")
    with col4:
        ui.info_card(
            "AI feedback",
            "Gemini adds qualitative strengths and suggestions when a free API key is configured.",
        )
    with col5:
        ui.info_card(
            "History and analytics",
            "Every evaluation is stored in a local SQLite database with trend charts.",
        )
    with col6:
        ui.info_card(
            "PDF reports",
            "Download a formatted evaluation report for any current or past analysis.",
        )

    st.markdown("### How it works")
    st.markdown(
        """
1. **Upload Audio** - add the student's recording and the expected answer.
2. **AI Analysis** - the recording is transcribed, measured and compared.
3. **Dashboard** - inspect the score breakdown, waveform and delivery metrics.
4. **Report** - download a PDF summary.
5. **History** - revisit, export or delete earlier evaluations.
        """
    )

    st.markdown("### Scoring model")
    st.markdown(
        f"""
The overall score is a weighted combination of two measured components:

- **Semantic similarity ({settings.semantic_weight * 100:.0f}%)** - how closely the meaning of the
  spoken answer matches the expected answer.
- **Fluency ({settings.fluency_weight * 100:.0f}%)** - filler-word rate, speaking pace, pause ratio
  and energy consistency.

Grades: A+ ≥ 90, A ≥ 80, B ≥ 70, C ≥ 60, D ≥ 50, otherwise F.
        """
    )

    if database_error:
        st.error(
            f"The results database could not be opened ({database_error}). "
            "Analyses will still run, but nothing will be saved."
        )


PROGRESS_STEPS = {
    "Analysing audio signal...": 15,
    "Converting speech to text with Whisper...": 45,
    "Measuring semantic similarity...": 70,
    "Building RAG-grounded concept map with Gemini...": 78,
    "Requesting AI feedback from Gemini...": 88,
    "Saving results...": 95,
    "Done.": 100,
}


def page_analysis() -> None:
    ui.hero("AI analysis", "Transcribe, measure and evaluate the saved response.")

    if not st.session_state.get("audio"):
        ui.empty_state("No audio saved", "Upload a recording on the Upload Audio page first.")
        return
    if not (st.session_state.get("reference_answer") or "").strip() and not st.session_state.get("rag_context"):
        ui.empty_state(
            "No evaluation source", "Enter an expected answer or retrieve study material from a PDF first."
        )
        return

    col_info, col_run = st.columns([3, 1])
    with col_info:
        st.markdown(
            f"**Recording:** {st.session_state.get('audio_name', 'recording')}  \n"
            f"**Expected answer:** {len(st.session_state.get('reference_answer', '').split())} words"
        )
        if st.session_state.get("rag_context"):
            st.success(
                f"RAG grounding active: {st.session_state.get('rag_document_name', 'study material')} "
                f"→ {st.session_state.get('rag_query', 'retrieved concept')}"
            )
            with st.expander("View retrieved study material", expanded=False):
                st.write(st.session_state["rag_context"])
    with col_run:
        run = st.button("Run analysis", type="primary", width="stretch")

    if st.session_state.get("result") is not None and not run:
        st.success("An evaluation is already available for this input.")
        ui.render_result_details(st.session_state["result"])
        ui.notices(getattr(st.session_state["result"], "notices", []))
        return

    if not run:
        st.info(
            "The first run downloads the Whisper and embedding models (a few hundred MB) and may "
            "take a couple of minutes. Subsequent runs are much faster."
        )
        return

    status = st.status("Starting analysis...", expanded=True)
    progress_bar = st.progress(0)

    def progress(message: str) -> None:
        status.update(label=message)
        status.write(message)
        progress_bar.progress(PROGRESS_STEPS.get(message, 50))

    try:
        result = analyse(
            st.session_state["audio"],
            st.session_state.get("audio_name", "recording.wav"),
            st.session_state.get("reference_answer", ""),
            rag_context=st.session_state.get("rag_context", ""),
            rag_query=st.session_state.get("rag_query", ""),
            settings=settings,
            database=database,
            progress=progress,
        )
    except AnalysisError as exc:
        progress_bar.empty()
        status.update(label="Analysis stopped", state="error")
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected analysis failure")
        progress_bar.empty()
        status.update(label="Analysis failed", state="error")
        st.error(
            "The analysis could not be completed because of an unexpected error. "
            f"Details: {exc}"
        )
        return

    progress_bar.progress(100)
    status.update(label="Analysis complete", state="complete", expanded=False)
    st.session_state.result = result

    st.success("Analysis completed successfully.")
    ui.render_result_details(result)
    ui.notices(result.notices)

    st.markdown("")
    st.info("Open the **Dashboard** for charts, or the **Report** page to download the PDF.")


def page_tutor() -> None:
    ui.hero("Tutor Mode", "A grounded Socratic loop that targets the concepts your explanation needs to demonstrate.")
    if not st.session_state.get("tutor_mode"):
        st.info("Tutor Mode is OFF. Turn it on from the sidebar to start targeted follow-up questions.")
        return
    result = st.session_state.get("result")
    if result is None:
        st.warning("Run the normal AI Analysis first. Tutor Mode uses that transcript and its concept map.")
        return
    if not getattr(result, "concepts", None):
        st.warning("No concept map is available yet. In Tutor Mode, use a PDF + retrieval query, then run AI Analysis.")
        return

    if not st.session_state.get("tutor_concepts"):
        st.session_state.tutor_concepts = list(result.concepts)
        st.session_state.tutor_scores = tutor.score_concepts(
            st.session_state.tutor_concepts, result.transcript, settings.embedding_model
        )
        st.session_state.tutor_round = 0
        st.session_state.tutor_history = []

    scores = st.session_state.tutor_scores
    st.markdown("### Concept mastery")
    for item in scores:
        score = float(item.get("score", 0))
        st.progress(max(0.0, min(1.0, score / 100.0)), text=f"{item.get('name', 'Concept')} · {score:.0f}/100")

    if st.session_state.tutor_round >= 3:
        st.success("Tutor session complete — 3 targeted rounds reached.")
        return

    if st.session_state.tutor_current is None:
        asked = {str(x.get("concept", "")) for x in st.session_state.tutor_history}
        current = tutor.weakest_concept(scores, asked)
        if current is None:
            st.success("All extracted concepts have been covered.")
            return
        st.session_state.tutor_current = current
        st.session_state.tutor_question = tutor.generate_question(
            concept=current,
            study_context=st.session_state.get("rag_context", ""),
            settings=settings, database=database,
        )

    current = st.session_state.tutor_current
    st.markdown(f"### Round {st.session_state.tutor_round + 1}/3")
    st.info(st.session_state.tutor_question)
    followup = st.audio_input("Answer the tutor by voice", key=f"tutor_audio_{st.session_state.tutor_round}")
    if followup is not None and st.button("Evaluate follow-up", type="primary"):
        audio_bytes = followup.getvalue()
        try:
            from database import sha256_bytes
            from transcription import transcribe_bytes, TranscriptionError
            audio_hash = sha256_bytes(audio_bytes)
            tr = transcribe_bytes(audio_bytes, "tutor_followup.wav", model_name=settings.whisper_model, database=database, audio_hash=audio_hash)
            answer = tr.text.strip()
            if not answer:
                st.error("Whisper produced an empty follow-up transcript.")
                return
            evaluation = tutor.evaluate_followup(
                concept=current, answer=answer,
                study_context=st.session_state.get("rag_context", ""),
                settings=settings, database=database,
            )
            local = tutor.score_concepts([current], answer, settings.embedding_model)[0]
            combined_score = float(evaluation.get("score", local["score"]))
            st.session_state.tutor_history.append({
                "concept": current.get("name", "Concept"),
                "question": st.session_state.tutor_question,
                "answer": answer,
                "score": combined_score,
                "feedback": evaluation.get("feedback", ""),
            })
            st.success(f"Follow-up score: {combined_score:.0f}/100")
            st.write(evaluation.get("feedback", ""))
            if evaluation.get("missing"):
                st.caption("Still missing: " + ", ".join(map(str, evaluation["missing"])))
            st.session_state.tutor_round += 1
            st.session_state.tutor_current = None
            st.session_state.tutor_question = ""
            st.rerun()
        except TranscriptionError as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Tutor follow-up failed")
            st.error(f"Tutor follow-up could not be evaluated: {exc}")

    if st.session_state.tutor_history:
        with st.expander("Tutor session history", expanded=False):
            for i, item in enumerate(st.session_state.tutor_history, 1):
                st.markdown(f"**Round {i}: {item['concept']} — {float(item['score']):.0f}/100**")
                st.write(item["answer"])
                if item.get("feedback"):
                    st.caption(item["feedback"])


def page_report() -> None:
    ui.hero("Evaluation report", "Review the current evaluation and download it as a PDF.")

    result = st.session_state.get("result")
    if result is None:
        ui.empty_state(
            "No evaluation to report",
            "Run an analysis first, or open the History page to export a past evaluation.",
        )
        return

    ui.render_result_details(result)

    st.divider()
    try:
        pdf_bytes = generate_pdf_report(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF generation failed")
        st.error(f"The PDF report could not be generated: {exc}")
        return

    safe_name = (
        "".join(
            ch
            for ch in Path(getattr(result, "audio_name", "report") or "report").stem
            if ch.isalnum() or ch in "-_"
        )
        or "report"
    )

    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"VBCUA_{safe_name}.pdf",
        mime="application/pdf",
        width="stretch",
    )
    st.caption(f"Report size: {len(pdf_bytes) / 1024:.0f} KB")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def main() -> None:
    page = render_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "🎤 Upload Audio":
        upload_page.render(settings, database)
    elif page == "🧠 AI Analysis":
        page_analysis()
    elif page == "🧑‍🏫 Tutor Mode":
        page_tutor()
    elif page == "🔴 Real-time Mode":
         realtime_mode.render_realtime_panel()
    elif page == "📊 Dashboard":
        if database is None:
            st.error("The dashboard needs the results database, which could not be opened.")
        else:
            dashboard.render(settings, database)
    elif page == "📄 Report":
        page_report()
    elif page == "🗂 History":
        if database is None:
            st.error("History needs the results database, which could not be opened.")
        else:
            results_page.render(settings, database)


main()
