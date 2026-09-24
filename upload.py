
"""
Upload page - collects the student's recording and the expected answer.

Audio can be provided either by uploading a file or recording directly
from the microphone. Validation happens here so the analysis page can
assume clean input.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import ui
from config import ALLOWED_AUDIO_TYPES, Settings
from database import sha256_bytes


def render(settings: Settings) -> None:
    ui.hero(
        "Upload or record the student's answer",
        "Provide a spoken explanation and the expected answer it should be compared against.",
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 1. Audio recording")

        # ---------------------------------------------------------
        # Two ways to provide audio:
        #   1. Upload an existing audio file
        #   2. Record directly from the microphone
        # ---------------------------------------------------------
        upload_col, mic_col = st.columns(2)

        with upload_col:
            st.markdown("#### 📁 Upload")
            uploaded_audio = st.file_uploader(
                "Audio file",
                type=ALLOWED_AUDIO_TYPES,
                help=(
                    f"Supported: {', '.join(ALLOWED_AUDIO_TYPES)}. "
                    f"Maximum {settings.max_audio_mb} MB and "
                    f"{settings.max_audio_seconds // 60} minutes."
                ),
                label_visibility="visible",
            )

        with mic_col:
            st.markdown("#### 🎙️ Microphone")
            recorded_audio = st.audio_input(
                "Record from microphone",
                help="Click the microphone button and speak your answer.",
            )

        # ---------------------------------------------------------
        # Give uploaded file priority if both are supplied.
        # Otherwise use microphone recording.
        # ---------------------------------------------------------
        selected_audio = None
        selected_audio_name = None

        if uploaded_audio is not None:
            selected_audio = uploaded_audio.getvalue()
            selected_audio_name = uploaded_audio.name

            # If the user also recorded something, make it clear that
            # the uploaded file is the one being used.
            if recorded_audio is not None:
                st.info(
                    "Both an uploaded file and microphone recording are available. "
                    "The uploaded file will be used."
                )

        elif recorded_audio is not None:
            selected_audio = recorded_audio.getvalue()
            selected_audio_name = "microphone_recording.wav"

        # ---------------------------------------------------------
        # Preview and validate the selected audio.
        # ---------------------------------------------------------
        if selected_audio is not None:
            audio_bytes = selected_audio
            size_mb = len(audio_bytes) / (1024 * 1024)

            st.audio(audio_bytes)

            if size_mb > settings.max_audio_mb:
                st.error(
                    f"This recording is {size_mb:.1f} MB, above the "
                    f"{settings.max_audio_mb} MB limit. "
                    "Please record a shorter answer or upload a smaller file."
                )
            else:
                st.caption(
                    f"{selected_audio_name} - {size_mb:.2f} MB"
                )

        elif st.session_state.get("audio"):
            st.caption(
                f"Currently stored: "
                f"{st.session_state.get('audio_name', 'recording')}"
            )
            st.audio(st.session_state["audio"])

    with col_right:
        st.markdown("### 2. Expected answer")

        expected_answer = st.text_area(
            "Reference answer",
            value=st.session_state.get("reference_answer", ""),
            height=280,
            placeholder=(
                "Example: Machine learning is a branch of artificial intelligence "
                "where a model learns statistical patterns from data rather than "
                "following explicitly programmed rules, and then generalises those "
                "patterns to unseen inputs."
            ),
            help="The richer this answer, the more precise the semantic comparison.",
        )

        words = len(expected_answer.split())
        st.caption(
            f"{words} word(s) - at least one full sentence is recommended."
        )

    st.markdown("")

    save_col, clear_col = st.columns([3, 1])

    with save_col:
        if st.button(
            "Save input",
            type="primary",
            width="stretch"
        ):
            _save(
                selected_audio,
                selected_audio_name,
                expected_answer,
                settings,
            )

    with clear_col:
        if st.button(
            "Clear",
            width="stretch"
        ):
            for key in (
                "audio",
                "audio_name",
                "audio_hash",
                "reference_answer",
                "result",
            ):
                st.session_state[key] = (
                    ""
                    if key.endswith("name") or key == "reference_answer"
                    else None
                )

            st.success("Input cleared.")
            st.rerun()

    if (
        st.session_state.get("audio")
        and st.session_state.get("reference_answer")
    ):
        st.success(
            "Input is ready. Open **AI Analysis** in the sidebar "
            "to evaluate this response."
        )
    else:
        ui.empty_state(
            "Nothing saved yet",
            "Upload an audio file or record from the microphone, "
            "then enter the expected answer and choose Save input.",
        )


def _save(
    audio_bytes: bytes | None,
    audio_name: str | None,
    expected_answer: str,
    settings: Settings,
) -> None:
    """Validate and save either uploaded or microphone audio."""

    if not audio_bytes and not st.session_state.get("audio"):
        st.warning(
            "Please upload an audio file or record an answer "
            "from the microphone first."
        )
        return

    reference = (expected_answer or "").strip()

    if not reference:
        st.warning("Please enter the expected answer.")
        return

    if len(reference) < 15:
        st.warning(
            "The expected answer is too short. "
            "Please write at least one full sentence."
        )
        return

    # -------------------------------------------------------------
    # New audio supplied by either uploader or microphone.
    # -------------------------------------------------------------
    if audio_bytes:
        if not audio_bytes:
            st.error("The audio recording is empty.")
            return

        size_mb = len(audio_bytes) / (1024 * 1024)

        if size_mb > settings.max_audio_mb:
            st.error(
                f"The recording exceeds the "
                f"{settings.max_audio_mb} MB limit."
            )
            return

        # Uploaded files use their real extension.
        # Microphone recordings are WAV from st.audio_input().
        extension = (
            Path(audio_name or "recording.wav")
            .suffix
            .lower()
            .lstrip(".")
        )

        if extension and extension not in ALLOWED_AUDIO_TYPES:
            st.error(
                f"'{extension}' is not a supported audio format."
            )
            return

        new_hash = sha256_bytes(audio_bytes)

        if new_hash != st.session_state.get("audio_hash"):
            # New recording invalidates the previous evaluation.
            st.session_state.result = None

        st.session_state.audio = audio_bytes
        st.session_state.audio_name = audio_name or "recording.wav"
        st.session_state.audio_hash = new_hash

    # -------------------------------------------------------------
    # Reference answer changed -> invalidate old evaluation.
    # -------------------------------------------------------------
    if reference != st.session_state.get("reference_answer"):
        st.session_state.result = None

    st.session_state.reference_answer = reference

    st.success(
        "Audio and expected answer saved. "
        "You can now run the AI analysis."
    )


__all__ = ["render"]