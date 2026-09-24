"""End-to-end evaluation pipeline for VBCUA.

This module deliberately contains no Streamlit UI code.  It coordinates the
local audio analysis, Whisper transcription, semantic similarity, fluency
scoring, optional Gemini feedback, and SQLite persistence used by the pages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from audio_features import AudioFeatures, extract_features
from config import Settings
from database import Database, sha256_bytes
from scoring import (
    build_offline_feedback,
    final_score,
    fluency_score,
    format_feedback_text,
    grade_for_score,
)
from semantics import compute_similarity
from transcription import TranscriptionError, transcribe_bytes

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str], None]]


class AnalysisError(RuntimeError):
    """A user-facing failure during an evaluation."""


@dataclass
class AnalysisResult:
    """Complete result consumed by the Streamlit pages and PDF renderer."""

    audio_name: str = "recording.wav"
    reference_answer: str = ""
    transcript: str = ""
    language: Optional[str] = None
    similarity: float = 0.0
    semantic_similarity: float = 0.0
    fluency_score: float = 0.0
    score: float = 0.0
    grade: str = "F"
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    summary: str = ""
    feedback: str = ""
    coverage: Dict[str, List[str]] = field(default_factory=dict)
    fluency_detail: Dict[str, float] = field(default_factory=dict)
    audio_metrics: Dict[str, Any] = field(default_factory=dict)
    waveform: List[float] = field(default_factory=list)
    engine: str = "offline"
    semantic_method: str = "lexical"
    transcription_model: str = "base"
    transcript_cached: bool = False
    notices: List[str] = field(default_factory=list)
    result_id: Optional[int] = None

    def to_record(self) -> Dict[str, Any]:
        """Return the subset of fields persisted by Database.save_result."""
        return {
            "audio_name": self.audio_name,
            "reference_answer": self.reference_answer,
            "transcript": self.transcript,
            "feedback": self.feedback,
            "similarity": self.similarity,
            "semantic_similarity": self.semantic_similarity,
            "fluency_score": self.fluency_score,
            "score": self.score,
            "grade": self.grade,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "suggestions": list(self.suggestions),
            "audio_metrics": dict(self.audio_metrics),
            "engine": self.engine,
        }


def _notify(progress: ProgressCallback, message: str) -> None:
    if progress:
        try:
            progress(message)
        except Exception:  # UI callbacks must never break the pipeline.
            logger.debug("Progress callback failed", exc_info=True)


def _validate_inputs(audio_bytes: bytes, reference_answer: str, settings: Settings) -> None:
    if not audio_bytes:
        raise AnalysisError("The uploaded audio file is empty.")
    reference = (reference_answer or "").strip()
    if len(reference.split()) < 2:
        raise AnalysisError("Please provide an expected answer containing at least two words.")
    max_bytes = max(1, settings.max_audio_mb) * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise AnalysisError(
            f"The audio file is too large ({len(audio_bytes) / 1024 / 1024:.1f} MB). "
            f"The limit is {settings.max_audio_mb} MB."
        )


def _audio_metrics(features: AudioFeatures) -> Dict[str, Any]:
    return features.to_dict()


def _feedback_from_gemini(
    reference: str,
    transcript: str,
    similarity: float,
    fluency: Dict[str, float],
    audio_metrics: Dict[str, Any],
    settings: Settings,
    database: Database,
) -> Optional[Dict[str, object]]:
    """Ask Gemini through gemini_client when enabled; failures are non-fatal."""
    if not settings.gemini_enabled:
        return None
    try:
        from gemini_client import GeminiError, generate_feedback

        return generate_feedback(
            reference=reference,
            transcript=transcript,
            semantic_similarity=similarity,
            fluency=fluency,
            audio_metrics=audio_metrics,
            settings=settings,
            database=database,
        )
    except GeminiError as exc:
        logger.info("Gemini feedback unavailable: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - provider-specific failures
        logger.warning("Unexpected Gemini failure: %s", exc, exc_info=True)
        return None


def analyse(
    audio_bytes: bytes,
    audio_name: str,
    reference_answer: str,
    *,
    settings: Settings,
    database: Database,
    progress: ProgressCallback = None,
) -> AnalysisResult:
    """Run one complete evaluation and persist it in SQLite."""
    _validate_inputs(audio_bytes, reference_answer, settings)
    reference_answer = reference_answer.strip()
    audio_name = audio_name or "recording.wav"
    audio_hash = sha256_bytes(audio_bytes)
    notices: List[str] = []

    _notify(progress, "Analysing audio signal...")
    try:
        # extract_features handles librosa first and a stdlib WAV fallback.
        features = extract_features_from_bytes(audio_bytes, audio_name)
    except Exception as exc:
        logger.warning("Audio feature extraction failed: %s", exc)
        features = AudioFeatures(error=str(exc), backend="unavailable")
        notices.append(f"Audio signal analysis was unavailable: {exc}")

    if features.duration is not None and features.duration > settings.max_audio_seconds:
        raise AnalysisError(
            f"The recording is {features.duration:.1f} seconds long. "
            f"The maximum allowed duration is {settings.max_audio_seconds} seconds."
        )

    _notify(progress, "Converting speech to text with Whisper...")
    try:
        transcription = transcribe_bytes(
            audio_bytes,
            audio_name,
            model_name=settings.whisper_model,
            database=database,
            audio_hash=audio_hash,
        )
    except TranscriptionError as exc:
        raise AnalysisError(str(exc)) from exc

    if transcription.cached:
        notices.append("Transcript was reused from the local SQLite cache.")

    _notify(progress, "Measuring semantic similarity...")
    similarity_result = compute_similarity(
        reference_answer,
        transcription.text,
        model_name=settings.embedding_model,
        allow_embeddings=True,
    )
    if similarity_result.method == "lexical":
        notices.append("Sentence-transformer embeddings were unavailable; TF-IDF + keyword recall was used.")

    metrics = _audio_metrics(features)
    fluency = fluency_score(
        transcription.text,
        duration_seconds=features.duration or transcription.duration,
        pause_ratio=features.pause_ratio,
        energy_consistency=features.energy_consistency,
    )

    score = final_score(
        similarity_result.similarity,
        fluency.get("overall", 0.0),
        settings.semantic_weight,
        settings.fluency_weight,
    )
    grade = grade_for_score(score)

    _notify(progress, "Requesting AI feedback from Gemini...")
    feedback = _feedback_from_gemini(
        reference_answer,
        transcription.text,
        similarity_result.similarity,
        fluency,
        metrics,
        settings,
        database,
    )

    if feedback is None:
        feedback = build_offline_feedback(
            reference_answer,
            transcription.text,
            similarity_result.similarity,
            fluency,
            metrics,
        )
        engine = "offline"
        if settings.gemini_enabled:
            notices.append("Gemini feedback was unavailable; the local evaluator was used instead.")
    else:
        engine = "gemini"

    _notify(progress, "Saving results...")
    result = AnalysisResult(
        audio_name=audio_name,
        reference_answer=reference_answer,
        transcript=transcription.text,
        language=transcription.language,
        similarity=similarity_result.similarity,
        semantic_similarity=similarity_result.similarity,
        fluency_score=fluency.get("overall", 0.0),
        score=score,
        grade=grade,
        strengths=[str(x) for x in feedback.get("strengths", [])],
        weaknesses=[str(x) for x in feedback.get("weaknesses", [])],
        suggestions=[str(x) for x in feedback.get("suggestions", [])],
        summary=str(feedback.get("summary", "")),
        feedback=format_feedback_text(feedback),
        coverage=feedback.get("coverage", {}) if isinstance(feedback.get("coverage", {}), dict) else {},
        fluency_detail=fluency,
        audio_metrics=metrics,
        waveform=list(features.waveform),
        engine=engine,
        semantic_method=similarity_result.method,
        transcription_model=transcription.model,
        transcript_cached=transcription.cached,
        notices=notices,
    )

    try:
        result.result_id = database.save_result(result.to_record())
    except Exception as exc:
        raise AnalysisError(f"The evaluation was completed but could not be saved: {exc}") from exc

    _notify(progress, "Done.")
    return result


def extract_features_from_bytes(audio_bytes: bytes, filename: str) -> AudioFeatures:
    """Write upload bytes to a temporary file and extract real audio features."""
    import os
    import tempfile
    from pathlib import Path

    suffix = Path(filename or "audio.wav").suffix or ".wav"
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as file_obj:
            file_obj.write(audio_bytes)
        return extract_features(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            logger.debug("Could not remove temporary audio file %s", temp_path)


def result_from_record(record: Dict[str, Any]) -> AnalysisResult:
    """Convert a SQLite result row into the object expected by the UI/report."""
    audio_metrics = record.get("audio_metrics") or {}
    return AnalysisResult(
        audio_name=record.get("audio_name") or "recording.wav",
        reference_answer=record.get("reference_answer") or "",
        transcript=record.get("transcript") or "",
        similarity=float(record.get("similarity") or record.get("semantic_similarity") or 0),
        semantic_similarity=float(record.get("semantic_similarity") or record.get("similarity") or 0),
        fluency_score=float(record.get("fluency_score") or 0),
        score=float(record.get("score") or 0),
        grade=record.get("grade") or "N/A",
        strengths=list(record.get("strengths") or []),
        weaknesses=list(record.get("weaknesses") or []),
        suggestions=list(record.get("suggestions") or []),
        feedback=record.get("feedback") or "",
        summary=_extract_summary(record.get("feedback") or ""),
        audio_metrics=audio_metrics if isinstance(audio_metrics, dict) else {},
        engine=record.get("engine") or "offline",
        result_id=record.get("id"),
    )


def _extract_summary(feedback: str) -> str:
    lines = [line.strip() for line in str(feedback).splitlines() if line.strip()]
    return lines[0] if lines and not lines[0].endswith(":") else ""


__all__ = ["AnalysisError", "AnalysisResult", "analyse", "result_from_record"]
