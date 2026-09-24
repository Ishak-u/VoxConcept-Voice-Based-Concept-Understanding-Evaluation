"""
Speech-to-text using OpenAI Whisper, running fully locally (no API, no cost).

The model is loaded once per process and cached; transcripts are cached in
SQLite keyed by the SHA-256 of the audio bytes, so re-analysing the same
recording never re-runs the model.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}

VALID_WHISPER_MODELS = ("tiny", "base", "small", "medium", "large")


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    model: str = "base"
    cached: bool = False


# --------------------------------------------------------------------------
# Environment checks
# --------------------------------------------------------------------------

def ffmpeg_available() -> bool:
    """Whisper shells out to ffmpeg for anything that is not a plain WAV."""
    return shutil.which("ffmpeg") is not None


def whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
    except Exception:
        return False
    return True


def probe_duration(path: str | Path) -> Optional[float]:
    """Best-effort duration probe used to reject over-long uploads early."""
    from audio_features import extract_features

    features = extract_features(path)
    if features.duration is not None:
        return features.duration

    if shutil.which("ffprobe"):
        try:
            output = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            value = output.stdout.strip()
            if value:
                return float(value)
        except (subprocess.SubprocessError, ValueError) as exc:
            logger.debug("ffprobe duration probe failed: %s", exc)
    return None


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_whisper_model(model_name: str = "base"):
    """Load (and memoise) a Whisper model. Raises :class:`TranscriptionError`."""
    name = (model_name or "base").strip().lower()
    if name not in VALID_WHISPER_MODELS:
        logger.warning("Unknown Whisper model '%s'; falling back to 'base'.", name)
        name = "base"

    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]

    try:
        import whisper
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TranscriptionError(
            "openai-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        model = whisper.load_model(name)
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(
            f"Could not load the Whisper '{name}' model: {exc}. "
            "The first run downloads the model once (~140 MB for 'base'); "
            "check your internet connection and free disk space."
        ) from exc

    _MODEL_CACHE[name] = model
    return model


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def transcribe_file(path: str | Path, model_name: str = "base") -> TranscriptionResult:
    """Transcribe a file on disk with Whisper."""
    path = Path(path)
    if not path.exists():
        raise TranscriptionError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    if suffix != ".wav" and not ffmpeg_available():
        raise TranscriptionError(
            "ffmpeg is required to decode "
            f"'{suffix or 'this file type'}'. Install ffmpeg and restart the app, "
            "or upload a .wav file instead."
        )

    model = load_whisper_model(model_name)

    try:
        # fp16 is only meaningful on CUDA; forcing it off keeps CPU runs correct.
        raw = model.transcribe(str(path), fp16=False)
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Whisper could not transcribe this audio: {exc}") from exc

    text = (raw.get("text") or "").strip()
    if not text:
        raise TranscriptionError(
            "No speech was detected in the recording. Please upload a clearer audio file."
        )

    duration = None
    segments = raw.get("segments") or []
    if segments:
        try:
            duration = float(segments[-1].get("end"))
        except (TypeError, ValueError):
            duration = None

    return TranscriptionResult(
        text=text,
        language=raw.get("language"),
        duration=duration,
        model=model_name,
        cached=False,
    )


def transcribe_bytes(
    audio_bytes: bytes,
    filename: str,
    model_name: str = "base",
    database=None,
    audio_hash: Optional[str] = None,
) -> TranscriptionResult:
    """
    Transcribe raw upload bytes, using the SQLite transcript cache when possible.

    The temporary file is always removed, even on failure.
    """
    from database import sha256_bytes

    if not audio_bytes:
        raise TranscriptionError("The uploaded audio file is empty.")

    audio_hash = audio_hash or sha256_bytes(audio_bytes)

    if database is not None:
        try:
            cached = database.get_cached_transcript(audio_hash, model_name)
        except Exception as exc:  # noqa: BLE001 - cache problems must not break analysis
            logger.warning("Transcript cache lookup failed: %s", exc)
            cached = None
        if cached and cached.get("transcript"):
            return TranscriptionResult(
                text=cached["transcript"],
                language=cached.get("language"),
                duration=cached.get("duration"),
                model=model_name,
                cached=True,
            )

    suffix = Path(filename or "audio.wav").suffix or ".wav"
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as file_obj:
            file_obj.write(audio_bytes)
        result = transcribe_file(temp_path, model_name=model_name)
    finally:
        try:
            os.remove(temp_path)
        except OSError:  # pragma: no cover - best effort cleanup
            logger.debug("Could not remove temporary file %s", temp_path)

    if database is not None:
        try:
            database.cache_transcript(
                audio_hash, result.text, model_name, result.language, result.duration
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not cache transcript: %s", exc)

    return result


__all__ = [
    "TranscriptionError",
    "TranscriptionResult",
    "transcribe_bytes",
    "transcribe_file",
    "load_whisper_model",
    "ffmpeg_available",
    "whisper_available",
    "probe_duration",
]
