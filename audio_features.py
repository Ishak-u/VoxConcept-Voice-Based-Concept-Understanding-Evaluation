"""
Audio feature extraction (duration, pause ratio, energy, waveform envelope).

Librosa is the primary backend, exactly as documented in the project's
technology stack. When librosa (or its ffmpeg/audioread backend) cannot decode
a file, the module falls back to Python's stdlib ``wave`` module plus NumPy for
PCM WAV files, and finally to a "no features" result. The analysis pipeline
degrades gracefully in every case - fluency is then scored from the transcript
alone rather than failing.
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Analysis window used for frame-level energy, in seconds.
FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010
TARGET_SAMPLE_RATE = 16000
WAVEFORM_POINTS = 800


@dataclass
class AudioFeatures:
    """Container for everything the rest of the app needs from the waveform."""

    duration: Optional[float] = None
    pause_ratio: Optional[float] = None
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None
    energy_consistency: Optional[float] = None
    silence_segments: int = 0
    longest_pause: Optional[float] = None
    sample_rate: Optional[int] = None
    waveform: List[float] = field(default_factory=list)
    backend: str = "unavailable"
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.duration is not None

    def to_dict(self) -> Dict[str, object]:
        """Compact, JSON-serialisable form (the waveform is intentionally excluded)."""
        return {
            "duration": _round(self.duration, 2),
            "pause_ratio": _round(self.pause_ratio, 4),
            "energy_mean": _round(self.energy_mean, 6),
            "energy_std": _round(self.energy_std, 6),
            "energy_consistency": _round(self.energy_consistency, 4),
            "silence_segments": self.silence_segments,
            "longest_pause": _round(self.longest_pause, 2),
            "sample_rate": self.sample_rate,
            "backend": self.backend,
        }


def _round(value: Optional[float], digits: int) -> Optional[float]:
    if value is None:
        return None
    try:
        rounded = round(float(value), digits)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(rounded) or math.isinf(rounded) else rounded


# --------------------------------------------------------------------------
# Loading backends
# --------------------------------------------------------------------------

def _load_with_librosa(path: Path):
    import librosa  # imported lazily - heavy and optional at import time

    samples, sample_rate = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    return np.asarray(samples, dtype=np.float32), int(sample_rate), "librosa"


def _load_with_wave(path: Path):
    """Decode uncompressed PCM WAV using only the standard library."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtype_map:
        raise ValueError(f"Unsupported PCM sample width: {width} bytes")

    data = np.frombuffer(frames, dtype=dtype_map[width]).astype(np.float32)
    if width == 1:  # 8-bit PCM is unsigned and centred on 128
        data = (data - 128.0) / 128.0
    else:
        data = data / float(2 ** (8 * width - 1))

    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)

    return data.astype(np.float32), int(sample_rate), "wave"


def load_audio(path: str | Path):
    """Return ``(samples, sample_rate, backend)``; raises on total failure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    errors: List[str] = []
    for loader in (_load_with_librosa, _load_with_wave):
        try:
            return loader(path)
        except Exception as exc:  # noqa: BLE001 - we intentionally try the next backend
            errors.append(f"{loader.__name__}: {exc}")
            logger.debug("Audio backend failed (%s): %s", loader.__name__, exc)
    raise RuntimeError("; ".join(errors))


# --------------------------------------------------------------------------
# Feature computation
# --------------------------------------------------------------------------

def frame_energy(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Short-time RMS energy per frame."""
    frame_length = max(1, int(FRAME_SECONDS * sample_rate))
    hop_length = max(1, int(HOP_SECONDS * sample_rate))
    if samples.size < frame_length:
        return np.array([float(np.sqrt(np.mean(np.square(samples))))]) if samples.size else np.array([])

    frame_count = 1 + (samples.size - frame_length) // hop_length
    # Strided view avoids copying the signal for every frame.
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        samples, shape=(frame_count, frame_length), strides=strides, writeable=False
    )
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def downsample_waveform(samples: np.ndarray, points: int = WAVEFORM_POINTS) -> List[float]:
    """Peak-preserving envelope for plotting, capped at ``points`` values."""
    if samples.size == 0:
        return []
    if samples.size <= points:
        return [float(value) for value in samples]
    bucket = samples.size // points
    trimmed = samples[: bucket * points].reshape(points, bucket)
    # Keep whichever of the min/max has the larger magnitude so spikes survive.
    maxima = trimmed.max(axis=1)
    minima = trimmed.min(axis=1)
    envelope = np.where(np.abs(maxima) >= np.abs(minima), maxima, minima)
    return [float(value) for value in envelope]


def extract_features(path: str | Path) -> AudioFeatures:
    """Extract delivery features from an audio file, never raising."""
    try:
        samples, sample_rate, backend = load_audio(path)
    except Exception as exc:  # noqa: BLE001 - feature extraction is best-effort
        logger.warning("Audio feature extraction unavailable: %s", exc)
        return AudioFeatures(error=str(exc), backend="unavailable")

    if samples.size == 0 or sample_rate <= 0:
        return AudioFeatures(error="Audio file contains no samples", backend=backend)

    duration = float(samples.size) / float(sample_rate)
    energies = frame_energy(samples, sample_rate)

    if energies.size == 0:
        return AudioFeatures(duration=round(duration, 2), sample_rate=sample_rate, backend=backend)

    peak = float(np.max(energies))
    # Adaptive silence threshold: relative to the loudest frame, with a floor so
    # a completely silent clip is not reported as fully voiced.
    threshold = max(peak * 0.12, 1e-4)
    voiced_mask = energies >= threshold
    silent_mask = ~voiced_mask

    pause_ratio = float(np.mean(silent_mask))

    # Count contiguous silent runs longer than 300 ms as real pauses.
    hop_seconds = HOP_SECONDS
    min_pause_frames = max(1, int(0.3 / hop_seconds))
    silence_segments = 0
    longest_run = 0
    current_run = 0
    for is_silent in silent_mask:
        if is_silent:
            current_run += 1
        else:
            if current_run >= min_pause_frames:
                silence_segments += 1
            longest_run = max(longest_run, current_run)
            current_run = 0
    if current_run >= min_pause_frames:
        silence_segments += 1
    longest_run = max(longest_run, current_run)

    voiced_energies = energies[voiced_mask]
    if voiced_energies.size:
        energy_mean = float(np.mean(voiced_energies))
        energy_std = float(np.std(voiced_energies))
        # Consistency = 1 - coefficient of variation, clipped into [0, 1].
        consistency = 1.0 - (energy_std / energy_mean) if energy_mean > 0 else 0.0
        energy_consistency = float(min(1.0, max(0.0, consistency)))
    else:
        energy_mean = energy_std = 0.0
        energy_consistency = 0.0

    return AudioFeatures(
        duration=round(duration, 2),
        pause_ratio=round(pause_ratio, 4),
        energy_mean=energy_mean,
        energy_std=energy_std,
        energy_consistency=round(energy_consistency, 4),
        silence_segments=silence_segments,
        longest_pause=round(longest_run * hop_seconds, 2),
        sample_rate=sample_rate,
        waveform=downsample_waveform(samples),
        backend=backend,
    )


__all__ = ["AudioFeatures", "extract_features", "load_audio", "frame_energy", "downsample_waveform"]
