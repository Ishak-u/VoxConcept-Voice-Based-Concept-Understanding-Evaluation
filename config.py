"""
Central configuration for the Voice Based Concept Understanding Analyser.

Configuration precedence:

1. Real environment variables
2. Local .env file
3. Streamlit secrets
4. Built-in defaults

No API key is hard-coded in this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ============================================================================
# Paths
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOTENV_PATH = BASE_DIR / ".env"


# ============================================================================
# .env loader
# ============================================================================

def _load_dotenv(path: Path = DOTENV_PATH) -> None:
    """
    Load simple KEY=VALUE pairs from .env.

    Existing environment variables are never overwritten.
    """

    if not path.exists():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, _, value = line.partition("=")

            key = key.strip()
            value = value.strip()

            # Remove matching quotes.
            if len(value) >= 2:
                if (
                    value.startswith('"')
                    and value.endswith('"')
                ) or (
                    value.startswith("'")
                    and value.endswith("'")
                ):
                    value = value[1:-1]

            if key and key not in os.environ:
                os.environ[key] = value

    except OSError:
        # .env is optional.
        pass


_load_dotenv()


# ============================================================================
# Streamlit secrets
# ============================================================================

def _from_streamlit_secrets(name: str) -> Optional[str]:
    """
    Read a setting from Streamlit secrets.

    This function intentionally tolerates missing secrets.toml.
    """

    try:
        import streamlit as st
    except Exception:
        return None

    try:
        if name in st.secrets:
            value = st.secrets[name]

            if value is None:
                return None

            return str(value)

    except Exception:
        return None

    return None


# ============================================================================
# Generic configuration helpers
# ============================================================================

def get_setting(
    name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve configuration in the following order:

    environment -> Streamlit secrets -> default

    .env has already been loaded into os.environ.
    """

    value = os.environ.get(name)

    if value not in (None, ""):
        return value

    value = _from_streamlit_secrets(name)

    if value not in (None, ""):
        return value

    return default


def _get_int(name: str, default: int) -> int:
    try:
        value = get_setting(name, str(default))
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        value = get_setting(name, str(default))
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = str(
        get_setting(name, str(default))
    ).strip().lower()

    return raw in {
        "1",
        "true",
        "yes",
        "on",
    }


# ============================================================================
# Gemini models
# ============================================================================

# These are the models you said are available to your API key.
#
# Order matters:
#   1. Try Gemini 3.6 Flash first.
#   2. If temporarily unavailable/overloaded, try 3.5 Flash-Lite.
#   3. Then 2.5 Flash.
#   4. Finally 2.5 Flash-Lite.
#
# Do NOT add the old 2.0/1.5 models here.

DEFAULT_GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


# ============================================================================
# Application settings
# ============================================================================

@dataclass(frozen=True)
class Settings:
    """
    Immutable snapshot of application configuration.
    """

    # ----------------------------------------------------------------------
    # Gemini
    # ----------------------------------------------------------------------

    gemini_api_key: Optional[str] = None

    gemini_models: List[str] = field(
        default_factory=lambda: list(DEFAULT_GEMINI_MODELS)
    )

    # Number of retry attempts for temporary provider errors.
    gemini_retry_attempts: int = 2

    # Initial delay between retry attempts.
    gemini_retry_delay_seconds: float = 1.5

    # ----------------------------------------------------------------------
    # Local models
    # ----------------------------------------------------------------------

    whisper_model: str = "base"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ----------------------------------------------------------------------
    # Storage
    # ----------------------------------------------------------------------

    database_path: Path = DATA_DIR / "results.db"

    # ----------------------------------------------------------------------
    # Audio limits
    # ----------------------------------------------------------------------

    max_audio_mb: int = 25

    max_audio_seconds: int = 300

    # ----------------------------------------------------------------------
    # Gemini local rate limits
    # ----------------------------------------------------------------------

    gemini_calls_per_minute: int = 10

    gemini_calls_per_day: int = 180

    # ----------------------------------------------------------------------
    # AI cache
    # ----------------------------------------------------------------------

    ai_cache_ttl_hours: int = 720

    # ----------------------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------------------

    semantic_weight: float = 0.7

    fluency_weight: float = 0.3

    # ----------------------------------------------------------------------
    # Behaviour
    # ----------------------------------------------------------------------

    offline_mode: bool = False

    history_page_size: int = 25

    # ----------------------------------------------------------------------
    # Derived properties
    # ----------------------------------------------------------------------

    @property
    def gemini_enabled(self) -> bool:
        """
        Gemini is enabled only when:

        - an API key exists
        - offline mode is not forced
        """

        return (
            bool(self.gemini_api_key)
            and not self.offline_mode
            and bool(self.gemini_models)
        )


# ============================================================================
# Settings loader
# ============================================================================

def load_settings() -> Settings:
    """
    Build Settings from the current environment.
    """

    # ----------------------------------------------------------------------
    # Gemini models
    # ----------------------------------------------------------------------

    models_raw = get_setting(
        "GEMINI_MODEL",
        "",
    )

    models: List[str] = []

    if models_raw:
        models.extend(
            model.strip()
            for model in str(models_raw).split(",")
            if model.strip()
        )

    # Always append the current supported models.
    for model in DEFAULT_GEMINI_MODELS:
        if model not in models:
            models.append(model)

    # ----------------------------------------------------------------------
    # Database
    # ----------------------------------------------------------------------

    db_raw = get_setting(
        "DATABASE_PATH",
        "",
    )

    if db_raw:
        db_path = Path(db_raw).expanduser()

        if not db_path.is_absolute():
            db_path = (
                BASE_DIR / db_path
            ).resolve()
    else:
        db_path = DATA_DIR / "results.db"

    # ----------------------------------------------------------------------
    # Scoring weights
    # ----------------------------------------------------------------------

    semantic_weight = _get_float(
        "SEMANTIC_WEIGHT",
        0.7,
    )

    fluency_weight = _get_float(
        "FLUENCY_WEIGHT",
        0.3,
    )

    total = (
        semantic_weight
        + fluency_weight
    )

    if total <= 0:
        semantic_weight = 0.7
        fluency_weight = 0.3
    else:
        semantic_weight = (
            semantic_weight / total
        )

        fluency_weight = (
            fluency_weight / total
        )

    # ----------------------------------------------------------------------
    # Build settings
    # ----------------------------------------------------------------------

    return Settings(
        # Gemini
        gemini_api_key=(
            get_setting("GEMINI_API_KEY")
            or None
        ),

        gemini_models=models,

        gemini_retry_attempts=max(
            0,
            _get_int(
                "GEMINI_RETRY_ATTEMPTS",
                2,
            ),
        ),

        gemini_retry_delay_seconds=max(
            0.1,
            _get_float(
                "GEMINI_RETRY_DELAY_SECONDS",
                1.5,
            ),
        ),

        # Local models
        whisper_model=(
            get_setting(
                "WHISPER_MODEL",
                "base",
            )
            or "base"
        ),

        embedding_model=(
            get_setting(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            or "sentence-transformers/all-MiniLM-L6-v2"
        ),

        # Storage
        database_path=db_path,

        # Audio
        max_audio_mb=_get_int(
            "MAX_AUDIO_MB",
            25,
        ),

        max_audio_seconds=_get_int(
            "MAX_AUDIO_SECONDS",
            300,
        ),

        # Gemini limits
        gemini_calls_per_minute=_get_int(
            "GEMINI_CALLS_PER_MINUTE",
            10,
        ),

        gemini_calls_per_day=_get_int(
            "GEMINI_CALLS_PER_DAY",
            180,
        ),

        # Cache
        ai_cache_ttl_hours=_get_int(
            "AI_CACHE_TTL_HOURS",
            720,
        ),

        # Scoring
        semantic_weight=semantic_weight,

        fluency_weight=fluency_weight,

        # Behaviour
        offline_mode=_get_bool(
            "OFFLINE_MODE",
            False,
        ),

        history_page_size=_get_int(
            "HISTORY_PAGE_SIZE",
            25,
        ),
    )


# ============================================================================
# Module-level settings snapshot
# ============================================================================

settings = load_settings()


# ============================================================================
# Audio formats
# ============================================================================

ALLOWED_AUDIO_TYPES = [
    "mp3",
    "wav",
    "m4a",
    "ogg",
    "flac",
    "webm",
]


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DOTENV_PATH",
    "Settings",
    "settings",
    "load_settings",
    "get_setting",
    "ALLOWED_AUDIO_TYPES",
    "DEFAULT_GEMINI_MODELS",
]