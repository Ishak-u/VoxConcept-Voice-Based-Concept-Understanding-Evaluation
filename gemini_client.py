"""
Optional Gemini feedback integration.

Uses Google's modern ``google-genai`` SDK.

Features:

- Gemini 3.6 Flash
- Gemini 3.5 Flash-Lite
- Gemini 2.5 Flash
- Gemini 2.5 Flash-Lite
- Automatic fallback between models
- Automatic retry for temporary 429/503 errors
- Structured JSON output
- SQLite response caching
- Local per-minute and daily rate limits
- Safe operation when Gemini is unavailable

The rest of the application remains functional when Gemini fails.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from config import Settings
from database import Database, sha256_text


logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class GeminiError(RuntimeError):
    """
    Raised for recoverable Gemini configuration/provider failures.
    """


# ============================================================================
# Prompt
# ============================================================================

def _prompt(
    reference: str,
    transcript: str,
    similarity: float,
    fluency: Dict[str, float],
    audio_metrics: Dict[str, Any],
) -> str:
    """
    Build the Gemini evaluation prompt.
    """

    compact_metrics = {
        "semantic_similarity": round(
            float(similarity),
            2,
        ),
        "fluency": {
            key: round(
                float(value),
                2,
            )
            for key, value in fluency.items()
            if isinstance(
                value,
                (int, float),
            )
        },
        "audio": {
            key: value
            for key, value in audio_metrics.items()
            if key != "backend"
        },
    }

    return f"""
You are an examiner giving concise and constructive feedback
on a student's spoken explanation.

Evaluate ONLY the information provided below.

Expected answer:
{reference}

Student transcript:
{transcript}

Measured signals:
{json.dumps(compact_metrics, ensure_ascii=False)}

Return feedback with exactly these fields:

summary
strengths
weaknesses
suggestions

Rules:

1. Base comments only on the supplied transcript and measurements.
2. Do not invent facts about the recording.
3. Do not claim that the student said something that is not in the transcript.
4. Do not introduce unrelated information.
5. Keep the summary concise.
6. Give practical suggestions.
7. Strengths, weaknesses and suggestions should be short.
8. Return valid JSON matching the requested schema.
""".strip()


# ============================================================================
# Response extraction
# ============================================================================

def _extract_text(response: Any) -> str:
    """
    Extract generated text from google-genai response objects.
    """

    # Modern SDK usually exposes response.text.
    text = getattr(
        response,
        "text",
        None,
    )

    if text:
        return str(text).strip()

    # Fallback for candidate/content/parts response shapes.
    try:
        candidates = getattr(
            response,
            "candidates",
            None,
        ) or []

        if not candidates:
            return ""

        candidate = candidates[0]

        content = getattr(
            candidate,
            "content",
            None,
        )

        if content is None:
            return ""

        parts = getattr(
            content,
            "parts",
            None,
        ) or []

        texts: List[str] = []

        for part in parts:
            part_text = getattr(
                part,
                "text",
                None,
            )

            if part_text:
                texts.append(
                    str(part_text)
                )

        return "".join(texts).strip()

    except Exception:
        return ""


# ============================================================================
# JSON parsing
# ============================================================================

def _remove_code_fences(text: str) -> str:
    """
    Remove Markdown JSON code fences if a provider returns them.
    """

    cleaned = (text or "").strip()

    # ```json
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ```
    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    )

    return cleaned.strip()


def _extract_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from surrounding text.

    This is used only as a recovery mechanism.
    """

    cleaned = _remove_code_fences(text)

    if not cleaned:
        return ""

    # First try the whole response.
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Find a balanced {...} object.
    start = cleaned.find("{")

    if start == -1:
        return ""

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start,
        len(cleaned),
    ):
        char = cleaned[index]

        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                return cleaned[
                    start:index + 1
                ]

    return ""


def _parse_json(
    text: str,
) -> Dict[str, object]:
    """
    Parse Gemini feedback safely.
    """

    candidate = _extract_json_object(
        text
    )

    if not candidate:
        raise GeminiError(
            "Gemini returned a response that "
            "was not valid JSON."
        )

    try:
        parsed = json.loads(
            candidate
        )
    except json.JSONDecodeError as exc:
        raise GeminiError(
            "Gemini returned malformed JSON feedback."
        ) from exc

    if not isinstance(
        parsed,
        dict,
    ):
        raise GeminiError(
            "Gemini feedback was not a JSON object."
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def items(
        key: str,
    ) -> List[str]:
        value = parsed.get(
            key,
            [],
        )

        if isinstance(
            value,
            str,
        ):
            value = value.strip()

            return (
                [value]
                if value
                else []
            )

        if isinstance(
            value,
            list,
        ):
            result: List[str] = []

            for item in value:
                item_text = str(
                    item
                ).strip()

                if item_text:
                    result.append(
                        item_text
                    )

            return result

        return []

    # ----------------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------------

    summary = str(
        parsed.get(
            "summary",
            "",
        )
    ).strip()

    strengths = items(
        "strengths"
    )

    weaknesses = items(
        "weaknesses"
    )

    suggestions = items(
        "suggestions"
    )

    # ----------------------------------------------------------------------
    # Empty response check
    # ----------------------------------------------------------------------

    if not summary and not (
        strengths
        or weaknesses
        or suggestions
    ):
        raise GeminiError(
            "Gemini returned empty feedback."
        )

    return {
        "summary": summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
    }


# ============================================================================
# Provider error helpers
# ============================================================================

def _error_text(exc: Exception) -> str:
    """
    Convert a provider exception to searchable lowercase text.
    """

    return str(exc).strip().lower()


def _is_retryable_error(
    exc: Exception,
) -> bool:
    """
    Identify temporary Gemini failures.

    429 = rate limit / resource exhaustion
    503 = temporarily unavailable / overloaded

    We also recognize common textual variants because SDK exception
    formatting can differ between versions.
    """

    message = _error_text(exc)

    retryable_markers = (
        "429",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "503",
        "service unavailable",
        "unavailable",
        "temporarily unavailable",
        "high demand",
        "overloaded",
        "internal server error",
        "500",
        "502",
        "504",
    )

    return any(
        marker in message
        for marker in retryable_markers
    )


def _is_model_not_found(
    exc: Exception,
) -> bool:
    """
    Detect invalid/deprecated model IDs.

    These should not be retried repeatedly.
    """

    message = _error_text(exc)

    markers = (
        "404",
        "not_found",
        "not found",
        "no longer available",
        "model not found",
        "unknown model",
    )

    return any(
        marker in message
        for marker in markers
    )


# ============================================================================
# Google GenAI call
# ============================================================================

def _call_google_genai(
    api_key: str,
    model: str,
    prompt: str,
    retry_attempts: int = 2,
    retry_delay_seconds: float = 1.5,
) -> str:
    """
    Call Gemini using the modern google-genai SDK.

    Temporary 429/503 errors are retried with exponential backoff.
    """

    try:
        from google import genai
        from google.genai import types

    except ImportError as exc:
        raise GeminiError(
            "The 'google-genai' package is not installed. "
            "Install it with: pip install -U google-genai"
        ) from exc

    try:
        client = genai.Client(
            api_key=api_key
        )
    except Exception as exc:
        raise GeminiError(
            f"Could not initialise Gemini client: {exc}"
        ) from exc

    # ----------------------------------------------------------------------
    # Structured output schema
    # ----------------------------------------------------------------------

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "summary": {
                "type": "STRING",
            },
            "strengths": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                },
            },
            "weaknesses": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                },
            },
            "suggestions": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                },
            },
        },
        "required": [
            "summary",
            "strengths",
            "weaknesses",
            "suggestions",
        ],
    }

    # ----------------------------------------------------------------------
    # Generation config
    # ----------------------------------------------------------------------

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.2,
        max_output_tokens=700,
    )

    # ----------------------------------------------------------------------
    # Retry loop
    # ----------------------------------------------------------------------

    attempts = max(
        0,
        int(retry_attempts),
    )

    for attempt in range(
        attempts + 1
    ):
        try:
            logger.info(
                "Calling Gemini model: %s "
                "(attempt %s/%s)",
                model,
                attempt + 1,
                attempts + 1,
            )

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config,
            )

            text = _extract_text(
                response
            )

            if not text:
                raise GeminiError(
                    "Gemini returned no text."
                )

            return text

        except GeminiError:
            raise

        except Exception as exc:
            # --------------------------------------------------------------
            # Model doesn't exist / isn't available.
            # Don't waste retries.
            # --------------------------------------------------------------

            if _is_model_not_found(
                exc
            ):
                raise GeminiError(
                    f"Gemini model {model} "
                    f"is unavailable: {exc}"
                ) from exc

            # --------------------------------------------------------------
            # Temporary provider problem.
            # --------------------------------------------------------------

            if _is_retryable_error(
                exc
            ):
                if attempt < attempts:
                    delay = (
                        retry_delay_seconds
                        * (2 ** attempt)
                    )

                    logger.warning(
                        "Gemini model %s temporarily "
                        "unavailable. Retrying in %.1f "
                        "seconds: %s",
                        model,
                        delay,
                        exc,
                    )

                    time.sleep(
                        delay
                    )

                    continue

                raise GeminiError(
                    f"Gemini request failed after "
                    f"{attempts + 1} attempts: {exc}"
                ) from exc

            # --------------------------------------------------------------
            # Non-retryable provider error.
            # --------------------------------------------------------------

            raise GeminiError(
                f"Gemini request failed: {exc}"
            ) from exc

    raise GeminiError(
        f"Gemini model {model} failed."
    )


# ============================================================================
# Feedback generation
# ============================================================================

def generate_feedback(
    *,
    reference: str,
    transcript: str,
    semantic_similarity: float,
    fluency: Dict[str, float],
    audio_metrics: Dict[str, Any],
    settings: Settings,
    database: Database,
) -> Dict[str, object]:
    """
    Generate and cache structured Gemini feedback.

    Model order comes from settings.gemini_models.

    Example:

        gemini-3.6-flash
        gemini-3.5-flash-lite
        gemini-2.5-flash
        gemini-2.5-flash-lite
    """

    # =========================================================================
    # Gemini availability
    # =========================================================================

    if not settings.gemini_enabled:
        raise GeminiError(
            "Gemini is disabled."
        )

    if not settings.gemini_api_key:
        raise GeminiError(
            "Gemini API key is not configured."
        )

    if not settings.gemini_models:
        raise GeminiError(
            "No Gemini models are configured."
        )

    # =========================================================================
    # Build prompt
    # =========================================================================

    prompt = _prompt(
        reference=reference,
        transcript=transcript,
        similarity=semantic_similarity,
        fluency=fluency,
        audio_metrics=audio_metrics,
    )

    prompt_hash = sha256_text(
        prompt
    )

    # =========================================================================
    # Cache
    # =========================================================================

    try:
        cached = database.get_cached_ai_response(
            prompt_hash,
            settings.ai_cache_ttl_hours,
        )
    except Exception as exc:
        logger.warning(
            "Gemini cache lookup failed: %s",
            exc,
        )
        cached = None

    if cached:
        logger.info(
            "Using cached Gemini response."
        )

        return _parse_json(
            cached
        )

    # =========================================================================
    # Local rate limit
    # =========================================================================

    try:
        minute_calls = database.count_api_calls(
            "gemini",
            60,
        )

        if (
            minute_calls
            >= settings.gemini_calls_per_minute
        ):
            raise GeminiError(
                "The local Gemini per-minute "
                "rate limit has been reached."
            )

        daily_calls = database.count_api_calls(
            "gemini",
            86400,
        )

        if (
            daily_calls
            >= settings.gemini_calls_per_day
        ):
            raise GeminiError(
                "The local Gemini daily "
                "rate limit has been reached."
            )

    except GeminiError:
        raise

    except Exception as exc:
        logger.warning(
            "Gemini rate-limit check failed: %s",
            exc,
        )

    # =========================================================================
    # Model fallback chain
    # =========================================================================

    last_error: Optional[Exception] = None

    for model in settings.gemini_models:

        try:
            raw = _call_google_genai(
                api_key=settings.gemini_api_key,
                model=model,
                prompt=prompt,
                retry_attempts=settings.gemini_retry_attempts,
                retry_delay_seconds=settings.gemini_retry_delay_seconds,
            )

            # --------------------------------------------------------------
            # Validate structured response
            # --------------------------------------------------------------

            feedback = _parse_json(
                raw
            )

            # --------------------------------------------------------------
            # Record successful API call
            # --------------------------------------------------------------

            try:
                database.record_api_call(
                    "gemini"
                )
            except Exception as exc:
                logger.warning(
                    "Could not record Gemini API call: %s",
                    exc,
                )

            # --------------------------------------------------------------
            # Cache response
            # --------------------------------------------------------------

            try:
                database.cache_ai_response(
                    prompt_hash,
                    raw,
                    model,
                )
            except Exception as exc:
                logger.warning(
                    "Could not cache Gemini response: %s",
                    exc,
                )

            logger.info(
                "Gemini feedback generated successfully "
                "with model: %s",
                model,
            )

            return feedback

        except GeminiError as exc:

            last_error = exc

            logger.warning(
                "Gemini model %s failed: %s",
                model,
                exc,
            )

            # --------------------------------------------------------------
            # Important:
            #
            # Do NOT stop on 503/429.
            #
            # Move to the next configured model.
            # --------------------------------------------------------------

            continue

        except Exception as exc:

            last_error = exc

            logger.warning(
                "Unexpected Gemini error with model %s: %s",
                model,
                exc,
            )

            continue

    # =========================================================================
    # Everything failed
    # =========================================================================

    raise GeminiError(
        str(
            last_error
            or "No Gemini model could generate feedback."
        )
    )


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "GeminiError",
    "generate_feedback",
]