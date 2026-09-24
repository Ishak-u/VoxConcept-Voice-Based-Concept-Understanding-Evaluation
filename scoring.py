"""
Pure, dependency-light scoring logic.

Nothing in this module imports Streamlit, torch or any network client, which
makes the grading rules directly unit-testable and keeps the Streamlit pages
free of business logic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")

# A small, self-contained stop-word list. Deliberately not pulled from NLTK so
# the app never needs a download step at first run.
STOP_WORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with
    without into onto about as is are was were be been being am do does did doing have has
    had having it its it's i you he she we they them us our your their my me his her not no
    so such very can could should would may might will shall just also there here what which
    who whom when where why how all any both each few more most other some only own same too
    s t don now
    """.split()
)

FILLER_WORDS = (
    "um",
    "uh",
    "erm",
    "hmm",
    "like",
    "actually",
    "basically",
    "literally",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "so yeah",
    "right",
)


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens, punctuation stripped."""
    return _WORD_RE.findall((text or "").lower())


def content_tokens(text: str) -> List[str]:
    """Tokens with stop-words removed, used for keyword coverage."""
    return [token for token in tokenize(text) if token not in STOP_WORDS and len(token) > 1]


def word_count(text: str) -> int:
    return len(tokenize(text))


# --------------------------------------------------------------------------
# Lexical similarity (offline fallback when embeddings are unavailable)
# --------------------------------------------------------------------------

def _tf_idf_vectors(
    doc_a: Sequence[str], doc_b: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Build length-normalised TF-IDF vectors over a two-document corpus."""
    count_a, count_b = Counter(doc_a), Counter(doc_b)
    vocabulary = set(count_a) | set(count_b)

    vec_a: Dict[str, float] = {}
    vec_b: Dict[str, float] = {}
    for term in vocabulary:
        document_frequency = (1 if term in count_a else 0) + (1 if term in count_b else 0)
        # Smoothed IDF over a 2-document corpus: shared terms still contribute.
        idf = math.log((2 + 1) / (document_frequency + 1)) + 1.0
        if count_a[term]:
            vec_a[term] = (1 + math.log(count_a[term])) * idf
        if count_b[term]:
            vec_b[term] = (1 + math.log(count_b[term])) * idf
    return vec_a, vec_b


def cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared = set(vec_a) & set(vec_b)
    numerator = sum(vec_a[term] * vec_b[term] for term in shared)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def lexical_similarity(reference: str, answer: str) -> float:
    """
    Similarity in the range 0-100 using TF-IDF cosine blended with keyword
    recall. Used when sentence-transformers is not installed or fails to load.
    """
    ref_tokens = content_tokens(reference)
    ans_tokens = content_tokens(answer)
    if not ref_tokens or not ans_tokens:
        return 0.0

    vec_ref, vec_ans = _tf_idf_vectors(ref_tokens, ans_tokens)
    cosine_score = cosine(vec_ref, vec_ans)

    ref_set = set(ref_tokens)
    recall = len(ref_set & set(ans_tokens)) / len(ref_set)

    blended = 0.6 * cosine_score + 0.4 * recall
    return round(clamp(blended * 100, 0, 100), 2)


def keyword_coverage(reference: str, answer: str, top_n: int = 12) -> Dict[str, List[str]]:
    """Which of the reference answer's key terms the student did and did not use."""
    ref_counts = Counter(content_tokens(reference))
    key_terms = [term for term, _ in ref_counts.most_common(top_n)]
    answer_set = set(content_tokens(answer))
    covered = [term for term in key_terms if term in answer_set]
    missing = [term for term in key_terms if term not in answer_set]
    return {"covered": covered, "missing": missing, "key_terms": key_terms}


# --------------------------------------------------------------------------
# Fluency
# --------------------------------------------------------------------------

def count_fillers(text: str) -> Dict[str, int]:
    """Count filler words/phrases in a transcript (whole-word matching)."""
    lowered = " " + " ".join(tokenize(text)) + " "
    counts: Dict[str, int] = {}
    for filler in FILLER_WORDS:
        pattern = " " + " ".join(tokenize(filler)) + " "
        if not pattern.strip():
            continue
        occurrences = lowered.count(pattern)
        # Overlapping search for repeated adjacent fillers ("um um um").
        if occurrences:
            counts[filler] = occurrences
    return counts


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def speech_rate(text: str, duration_seconds: Optional[float]) -> Optional[float]:
    """Words per minute, or ``None`` when the duration is unknown."""
    if not duration_seconds or duration_seconds <= 0:
        return None
    return round(word_count(text) * 60.0 / duration_seconds, 1)


def fluency_score(
    transcript: str,
    duration_seconds: Optional[float] = None,
    pause_ratio: Optional[float] = None,
    energy_consistency: Optional[float] = None,
) -> Dict[str, float]:
    """
    Combine delivery signals into a 0-100 fluency score.

    Components (each 0-100, missing components are dropped and the remaining
    weights re-normalised):

    * filler penalty  - fewer filler words is better
    * pace            - 110-160 wpm scores best
    * pause ratio     - under ~25 % silence scores best
    * energy          - steady loudness scores best
    """
    components: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    words = max(word_count(transcript), 1)
    filler_total = sum(count_fillers(transcript).values())
    filler_rate = filler_total / words
    components["fillers"] = clamp(100.0 - (filler_rate * 100.0 * 6.0), 0, 100)
    weights["fillers"] = 0.35

    wpm = speech_rate(transcript, duration_seconds)
    if wpm is not None:
        if 110 <= wpm <= 160:
            pace = 100.0
        elif wpm < 110:
            pace = clamp(100.0 - (110 - wpm) * 1.2, 0, 100)
        else:
            pace = clamp(100.0 - (wpm - 160) * 1.0, 0, 100)
        components["pace"] = pace
        weights["pace"] = 0.25

    if pause_ratio is not None:
        excess = max(0.0, pause_ratio - 0.25)
        components["pauses"] = clamp(100.0 - excess * 220.0, 0, 100)
        weights["pauses"] = 0.25

    if energy_consistency is not None:
        components["energy"] = clamp(energy_consistency * 100.0, 0, 100)
        weights["energy"] = 0.15

    total_weight = sum(weights.values()) or 1.0
    overall = sum(components[key] * weights[key] for key in components) / total_weight

    result = {key: round(value, 2) for key, value in components.items()}
    result["overall"] = round(clamp(overall, 0, 100), 2)
    result["filler_count"] = float(filler_total)
    if wpm is not None:
        result["words_per_minute"] = wpm
    return result


# --------------------------------------------------------------------------
# Final score & grade
# --------------------------------------------------------------------------

GRADE_BANDS: Tuple[Tuple[float, str], ...] = (
    (90.0, "A+"),
    (80.0, "A"),
    (70.0, "B"),
    (60.0, "C"),
    (50.0, "D"),
    (0.0, "F"),
)


def grade_for_score(score: float) -> str:
    """Map a 0-100 score onto a letter grade."""
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return "N/A"
    numeric = clamp(numeric, 0, 100)
    for threshold, letter in GRADE_BANDS:
        if numeric >= threshold:
            return letter
    return "F"


def final_score(
    semantic_similarity: float,
    fluency: float,
    semantic_weight: float = 0.7,
    fluency_weight: float = 0.3,
) -> float:
    """Weighted combination of content accuracy and delivery quality."""
    total = semantic_weight + fluency_weight
    if total <= 0:
        semantic_weight, fluency_weight, total = 0.7, 0.3, 1.0
    combined = (
        clamp(float(semantic_similarity), 0, 100) * semantic_weight
        + clamp(float(fluency), 0, 100) * fluency_weight
    ) / total
    return round(clamp(combined, 0, 100), 2)


def build_offline_feedback(
    reference: str,
    transcript: str,
    semantic_similarity: float,
    fluency: Dict[str, float],
    audio_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Deterministic, rule-based feedback used when Gemini is unavailable.

    This is a real evaluation - it is derived from the actual similarity,
    keyword coverage and delivery metrics of the submission, not canned text.
    """
    audio_metrics = audio_metrics or {}
    coverage = keyword_coverage(reference, transcript)
    strengths: List[str] = []
    weaknesses: List[str] = []
    suggestions: List[str] = []

    if semantic_similarity >= 80:
        strengths.append(
            f"The explanation closely matches the expected answer ({semantic_similarity:.0f}% semantic match)."
        )
    elif semantic_similarity >= 60:
        strengths.append(
            f"The core idea is recognisable and broadly correct ({semantic_similarity:.0f}% semantic match)."
        )
    else:
        weaknesses.append(
            f"The explanation diverges substantially from the expected answer ({semantic_similarity:.0f}% semantic match)."
        )

    if coverage["covered"]:
        strengths.append(
            "Key terms used correctly: " + ", ".join(coverage["covered"][:6]) + "."
        )
    if coverage["missing"]:
        weaknesses.append(
            "Important concepts that were not mentioned: "
            + ", ".join(coverage["missing"][:6])
            + "."
        )
        suggestions.append(
            "Revise the answer to explicitly define and use: "
            + ", ".join(coverage["missing"][:4])
            + "."
        )

    filler_count = int(fluency.get("filler_count", 0))
    if filler_count == 0:
        strengths.append("Delivery is clean - no filler words were detected.")
    elif filler_count <= 3:
        strengths.append(f"Only {filler_count} filler word(s) detected - delivery is mostly clean.")
    else:
        weaknesses.append(f"{filler_count} filler words were detected, which breaks up the explanation.")
        suggestions.append("Pause silently instead of using fillers such as 'um', 'like' or 'basically'.")

    wpm = fluency.get("words_per_minute")
    if wpm:
        if 110 <= wpm <= 160:
            strengths.append(f"Speaking pace is comfortable at about {wpm:.0f} words per minute.")
        elif wpm < 110:
            weaknesses.append(f"Speaking pace is slow at about {wpm:.0f} words per minute.")
            suggestions.append("Practise delivering the explanation with fewer long hesitations.")
        else:
            weaknesses.append(f"Speaking pace is fast at about {wpm:.0f} words per minute.")
            suggestions.append("Slow down slightly so each idea has time to land.")

    pause_ratio = audio_metrics.get("pause_ratio")
    if isinstance(pause_ratio, (int, float)):
        if pause_ratio > 0.4:
            weaknesses.append(
                f"About {pause_ratio * 100:.0f}% of the recording is silence, suggesting frequent hesitation."
            )
            suggestions.append("Rehearse the answer once before recording to reduce long pauses.")
        elif pause_ratio < 0.25:
            strengths.append("The explanation flows continuously with few long pauses.")

    spoken_words = word_count(transcript)
    expected_words = word_count(reference)
    if expected_words and spoken_words < expected_words * 0.5:
        weaknesses.append("The answer is noticeably shorter than the expected explanation.")
        suggestions.append("Expand the answer with an example or a short justification.")

    if not strengths:
        strengths.append("The response was captured and transcribed successfully.")
    if not weaknesses:
        weaknesses.append("No significant weaknesses were detected in this response.")
    if not suggestions:
        suggestions.append("Keep practising with slightly more advanced concepts to build depth.")

    summary = (
        f"Semantic match with the expected answer is {semantic_similarity:.0f}% and delivery "
        f"fluency scores {fluency.get('overall', 0):.0f}/100. "
        + ("Strong overall understanding." if semantic_similarity >= 75 else
           "Partial understanding - review the missing concepts above." if semantic_similarity >= 50 else
           "The concept needs to be revisited before re-attempting.")
    )

    return {
        "summary": summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "coverage": coverage,
    }


def format_feedback_text(feedback: Dict[str, object]) -> str:
    """Render a structured feedback dict into readable plain text."""
    lines: List[str] = []
    summary = feedback.get("summary")
    if summary:
        lines.append(str(summary))
        lines.append("")
    for title, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses"), ("Suggestions", "suggestions")):
        items = feedback.get(key) or []
        if not items:
            continue
        lines.append(f"{title}:")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    return "\n".join(lines).strip()


__all__ = [
    "tokenize",
    "content_tokens",
    "word_count",
    "lexical_similarity",
    "keyword_coverage",
    "count_fillers",
    "speech_rate",
    "fluency_score",
    "grade_for_score",
    "final_score",
    "build_offline_feedback",
    "format_feedback_text",
    "clamp",
    "FILLER_WORDS",
    "STOP_WORDS",
]
