"""
Semantic similarity between the expected answer and the spoken answer.

Primary backend: ``sentence-transformers`` (all-MiniLM-L6-v2, ~90 MB, runs
locally on CPU, free and offline after the first download).

Fallback: the TF-IDF + keyword-recall metric in :mod:`scoring`, which needs no
model at all. The application therefore always produces a similarity number,
even on a machine where the embedding model cannot be downloaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from scoring import clamp, lexical_similarity

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}
_FAILED_MODELS: set[str] = set()


@dataclass
class SimilarityResult:
    similarity: float
    method: str
    detail: Optional[str] = None


def load_embedding_model(model_name: str):
    """Load and memoise a SentenceTransformer, or return ``None`` if unavailable."""
    if model_name in _FAILED_MODELS:
        return None
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        logger.info("sentence-transformers unavailable (%s); using lexical similarity.", exc)
        _FAILED_MODELS.add(model_name)
        return None

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 - offline machines land here
        logger.warning("Could not load embedding model '%s': %s", model_name, exc)
        _FAILED_MODELS.add(model_name)
        return None

    _MODEL_CACHE[model_name] = model
    return model


def embedding_similarity(reference: str, answer: str, model) -> float:
    """Cosine similarity of two sentence embeddings, rescaled to 0-100."""
    import numpy as np

    vectors = model.encode([reference, answer], convert_to_numpy=True, normalize_embeddings=True)
    cosine = float(np.dot(vectors[0], vectors[1]))

    # MiniLM cosine for unrelated sentences hovers around 0.0-0.1 while
    # paraphrases sit near 0.8-0.95. Rescaling from [0.1, 0.95] spreads the
    # usable range across 0-100 instead of compressing everything above 50.
    rescaled = (cosine - 0.10) / (0.95 - 0.10)
    return round(clamp(rescaled * 100.0, 0.0, 100.0), 2)


def compute_similarity(
    reference: str,
    answer: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    allow_embeddings: bool = True,
) -> SimilarityResult:
    """Return the best available semantic similarity in the range 0-100."""
    reference = (reference or "").strip()
    answer = (answer or "").strip()
    if not reference or not answer:
        return SimilarityResult(0.0, "none", "Reference answer or transcript was empty.")

    if allow_embeddings:
        model = load_embedding_model(model_name)
        if model is not None:
            try:
                score = embedding_similarity(reference, answer, model)
                return SimilarityResult(score, "embeddings", model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding similarity failed, falling back: %s", exc)

    score = lexical_similarity(reference, answer)
    return SimilarityResult(score, "lexical", "TF-IDF cosine blended with keyword recall")


__all__ = ["compute_similarity", "SimilarityResult", "load_embedding_model", "embedding_similarity"]
