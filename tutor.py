"""Tutor Mode: grounded Socratic follow-up loop for VBCUA Phase 3."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from database import Database
from semantics import compute_similarity

logger = logging.getLogger(__name__)


def score_concepts(concepts: List[Dict[str, Any]], answer: str, model_name: str) -> List[Dict[str, Any]]:
    scored = []
    for c in concepts:
        desc = str(c.get("description", "")).strip()
        if not desc:
            continue
        sim = compute_similarity(desc, answer, model_name=model_name, allow_embeddings=True)
        scored.append({**c, "score": sim.similarity, "method": sim.method})
    return scored


def weakest_concept(scores: List[Dict[str, Any]], asked: set[str] | None = None) -> Dict[str, Any] | None:
    asked = asked or set()
    remaining = [x for x in scores if str(x.get("name", "")) not in asked]
    return min(remaining, key=lambda x: float(x.get("score", 0))) if remaining else None


def _fallback_question(concept: Dict[str, Any]) -> str:
    return f"Can you explain {concept.get('name', 'this concept')} in your own words and give the key idea that makes it correct?"


def generate_question(*, concept: Dict[str, Any], study_context: str, settings, database: Database) -> str:
    try:
        from gemini_client import generate_tutor_question
        result = generate_tutor_question(concept=concept, study_context=study_context, settings=settings, database=database)
        return str(result.get("question", "")).strip() or _fallback_question(concept)
    except Exception as exc:
        logger.info("Tutor question generation unavailable: %s", exc)
        return _fallback_question(concept)


def evaluate_followup(*, concept: Dict[str, Any], answer: str, study_context: str, settings, database: Database) -> Dict[str, Any]:
    try:
        from gemini_client import evaluate_tutor_answer
        result = evaluate_tutor_answer(concept=concept, answer=answer, study_context=study_context, settings=settings, database=database)
        if isinstance(result, dict):
            return result
    except Exception as exc:
        logger.info("Tutor answer evaluation unavailable: %s", exc)
    sim = compute_similarity(str(concept.get("description", "")), answer, model_name=settings.embedding_model, allow_embeddings=True)
    return {"correct": sim.similarity >= 65, "score": sim.similarity, "feedback": "Follow-up evaluated with the local semantic scorer."}
