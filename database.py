"""
SQLite persistence layer.

Design notes
------------
* Every call opens a short-lived connection through a context manager so the
  app never shares a cursor between Streamlit script runs (the original code
  used one global connection with ``check_same_thread=False``, which is racy).
* All statements are parameterised - there is no string interpolation of user
  data anywhere, so SQL injection is structurally impossible.
* Two cache tables (``transcripts`` and ``ai_cache``) exist purely to keep the
  project inside free-tier limits: identical audio is never transcribed twice
  and an identical evaluation prompt is never sent to Gemini twice.
* ``migrate()`` upgrades the original 6-column ``results`` table in place, so
  an existing ``results.db`` keeps its rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# Columns expected on the modern ``results`` table.
RESULT_COLUMNS = (
    "created_at",
    "audio_name",
    "reference_answer",
    "transcript",
    "feedback",
    "similarity",
    "semantic_similarity",
    "fluency_score",
    "score",
    "grade",
    "strengths",
    "weaknesses",
    "suggestions",
    "audio_metrics",
    "engine",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    """Stable content hash used as a cache key."""
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


class Database:
    """Thin, dependency-free wrapper around a SQLite file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def migrate(self) -> None:
        """Create or upgrade the schema without destroying existing rows."""
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcript TEXT,
                    feedback TEXT,
                    similarity TEXT,
                    score TEXT,
                    grade TEXT
                )
                """
            )

            existing = {row["name"] for row in conn.execute("PRAGMA table_info(results)")}
            additions = {
                "created_at": "TEXT",
                "audio_name": "TEXT",
                "reference_answer": "TEXT",
                "semantic_similarity": "REAL",
                "fluency_score": "REAL",
                "strengths": "TEXT",
                "weaknesses": "TEXT",
                "suggestions": "TEXT",
                "audio_metrics": "TEXT",
                "engine": "TEXT",
            }
            for column, coltype in additions.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE results ADD COLUMN {column} {coltype}")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    audio_hash TEXT PRIMARY KEY,
                    transcript TEXT NOT NULL,
                    language TEXT,
                    model TEXT,
                    duration REAL,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    model TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    called_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_usage_called_at ON api_usage(provider, called_at)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def save_result(self, result: Dict[str, Any]) -> int:
        """Persist one evaluation. Returns the new row id."""
        payload = {key: result.get(key) for key in RESULT_COLUMNS}
        payload["created_at"] = payload.get("created_at") or _utcnow()

        for list_field in ("strengths", "weaknesses", "suggestions"):
            value = payload.get(list_field)
            if isinstance(value, (list, tuple)):
                payload[list_field] = json.dumps(list(value))
            elif value is None:
                payload[list_field] = json.dumps([])

        metrics = payload.get("audio_metrics")
        if isinstance(metrics, dict):
            payload["audio_metrics"] = json.dumps(metrics)
        elif metrics is None:
            payload["audio_metrics"] = json.dumps({})

        columns = ", ".join(RESULT_COLUMNS)
        placeholders = ", ".join("?" for _ in RESULT_COLUMNS)
        values = [payload[key] for key in RESULT_COLUMNS]

        with self.connect() as conn:
            cursor = conn.execute(
                f"INSERT INTO results ({columns}) VALUES ({placeholders})", values
            )
            return int(cursor.lastrowid)

    def fetch_results(self, limit: int = 25, offset: int = 0) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM results
                ORDER BY COALESCE(created_at, '') DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def fetch_result(self, result_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM results WHERE id = ?", (int(result_id),)).fetchone()
        return self._row_to_result(row) if row else None

    def count_results(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM results").fetchone()[0])

    def delete_result(self, result_id: int) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM results WHERE id = ?", (int(result_id),))
            return cursor.rowcount > 0

    def clear_results(self) -> int:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM results")
            return cursor.rowcount

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(row)

        for list_field in ("strengths", "weaknesses", "suggestions"):
            raw = data.get(list_field)
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                    data[list_field] = parsed if isinstance(parsed, list) else [str(parsed)]
                except json.JSONDecodeError:
                    data[list_field] = [raw]
            else:
                data[list_field] = []

        raw_metrics = data.get("audio_metrics")
        if isinstance(raw_metrics, str) and raw_metrics.strip():
            try:
                parsed = json.loads(raw_metrics)
                data["audio_metrics"] = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                data["audio_metrics"] = {}
        elif not isinstance(raw_metrics, dict):
            data["audio_metrics"] = {}

        # Legacy rows stored numbers as TEXT; normalise for charting.
        for numeric in ("similarity", "score", "semantic_similarity", "fluency_score"):
            value = data.get(numeric)
            if value in (None, ""):
                data[numeric] = None
                continue
            try:
                data[numeric] = float(value)
            except (TypeError, ValueError):
                data[numeric] = None

        return data

    # ------------------------------------------------------------------
    # Transcript cache (saves CPU time, keeps Whisper runs to a minimum)
    # ------------------------------------------------------------------
    def get_cached_transcript(self, audio_hash: str, model: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM transcripts WHERE audio_hash = ? AND model = ?",
                (audio_hash, model),
            ).fetchone()
        return dict(row) if row else None

    def cache_transcript(
        self,
        audio_hash: str,
        transcript: str,
        model: str,
        language: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO transcripts
                    (audio_hash, transcript, language, model, duration, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (audio_hash, transcript, language, model, duration, _utcnow()),
            )

    # ------------------------------------------------------------------
    # Gemini response cache (this is what keeps us inside the free tier)
    # ------------------------------------------------------------------
    def get_cached_ai_response(self, prompt_hash: str, ttl_hours: int = 720) -> Optional[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max(1, ttl_hours))).isoformat(
            timespec="seconds"
        )
        with self.connect() as conn:
            row = conn.execute(
                "SELECT response FROM ai_cache WHERE prompt_hash = ? AND created_at >= ?",
                (prompt_hash, cutoff),
            ).fetchone()
        return row["response"] if row else None

    def cache_ai_response(self, prompt_hash: str, response: str, model: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ai_cache (prompt_hash, response, model, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (prompt_hash, response, model, _utcnow()),
            )

    # ------------------------------------------------------------------
    # API usage accounting (client-side rate limiting)
    # ------------------------------------------------------------------
    def record_api_call(self, provider: str = "gemini") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO api_usage (provider, called_at) VALUES (?, ?)",
                (provider, _utcnow()),
            )
            # Keep the table tiny - anything older than 2 days is useless.
            cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec="seconds")
            conn.execute("DELETE FROM api_usage WHERE called_at < ?", (cutoff,))

    def count_api_calls(self, provider: str = "gemini", within_seconds: int = 60) -> int:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=max(1, within_seconds))
        ).isoformat(timespec="seconds")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM api_usage WHERE provider = ? AND called_at >= ?",
                (provider, cutoff),
            ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # Analytics helpers used by the dashboard
    # ------------------------------------------------------------------
    def summary_stats(self) -> Dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                      AS total,
                    AVG(CAST(score AS REAL))      AS avg_score,
                    AVG(CAST(similarity AS REAL)) AS avg_similarity,
                    MAX(CAST(score AS REAL))      AS best_score
                FROM results
                """
            ).fetchone()
            grades = conn.execute(
                """
                SELECT COALESCE(grade, 'N/A') AS grade, COUNT(*) AS count
                FROM results GROUP BY grade ORDER BY count DESC
                """
            ).fetchall()
        return {
            "total": int(row["total"] or 0),
            "avg_score": float(row["avg_score"]) if row["avg_score"] is not None else None,
            "avg_similarity": (
                float(row["avg_similarity"]) if row["avg_similarity"] is not None else None
            ),
            "best_score": float(row["best_score"]) if row["best_score"] is not None else None,
            "grades": [(g["grade"], int(g["count"])) for g in grades],
        }

    def score_trend(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, CAST(score AS REAL) AS score,
                       CAST(similarity AS REAL) AS similarity
                FROM results
                WHERE score IS NOT NULL
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]


_INSTANCES: Dict[str, Database] = {}


def get_database(path: Path | str) -> Database:
    """Return a process-wide :class:`Database` for ``path`` (cheap to call)."""
    key = str(Path(path).resolve())
    if key not in _INSTANCES:
        _INSTANCES[key] = Database(key)
    return _INSTANCES[key]


__all__ = ["Database", "get_database", "sha256_bytes", "sha256_text", "RESULT_COLUMNS"]
