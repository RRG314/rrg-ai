from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .recursive_adic import depth_laplace_weight, recursive_adic_score, recursive_depth


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "have",
    "just",
    "about",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "their",
    "there",
    "been",
    "also",
    "will",
    "them",
}


@dataclass
class MemoryFact:
    key: str
    value: str
    updated_at: int


@dataclass
class TaskRecord:
    task_id: str
    session_id: str
    title: str
    status: str
    created_at: int
    updated_at: int
    steps: list[dict[str, object]]
    outputs: dict[str, object]
    provenance: list[dict[str, object]]


@dataclass
class StructuredMemoryItem:
    id: int
    session_id: str
    key: str
    value: str
    source: str
    created_at: int
    updated_at: int


@dataclass
class OutcomeItem:
    id: int
    session_id: str
    title: str
    summary: str
    status: str
    score: float
    created_at: int
    updated_at: int


@dataclass
class ArtifactItem:
    id: int
    session_id: str
    artifact_type: str
    location: str
    source: str
    doc_id: str
    description: str
    created_at: int


@dataclass
class PlanningHeuristicItem:
    key: str
    value: float
    source: str
    updated_at: int


@dataclass
class ImprovementRuleItem:
    id: int
    session_id: str
    task_id: str
    rule: str
    trigger: str
    confidence: float
    created_at: int


class SQLiteStore:
    def __init__(
        self,
        db_path: Path,
        use_recursive_adic: bool = True,
        radf_beta: float = 0.35,
        radf_alpha: float = 1.5,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.use_recursive_adic = use_recursive_adic
        self.radf_beta = float(radf_beta)
        self.radf_alpha = float(radf_alpha)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(session_id, key)
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    steps_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(session_id, key)
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(session_id, key)
                );

                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    location TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    doc_id TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS planning_heuristics (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT 'adaptive',
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS improvement_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    rule TEXT NOT NULL,
                    trigger TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                );
                """
            )

    def ensure_session(self, session_id: str | None, title: str = "Chat") -> str:
        sid = session_id or str(uuid.uuid4())
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT id FROM sessions WHERE id = ?", (sid,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (sid, title, now, now),
                )
            else:
                conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
        return sid

    def add_message(self, session_id: str, role: str, content: str) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

    def recent_messages(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def upsert_memory(self, session_id: str, key: str, value: str) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory(session_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, key)
                DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (session_id, key, value, now),
            )

    def memory_for_session(self, session_id: str) -> list[MemoryFact]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value, updated_at FROM memory WHERE session_id = ? ORDER BY updated_at DESC",
                (session_id,),
            ).fetchall()
        return [MemoryFact(key=r["key"], value=r["value"], updated_at=r["updated_at"]) for r in rows]

    def list_sessions(self, limit: int = 100) -> list[dict[str, str | int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_fact(self, session_id: str, key: str, value: str, source: str = "user") -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO facts(session_id, key, value, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, key)
                DO UPDATE SET value = excluded.value, source = excluded.source, updated_at = excluded.updated_at
                """,
                (session_id, key, value, source, now, now),
            )

    def upsert_preference(self, session_id: str, key: str, value: str, source: str = "user") -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO preferences(session_id, key, value, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, key)
                DO UPDATE SET value = excluded.value, source = excluded.source, updated_at = excluded.updated_at
                """,
                (session_id, key, value, source, now, now),
            )

    def add_outcome(
        self,
        session_id: str,
        title: str,
        summary: str,
        status: str = "completed",
        score: float = 0.0,
    ) -> int:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO outcomes(session_id, title, summary, status, score, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, title, summary, status, float(score), now, now),
            )
        return int(cursor.lastrowid)

    def add_artifact(
        self,
        session_id: str,
        artifact_type: str,
        location: str,
        source: str = "",
        doc_id: str = "",
        description: str = "",
    ) -> int:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO artifacts(session_id, artifact_type, location, source, doc_id, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    artifact_type,
                    location,
                    source,
                    doc_id,
                    description,
                    now,
                ),
            )
        return int(cursor.lastrowid)

    def list_facts(self, session_id: str, limit: int = 200) -> list[StructuredMemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, key, value, source, created_at, updated_at
                FROM facts
                WHERE session_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            StructuredMemoryItem(
                id=int(r["id"]),
                session_id=str(r["session_id"]),
                key=str(r["key"]),
                value=str(r["value"]),
                source=str(r["source"]),
                created_at=int(r["created_at"]),
                updated_at=int(r["updated_at"]),
            )
            for r in rows
        ]

    def list_preferences(self, session_id: str, limit: int = 200) -> list[StructuredMemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, key, value, source, created_at, updated_at
                FROM preferences
                WHERE session_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            StructuredMemoryItem(
                id=int(r["id"]),
                session_id=str(r["session_id"]),
                key=str(r["key"]),
                value=str(r["value"]),
                source=str(r["source"]),
                created_at=int(r["created_at"]),
                updated_at=int(r["updated_at"]),
            )
            for r in rows
        ]

    def list_outcomes(self, session_id: str, limit: int = 200) -> list[OutcomeItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, title, summary, status, score, created_at, updated_at
                FROM outcomes
                WHERE session_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            OutcomeItem(
                id=int(r["id"]),
                session_id=str(r["session_id"]),
                title=str(r["title"]),
                summary=str(r["summary"]),
                status=str(r["status"]),
                score=float(r["score"]),
                created_at=int(r["created_at"]),
                updated_at=int(r["updated_at"]),
            )
            for r in rows
        ]

    def list_artifacts(self, session_id: str, limit: int = 500) -> list[ArtifactItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, artifact_type, location, source, doc_id, description, created_at
                FROM artifacts
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            ArtifactItem(
                id=int(r["id"]),
                session_id=str(r["session_id"]),
                artifact_type=str(r["artifact_type"]),
                location=str(r["location"]),
                source=str(r["source"]),
                doc_id=str(r["doc_id"]),
                description=str(r["description"]),
                created_at=int(r["created_at"]),
            )
            for r in rows
        ]

    def upsert_planning_heuristic(self, key: str, value: float, source: str = "adaptive") -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planning_heuristics(key, value, source, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key)
                DO UPDATE SET value = excluded.value, source = excluded.source, updated_at = excluded.updated_at
                """,
                (key, float(value), source, now),
            )

    def list_planning_heuristics(self, limit: int = 100) -> list[PlanningHeuristicItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, value, source, updated_at
                FROM planning_heuristics
                ORDER BY updated_at DESC, key ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            PlanningHeuristicItem(
                key=str(r["key"]),
                value=float(r["value"]),
                source=str(r["source"]),
                updated_at=int(r["updated_at"]),
            )
            for r in rows
        ]

    def get_planning_heuristics(self, defaults: dict[str, float] | None = None) -> dict[str, float]:
        values = dict(defaults or {})
        for item in self.list_planning_heuristics(limit=500):
            values[item.key] = float(item.value)
        return values

    def add_improvement_rule(
        self,
        session_id: str,
        task_id: str,
        rule: str,
        trigger: str = "",
        confidence: float = 0.0,
    ) -> int:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO improvement_rules(session_id, task_id, rule, trigger, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, task_id, rule, trigger, float(confidence), now),
            )
        return int(cursor.lastrowid)

    def list_improvement_rules(self, session_id: str, limit: int = 200) -> list[ImprovementRuleItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, task_id, rule, trigger, confidence, created_at
                FROM improvement_rules
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            ImprovementRuleItem(
                id=int(r["id"]),
                session_id=str(r["session_id"]),
                task_id=str(r["task_id"]),
                rule=str(r["rule"]),
                trigger=str(r["trigger"]),
                confidence=float(r["confidence"]),
                created_at=int(r["created_at"]),
            )
            for r in rows
        ]

    def memory_snapshot(self, session_id: str, limit: int = 200) -> dict[str, object]:
        return {
            "facts": [x.__dict__ for x in self.list_facts(session_id, limit=limit)],
            "preferences": [x.__dict__ for x in self.list_preferences(session_id, limit=limit)],
            "outcomes": [x.__dict__ for x in self.list_outcomes(session_id, limit=limit)],
            "artifacts": [x.__dict__ for x in self.list_artifacts(session_id, limit=max(200, limit))],
            "planning_heuristics": [x.__dict__ for x in self.list_planning_heuristics(limit=100)],
            "improvement_rules": [x.__dict__ for x in self.list_improvement_rules(session_id, limit=limit)],
        }

    def create_task(
        self,
        session_id: str,
        title: str,
        status: str = "running",
        steps: list[dict[str, object]] | None = None,
        outputs: dict[str, object] | None = None,
        provenance: list[dict[str, object]] | None = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        now = int(time.time())
        steps = steps or []
        outputs = outputs or {}
        provenance = provenance or []

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(
                    task_id, session_id, title, status, created_at, updated_at,
                    steps_json, outputs_json, provenance_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    session_id,
                    title,
                    status,
                    now,
                    now,
                    _to_json(steps),
                    _to_json(outputs),
                    _to_json(provenance),
                ),
            )
        return task_id

    def update_task(
        self,
        task_id: str,
        status: str | None = None,
        steps: list[dict[str, object]] | None = None,
        outputs: dict[str, object] | None = None,
        provenance: list[dict[str, object]] | None = None,
    ) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT status, steps_json, outputs_json, provenance_json FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"task not found: {task_id}")

            next_status = status or str(row["status"])
            next_steps = steps if steps is not None else _from_json(str(row["steps_json"]), fallback=[])
            next_outputs = outputs if outputs is not None else _from_json(str(row["outputs_json"]), fallback={})
            next_provenance = (
                provenance if provenance is not None else _from_json(str(row["provenance_json"]), fallback=[])
            )

            conn.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?, steps_json = ?, outputs_json = ?, provenance_json = ?
                WHERE task_id = ?
                """,
                (
                    next_status,
                    now,
                    _to_json(next_steps),
                    _to_json(next_outputs),
                    _to_json(next_provenance),
                    task_id,
                ),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT task_id, session_id, title, status, created_at, updated_at,
                       steps_json, outputs_json, provenance_json
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None

        return TaskRecord(
            task_id=str(row["task_id"]),
            session_id=str(row["session_id"]),
            title=str(row["title"]),
            status=str(row["status"]),
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
            steps=_from_json(str(row["steps_json"]), fallback=[]),
            outputs=_from_json(str(row["outputs_json"]), fallback={}),
            provenance=_from_json(str(row["provenance_json"]), fallback=[]),
        )

    def list_tasks(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        query = (
            """
            SELECT task_id, session_id, title, status, created_at, updated_at,
                   steps_json, outputs_json, provenance_json
            FROM tasks
            {where_clause}
            ORDER BY updated_at DESC
            LIMIT ?
            """
        )
        params: tuple[object, ...]
        if session_id:
            sql = query.format(where_clause="WHERE session_id = ?")
            params = (session_id, limit)
        else:
            sql = query.format(where_clause="")
            params = (limit,)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        items: list[dict[str, object]] = []
        for row in rows:
            items.append(
                {
                    "task_id": str(row["task_id"]),
                    "session_id": str(row["session_id"]),
                    "title": str(row["title"]),
                    "status": str(row["status"]),
                    "created_at": int(row["created_at"]),
                    "updated_at": int(row["updated_at"]),
                    "steps": _from_json(str(row["steps_json"]), fallback=[]),
                    "outputs": _from_json(str(row["outputs_json"]), fallback={}),
                    "provenance": _from_json(str(row["provenance_json"]), fallback=[]),
                }
            )
        return items

    def add_document(self, name: str, source: str | None, kind: str, text: str) -> str:
        doc_id = str(uuid.uuid4())
        created_at = int(time.time())
        clean_text = text.strip()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO documents(id, name, source, kind, text, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, name, source, kind, clean_text, created_at),
            )
            for idx, chunk in enumerate(_chunk_text(clean_text, 1100, 200)):
                conn.execute(
                    "INSERT INTO chunks(doc_id, chunk_index, text) VALUES (?, ?, ?)",
                    (doc_id, idx, chunk),
                )
        return doc_id

    def list_documents(self, limit: int = 200) -> list[dict[str, str | int]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, source, kind, created_at,
                       LENGTH(text) AS char_count
                FROM documents
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_chunks(self, query: str, limit: int = 6) -> list[dict[str, str | float]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.id AS doc_id,
                    c.text AS chunk_text,
                    c.chunk_index AS chunk_index,
                    d.name AS doc_name,
                    d.source AS source,
                    d.created_at AS created_at
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                ORDER BY d.created_at DESC, c.chunk_index ASC, c.id ASC
                """
            ).fetchall()

        scored: list[dict[str, str | float]] = []
        for rank_idx, row in enumerate(rows, start=1):
            text = row["chunk_text"].lower()
            base_score = float(sum(text.count(tok) for tok in tokens))
            if base_score <= 0:
                continue

            if self.use_recursive_adic:
                score = recursive_adic_score(
                    base_score,
                    rank_index=rank_idx,
                    beta=self.radf_beta,
                    alpha=self.radf_alpha,
                )
                radf_weight = depth_laplace_weight(
                    rank_idx,
                    beta=self.radf_beta,
                    alpha=self.radf_alpha,
                )
                radf_depth = float(recursive_depth(rank_idx, alpha=self.radf_alpha))
            else:
                score = base_score
                radf_weight = 1.0
                radf_depth = 0.0

            scored.append(
                {
                    "doc_id": row["doc_id"],
                    "doc_name": row["doc_name"],
                    "source": row["source"] or "",
                    "text": row["chunk_text"],
                    "base_score": base_score,
                    "score": float(score),
                    "radf_weight": float(radf_weight),
                    "radf_depth": radf_depth,
                }
            )

        scored.sort(
            key=lambda x: (
                float(x["score"]),
                float(x["base_score"]),
                -float(x["radf_depth"]),
            ),
            reverse=True,
        )
        return scored[:limit]


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9_\-]{2,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _chunk_text(text: str, size: int, overlap: int) -> Iterable[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _to_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True)


def _from_json(raw: str, fallback: object) -> object:
    try:
        return json.loads(raw)
    except Exception:
        return fallback
