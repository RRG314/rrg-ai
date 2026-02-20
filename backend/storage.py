from __future__ import annotations

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
