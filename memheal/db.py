"""SQLite schema and helpers for the instrumented memory store (proposal Section 6.2).

One SQLite file per experiment run. Snapshotting = copying the file (Section 6.6).
Ground-truth attack labels are never written here; they live in the harness.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,
    source_type   TEXT NOT NULL,             -- user | webpage | ...
    content_hash  TEXT NOT NULL,
    trust_class   TEXT NOT NULL,             -- trusted | untrusted
    content       TEXT NOT NULL,             -- raw text, to measure what survives extraction
    session       INTEGER
);

CREATE TABLE IF NOT EXISTS memories (
    id                 TEXT PRIMARY KEY,
    text               TEXT NOT NULL,
    source_type        TEXT NOT NULL,        -- user | webpage | derived
    source_id          TEXT,
    operation          TEXT NOT NULL,        -- extract | summarize | reflect | reconstruct
    session            INTEGER NOT NULL,
    status             TEXT NOT NULL DEFAULT 'active',  -- active | quarantined | deleted
    created_by_prompt  TEXT,                 -- hash of the prompt template that produced it
    model              TEXT,
    embedding          BLOB,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS memory_edges (
    parent_id  TEXT NOT NULL REFERENCES memories(id),
    child_id   TEXT NOT NULL REFERENCES memories(id),
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    run_id     TEXT NOT NULL,
    query_id   TEXT NOT NULL,
    session    INTEGER,
    memory_id  TEXT NOT NULL REFERENCES memories(id),
    rank       INTEGER NOT NULL,
    score      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    session         INTEGER,
    step            INTEGER,
    tool_name       TEXT NOT NULL,
    arguments_json  TEXT NOT NULL,
    outcome_json    TEXT
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id       TEXT PRIMARY KEY,
    case_id      TEXT NOT NULL,
    condition    TEXT NOT NULL,
    model        TEXT,
    seed         INTEGER,
    config_json  TEXT NOT NULL,
    started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_edges_child ON memory_edges(child_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_query ON retrieval_events(run_id, query_id);
"""


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def snapshot(conn: sqlite3.Connection, dest_path: str | Path) -> None:
    """Copy the live database to dest_path (safe while conn is open)."""
    dest = sqlite3.connect(str(dest_path))
    with dest:
        conn.backup(dest)
    dest.close()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- inserts -----------------------------------------------------------------

def insert_source(
    conn: sqlite3.Connection,
    source_id: str,
    source_type: str,
    content: str,
    trust_class: str,
    session: int | None = None,
) -> str:
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (source_id, source_type, content_hash, trust_class, content, session)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, source_type, content_hash(content), trust_class, content, session),
        )
    return source_id


def insert_memory(
    conn: sqlite3.Connection,
    text: str,
    source_type: str,
    session: int,
    *,
    source_id: str | None = None,
    operation: str = "extract",
    status: str = "active",
    created_by_prompt: str | None = None,
    model: str | None = None,
    embedding: bytes | None = None,
    parent_ids: Sequence[str] = (),
    memory_id: str | None = None,
) -> str:
    mem_id = memory_id or f"m_{uuid.uuid4().hex[:8]}"
    with conn:
        conn.execute(
            "INSERT INTO memories (id, text, source_type, source_id, operation, session, status,"
            " created_by_prompt, model, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mem_id, text, source_type, source_id, operation, session, status,
             created_by_prompt, model, embedding),
        )
    if parent_ids:
        insert_edges(conn, mem_id, parent_ids)
    return mem_id


def insert_edges(conn: sqlite3.Connection, child_id: str, parent_ids: Iterable[str]) -> None:
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO memory_edges (parent_id, child_id) VALUES (?, ?)",
            [(p, child_id) for p in parent_ids],
        )


def log_action(
    conn: sqlite3.Connection,
    run_id: str,
    session: int | None,
    step: int | None,
    tool: str,
    args: dict[str, Any],
    outcome: Any = None,
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO actions (run_id, session, step, tool_name, arguments_json, outcome_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, session, step, tool, json.dumps(args, sort_keys=True), json.dumps(outcome)),
        )


def log_retrieval(
    conn: sqlite3.Connection,
    run_id: str,
    query_id: str,
    session: int | None,
    memory_ids: Sequence[str],
    scores: Sequence[float],
) -> None:
    with conn:
        conn.executemany(
            "INSERT INTO retrieval_events (run_id, query_id, session, memory_id, rank, score)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(run_id, query_id, session, m, rank, float(s))
             for rank, (m, s) in enumerate(zip(memory_ids, scores), start=1)],
        )


def insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    case_id: str,
    condition: str,
    model: str,
    config: dict[str, Any],
    seed: int | None = None,
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO experiment_runs (run_id, case_id, condition, model, seed, config_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, case_id, condition, model, seed, json.dumps(config, sort_keys=True, default=str)),
        )


def update_run_config(conn: sqlite3.Connection, run_id: str, **updates: Any) -> None:
    row = conn.execute("SELECT config_json FROM experiment_runs WHERE run_id = ?", (run_id,)).fetchone()
    config = json.loads(row["config_json"])
    config.update(updates)
    with conn:
        conn.execute(
            "UPDATE experiment_runs SET config_json = ? WHERE run_id = ?",
            (json.dumps(config, sort_keys=True, default=str), run_id),
        )


def set_status(conn: sqlite3.Connection, memory_id: str, status: str) -> bool:
    """Flag a memory (e.g. quarantined/deleted) without removing its row or edges."""
    with conn:
        cur = conn.execute("UPDATE memories SET status = ? WHERE id = ?", (status, memory_id))
    return cur.rowcount > 0


# --- queries -----------------------------------------------------------------

def next_session(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(session) FROM memories").fetchone()
    return (row[0] or 0) + 1


def get_memory(conn: sqlite3.Connection, memory_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()


def all_memories(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM memories ORDER BY session, created_at, rowid").fetchall()


def active_memories(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM memories WHERE status = 'active' ORDER BY session, created_at, rowid"
    ).fetchall()


def get_actions(conn: sqlite3.Connection, run_id: str, session: int | None = None) -> list[sqlite3.Row]:
    if session is None:
        return conn.execute("SELECT * FROM actions WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return conn.execute(
        "SELECT * FROM actions WHERE run_id = ? AND session = ? ORDER BY id", (run_id, session)
    ).fetchall()


def get_retrievals(conn: sqlite3.Connection, run_id: str, query_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM retrieval_events WHERE run_id = ? AND query_id = ? ORDER BY rank",
        (run_id, query_id),
    ).fetchall()


def get_descendants(conn: sqlite3.Connection, memory_id: str) -> set[str]:
    """Every memory reachable via parent->child edges, excluding memory_id itself."""
    rows = conn.execute(
        """
        WITH RECURSIVE descendants(id) AS (
            SELECT child_id FROM memory_edges WHERE parent_id = ?
            UNION
            SELECT e.child_id FROM memory_edges e JOIN descendants d ON e.parent_id = d.id
        )
        SELECT id FROM descendants
        """,
        (memory_id,),
    ).fetchall()
    return {r["id"] for r in rows} - {memory_id}


def get_ancestors(conn: sqlite3.Connection, memory_id: str) -> set[str]:
    """Every memory reachable via child->parent edges, excluding memory_id itself."""
    rows = conn.execute(
        """
        WITH RECURSIVE ancestors(id) AS (
            SELECT parent_id FROM memory_edges WHERE child_id = ?
            UNION
            SELECT e.parent_id FROM memory_edges e JOIN ancestors a ON e.child_id = a.id
        )
        SELECT id FROM ancestors
        """,
        (memory_id,),
    ).fetchall()
    return {r["id"] for r in rows} - {memory_id}
