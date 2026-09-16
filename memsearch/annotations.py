"""SQLite persistence for user-authored graph content -- notes, manually
drawn links, and gap flags (both auto-detected and manual). Kept entirely
separate from chromadb and from the regenerable graph.json, so re-running
`memsearch graph build` never touches anything the user wrote."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = os.path.expanduser(r"~\.memsearch\annotations.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_node ON notes(node_id);

CREATE TABLE IF NOT EXISTS manual_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_a TEXT NOT NULL,
    node_b TEXT NOT NULL,
    label TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_a ON manual_links(node_a);
CREATE INDEX IF NOT EXISTS idx_links_b ON manual_links(node_b);

CREATE TABLE IF NOT EXISTS gaps (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,            -- 'auto' | 'manual'
    node_a TEXT NOT NULL,
    node_b TEXT,
    description TEXT NOT NULL,
    score REAL,
    status TEXT NOT NULL DEFAULT 'open',   -- 'open' | 'dismissed'
    created_at REAL NOT NULL
);
"""


def get_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def gap_id(kind: str, node_a: str, node_b: str | None, description: str) -> str:
    raw = f"{kind}:{node_a}:{node_b or ''}:{description}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---- notes ----

def add_note(conn: sqlite3.Connection, node_id: str, text: str) -> int:
    now = time.time()
    cur = conn.execute(
        "INSERT INTO notes (node_id, text, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (node_id, text, now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def notes_for_node(conn: sqlite3.Connection, node_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, node_id, text, created_at, updated_at FROM notes WHERE node_id = ? ORDER BY created_at",
        (node_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def all_notes(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT id, node_id, text, created_at, updated_at FROM notes").fetchall()
    return [dict(r) for r in rows]


def delete_note(conn: sqlite3.Connection, note_id: int) -> None:
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()


# ---- manual links ----

def add_manual_link(conn: sqlite3.Connection, node_a: str, node_b: str, label: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO manual_links (node_a, node_b, label, created_at) VALUES (?, ?, ?, ?)",
        (node_a, node_b, label, time.time()),
    )
    conn.commit()
    return cur.lastrowid


def all_manual_links(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT id, node_a, node_b, label, created_at FROM manual_links").fetchall()
    return [dict(r) for r in rows]


def delete_manual_link(conn: sqlite3.Connection, link_id: int) -> None:
    conn.execute("DELETE FROM manual_links WHERE id = ?", (link_id,))
    conn.commit()


# ---- gaps ----

def upsert_auto_gaps(conn: sqlite3.Connection, gaps: list[dict]) -> None:
    """Insert freshly-detected auto gaps, preserving `status` (in particular
    `dismissed`) for any gap that already exists under the same stable id."""
    now = time.time()
    for g in gaps:
        gid = gap_id("auto", g["node_a"], g.get("node_b"), g["description"])
        existing = conn.execute("SELECT status FROM gaps WHERE id = ?", (gid,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE gaps SET description = ?, score = ? WHERE id = ?",
                (g["description"], g.get("score"), gid),
            )
        else:
            conn.execute(
                "INSERT INTO gaps (id, kind, node_a, node_b, description, score, status, created_at) "
                "VALUES (?, 'auto', ?, ?, ?, ?, 'open', ?)",
                (gid, g["node_a"], g.get("node_b"), g["description"], g.get("score"), now),
            )
    conn.commit()


def add_gap(conn: sqlite3.Connection, kind: str, node_a: str, node_b: str | None,
            description: str, status: str = "open") -> str:
    """Insert or update a gap under the SAME id-computation upsert_auto_gaps
    uses for kind='auto', so dismissing an auto-detected gap before it has
    ever been through a `graph build` upsert still lands on the id a later
    build will look for -- otherwise the dismissal (stored under a
    different hash) would silently fail to stick past the next rebuild."""
    gid = gap_id(kind, node_a, node_b, description)
    conn.execute(
        "INSERT OR REPLACE INTO gaps (id, kind, node_a, node_b, description, score, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
        (gid, kind, node_a, node_b, description, status, time.time()),
    )
    conn.commit()
    return gid


def add_manual_gap(conn: sqlite3.Connection, node_a: str, node_b: str | None, description: str) -> str:
    return add_gap(conn, "manual", node_a, node_b, description, status="open")


def all_gaps(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, kind, node_a, node_b, description, score, status, created_at FROM gaps"
    ).fetchall()
    return [dict(r) for r in rows]


def set_gap_status(conn: sqlite3.Connection, gap_id_: str, status: str) -> None:
    conn.execute("UPDATE gaps SET status = ? WHERE id = ?", (status, gap_id_))
    conn.commit()
