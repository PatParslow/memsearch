"""Mine Claude Code's own session transcripts (~/.claude/projects/*/*.jsonl).

Chunk by exchange pair: one user turn + the assistant response that
follows it = one chunk. Real schema, verified against actual session
files: {"type": "user"|"assistant", "message": {"content": ...}}.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from . import store
from .chunking import MAX_CHUNK_SIZE, overlap_prefix, split_bounded

CLAUDE_PROJECTS_DIR = os.path.expanduser(r"~\.claude\projects")


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _decode_project_dir_name(name: str) -> str:
    """`D--Projects-Organized-parslow-soft-editorial` -> a readable label.
    Best-effort only -- used as the `project` tag, not a real path."""
    return name


def parse_session(path: Path) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return messages
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        msg_type = entry.get("type", "")
        message = entry.get("message", {})
        if not isinstance(message, dict):
            continue
        text = _extract_text(message.get("content", ""))
        if not text.strip():
            continue
        if msg_type == "user":
            messages.append(("user", text))
        elif msg_type == "assistant":
            messages.append(("assistant", text))
    return messages


def chunk_exchanges(messages: list[tuple[str, str]]) -> list[str]:
    """One chunk per exchange pair, with the same two protections
    chunking.py's split_bounded already gives project files:

    1. Size-bounded: an exchange with no bound at all (the original
       behaviour here) can run far past the embedding model's own
       256-token truncation limit (see gpu/store.py's ONNXMiniLM_L6_V2) --
       a long assistant response full of code becomes one giant chunk
       whose embedding reflects only its first ~256 tokens, silently,
       even though the full text is still stored and returned. Each
       exchange is now run through split_bounded, same as any other
       oversized block of text.

    2. Overlapping: split_bounded only adds overlap BETWEEN pieces it
       itself splits from the SAME exchange. The far more common boundary
       here is exchange-to-exchange (one user/assistant pair to the
       next), which had no overlap at all -- real conversational context
       (anaphora, a continuing thread) routinely spans that boundary.
       Each exchange's first piece is prefixed with a word-overlap tail
       from the PREVIOUS exchange's last piece, the same mechanism and
       bound (MIN_OVERLAP_WORDS) chunking.py uses.
    """
    raw = []
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            piece = f"> {text}"
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                piece += f"\n\n{messages[i + 1][1]}"
                i += 1
            raw.append(piece)
        else:
            raw.append(text)
        i += 1

    kept_raw = [c for c in raw if len(c.strip()) >= 40]
    all_pieces: list[str] = []
    for raw_piece in kept_raw:
        sub_pieces = split_bounded(raw_piece, max_size=MAX_CHUNK_SIZE)
        if all_pieces and sub_pieces:
            tail = overlap_prefix(all_pieces[-1])
            if tail:
                sub_pieces[0] = f"{tail}\n\n{sub_pieces[0]}"
        all_pieces.extend(sub_pieces)
    return all_pieces


def mine_convos(
    claude_projects_dir: str = CLAUDE_PROJECTS_DIR,
    store_path: str = store.DEFAULT_STORE_PATH,
    dry_run: bool = False,
    force: bool = False,
    progress_every: int = 0,
) -> dict:
    """`force=True` bypasses the unchanged-content-hash skip for every
    session, the same purpose as mine_project's force_paths/force_all --
    needed because a chunking-LOGIC change here (see chunk_exchanges) is
    otherwise invisible to a plain re-mine of transcripts whose bytes
    haven't changed. `progress_every`: 0 (default) prints nothing; a
    positive value prints an elapsed-time status line every that many
    sessions SEEN, same convention as miner.mine_files/contextual's
    build_or_update."""
    t0 = time.monotonic()
    root = Path(claude_projects_dir)
    col = None if dry_run else store.get_collection(store_path)
    stats = {"sessions_processed": 0, "sessions_unchanged": 0, "chunks": 0}

    if not root.is_dir():
        return stats

    seen = 0
    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        project = _decode_project_dir_name(project_dir.name)
        # Main session transcripts live directly in the project dir; forked
        # subagent transcripts live one level deeper under
        # <session-id>/subagents/agent-*.jsonl -- rglob catches both.
        for session_file in project_dir.rglob("*.jsonl"):
            seen += 1
            if progress_every and seen % progress_every == 0:
                elapsed = time.monotonic() - t0
                print(f"  [{elapsed:7.1f}s] seen {seen:,} sessions "
                      f"(processed {stats['sessions_processed']:,}, unchanged {stats['sessions_unchanged']:,}, "
                      f"chunks {stats['chunks']:,})", file=sys.stderr, flush=True)
            try:
                raw = session_file.read_bytes()
            except OSError:
                continue
            content_hash = hashlib.sha256(raw).hexdigest()[:16]
            source_file = str(session_file)
            # See miner.py's identical comment: id must include the path,
            # not just content_hash, or two files with identical bytes
            # (e.g. two aborted, near-empty sessions) collide.
            path_hash = hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:12]

            if not dry_run and not force and store.already_current(col, source_file, content_hash):
                stats["sessions_unchanged"] += 1
                continue

            messages = parse_session(session_file)
            pieces = chunk_exchanges(messages)
            stats["sessions_processed"] += 1
            stats["chunks"] += len(pieces)

            if dry_run:
                continue

            chunks = [
                {
                    "id": f"{path_hash}-{content_hash}-{i}",
                    "text": piece,
                    "metadata": {
                        "project": project,
                        "category": "conversation",
                        "source_file": source_file,
                        "content_hash": content_hash,
                        "mtime": session_file.stat().st_mtime,
                        "kind": "conversation",
                    },
                }
                for i, piece in enumerate(pieces)
            ]
            store.replace_file(col, source_file, chunks)

    return stats
