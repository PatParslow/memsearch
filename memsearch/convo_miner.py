"""Mine Claude Code's own session transcripts (~/.claude/projects/*/*.jsonl).

Chunk by exchange pair: one user turn + the assistant response that
follows it = one chunk. Real schema, verified against actual session
files: {"type": "user"|"assistant", "message": {"content": ...}}.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import store

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
    chunks = []
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            piece = f"> {text}"
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                piece += f"\n\n{messages[i + 1][1]}"
                i += 1
            chunks.append(piece)
        else:
            chunks.append(text)
        i += 1
    return [c for c in chunks if len(c.strip()) >= 40]


def mine_convos(
    claude_projects_dir: str = CLAUDE_PROJECTS_DIR,
    store_path: str = store.DEFAULT_STORE_PATH,
    dry_run: bool = False,
) -> dict:
    root = Path(claude_projects_dir)
    col = None if dry_run else store.get_collection(store_path)
    stats = {"sessions_processed": 0, "sessions_unchanged": 0, "chunks": 0}

    if not root.is_dir():
        return stats

    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        project = _decode_project_dir_name(project_dir.name)
        # Main session transcripts live directly in the project dir; forked
        # subagent transcripts live one level deeper under
        # <session-id>/subagents/agent-*.jsonl -- rglob catches both.
        for session_file in project_dir.rglob("*.jsonl"):
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

            if not dry_run and store.already_current(col, source_file, content_hash):
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
