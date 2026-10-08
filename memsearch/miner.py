"""Project file mining: walk a directory tree, skip binary/generated
noise, chunk what's left content-aware, and store it incrementally."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

from . import chunking, store
from .gitignore import GitignoreMatcher

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", "coverage", ".ruff_cache", ".mypy_cache",
    ".pytest_cache", ".cache", ".tox", ".nox", ".idea", ".vscode",
    ".ipynb_checkpoints", ".eggs", "htmlcov", "target",
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def _category(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "general"


def _walk(root: Path, gi: GitignoreMatcher, exclude_dirs: set[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and d not in exclude_dirs
            and not gi.is_ignored(Path(dirpath) / d)
        ]
        for name in filenames:
            path = Path(dirpath) / name
            if gi.is_ignored(path):
                continue
            yield path


def _process_file(
    path: Path,
    project: str,
    category: str,
    col,
    dry_run: bool,
    force: bool,
    stats: dict,
) -> None:
    """Chunk and (unless dry_run) store one file under the given
    project/category, honouring the same skip rules mine_project always
    has -- shared so mine_files (re-mining a specific file list without a
    directory walk) can't drift from mine_project's own per-file logic."""
    if chunking.is_binary_extension(path):
        stats["skipped_binary"] += 1
        return
    is_pdf = path.suffix.lower() in chunking.PDF_EXTENSIONS
    try:
        if path.stat().st_size > MAX_FILE_SIZE and not is_pdf:
            return
        raw = path.read_bytes()
    except OSError:
        return
    # PDFs are binary by nature (null bytes are routine) and are
    # extracted via PyMuPDF in chunk_pdf(), not decoded as UTF-8 here.
    if not is_pdf and b"\x00" in raw[:1024]:
        stats["skipped_binary"] += 1
        return

    content_hash = hashlib.sha256(raw).hexdigest()[:16]
    source_file = str(path)
    # Chunk ids need the path baked in, not just content_hash: two
    # different files with byte-identical content (seen in practice with
    # generated .dropcap_cache SVGs) produce the same content_hash, and
    # an id keyed on content_hash alone collides across them -- the
    # second file's add() then silently no-ops against the first file's
    # already-stored id instead of being indexed under its own path,
    # every single mining run forever (found via ~8,600 "Add of existing
    # embedding ID" warnings in one scheduled run's log).
    path_hash = hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:12]

    if not dry_run and not force and store.already_current(col, source_file, content_hash):
        stats["skipped_unchanged"] += 1
        return

    text = None
    if not is_pdf:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            stats["skipped_binary"] += 1
            return

    pieces = chunking.chunk_file(path, text)
    stats["processed"] += 1
    stats["chunks"] += len(pieces)

    if dry_run:
        return

    chunks = [
        {
            "id": f"{path_hash}-{content_hash}-{i}",
            "text": piece,
            "metadata": {
                "project": project,
                "category": category,
                "source_file": source_file,
                "content_hash": content_hash,
                "mtime": path.stat().st_mtime,
                "kind": "prose" if path.suffix.lower() in (".html", ".htm", ".md", ".pdf") else "code",
            },
        }
        for i, piece in enumerate(pieces)
    ]
    store.replace_file(col, source_file, chunks)


def mine_project(
    directory: str,
    project: str | None = None,
    store_path: str = store.DEFAULT_STORE_PATH,
    dry_run: bool = False,
    exclude_dirs: set[str] | None = None,
    force_paths: set[str] | None = None,
    force_all: bool = False,
) -> dict:
    """Walk `directory` and mine every file found, same as always, except
    that any file in `force_paths` (or every file, if `force_all`) is
    re-chunked and re-stored even when its content hash hasn't changed --
    the hook a chunking-logic change (not a content change) needs to take
    effect on already-mined files."""
    root = Path(directory).expanduser().resolve()
    project = project or root.name
    gi = GitignoreMatcher(root)
    col = None if dry_run else store.get_collection(store_path)
    exclude_dirs = exclude_dirs or set()

    stats = {"processed": 0, "skipped_unchanged": 0, "skipped_binary": 0, "chunks": 0}

    for path in _walk(root, gi, exclude_dirs):
        source_file = str(path)
        force = force_all or (force_paths is not None and source_file in force_paths)
        _process_file(path, project, _category(root, path), col, dry_run, force, stats)

    return stats


def mine_files(
    paths: list[str],
    store_path: str = store.DEFAULT_STORE_PATH,
    dry_run: bool = False,
    progress_every: int = 0,
) -> dict:
    """Re-chunk and re-store a specific list of already-mined source files,
    bypassing the unchanged-content-hash skip -- without needing their
    original project directory root. Each file's project/category is read
    back from its own existing chunk metadata rather than re-derived from a
    walk, so the list can span several different mined projects at once.
    The counterpart to mine_project's whole-directory walk, for a targeted
    "patch" re-mine of only the files a chunking change actually affects.

    `progress_every`: 0 (default) prints nothing; a positive value prints
    an elapsed-time status line every that many files SEEN (same
    convention as contextual's build_or_update) -- this list can include
    large PDFs, so a real re-mine over thousands of them needs to be
    told "slow" from "stuck" rather than run on faith."""
    t0 = time.monotonic()
    col = store.get_collection(store_path)
    stats = {
        "processed": 0, "skipped_unchanged": 0, "skipped_binary": 0,
        "chunks": 0, "not_found": 0, "wrong_miner": 0,
    }

    for seen, sf in enumerate(paths, start=1):
        if progress_every and seen % progress_every == 0:
            elapsed = time.monotonic() - t0
            print(f"  [{elapsed:7.1f}s] seen {seen:,}/{len(paths):,} files "
                  f"(processed {stats['processed']:,}, chunks {stats['chunks']:,})",
                  file=sys.stderr, flush=True)
        existing = col.get(where={"source_file": sf}, limit=1, include=["metadatas"])
        if not existing["ids"]:
            stats["not_found"] += 1
            continue
        meta0 = existing["metadatas"][0]
        if meta0.get("kind") == "conversation":
            # Stamped by convo_miner.py, which never calls chunking.py --
            # re-chunking it here with chunking.chunk_file would silently
            # corrupt a conversation transcript's chunks, not refresh them.
            stats["wrong_miner"] += 1
            continue
        project = meta0.get("project", "?")
        category = meta0.get("category", "general")
        _process_file(Path(sf), project, category, col, dry_run, force=True, stats=stats)

    return stats
