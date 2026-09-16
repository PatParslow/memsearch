"""Project file mining: walk a directory tree, skip binary/generated
noise, chunk what's left content-aware, and store it incrementally."""

from __future__ import annotations

import hashlib
import os
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


def mine_project(
    directory: str,
    project: str | None = None,
    store_path: str = store.DEFAULT_STORE_PATH,
    dry_run: bool = False,
    exclude_dirs: set[str] | None = None,
) -> dict:
    root = Path(directory).expanduser().resolve()
    project = project or root.name
    gi = GitignoreMatcher(root)
    col = None if dry_run else store.get_collection(store_path)
    exclude_dirs = exclude_dirs or set()

    stats = {"processed": 0, "skipped_unchanged": 0, "skipped_binary": 0, "chunks": 0}

    for path in _walk(root, gi, exclude_dirs):
        if chunking.is_binary_extension(path):
            stats["skipped_binary"] += 1
            continue
        is_pdf = path.suffix.lower() in chunking.PDF_EXTENSIONS
        try:
            if path.stat().st_size > MAX_FILE_SIZE and not is_pdf:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        # PDFs are binary by nature (null bytes are routine) and are
        # extracted via PyMuPDF in chunk_pdf(), not decoded as UTF-8 here.
        if not is_pdf and b"\x00" in raw[:1024]:
            stats["skipped_binary"] += 1
            continue

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

        if not dry_run and store.already_current(col, source_file, content_hash):
            stats["skipped_unchanged"] += 1
            continue

        text = None
        if not is_pdf:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                stats["skipped_binary"] += 1
                continue

        pieces = chunking.chunk_file(path, text)
        stats["processed"] += 1
        stats["chunks"] += len(pieces)

        if dry_run:
            continue

        chunks = [
            {
                "id": f"{path_hash}-{content_hash}-{i}",
                "text": piece,
                "metadata": {
                    "project": project,
                    "category": _category(root, path),
                    "source_file": source_file,
                    "content_hash": content_hash,
                    "mtime": path.stat().st_mtime,
                    "kind": "prose" if path.suffix.lower() in (".html", ".htm", ".md", ".pdf") else "code",
                },
            }
            for i, piece in enumerate(pieces)
        ]
        store.replace_file(col, source_file, chunks)

    return stats
