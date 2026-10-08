"""Chromadb storage -- created explicitly in cosine space, unlike
MemPalace's collections (which defaulted to L2, making every displayed
similarity score meaningless -- see memsearch-project memory notes).
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

DEFAULT_STORE_PATH = os.path.expanduser(r"~\.memsearch\store")
COLLECTION_NAME = "memsearch_chunks"

# chromadb's default embedding function always tries TensorrtExecutionProvider
# first regardless of whether TensorRT is installed -- it isn't here (only
# CUDA/cuDNN support was ever set up, deliberately; TensorRT is a separate,
# heavier install this small embedding model doesn't need). Naming the
# providers we actually have skips the doomed TensorRT load attempt instead
# of logging a scary-looking (but harmless) failure-then-fallback every run.
_EMBEDDING_FUNCTION = ONNXMiniLM_L6_V2(
    preferred_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)


def get_collection(store_path: str = DEFAULT_STORE_PATH):
    # chromadb defaults anonymized_telemetry=True (phones home to PostHog) --
    # this is meant to be a fully local, private tool, and that's also a
    # real (if usually non-fatal) network dependency a purely local run
    # shouldn't have.
    client = chromadb.PersistentClient(
        path=store_path, settings=Settings(anonymized_telemetry=False)
    )
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=_EMBEDDING_FUNCTION)
    except Exception:
        return client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
            embedding_function=_EMBEDDING_FUNCTION,
        )


def already_current(col, source_file: str, content_hash: str) -> bool:
    """True if this exact file content is already stored (skip re-mining)."""
    existing = col.get(
        where={"$and": [{"source_file": source_file}, {"content_hash": content_hash}]},
        limit=1,
        include=[],
    )
    return bool(existing["ids"])


def replace_file(col, source_file: str, chunks: list[dict]) -> int:
    """Delete any existing chunks for this source file, then add the new
    ones. `chunks` is a list of dicts with keys: id, text, metadata."""
    existing = col.get(where={"source_file": source_file}, include=[])
    if existing["ids"]:
        col.delete(ids=existing["ids"])
    if not chunks:
        return 0
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def search(col, query: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
    kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": doc,
                "project": meta.get("project", "?"),
                "category": meta.get("category", "?"),
                "source_file": Path(meta.get("source_file", "?")).name,
                # Real cosine space this time -- 1 - dist IS the similarity.
                "similarity": round(1 - dist, 3),
            }
        )
    return hits


def delete_path_prefix(path_prefix: str, store_path: str = DEFAULT_STORE_PATH) -> int:
    """Delete every chunk whose source_file starts with the given path --
    for removing a directory from scope after the fact (files still exist
    on disk, so prune_missing() won't touch them; this is the explicit
    opt-out counterpart to mine's --exclude)."""
    col = get_collection(store_path)
    total = col.count()
    batch_size = 5000
    offset = 0
    prefix = str(Path(path_prefix))
    stale_ids: list[str] = []
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["metadatas"])
        for id_, m in zip(batch["ids"], batch["metadatas"]):
            sf = m.get("source_file", "")
            if sf.startswith(prefix):
                stale_ids.append(id_)
        offset += batch_size
    for i in range(0, len(stale_ids), batch_size):
        col.delete(ids=stale_ids[i : i + batch_size])
    return len(stale_ids)


def prune_missing(store_path: str = DEFAULT_STORE_PATH) -> dict:
    """Delete every chunk whose source_file no longer exists on disk.
    File-existence is checked once per distinct source_file (a file can
    have many chunks), not once per chunk."""
    col = get_collection(store_path)
    total = col.count()
    batch_size = 5000
    offset = 0
    file_exists: dict[str, bool] = {}
    stale_ids: list[str] = []
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["metadatas"])
        for id_, m in zip(batch["ids"], batch["metadatas"]):
            sf = m.get("source_file")
            if not sf:
                continue
            if sf not in file_exists:
                file_exists[sf] = Path(sf).is_file()
            if not file_exists[sf]:
                stale_ids.append(id_)
        offset += batch_size
    for i in range(0, len(stale_ids), batch_size):
        col.delete(ids=stale_ids[i : i + batch_size])
    return {
        "files_checked": len(file_exists),
        "files_missing": sum(1 for exists in file_exists.values() if not exists),
        "chunks_deleted": len(stale_ids),
    }


def source_file_summary(store_path: str = DEFAULT_STORE_PATH) -> dict[str, dict]:
    """source_file -> {"count": n, "project": p, "kind": k}, one pass over
    the whole store. Used to find which already-mined files a chunking-
    logic change can actually affect, without re-reading or re-chunking
    anything: a file currently stored as exactly one chunk went through
    _split_bounded's single-piece path, which is provably unchanged by the
    word-boundary and overlap fix (see
    test_split_bounded_adds_no_overlap_when_text_fits_in_one_piece) -- only
    files with more than one stored chunk need a re-mine. kind is carried
    along too: convo_miner.py stamps "conversation" and has its own
    chunking entirely separate from chunking.py (chunk_file is never
    called on a .jsonl transcript), so those files are never affected by
    a chunking.py change regardless of chunk count. Project is carried
    along in the same scan so a project filter doesn't need a separate
    per-file lookup."""
    col = get_collection(store_path)
    total = col.count()
    batch_size = 5000
    offset = 0
    summary: dict[str, dict] = {}
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["metadatas"])
        for m in batch["metadatas"]:
            sf = m.get("source_file")
            if not sf:
                continue
            entry = summary.setdefault(
                sf, {"count": 0, "project": m.get("project"), "kind": m.get("kind")}
            )
            entry["count"] += 1
        offset += batch_size
    return summary


def total_count(store_path: str = DEFAULT_STORE_PATH) -> int:
    col = get_collection(store_path)
    return col.count()


def status_breakdown(store_path: str = DEFAULT_STORE_PATH) -> dict[str, dict[str, int]]:
    """Project -> category -> count, fetched in full (no hardcoded cap --
    this is the exact bug that made MemPalace's status silently truncate
    at 10,000 rows)."""
    col = get_collection(store_path)
    total = col.count()
    counts: dict[str, dict[str, int]] = {}
    batch_size = 5000
    offset = 0
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["metadatas"])
        for m in batch["metadatas"]:
            project = m.get("project", "?")
            category = m.get("category", "?")
            counts.setdefault(project, {}).setdefault(category, 0)
            counts[project][category] += 1
        offset += batch_size
    return counts
