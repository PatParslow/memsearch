"""Tests for the targeted 'patch' re-mine mechanism: mine_project's
already_current skip means a chunking-LOGIC change (content unchanged)
is invisible to a plain re-mine. mine_files (force=True, always) and
mine_project's force_paths/force_all are the hooks that bypass that skip
deliberately, and store.source_file_summary is how affected files are
found without re-chunking the whole corpus first.
"""

from __future__ import annotations

from pathlib import Path

from memsearch import chunking, miner, store


def _col(tmp_path):
    return store.get_collection(str(tmp_path / "chroma"))


def test_mine_project_skips_unchanged_content_by_default(tmp_path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "a.md").write_text("Some prose content here, long enough to count.")
    store_path = str(tmp_path / "chroma")

    stats1 = miner.mine_project(str(proj_dir), project="proj", store_path=store_path)
    assert stats1["processed"] == 1

    stats2 = miner.mine_project(str(proj_dir), project="proj", store_path=store_path)
    assert stats2["processed"] == 0
    assert stats2["skipped_unchanged"] == 1


def test_mine_project_force_paths_rechunks_despite_unchanged_content(tmp_path, monkeypatch):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    target = proj_dir / "a.md"
    target.write_text("Some prose content here, long enough to count.")
    store_path = str(tmp_path / "chroma")

    miner.mine_project(str(proj_dir), project="proj", store_path=store_path)

    # Simulate a chunking-LOGIC change with the file's content held fixed:
    # a plain re-mine must not notice it (content hash is identical), but
    # force_paths must pick it up anyway.
    monkeypatch.setattr(chunking, "chunk_file", lambda path, text: ["NEW CHUNKING OUTPUT " * 3])

    plain = miner.mine_project(str(proj_dir), project="proj", store_path=store_path)
    assert plain["processed"] == 0, "a content-unchanged file must be invisible to a plain re-mine"

    forced = miner.mine_project(
        str(proj_dir), project="proj", store_path=store_path,
        force_paths={str(target)},
    )
    assert forced["processed"] == 1

    col = store.get_collection(store_path)
    stored = col.get(where={"source_file": str(target)}, include=["documents"])
    assert stored["documents"] == ["NEW CHUNKING OUTPUT " * 3]


def test_mine_files_rechunks_by_path_without_a_project_root(tmp_path, monkeypatch):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    target = proj_dir / "sub" / "b.md"
    target.parent.mkdir()
    target.write_text("Original content, long enough to pass the floor.")
    store_path = str(tmp_path / "chroma")

    miner.mine_project(str(proj_dir), project="proj", store_path=store_path)
    col = store.get_collection(store_path)
    before = col.get(where={"source_file": str(target)}, include=["metadatas"])
    assert before["metadatas"][0]["project"] == "proj"
    assert before["metadatas"][0]["category"] == "sub"

    monkeypatch.setattr(chunking, "chunk_file", lambda path, text: ["REMINED TEXT " * 5])

    stats = miner.mine_files([str(target)], store_path=store_path)
    assert stats["processed"] == 1
    assert stats["not_found"] == 0

    after = col.get(where={"source_file": str(target)}, include=["documents", "metadatas"])
    assert after["documents"] == ["REMINED TEXT " * 5]
    # project/category carried over from the file's own prior metadata,
    # not re-derived -- mine_files has no directory root to derive it from.
    assert after["metadatas"][0]["project"] == "proj"
    assert after["metadatas"][0]["category"] == "sub"


def test_mine_files_reports_not_found_for_an_unmined_path(tmp_path):
    store_path = str(tmp_path / "chroma")
    store.get_collection(store_path)  # create the (empty) collection
    stats = miner.mine_files(["/no/such/file.md"], store_path=store_path)
    assert stats["not_found"] == 1
    assert stats["processed"] == 0


def test_source_file_summary_counts_and_project(tmp_path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "single.md").write_text("Short single-chunk content, comfortably over the floor.")
    big_para = "word " * 400  # forces _split_bounded into >1 piece
    (proj_dir / "multi.md").write_text(big_para)
    store_path = str(tmp_path / "chroma")

    miner.mine_project(str(proj_dir), project="proj", store_path=store_path)

    summary = store.source_file_summary(store_path)
    single_path = str(proj_dir / "single.md")
    multi_path = str(proj_dir / "multi.md")
    assert summary[single_path]["count"] == 1
    assert summary[single_path]["project"] == "proj"
    assert summary[multi_path]["count"] > 1


def test_mine_files_refuses_a_conversation_chunk(tmp_path):
    """convo_miner.py stamps kind="conversation" and never calls
    chunking.chunk_file -- mine_files must refuse those rather than
    re-chunk a .jsonl transcript as if it were a project file."""
    store_path = str(tmp_path / "chroma")
    col = store.get_collection(store_path)
    sf = str(tmp_path / "session.jsonl")
    col.add(
        ids=["x-0"],
        documents=["some conversation text"],
        metadatas=[{
            "project": "someproj", "source_file": sf,
            "content_hash": "abc", "kind": "conversation",
        }],
    )
    stats = miner.mine_files([sf], store_path=store_path)
    assert stats["wrong_miner"] == 1
    assert stats["processed"] == 0


def test_remine_affected_selection_only_targets_multi_chunk_files(tmp_path):
    """The actual criterion cmd_remine_affected uses: a file stored as
    exactly one chunk went through _split_bounded's single-piece path,
    which the chunking-boundary fix provably doesn't change -- so only
    multi-chunk files should ever be selected for a patch re-mine."""
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "single.md").write_text("Short single-chunk content, comfortably over the floor.")
    (proj_dir / "multi.md").write_text("word " * 400)
    store_path = str(tmp_path / "chroma")

    miner.mine_project(str(proj_dir), project="proj", store_path=store_path)
    summary = store.source_file_summary(store_path)
    affected = sorted(sf for sf, info in summary.items() if info["count"] > 1)

    assert affected == [str(proj_dir / "multi.md")]
