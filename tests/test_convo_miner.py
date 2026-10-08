"""chunk_exchanges had neither of the two protections chunking.py's
split_bounded already has for project files: no overlap between adjacent
chunks (so cross-turn context -- "it", "that", a continuing thread -- is
invisible to anything that only looks within one chunk), and no size
bound at all (so one huge exchange, e.g. a long assistant response full
of code, becomes a single giant chunk -- and since the embedding model
silently truncates at 256 tokens, most of a large exchange is invisible
to semantic search even though the full text is stored and returned).
"""

from __future__ import annotations

from memsearch import chunking
from memsearch.convo_miner import chunk_exchanges


def test_adjacent_exchanges_get_overlap_from_the_previous_one():
    messages = [
        ("user", "What's the plan for migrating the database schema?"),
        ("assistant", "We'll add a new column, backfill it in batches, then drop the old one."),
        ("user", "How long will the backfill take?"),
        ("assistant", "A few hours, depending on row count and batch size."),
    ]
    chunks = chunk_exchanges(messages)
    assert len(chunks) == 2
    tail = chunking.overlap_prefix(chunks[0])
    assert tail in chunks[1]


def test_first_chunk_has_no_overlap_prepended():
    messages = [
        ("user", "What's the plan for migrating the database schema?"),
        ("assistant", "We'll add a new column, backfill it in batches, then drop the old one."),
    ]
    chunks = chunk_exchanges(messages)
    assert chunks[0].startswith("> What's the plan")


def test_a_single_exchange_still_produces_one_chunk_when_short():
    messages = [
        ("user", "What's the plan for migrating the database schema?"),
        ("assistant", "We'll add a new column, backfill it in batches, then drop the old one."),
    ]
    chunks = chunk_exchanges(messages)
    assert len(chunks) == 1


def test_an_oversized_exchange_is_split_into_bounded_overlapping_pieces():
    huge_response = "word " * 1000  # well over MAX_CHUNK_SIZE (1200 chars)
    messages = [
        ("user", "Can you walk me through the whole refactor?"),
        ("assistant", huge_response),
    ]
    chunks = chunk_exchanges(messages)
    assert len(chunks) > 1
    assert all(len(c) <= chunking.MAX_CHUNK_SIZE + 200 for c in chunks)  # overlap adds a little


def test_oversized_exchange_pieces_also_get_overlap_between_them():
    huge_response = "word " * 1000
    messages = [
        ("user", "Can you walk me through the whole refactor?"),
        ("assistant", huge_response),
    ]
    chunks = chunk_exchanges(messages)
    tail = chunking.overlap_prefix(chunks[0])
    assert tail in chunks[1]


def test_short_exchanges_are_still_dropped_below_the_floor():
    messages = [("user", "ok"), ("assistant", "ok")]
    chunks = chunk_exchanges(messages)
    assert chunks == []
