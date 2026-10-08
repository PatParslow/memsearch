"""Real bug found and fixed (contextual-concept-algebra session, see
docs/phase3_calibration.md): chunk_pdf chunked each PDF page
independently, so a word or sentence spanning a page break -- the
ORDINARY case for a real book, not an edge case -- was split with no
way to recover it downstream. Measured on the real local books+papers
corpus: 82.99% of adjacent chunk-boundary pairs showed no sentence-
ending punctuation at the seam, including literal mid-word corruption
("...for survival. 8tu" | "dents should..." for "students"). Fixed by
(1) joining pages into one continuous document before chunking, (2) a
word-boundary-aware hard-cut instead of an exact character cut, and
(3) a bounded word-overlap between every adjacent piece.
"""

from __future__ import annotations

import sys

from memsearch import chunking


def test_overlap_prefix_returns_last_n_words():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen"
    prefix = chunking._overlap_prefix(text, min_words=3)
    assert prefix == "eleven twelve thirteen"


def test_overlap_prefix_handles_text_shorter_than_min_words():
    text = "only three words"
    assert chunking._overlap_prefix(text, min_words=12) == "only three words"


def test_overlap_prefix_empty_text_is_empty():
    assert chunking._overlap_prefix("", min_words=5) == ""


def test_cut_at_word_boundary_never_splits_a_word():
    words = [f"word{i}" for i in range(400)]  # no natural paragraph breaks
    text = " ".join(words)
    pieces = chunking._cut_at_word_boundary(text, max_size=100)
    assert len(pieces) > 1
    reassembled = "".join(pieces)
    assert reassembled == text  # nothing lost or duplicated by this function alone
    for piece in pieces[:-1]:
        # every piece but the last ends with the space that separated it
        # from the next word -- i.e. it never stops mid-word.
        assert piece.endswith(" ") or " " not in piece.strip()


def test_cut_at_word_boundary_falls_back_to_hard_cut_for_one_giant_token():
    text = "a" * 500  # a single unbroken token, e.g. a URL -- no whitespace to snap to
    pieces = chunking._cut_at_word_boundary(text, max_size=100)
    assert len(pieces) == 5
    assert "".join(pieces) == text


def test_split_bounded_adds_no_overlap_when_text_fits_in_one_piece():
    text = "A short paragraph that easily fits in one chunk."
    pieces = chunking._split_bounded(text, max_size=1200)
    assert pieces == [text]


def test_split_bounded_prepends_overlap_to_later_pieces():
    para_a = "Alpha " * 50  # long enough to force a split on its own
    para_b = "Beta " * 50
    text = f"{para_a.strip()}\n\n{para_b.strip()}"
    pieces = chunking._split_bounded(text, max_size=len(para_a) + 10)
    assert len(pieces) >= 2
    # the second piece must start with trailing words from the first piece's
    # own tail (the overlap), not just its own paragraph's content.
    first_tail = chunking._overlap_prefix(pieces[0])
    assert pieces[1].startswith(first_tail)


def test_split_bounded_reproduces_and_fixes_the_real_mid_word_bug():
    """The exact real-world failure mode: one long paragraph (common in
    PDF-extracted book prose, e.g. a dense preface) with no internal
    \\n\\n breaks, long enough that the OLD implementation's exact-
    character hard-cut would land mid-word. The word that would have
    been corrupted must now appear WHOLE in at least one piece."""
    filler_a = "word " * 230  # pushes well past a small max_size
    long_paragraph = f"{filler_a}essential for survival students should already be conversant"
    pieces = chunking._split_bounded(long_paragraph, max_size=1200)
    assert any("students" in piece for piece in pieces)
    # and specifically: no piece contains a truncated fragment of it
    # standing alone where "students" should be (the corrupted "8tu"/
    # "dents"-style split this bug produced in the real corpus).
    assert not any("stu dents" in piece or "stu" == piece.strip().split()[-1] for piece in pieces if piece.strip())


def test_chunk_html_and_markdown_oversized_section_also_gets_overlap():
    # chunk_html/chunk_markdown both fall through to _split_bounded for
    # an oversized section -- confirm the fix applies there too, not
    # just to the new chunk_pdf path.
    long_markdown = "# Heading\n\n" + ("Sentence about the topic. " * 100)
    pieces = chunking.chunk_markdown(long_markdown)
    assert len(pieces) >= 2
    # later pieces should carry some overlap from the previous piece's tail
    assert any(pieces[i].split()[0] in pieces[i - 1] for i in range(1, len(pieces)))


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self):
        return self._text


class _FakeDoc:
    def __init__(self, page_texts: list[str]):
        self._pages = [_FakePage(t) for t in page_texts]

    def __iter__(self):
        return iter(self._pages)

    def close(self):
        pass


def test_chunk_pdf_joins_pages_so_a_hyphenated_word_split_at_a_page_break_survives(monkeypatch, tmp_path):
    """Real-world shape: a typeset book hyphenates a line-wrapped word,
    and that wrap happens to fall exactly at a page boundary -- "stu-"
    ending page 1, "dents" starting page 2. normalize_pdf_text's own
    hyphen-rejoin (already relied on for ordinary within-page line
    wraps) can only catch this if pages are joined BEFORE normalising,
    not chunked independently per page (the actual bug: chunking each
    page separately meant this case was never recoverable)."""
    page_1 = "This course requires real effort and discipline. All stu-"
    page_2 = "dents should already be conversant with the prerequisites."

    fake_fitz = type(sys)("fitz")
    fake_fitz.open = lambda path: _FakeDoc([page_1, page_2])
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    chunks = chunking.chunk_pdf(tmp_path / "fake.pdf")
    assert len(chunks) == 1  # short enough to fit in one chunk once joined
    assert "students" in chunks[0]
    assert "stu-" not in chunks[0]
