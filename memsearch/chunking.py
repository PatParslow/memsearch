"""Content-type-aware chunking, with a general heuristic for dropping
embedded binary blobs (base64 WASM, minified JS, giant inline SVG path
data) instead of special-casing each one.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

MIN_CHUNK_SIZE = 40
MAX_CHUNK_SIZE = 1200

# How many whitespace-separated words of the END of one piece get
# prepended to the NEXT piece -- sized well above the longest n-gram
# phrase discovery actually builds (reached length 5 in testing; this
# leaves real margin for going further, e.g. 8-grams, per the explicit
# ask that drove this fix). Word-based, not a fixed character count,
# so it stays correct regardless of average word length in a given
# document.
MIN_OVERLAP_WORDS = 12

# Never even open these as text.
BINARY_EXTENSIONS = {
    ".wasm", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib", ".pdb",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".mp3", ".mp4", ".webm", ".avi", ".mov",
    ".sqlite3", ".db", ".npz", ".npy", ".bin",
}

HTML_EXTENSIONS = {".html", ".htm"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
PDF_EXTENSIONS = {".pdf"}

# Common PDF-extraction ligatures that don't decompose under plain str
# operations -- normalize them back to their letter sequences.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
}
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

_LONG_TOKEN_RE = re.compile(r"\S{200,}")


def is_binary_extension(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


def looks_like_embedded_blob(text: str) -> bool:
    """True if `text` looks like a binary blob embedded in a text file
    (base64 WASM, minified JS, giant SVG path data) rather than prose or
    ordinary code -- one general heuristic instead of pattern-matching
    each specific case.
    """
    if not text.strip():
        return False
    if _LONG_TOKEN_RE.search(text):
        return True
    whitespace_ratio = sum(1 for c in text if c.isspace()) / len(text)
    return whitespace_ratio < 0.02 and len(text) > 300


def _cut_at_word_boundary(p: str, max_size: int) -> list[str]:
    """Hard-cut `p` into <= max_size pieces, snapping each cut to the
    nearest PRECEDING whitespace instead of an arbitrary character
    position. Real bug found and fixed: the old exact-character cut
    corrupted words at the seam -- confirmed on the real corpus (a
    preface paragraph longer than MAX_CHUNK_SIZE produced "...for
    survival. 8tu" | "dents should..." for "students", found via
    chunk-boundary analysis in the contextual-concept-algebra project).
    Falls back to the exact character position only if there is no
    whitespace at all within the piece (a single pathologically long
    unbroken token, e.g. a URL) -- that case was never a word split to
    begin with."""
    pieces = []
    start = 0
    n = len(p)
    while start < n:
        end = min(start + max_size, n)
        if end < n:
            cut = p.rfind(" ", start, end)
            cut = cut + 1 if cut > start else end  # keep the space with THIS piece
        else:
            cut = end
        pieces.append(p[start:cut])
        start = cut
    return pieces


def overlap_prefix(text: str, min_words: int = MIN_OVERLAP_WORDS) -> str:
    """Last `min_words` whitespace-separated words of `text` (fewer if
    `text` is shorter) -- prepended to the NEXT piece so a word or
    sentence is never invisibly split across a chunk boundary with no
    way to recover it downstream."""
    words = text.split()
    return " ".join(words[-min_words:]) if words else ""


def split_bounded(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split a block of text into <= max_size pieces on paragraph/line
    boundaries where possible (falling back to a word-boundary-aware
    hard cut -- see _cut_at_word_boundary), then prefix every piece
    after the first with a bounded word-overlap from the end of the
    previous piece (see overlap_prefix).

    Real bug found and fixed, confirmed on the real corpus: without
    overlap, a word or sentence straddling ANY chunk boundary -- not
    just the hard-cut fallback case, ordinary paragraph-to-paragraph
    splits too -- was invisible to anything that only looks within one
    chunk's own text (82.99% of adjacent chunk-boundary pairs in real
    books/papers prose showed no sentence-ending punctuation at the
    seam). This widens every chunk slightly rather than keeping them
    perfectly disjoint -- a few extra words of harmless duplication is
    a better trade than silently losing or corrupting the boundary.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]
    pieces = []
    para = text.split("\n\n")
    buf = ""
    for p in para:
        if len(buf) + len(p) + 2 <= max_size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                pieces.append(buf)
            if len(p) > max_size:
                pieces.extend(_cut_at_word_boundary(p, max_size))
                buf = ""
            else:
                buf = p
    if buf:
        pieces.append(buf)

    if len(pieces) <= 1:
        return pieces
    stitched = [pieces[0]]
    for i in range(1, len(pieces)):
        overlap = overlap_prefix(pieces[i - 1])
        stitched.append(f"{overlap} {pieces[i]}" if overlap else pieces[i])
    return stitched


def chunk_html(text: str) -> list[str]:
    """Split on heading boundaries, then bound each section's size.
    Blobs (embedded WASM/base64/etc.) are dropped, not chunked."""
    soup = BeautifulSoup(text, "lxml")

    # parslow.net's own build.py generates a "This page has moved to X."
    # redirect stub (render_redirect(), always titled exactly "Moved") for
    # every renamed/relocated page -- ~116 of them site-wide, all near-
    # identical apart from the target path. Mining them as prose added no
    # real searchable content and dominated cluster-label word frequency
    # sitewide ("topics"/"page"/"moved" winning nearly every large-group
    # label vote). The old URL is still fully searchable via source_file.
    title = soup.title
    if title and title.get_text(strip=True) == "Moved":
        return []

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    sections: list[str] = []
    current: list[str] = []

    def flush():
        joined = "\n".join(s for s in current if s.strip())
        if joined.strip():
            sections.append(joined)
        current.clear()

    body = soup.body or soup
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"], recursive=True):
        if el.name in ("h1", "h2", "h3", "h4"):
            flush()
        txt = el.get_text(" ", strip=True)
        if txt and not looks_like_embedded_blob(txt):
            current.append(txt)
    flush()

    chunks: list[str] = []
    for s in sections:
        chunks.extend(split_bounded(s))
    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE and not looks_like_embedded_blob(c)]


def chunk_markdown(text: str) -> list[str]:
    sections = re.split(r"(?m)^#{1,6}\s+.*$", text)
    chunks: list[str] = []
    for s in sections:
        if looks_like_embedded_blob(s):
            continue
        chunks.extend(split_bounded(s))
    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE and not looks_like_embedded_blob(c)]


def chunk_code(text: str) -> list[str]:
    """Blank-line-separated blocks, bounded, blobs dropped."""
    blocks = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for b in blocks:
        if looks_like_embedded_blob(b):
            continue
        if len(buf) + len(b) <= MAX_CHUNK_SIZE:
            buf = f"{buf}\n\n{b}" if buf else b
        else:
            if buf:
                chunks.append(buf)
            buf = b
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]


def normalize_pdf_text(text: str) -> str:
    """Fix the two most common PDF-extraction artifacts: unicode ligatures
    (fi/fl/ff/...) and words split by a line-wrap hyphen."""
    for lig, plain in _LIGATURES.items():
        text = text.replace(lig, plain)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    return text


def chunk_pdf(path: Path) -> list[str]:
    """Extract text page-by-page with PyMuPDF, normalize, join into one
    continuous document, THEN chunk by paragraph. Figure/table page-
    furniture (running headers, bare page numbers, short caption
    fragments) is dropped via the same blob/length heuristics used
    elsewhere rather than special-cased.

    Pages are joined into ONE string before chunking, not chunked
    independently per page as before. Real bug found and fixed:
    chunking each page separately meant a sentence or word spanning a
    page break -- the ORDINARY case, not an edge case, for a real
    book -- had no way to be recovered downstream, since nothing reads
    across a chunk boundary. Measured directly on the real local
    books+papers corpus: 82.99% of adjacent chunk-boundary pairs showed
    no sentence-ending punctuation at the seam, including literal
    mid-word corruption. De-paginating first, combined with
    split_bounded's own word-boundary-aware cut and overlap (see
    there), fixes both page-break splits and within-page oversized-
    paragraph splits the same way.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        raw_pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    # Joined with a single newline, BEFORE normalising (not normalised
    # per page first) -- so a hyphenated line-wrap split exactly at a
    # page boundary ("stu-" ending one page, "dents" starting the next)
    # is caught by normalize_pdf_text's own hyphen-rejoin regex the same
    # way an ordinary within-page line wrap already is; that regex needs
    # the hyphen and the continuation in the SAME string with exactly
    # one newline between them, which per-page normalisation could never
    # see across the boundary.
    full_text = normalize_pdf_text("\n".join(t for t in raw_pages if t.strip()))

    chunks: list[str] = []
    for piece in split_bounded(full_text):
        if len(piece) < MIN_CHUNK_SIZE or looks_like_embedded_blob(piece):
            continue
        chunks.append(piece)
    return chunks


def chunk_file(path: Path, text: str | None) -> list[str]:
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return chunk_pdf(path)
    assert text is not None, "text is only optional for the PDF path"
    if ext in HTML_EXTENSIONS:
        return chunk_html(text)
    if ext in MARKDOWN_EXTENSIONS:
        return chunk_markdown(text)
    return chunk_code(text)
