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


def _split_bounded(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split a block of text into <= max_size pieces on paragraph/line
    boundaries where possible, falling back to a hard cut."""
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
                for i in range(0, len(p), max_size):
                    pieces.append(p[i : i + max_size])
                buf = ""
            else:
                buf = p
    if buf:
        pieces.append(buf)
    return pieces


def chunk_html(text: str) -> list[str]:
    """Split on heading boundaries, then bound each section's size.
    Blobs (embedded WASM/base64/etc.) are dropped, not chunked."""
    soup = BeautifulSoup(text, "lxml")
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
        chunks.extend(_split_bounded(s))
    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE and not looks_like_embedded_blob(c)]


def chunk_markdown(text: str) -> list[str]:
    sections = re.split(r"(?m)^#{1,6}\s+.*$", text)
    chunks: list[str] = []
    for s in sections:
        if looks_like_embedded_blob(s):
            continue
        chunks.extend(_split_bounded(s))
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
    """Extract text page-by-page with PyMuPDF, normalize, and chunk by
    paragraph. Figure/table page-furniture (running headers, bare page
    numbers, short caption fragments) is dropped via the same blob/length
    heuristics used elsewhere rather than special-cased."""
    import fitz  # PyMuPDF

    chunks: list[str] = []
    doc = fitz.open(path)
    try:
        for page in doc:
            text = normalize_pdf_text(page.get_text())
            if not text.strip():
                continue
            for piece in _split_bounded(text):
                if len(piece) < MIN_CHUNK_SIZE or looks_like_embedded_blob(piece):
                    continue
                chunks.append(piece)
    finally:
        doc.close()
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
