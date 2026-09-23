"""Incremental fetcher for the UK AI Security Institute's published research
papers and blog posts (aisi.gov.uk), saved as a standing corpus under
F:\\books\\aisi-research so the nightly scheduled run can mine and graph it
alongside everything else.

Design: this only ever ADDS new items. It re-fetches the two listing pages
every run (cheap: two GETs) to discover anything published since the last
run, then skips any paper/blog slug that already has a file on disk --
AISI's own pages don't change once published, so there is nothing to gain
from re-fetching known items nightly, and every unnecessary request is a
request against someone else's server for no reason. Must never raise out
of run() -- a scheduled 3am job should log a failure and move on, not take
down the rest of that night's mining.

aisi.gov.uk is a Webflow site with no bot defence observed; the real
article content (title, date, category, authors, abstract, body) is present
in the plain server-rendered HTML, no JS execution needed. A handful of
external hosts a paper links out to for its full text (OpenReview, TechRxiv,
SSRN as observed 2026-09-23) do have real bot defence and will legitimately
fail to yield a PDF here -- that's expected, not a bug; see NOTES in
`fetch_new`'s return value.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://www.aisi.gov.uk"
RESEARCH_LIST_URL = f"{BASE}/research"
BLOG_LIST_URL = f"{BASE}/blog"

OUT_DIR = Path(r"F:\books\aisi-research")
SUMMARY_DIR = OUT_DIR / "papers" / "summaries"
PDF_DIR = OUT_DIR / "papers" / "pdf"
BLOG_DIR = OUT_DIR / "blog"

EXCLUDED_BLOG_CATEGORY = "Organisation"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

NOISE_CLASSES = ["toast", "navigation-wrap", "footer_component", "cookie-wrapper", "tweet-modal"]

REQUEST_DELAY_S = 0.6


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def _get(session, url, retries=3, timeout=30, **kw):
    last_err = None
    for i in range(retries):
        try:
            return session.get(url, timeout=timeout, **kw)
        except Exception as e:  # noqa: BLE001 -- genuinely want to retry on anything network-shaped
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise last_err


def _clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for sel in ["script", "style", "noscript"]:
        for t in soup.find_all(sel):
            t.decompose()
    for cls in NOISE_CLASSES:
        for t in soup.find_all(class_=cls):
            t.decompose()
    return soup


def _extract_text(soup: BeautifulSoup) -> str:
    body = soup.find("body") or soup
    lines = [l for l in body.get_text("\n", strip=True).split("\n") if l.strip()]
    return "\n".join(lines)


def _find_full_paper_link(raw_soup: BeautifulSoup) -> str | None:
    for a in raw_soup.find_all("a", href=True):
        if a.get_text(strip=True) == "Read the full paper" and a["href"] != "#":
            return urljoin(BASE, a["href"])
    return None


def list_research_slugs(session) -> list[str]:
    r = _get(session, RESEARCH_LIST_URL)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    slugs, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/research/") and href != "/research/":
            slug = href.rsplit("/", 1)[-1]
            if slug and slug not in seen:
                seen.add(slug)
                slugs.append(slug)
    return slugs


def list_blog_rows(session) -> list[tuple[str, str, str]]:
    """Returns (slug, category, date) for every blog post currently listed."""
    r = _get(session, BLOG_LIST_URL)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    rows = []
    for card in soup.find_all("div", class_="work-card-wrapper"):
        a = card.find("a", href=True)
        if not a:
            continue
        slug = a["href"].rsplit("/", 1)[-1]
        cat_el = card.find(attrs={"fs-list-field": "category"})
        date_el = card.find(attrs={"fs-list-field": "date"})
        rows.append((slug, cat_el.get_text(strip=True) if cat_el else "", date_el.get_text(strip=True) if date_el else ""))
    return rows


def _resolve_and_download_pdf(session, link: str, dest_path: Path) -> tuple[str, str]:
    """Best-effort: turn the paper's outbound link into a downloaded PDF.
    Returns (status, note); status in {pdf_ok, no_pdf_found, pdf_failed}."""
    parsed = urlparse(link)
    candidate = None

    if link.lower().endswith(".pdf"):
        candidate = link
    elif "arxiv.org/abs/" in link:
        candidate = f"https://arxiv.org/pdf/{link.rstrip('/').rsplit('/', 1)[-1]}"
    elif "arxiv.org/pdf/" in link:
        candidate = link
    elif parsed.netloc == "osf.io":
        # last path segment is the node id, possibly with a "_v<n>" version suffix
        # (e.g. /preprints/psyarxiv/zqngj_v1 -> node id zqngj); OSF's own /<id>/download
        # redirect resolves to the actual file regardless of version suffix.
        last = parsed.path.strip("/").split("/")[-1]
        node_id = re.sub(r"_v\d+$", "", last)
        if re.fullmatch(r"[a-z0-9]{4,8}", node_id):
            candidate = f"https://osf.io/{node_id}/download"
    elif parsed.netloc.endswith("aisi.gov.uk") and "/research/" not in link:
        # a dedicated AISI landing page for a native report -- look for its own PDF link
        try:
            r2 = _get(session, link)
            if r2.status_code == 200:
                soup2 = _clean_soup(r2.text)
                for a in soup2.find_all("a", href=True):
                    href = a["href"]
                    if href.lower().endswith(".pdf") or "download pdf" in a.get_text(strip=True).lower():
                        candidate = urljoin(link, href)
                        break
        except Exception as e:  # noqa: BLE001
            return ("pdf_failed", f"landing-page fetch failed: {e}")

    if not candidate:
        return ("no_pdf_found", f"outbound link {link} did not resolve to a PDF")

    try:
        r = _get(session, candidate, timeout=60)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code == 200 and (candidate.lower().endswith(".pdf") or "pdf" in ctype.lower() or len(r.content) > 20_000):
            dest_path.write_bytes(r.content)
            return ("pdf_ok", candidate)
        return ("pdf_failed", f"{candidate} -> HTTP {r.status_code} ctype={ctype}")
    except Exception as e:  # noqa: BLE001
        return ("pdf_failed", f"{candidate}: {e}")


def _fetch_paper(session, slug: str, log: list[str]) -> None:
    url = f"{BASE}/research/{slug}"
    r = _get(session, url)
    if r.status_code != 200:
        log.append(f"FAILED paper {slug}: HTTP {r.status_code}")
        return
    raw_soup = BeautifulSoup(r.text, "lxml")
    link = _find_full_paper_link(raw_soup)
    soup = _clean_soup(r.text)
    text = _extract_text(soup)
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else slug

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / f"{slug}.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\nSource: {url}\n")
        if link:
            f.write(f"Full paper link: {link}\n")
        f.write("\n---\n\n" + text)

    if link:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        status, note = _resolve_and_download_pdf(session, link, PDF_DIR / f"{slug}.pdf")
        log.append(f"paper {slug}: {status} ({note})")
    else:
        log.append(f"paper {slug}: no outbound link on AISI's page, summary only")


def _fetch_blog(session, slug: str, category: str, date: str, log: list[str]) -> None:
    url = f"{BASE}/blog/{slug}"
    r = _get(session, url)
    if r.status_code != 200:
        log.append(f"FAILED blog {slug}: HTTP {r.status_code}")
        return
    soup = _clean_soup(r.text)
    text = _extract_text(soup)
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else slug

    BLOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(BLOG_DIR / f"{slug}.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\nSource: {url}\nCategory: {category}\nDate: {date}\n\n---\n\n" + text)
    log.append(f"blog {slug}: ok")


def fetch_new() -> dict:
    """Check AISI's listings for anything not already on disk, fetch it, return a summary dict.
    Never raises -- a listing-fetch failure (site down, network issue) is reported, not thrown."""
    log: list[str] = []
    result = {"new_papers": 0, "new_blogs": 0, "excluded_organisation": 0, "log": log, "error": None}

    session = _session()
    try:
        paper_slugs = list_research_slugs(session)
        blog_rows = list_blog_rows(session)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"could not fetch AISI listings: {e}"
        log.append(result["error"])
        return result

    new_papers = [s for s in paper_slugs if not (SUMMARY_DIR / f"{s}.md").exists()]
    to_fetch_blogs = [(s, c, d) for (s, c, d) in blog_rows if c != EXCLUDED_BLOG_CATEGORY]
    new_blogs = [(s, c, d) for (s, c, d) in to_fetch_blogs if not (BLOG_DIR / f"{s}.md").exists()]
    result["excluded_organisation"] = sum(1 for (_, c, _) in blog_rows if c == EXCLUDED_BLOG_CATEGORY)

    log.append(f"listings: {len(paper_slugs)} papers, {len(blog_rows)} blog posts "
               f"({result['excluded_organisation']} Organisation, excluded)")
    log.append(f"new: {len(new_papers)} paper(s), {len(new_blogs)} blog post(s)")

    for slug in new_papers:
        try:
            _fetch_paper(session, slug, log)
            result["new_papers"] += 1
        except Exception as e:  # noqa: BLE001
            log.append(f"FAILED paper {slug}: {e}")
        time.sleep(REQUEST_DELAY_S)

    for slug, cat, date in new_blogs:
        try:
            _fetch_blog(session, slug, cat, date, log)
            result["new_blogs"] += 1
        except Exception as e:  # noqa: BLE001
            log.append(f"FAILED blog {slug}: {e}")
        time.sleep(REQUEST_DELAY_S)

    return result
