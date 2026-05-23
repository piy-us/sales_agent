"""
GitLab Knowledge Base Scraper
Handles both docs.gitlab.com (static) and about.gitlab.com (marketing/JS)
Outputs JSONL — one record per page, ready for chunking + embedding.
"""

import json
import time
import hashlib
import re
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
import trafilatura

# ── Config ────────────────────────────────────────────────────────────────────

SEED_URLS = [
    # Docs site — crawl deeply (follows links within docs.gitlab.com)
    "https://docs.gitlab.com/topics/plan_and_track/",
    "https://docs.gitlab.com/user/project/issues/",
    "https://docs.gitlab.com/user/group/epics/",
    "https://docs.gitlab.com/ci/",

    # Marketing / solutions site — scrape these pages directly
    "https://about.gitlab.com/solutions/continuous-integration/",
    "https://about.gitlab.com/solutions/source-code-management/",
    "https://about.gitlab.com/solutions/delivery-automation/",
]

# Which domains to crawl recursively (vs just scrape once)
CRAWLABLE_DOMAINS = {"docs.gitlab.com"}

# URL patterns to SKIP even on crawlable domains
SKIP_PATTERNS = [
    r"/api/",
    r"/-/",
    r"/releases/",
    r"\.xml$",
    r"\.json$",
    r"#",           # anchors
    r"/blog/",      # blog has its own cadence — add separately if needed
]

OUTPUT_FILE = Path("gitlab_knowledge_base.jsonl")
MAX_PAGES   = 500       # safety cap
DELAY_SEC   = 0.5       # polite crawl delay between requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Page:
    id:           str
    url:          str
    domain:       str
    title:        str
    text:         str
    headings:     list[str]
    source_type:  str          # "docs" | "marketing"
    scraped_at:   str
    word_count:   int

# ── Helpers ───────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "GitLabRAGBot/1.0 (enterprise knowledge base; contact: you@yourco.com)"
})

def should_skip(url: str) -> bool:
    for pat in SKIP_PATTERNS:
        if re.search(pat, url):
            return True
    return False

def url_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def clean_text(text: str) -> str:
    """Normalize whitespace, remove zero-width chars, etc."""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# ── Core extraction ───────────────────────────────────────────────────────────

def extract_docs_page(url: str, html: str) -> Optional[Page]:
    """
    docs.gitlab.com pages are clean static HTML.
    Strip nav/footer, extract main content + child links.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy elements
    for tag in soup(["nav", "footer", "header", "aside", "script",
                     "style", ".sidebar", ".breadcrumb", ".feedback"]):
        tag.decompose()

    # Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else (
        soup.title.string.split("|")[0].strip() if soup.title else url
    )

    # Main content area
    main = (soup.find("main") or soup.find("article") or
            soup.find(class_=re.compile(r"content|doc|markdown")) or soup.body)

    if not main:
        return None

    # Extract headings for metadata
    headings = [h.get_text(strip=True) for h in main.find_all(["h2", "h3"])]

    text = clean_text(main.get_text(separator="\n", strip=True))

    if len(text) < 100:   # skip stub/redirect pages
        return None

    return Page(
        id=url_id(url),
        url=url,
        domain="docs.gitlab.com",
        title=title,
        text=text,
        headings=headings,
        source_type="docs",
        scraped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        word_count=len(text.split()),
    )


def extract_marketing_page(url: str, html: str) -> Optional[Page]:
    """
    about.gitlab.com pages are JS-heavy marketing pages.
    Use trafilatura for robust boilerplate removal.
    """
    # trafilatura is the best open-source tool for marketing page extraction
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=True,
    )

    if not extracted or len(extracted) < 100:
        return None

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else (
        soup.title.string.split("|")[0].strip() if soup.title else url
    )

    headings = [h.get_text(strip=True)
                for h in soup.find_all(["h2", "h3"])
                if len(h.get_text(strip=True)) > 3]

    text = clean_text(extracted)

    return Page(
        id=url_id(url),
        url=url,
        domain="about.gitlab.com",
        title=title,
        text=text,
        headings=headings,
        source_type="marketing",
        scraped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        word_count=len(text.split()),
    )


def extract_child_links(url: str, html: str) -> list[str]:
    """Pull same-domain links from a docs page for recursive crawling."""
    base_domain = urlparse(url).netloc
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(url, href)
        parsed = urlparse(full)
        if parsed.netloc == base_domain and not should_skip(full):
            # Normalize: drop query strings and fragments
            clean = parsed._replace(query="", fragment="").geturl()
            links.append(clean)
    return list(set(links))

# ── Crawler ───────────────────────────────────────────────────────────────────

def scrape_url(url: str) -> tuple[Optional[Page], list[str]]:
    """Fetch a URL, extract content, return (Page, child_links)."""
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        log.warning(f"FETCH ERROR {url}: {e}")
        return None, []

    domain = urlparse(url).netloc
    child_links = []

    if domain == "docs.gitlab.com":
        page = extract_docs_page(url, html)
        if domain in CRAWLABLE_DOMAINS:
            child_links = extract_child_links(url, html)
    elif domain == "about.gitlab.com":
        page = extract_marketing_page(url, html)
    else:
        page = None

    return page, child_links


def run():
    visited   = set()
    queue     = list(SEED_URLS)
    pages_out = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        while queue and pages_out < MAX_PAGES:
            url = queue.pop(0)
            if url in visited or should_skip(url):
                continue
            visited.add(url)

            log.info(f"[{pages_out+1}] {url}")
            page, children = scrape_url(url)

            if page:
                fh.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
                pages_out += 1
                log.info(f"  ✓ '{page.title}' — {page.word_count} words")
            else:
                log.info(f"  ✗ skipped (no content)")

            # Only queue children for crawlable domains
            domain = urlparse(url).netloc
            if domain in CRAWLABLE_DOMAINS:
                new = [u for u in children if u not in visited]
                queue.extend(new)

            time.sleep(DELAY_SEC)

    log.info(f"\nDone. {pages_out} pages → {OUTPUT_FILE}")


if __name__ == "__main__":
    run()