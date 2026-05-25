"""
Phase 2: Scraper
- Reads all_english_urls.txt from Phase 1
- Scrapes each URL and saves raw page data to scraped_pages.jsonl
- Resume-safe: tracks progress in scrape_progress.json
- Does NOT chunk — that's Phase 3
"""

import json
import time
import hashlib
import re
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    print("⚠  trafilatura not installed — marketing pages will use fallback extractor")
    print("   Run:  pip install trafilatura")

# ── Config ────────────────────────────────────────────────────────────────────

URL_LIST_FILE  = Path("all_english_urls.txt")
OUTPUT_FILE    = Path("scraped_pages.jsonl")
PROGRESS_FILE  = Path("scrape_progress.json")

REQUEST_TIMEOUT = 20
MAX_RETRIES     = 4
DELAY_SEC       = 0.5
BACKOFF_FACTOR  = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrape.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RawPage:
    page_id:     str   # sha256[:16] of URL
    url:         str
    source_type: str   # "docs" | "marketing"
    title:       str
    text:        str   # full cleaned page text — chunking happens in Phase 3
    word_count:  int
    scraped_at:  str

# ── Helpers ───────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "GitLabRAGBot/2.0 (enterprise knowledge base builder)"
})

def url_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def clean_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# ── HTTP fetch with retry ─────────────────────────────────────────────────────

def fetch_html(url: str) -> Optional[str]:
    delay = DELAY_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", delay * 4))
                log.warning(f"  429 rate-limited — sleeping {retry_after}s")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            return resp.text

        except requests.exceptions.Timeout:
            log.warning(f"  Timeout (attempt {attempt}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError as e:
            log.warning(f"  Connection error (attempt {attempt}/{MAX_RETRIES}): {e}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status in (404, 410, 403):
                log.info(f"  HTTP {status} — skipping permanently")
                return None
            log.warning(f"  HTTP {status} (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            log.warning(f"  Unexpected error (attempt {attempt}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES:
            sleep_time = delay * (BACKOFF_FACTOR ** (attempt - 1))
            log.info(f"  Retrying in {sleep_time:.1f}s …")
            time.sleep(sleep_time)

    log.error(f"  FAILED after {MAX_RETRIES} attempts — {url}")
    return None

# ── Content extractors ────────────────────────────────────────────────────────

def extract_docs(url: str, html: str) -> Optional[tuple]:
    """Returns (title, text) for docs.gitlab.com pages."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["nav", "footer", "header", "aside", "script",
                     "style", ".sidebar", ".breadcrumb", ".feedback",
                     ".toc", ".pagination"]):
        tag.decompose()

    title_tag = soup.find("h1")
    title = (title_tag.get_text(strip=True) if title_tag
             else (soup.title.string.split("|")[0].strip() if soup.title else url))

    main = (soup.find("main") or soup.find("article") or
            soup.find(class_=re.compile(r"content|doc|markdown", re.I)) or
            soup.body)

    if not main:
        return None

    text = clean_text(main.get_text(separator="\n", strip=True))
    return (title, text) if len(text) >= 100 else None


def extract_marketing(url: str, html: str) -> Optional[tuple]:
    """Returns (title, text) for about.gitlab.com pages."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    title = (title_tag.get_text(strip=True) if title_tag
             else (soup.title.string.split("|")[0].strip() if soup.title else url))

    if HAS_TRAFILATURA:
        extracted = trafilatura.extract(
            html, url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )
        text = clean_text(extracted) if extracted else ""
    else:
        for tag in soup(["nav", "footer", "header", "script", "style", "aside"]):
            tag.decompose()
        body = soup.find("main") or soup.find("body")
        text = clean_text(body.get_text(separator="\n", strip=True)) if body else ""

    return (title, text) if len(text) >= 100 else None


def scrape_url(url: str) -> Optional[RawPage]:
    html = fetch_html(url)
    if not html:
        return None

    domain = urlparse(url).netloc
    source_type = "docs" if "docs.gitlab.com" in domain else "marketing"

    try:
        result = extract_docs(url, html) if source_type == "docs" else extract_marketing(url, html)
    except Exception:
        log.error(f"  Extraction error:\n{traceback.format_exc()}")
        return None

    if not result:
        return None

    title, text = result
    return RawPage(
        page_id=url_id(url),
        url=url,
        source_type=source_type,
        title=title,
        text=text,
        word_count=len(text.split()),
        scraped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress() -> set:
    if not PROGRESS_FILE.exists():
        return set()
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return set(data.get("done", []))
    except Exception:
        return set()

def save_progress(done: set):
    PROGRESS_FILE.write_text(
        json.dumps({"done": sorted(done)}, ensure_ascii=False),
        encoding="utf-8"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not URL_LIST_FILE.exists():
        print(f"ERROR: {URL_LIST_FILE} not found. Run phase1_sitemap_crawler.py first.")
        return

    urls = [u.strip() for u in URL_LIST_FILE.read_text(encoding="utf-8").splitlines()
            if u.strip() and not u.startswith("#")]

    print(f"Total URLs      : {len(urls)}")

    done = load_progress()
    remaining = [u for u in urls if u not in done]
    print(f"Already done    : {len(done)}")
    print(f"Remaining       : {len(remaining)}")

    if not remaining:
        print("All URLs already scraped! Delete scrape_progress.json to re-run.")
        return

    pages_saved = 0
    skipped     = 0

    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        for i, url in enumerate(remaining, 1):
            log.info(f"[{i}/{len(remaining)}] {url}")

            try:
                page = scrape_url(url)
            except KeyboardInterrupt:
                log.info("Interrupted — saving progress …")
                save_progress(done)
                print("Re-run to continue from where you left off.")
                return
            except Exception:
                log.error(f"  Unhandled error:\n{traceback.format_exc()}")
                skipped += 1
                done.add(url)
                continue

            if page:
                fh.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
                pages_saved += 1
                log.info(f"  ✓ '{page.title}' — {page.word_count} words")
            else:
                skipped += 1
                log.info(f"  ✗ no content extracted")

            done.add(url)

            if i % 50 == 0:
                save_progress(done)
                log.info(f"  Progress checkpoint: {len(done)} done, {pages_saved} saved")

            time.sleep(DELAY_SEC)

    save_progress(done)

    print(f"\n{'='*60}")
    print(f"Pages saved     : {pages_saved}  →  {OUTPUT_FILE}")
    print(f"Skipped/errors  : {skipped}")
    print(f"{'='*60}")
    print("Next step: python phase3_chunker.py")


if __name__ == "__main__":
    main()