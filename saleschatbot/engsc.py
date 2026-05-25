"""
Phase 1: Sitemap Crawler
- Recursively fetches all URLs from nested sitemaps
- Automatically filters out non-English locale pages (e.g. /it-it/, /ja-jp/)
- Saves to all_english_urls.txt (one URL per line) for Phase 2
"""

import requests
import xml.etree.ElementTree as ET
import re
import time
import logging
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

START_SITEMAPS = [
    "https://about.gitlab.com/sitemap.xml",
    "https://docs.gitlab.com/sitemap.xml",   # add or remove as needed
]

OUTPUT_FILE   = Path("all_english_urls.txt")
DELAY_SEC     = 0.3   # polite delay between sitemap fetches
REQUEST_TIMEOUT = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Non-English locale detection ──────────────────────────────────────────────
# Matches paths like /it-it/, /ja-jp/, /fr-fr/, /de/, /zh-cn/, /ko/, etc.
LOCALE_PATTERN = re.compile(
    r"/("
    r"af|ar|az|be|bg|bn|bs|ca|cs|cy|da|de|el|eo|es|et|eu|fa|fi|fr|ga|gl|"
    r"gu|he|hi|hr|hu|hy|id|is|it|ja|ka|kk|km|kn|ko|lt|lv|mk|ml|mn|mr|ms|"
    r"my|nb|ne|nl|no|or|pa|pl|pt|ro|ru|sk|sl|sq|sr|sv|sw|ta|te|th|tl|tr|"
    r"uk|ur|uz|vi|zh|"
    r"it-it|ja-jp|fr-fr|de-de|es-es|pt-br|pt-pt|zh-cn|zh-tw|ko-kr|"
    r"ru-ru|nl-nl|pl-pl|sv-se|nb-no|da-dk|fi-fi|cs-cz|sk-sk|ro-ro|"
    r"hu-hu|bg-bg|hr-hr|sr-rs|uk-ua|tr-tr|el-gr|ar-sa|he-il|fa-ir|"
    r"th-th|vi-vn|id-id|ms-my|tl-ph"
    r")(/|$)",
    re.IGNORECASE,
)

def is_non_english(url: str) -> bool:
    path = requests.utils.urlparse(url).path
    return bool(LOCALE_PATTERN.search(path))

# ── HTTP session ──────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "GitLabRAGBot/2.0 (enterprise knowledge base builder)"
})

def fetch_xml(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            log.warning(f"  Attempt {attempt+1}/3 failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return None

# ── Sitemap parser ────────────────────────────────────────────────────────────

NAMESPACE = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
visited_sitemaps: set[str] = set()
all_urls: set[str] = set()

def parse_sitemap(url: str):
    if url in visited_sitemaps:
        return
    visited_sitemaps.add(url)

    log.info(f"Sitemap: {url}")
    xml_data = fetch_xml(url)
    if not xml_data:
        return

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        log.warning(f"  XML parse error for {url}: {e}")
        return

    tag = root.tag.lower()

    if "sitemapindex" in tag:
        # Nested sitemap index — recurse into each child sitemap
        for sitemap in root.findall("ns:sitemap", NAMESPACE):
            loc = sitemap.find("ns:loc", NAMESPACE)
            if loc is not None and loc.text:
                time.sleep(DELAY_SEC)
                parse_sitemap(loc.text.strip())

    elif "urlset" in tag:
        added = skipped = 0
        for url_tag in root.findall("ns:url", NAMESPACE):
            loc = url_tag.find("ns:loc", NAMESPACE)
            if loc is not None and loc.text:
                page_url = loc.text.strip()
                if is_non_english(page_url):
                    skipped += 1
                else:
                    all_urls.add(page_url)
                    added += 1
        log.info(f"  +{added} English URLs, -{skipped} non-English skipped")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for seed in START_SITEMAPS:
        parse_sitemap(seed)

    sorted_urls = sorted(all_urls)

    OUTPUT_FILE.write_text("\n".join(sorted_urls), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Sitemaps visited : {len(visited_sitemaps)}")
    print(f"English URLs     : {len(sorted_urls)}")
    print(f"Saved to         : {OUTPUT_FILE}")
    print(f"{'='*60}")
    print("Now run:  python phase2_scraper.py")


if __name__ == "__main__":
    main()