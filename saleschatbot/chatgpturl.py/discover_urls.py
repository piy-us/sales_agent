"""
URL Discovery Pipeline
----------------------
Discovers GitLab URLs and stores lightweight metadata.

Output:
    discovered_urls.jsonl
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SEED_URLS = [
    "https://about.gitlab.com/",
    "https://docs.gitlab.com/",
]

ALLOWED_DOMAINS = {
    "about.gitlab.com",
    "docs.gitlab.com",
}

SKIP_PATTERNS = [
    r"/blog/",
    r"/jobs/",
    r"/events/",
    r"/press/",
    r"/community/",
    r"/partners/",
    r"\.pdf$",
    r"#",
    r"sign_in",
    r"/api/",
    r"/archives/",
]

OUTPUT_FILE = Path("discovered_urls.jsonl")
MAX_URLS = 5000
DELAY_SEC = 0.3

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "GitLabEnterpriseRAGBot/2.0"
})

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────


def should_skip(url: str) -> bool:
    return any(re.search(p, url) for p in SKIP_PATTERNS)



def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    parsed = parsed._replace(query="", fragment="")
    return parsed.geturl().rstrip("/")


# ─────────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────────


def extract_metadata(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)

    h1 = soup.find("h1")
    h1_text = h1.get_text(strip=True) if h1 else ""

    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta:
        meta_desc = meta.get("content", "")

    breadcrumbs = [
        b.get_text(strip=True)
        for b in soup.find_all(class_=re.compile(r"breadcrumb", re.I))
    ]

    return {
        "url": url,
        "domain": urlparse(url).netloc,
        "title": title,
        "h1": h1_text,
        "meta_description": meta_desc,
        "breadcrumbs": breadcrumbs,
        "path": urlparse(url).path,
    }



def extract_links(base_url: str, html: str) -> list:
    soup = BeautifulSoup(html, "lxml")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        full = normalize_url(full)

        parsed = urlparse(full)

        if parsed.netloc not in ALLOWED_DOMAINS:
            continue

        if should_skip(full):
            continue

        links.append(full)

    return list(set(links))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def run():
    queue = list(SEED_URLS)
    visited = set()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as fout:

        while queue and len(visited) < MAX_URLS:
            url = queue.pop(0)

            if url in visited:
                continue

            visited.add(url)

            print(f"DISCOVERING: {url}")

            try:
                r = SESSION.get(url, timeout=15)
                r.raise_for_status()
                html = r.text
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            metadata = extract_metadata(url, html)
            fout.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            links = extract_links(url, html)

            for link in links:
                if link not in visited:
                    queue.append(link)

            time.sleep(DELAY_SEC)

    print(f"Done. {len(visited)} URLs discovered.")


if __name__ == "__main__":
    run()
