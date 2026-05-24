"""
enterprise_sales_scraper.py
============================================================

Enterprise Sales RAG Scraper for GitLab

FIXES INCLUDED
---------------
✓ Keeps your ORIGINAL seed strategy
✓ Keeps homepage discovery
✓ Keeps 2-level crawl
✓ Locale URL rejection
✓ Markdown extraction
✓ Canonical URL normalization
✓ URL hierarchy depth limits
✓ Prevents deep docs recursion
✓ Better logging
✓ Cleaner extraction
✓ Chunker-compatible markdown output

WHY THIS VERSION WORKS
----------------------
Your original problem was NOT the seed list.

It was:
1. locale contamination
2. deep docs URL recursion
3. weak markdown extraction

This fixes ONLY those issues while preserving your workflow.

OUTPUT
------
gitlab_kb.jsonl

DEPENDENCIES
------------
pip install requests beautifulsoup4 lxml trafilatura
"""

import json
import time
import hashlib
import re

from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
import trafilatura

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = Path("gitlab_kb.jsonl")

DELAY_SEC = 1.0

# ============================================================
# ORIGINAL SEEDS (PRESERVED)
# ============================================================

ABOUT_SEEDS = [
    "https://about.gitlab.com/",

    "https://about.gitlab.com/pricing/",
    "https://about.gitlab.com/pricing/premium/",
    "https://about.gitlab.com/pricing/ultimate/",

    "https://about.gitlab.com/platform/",
    "https://about.gitlab.com/gitlab-duo-agent-platform/",
    "https://about.gitlab.com/why-gitlab/",

    "https://about.gitlab.com/solutions/continuous-integration/",
    "https://about.gitlab.com/solutions/source-code-management/",
    "https://about.gitlab.com/solutions/delivery-automation/",
    "https://about.gitlab.com/solutions/application-security-testing/",
    "https://about.gitlab.com/solutions/supply-chain/",
    "https://about.gitlab.com/solutions/software-compliance/",
    "https://about.gitlab.com/solutions/visibility-measurement/",
    "https://about.gitlab.com/solutions/value-stream-management/",
    "https://about.gitlab.com/solutions/analytics-and-insights/",
    "https://about.gitlab.com/solutions/agile-delivery/",
    "https://about.gitlab.com/solutions/gitops/",

    "https://about.gitlab.com/enterprise/",
    "https://about.gitlab.com/small-business/",
    "https://about.gitlab.com/solutions/public-sector/",
    "https://about.gitlab.com/solutions/finance/",

    "https://about.gitlab.com/topics/ci-cd/",
    "https://about.gitlab.com/topics/devops/",
    "https://about.gitlab.com/topics/devsecops/",
    "https://about.gitlab.com/topics/gitops/",
    "https://about.gitlab.com/topics/version-control/",
    "https://about.gitlab.com/topics/cloud-native/",
    "https://about.gitlab.com/topics/devops/ai-for-coding/",
    "https://about.gitlab.com/topics/agentic-ai/",

    "https://about.gitlab.com/customers/",

    "https://about.gitlab.com/company/",
    "https://about.gitlab.com/security/",
    "https://about.gitlab.com/services/",

    "https://about.gitlab.com/partners/",

    "https://about.gitlab.com/get-started/",
]

DOCS_SEEDS = [
    "https://docs.gitlab.com/user/",
    "https://docs.gitlab.com/tutorials/",
    "https://docs.gitlab.com/subscriptions/",
    "https://docs.gitlab.com/user/gitlab_duo/",
    "https://docs.gitlab.com/solutions/",
    "https://docs.gitlab.com/api/",
]

# ============================================================
# PATH DEPTH GOVERNANCE
# ============================================================
#
# THIS is what prevents:
#
# /user/duo_agent_platform/flows/foundational_flows/fix_pipeline
#
# while still allowing:
#
# /user/gitlab_duo/
# /subscriptions/
# /tutorials/
#
# ============================================================

MAX_ABOUT_PATH_DEPTH = 4

MAX_DOCS_PATH_DEPTH = 3

# ============================================================
# SKIP PATTERNS
# ============================================================

SKIP_PATTERNS = [
    r"/blog/",
    r"/jobs/",
    r"/press/",
    r"/events/",
    r"/community/",
    r"/search/",
    r"/releases/",
    r"/handbook/",
    r"/legal/",
    r"/support/",
    r"/forum/",
    r"/university/",
    r"/customers\.gitlab",
    r"/status\.gitlab",
    r"/editor/",
    r"/-/",
    r"\.pdf$",
]

# ============================================================
# LOCALE REJECTION
# ============================================================
#
# Rejects:
#
# /de-de/
# /fr-fr/
# /pt-br/
# /it-it/
#
# and future locale routes automatically
#
# ============================================================

LOCALE_RE = re.compile(
    r"^/[a-z]{2}-[a-z]{2}(/|$)",
    re.I
)

# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; EnterpriseSalesRAGBot/1.0)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})

# ============================================================
# HELPERS
# ============================================================

def url_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def normalize_url(url: str) -> str:
    """
    Remove:
      - query params
      - fragments

    Normalize:
      - trailing slash
    """

    parsed = urlparse(url)

    cleaned = parsed._replace(
        query="",
        fragment=""
    )

    normalized = urlunparse(cleaned)

    if (
        normalized.endswith("/")
        and normalized.count("/") > 3
    ):
        normalized = normalized.rstrip("/")

    return normalized


def path_depth(url: str) -> int:
    """
    Examples:

      /pricing/                  -> 1
      /solutions/gitops/         -> 2
      /user/gitlab_duo/          -> 2
      /a/b/c/d                   -> 4
    """

    path = urlparse(url).path.strip("/")

    if not path:
        return 0

    return len(path.split("/"))


def should_skip(url: str) -> bool:

    parsed = urlparse(url)

    # --------------------------------------------------------
    # DOMAIN CHECK
    # --------------------------------------------------------

    if parsed.netloc not in {
        "about.gitlab.com",
        "docs.gitlab.com",
    }:
        return True

    # --------------------------------------------------------
    # LOCALE REJECTION
    # --------------------------------------------------------

    if LOCALE_RE.match(parsed.path):
        return True

    # --------------------------------------------------------
    # GENERIC SKIPS
    # --------------------------------------------------------

    for pattern in SKIP_PATTERNS:
        if re.search(pattern, url):
            return True

    # --------------------------------------------------------
    # PATH DEPTH GOVERNANCE
    # --------------------------------------------------------

    depth = path_depth(url)

    if "about.gitlab.com" in url:
        if depth > MAX_ABOUT_PATH_DEPTH:
            return True

    if "docs.gitlab.com" in url:
        if depth > MAX_DOCS_PATH_DEPTH:
            return True

    return False

# ============================================================
# FETCH
# ============================================================

def fetch(url: str):

    try:

        response = SESSION.get(
            url,
            timeout=20,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if "text/html" not in content_type:
            return None

        return response.text

    except Exception as e:

        print(f"FETCH ERROR: {url}")
        print(f"  -> {e}")

        return None

# ============================================================
# LINK EXTRACTION
# ============================================================

def extract_links(html: str, base_url: str):

    soup = BeautifulSoup(html, "lxml")

    discovered = set()

    for tag in soup.find_all("a", href=True):

        href = tag["href"]

        if href.startswith((
            "mailto:",
            "javascript:",
            "#",
        )):
            continue

        absolute = normalize_url(
            urljoin(base_url, href)
        )

        if should_skip(absolute):
            continue

        discovered.add(absolute)

    return list(discovered)

# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text: str) -> str:

    text = re.sub(
        r"[\u200b\u200c\u200d\ufeff]",
        "",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()

# ============================================================
# EXTRACTION
# ============================================================

def extract_page(url: str, html: str):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # output_format="markdown"
    #
    # This is CRITICAL because your chunker expects:
    #
    #   # headings
    #   ## sections
    #
    # Your old scraper did not guarantee this.
    # --------------------------------------------------------

    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        favor_precision=True,
        no_fallback=False,
    )

    if not markdown:
        return None

    markdown = clean_text(markdown)

    # reject low-content pages
    if len(markdown) < 300:
        return None

    soup = BeautifulSoup(html, "lxml")

    title = (
        soup.title.string.strip()
        if soup.title and soup.title.string
        else url
    )

    path = urlparse(url).path.strip("/")

    section = (
        path.split("/")[0]
        if path
        else "home"
    )

    page = {
        "id": url_id(url),
        "url": url,
        "title": title,
        "text": markdown,
        "source": (
            "docs"
            if "docs.gitlab.com" in url
            else "about"
        ),
        "section": section,
        "scraped_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime()
        ),
        "word_count": len(markdown.split()),
    }

    return page

# ============================================================
# MAIN CRAWLER
# ============================================================

def run():

    visited = set()

    written = 0

    queue = []

    # --------------------------------------------------------
    # ABOUT QUEUE
    # --------------------------------------------------------
    #
    # homepage discovery preserved
    #
    # depth 0 = seed
    # depth 1 = child
    # depth 2 = grandchild
    #
    # --------------------------------------------------------

    for url in ABOUT_SEEDS:
        queue.append((
            normalize_url(url),
            0,
            2,
            "about"
        ))

    # --------------------------------------------------------
    # DOCS QUEUE
    # --------------------------------------------------------
    #
    # docs crawl intentionally shallower
    #
    # --------------------------------------------------------

    for url in DOCS_SEEDS:
        queue.append((
            normalize_url(url),
            0,
            1,
            "docs"
        ))

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as fh:

        while queue:

            (
                url,
                crawl_depth,
                max_depth,
                crawl_type
            ) = queue.pop(0)

            if url in visited:
                continue

            if should_skip(url):
                continue

            visited.add(url)

            # ------------------------------------------------
            # BETTER LOGGING
            # ------------------------------------------------

            print(
                f"[{written+1}] "
                f"[crawl_depth={crawl_depth}] "
                f"[path_depth={path_depth(url)}] "
                f"[{crawl_type}] "
                f"{url}"
            )

            html = fetch(url)

            if not html:
                continue

            # ------------------------------------------------
            # EXTRACTION
            # ------------------------------------------------

            page = extract_page(url, html)

            if page:

                fh.write(
                    json.dumps(
                        page,
                        ensure_ascii=False
                    ) + "\n"
                )

                written += 1

                print(
                    f"  ✓ {page['title']} "
                    f"({page['word_count']} words)"
                )

            else:

                print(
                    "  ✗ skipped "
                    "(insufficient content)"
                )

            # ------------------------------------------------
            # DISCOVERY EXPANSION
            # ------------------------------------------------

            if crawl_depth < max_depth:

                child_links = extract_links(
                    html,
                    url,
                )

                for child in child_links:

                    if child not in visited:

                        queue.append((
                            child,
                            crawl_depth + 1,
                            max_depth,
                            crawl_type
                        ))

            time.sleep(DELAY_SEC)

    print("\n" + "=" * 60)
    print(f"Done. {written} pages scraped.")
    print(f"Output: {OUTPUT_FILE}")

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    run()