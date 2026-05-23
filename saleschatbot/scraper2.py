"""
GitLab Sales Knowledge Base Scraper
====================================
Scrapes about.gitlab.com pages curated for a sales agent RAG pipeline.
Covers: pricing, solutions, topics, platform, customers, company pages.

Usage:
    pip install -r requirements.txt
    python scraper.py

Output: gitlab_kb.jsonl  (one JSON record per page)
Then run: python chunker.py  → chunked_dataset.jsonl
"""

import json, time, hashlib, re, logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
import trafilatura

# ─────────────────────────────────────────────────────────────────────────────
# SEED URLS — every footer link, categorized by sales relevance
# ─────────────────────────────────────────────────────────────────────────────

SEED_URLS = [

    # ── PRICING (highest value for a sales agent) ─────────────────────────
    ("https://about.gitlab.com/pricing/",           "pricing",   "HIGH"),
    ("https://about.gitlab.com/pricing/premium/",   "pricing",   "HIGH"),
    ("https://about.gitlab.com/pricing/ultimate/",  "pricing",   "HIGH"),

    # ── PLATFORM & PRODUCT ────────────────────────────────────────────────
    ("https://about.gitlab.com/platform/",                      "product", "HIGH"),
    ("https://about.gitlab.com/gitlab-duo-agent-platform/",     "product", "HIGH"),
    ("https://about.gitlab.com/why-gitlab/",                    "product", "HIGH"),

    # ── SOLUTIONS (what a sales rep maps to client pain points) ──────────
    ("https://about.gitlab.com/solutions/continuous-integration/",      "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/source-code-management/",      "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/delivery-automation/",         "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/application-security-testing/","solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/supply-chain/",                "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/software-compliance/",         "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/visibility-measurement/",      "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/value-stream-management/",     "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/analytics-and-insights/",      "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/agile-delivery/",              "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/gitops/",                      "solutions", "HIGH"),
    ("https://about.gitlab.com/solutions/value-stream-management/",     "solutions", "HIGH"),

    # ── VERTICALS / SEGMENTS ──────────────────────────────────────────────
    ("https://about.gitlab.com/enterprise/",                    "segment", "HIGH"),
    ("https://about.gitlab.com/small-business/",                "segment", "MEDIUM"),
    ("https://about.gitlab.com/solutions/public-sector/",       "segment", "MEDIUM"),
    ("https://about.gitlab.com/solutions/education/",           "segment", "LOW"),
    ("https://about.gitlab.com/solutions/finance/",             "segment", "MEDIUM"),

    # ── TOPICS (educational, good for explaining concepts to non-tech) ────
    ("https://about.gitlab.com/topics/ci-cd/",                  "topics", "HIGH"),
    ("https://about.gitlab.com/topics/devops/",                 "topics", "HIGH"),
    ("https://about.gitlab.com/topics/devsecops/",              "topics", "HIGH"),
    ("https://about.gitlab.com/topics/gitops/",                 "topics", "MEDIUM"),
    ("https://about.gitlab.com/topics/version-control/",        "topics", "MEDIUM"),
    ("https://about.gitlab.com/topics/cloud-native/",           "topics", "MEDIUM"),
    ("https://about.gitlab.com/topics/devops/ai-for-coding/",   "topics", "HIGH"),
    ("https://about.gitlab.com/topics/agentic-ai/",             "topics", "HIGH"),

    # ── CUSTOMERS / CASE STUDIES (social proof — gold for sales) ─────────
    ("https://about.gitlab.com/customers/",                     "customers", "HIGH"),

    # ── COMPANY (trust, credibility) ──────────────────────────────────────
    ("https://about.gitlab.com/company/",                       "company", "MEDIUM"),
    ("https://about.gitlab.com/security/",                      "company", "MEDIUM"),  # Trust Center
    ("https://about.gitlab.com/services/",                      "company", "MEDIUM"),

    # ── RESOURCES ─────────────────────────────────────────────────────────
    ("https://about.gitlab.com/get-started/",                   "resources", "MEDIUM"),
    ("https://about.gitlab.com/install/",                       "resources", "LOW"),
]

# Pages/domains NOT worth scraping for a sales agent KB
SKIP_PATTERNS = [
    r"/blog/",          # blog = high volume, low signal; add curated posts manually
    r"/jobs/",
    r"/press/",
    r"/events/",
    r"/community/",
    r"/partners/",
    r"sign_in",
    r"#",
    r"\.pdf$",
    r"forum\.gitlab",
    r"university\.gitlab",
    r"customers\.gitlab",   # login-walled portal
    r"status\.gitlab",
    r"ir\.gitlab",          # investor relations
    r"handbook\.gitlab",    # internal handbook — too much noise for sales agent
]

OUTPUT_FILE = Path("gitlab_kb.jsonl")
DELAY_SEC   = 1.0   # be polite — 1 req/sec

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HTTP SESSION
# ─────────────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; GitLabSalesRAGBot/1.0; "
        "enterprise knowledge base scraper; not for redistribution)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

# ─────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Page:
    id:           str
    url:          str
    title:        str
    text:         str
    headings:     list
    category:     str   # pricing | solutions | topics | customers | etc.
    relevance:    str   # HIGH | MEDIUM | LOW
    scraped_at:   str
    word_count:   int

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def url_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def clean(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def should_skip(url: str) -> bool:
    return any(re.search(p, url) for p in SKIP_PATTERNS)

def extract_page(url: str, html: str, category: str, relevance: str) -> Optional[Page]:
    """
    Dual-strategy extraction:
    1. trafilatura for main body text (strips nav/footer/ads automatically)
    2. BeautifulSoup for structured metadata (title, headings)

    trafilatura is specifically designed for news/marketing pages and handles
    about.gitlab.com's JS-rendered content better than raw BS4.
    """
    soup = BeautifulSoup(html, "lxml")

    # ── Title ──────────────────────────────────────────────────────────────
    h1 = soup.find("h1")
    title = (
        h1.get_text(strip=True) if h1
        else (soup.title.string.split("|")[0].strip() if soup.title else url)
    )

    # ── Headings (useful metadata for RAG filtering) ───────────────────────
    headings = [
        h.get_text(strip=True)
        for h in soup.find_all(["h2", "h3"])
        if len(h.get_text(strip=True)) > 3
    ][:20]   # cap at 20 to keep record size sane

    # ── Main text via trafilatura ──────────────────────────────────────────
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,    # pricing tables are valuable
        favor_precision=True,   # drop boilerplate aggressively
        no_fallback=False,
    )

    # Fallback: if trafilatura gets nothing, try BS4 on <main>/<article>
    if not extracted or len(extracted) < 150:
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"content|main", re.I))
        )
        if main:
            for tag in main(["nav", "footer", "script", "style", "aside"]):
                tag.decompose()
            extracted = main.get_text(separator="\n", strip=True)

    if not extracted or len(extracted) < 150:
        return None

    text = clean(extracted)

    # ── Special enrichment for pricing pages ──────────────────────────────
    # Pricing tables in trafilatura may lose structure; pull plan names + prices
    # explicitly and prepend as a summary block.
    if "pricing" in url:
        price_summary = extract_pricing_summary(soup)
        if price_summary:
            text = price_summary + "\n\n" + text

    return Page(
        id=url_id(url),
        url=url,
        title=title,
        text=text,
        headings=headings,
        category=category,
        relevance=relevance,
        scraped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        word_count=len(text.split()),
    )


def extract_pricing_summary(soup: BeautifulSoup) -> str:
    """
    Pull plan names + prices as clean structured text.
    This ensures pricing info survives even if trafilatura collapses tables.
    """
    lines = ["=== PRICING SUMMARY ==="]
    # Look for price blocks (GitLab pricing page uses h2 for plan names)
    for h2 in soup.find_all("h2"):
        text = h2.get_text(strip=True)
        if text in ("Free", "Premium", "Ultimate"):
            price_el = h2.find_next(string=re.compile(r"\$\d+|custom|contact", re.I))
            price = price_el.strip() if price_el else "See site"
            lines.append(f"Plan: {text} — {price}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER PAGE EXPANSION
# Customers page lists case studies — extract individual customer links too
# ─────────────────────────────────────────────────────────────────────────────

def get_customer_links(html: str) -> list:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/customers/" in href and href != "https://about.gitlab.com/customers/":
            full = href if href.startswith("http") else "https://about.gitlab.com" + href
            if not should_skip(full):
                links.append(full)
    return list(set(links))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CRAWLER
# ─────────────────────────────────────────────────────────────────────────────

def fetch(url: str) -> Optional[str]:
    try:
        r = SESSION.get(url, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.warning(f"FETCH ERROR {url}: {e}")
        return None


def run():
    visited = set()
    queue   = [(url, cat, rel) for url, cat, rel in SEED_URLS]
    written = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        while queue:
            url, category, relevance = queue.pop(0)

            if url in visited or should_skip(url):
                continue
            visited.add(url)

            log.info(f"[{written+1}] [{category.upper()}] {url}")
            html = fetch(url)
            if not html:
                continue

            # Expand customer listing page into individual case studies
            if url == "https://about.gitlab.com/customers/":
                customer_links = get_customer_links(html)
                for clink in customer_links:
                    if clink not in visited:
                        queue.append((clink, "customers", "HIGH"))
                log.info(f"  → Found {len(customer_links)} customer case study links")

            page = extract_page(url, html, category, relevance)

            if page:
                fh.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
                written += 1
                log.info(f"  ✓ '{page.title}' — {page.word_count} words")
            else:
                log.info(f"  ✗ skipped (insufficient content)")

            time.sleep(DELAY_SEC)

    log.info(f"\n{'='*50}")
    log.info(f"Done. {written} pages scraped → {OUTPUT_FILE}")
    log.info(f"Next: run  python chunker.py")


if __name__ == "__main__":
    run()