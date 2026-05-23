import json
import re
from pathlib import Path

INPUT_FILE = Path("discovered_urls.jsonl")
OUTPUT_FILE = Path("classified_urls.jsonl")

# ─────────────────────────────────────────────────────────────
# CATEGORY RULES
# ─────────────────────────────────────────────────────────────

CATEGORY_RULES = {

    "pricing": [
        "/pricing/",
        "premium",
        "ultimate",
    ],

    "solutions": [
        "/solutions/",
        "devsecops",
        "delivery automation",
        "value stream",
    ],

    "security": [
        "security",
        "compliance",
        "sast",
        "dast",
        "secret detection",
        "dependency scanning",
        "container scanning",
    ],

    "ci_cd": [
        "ci/cd",
        "pipeline",
        "continuous integration",
        "continuous delivery",
    ],

    "source_control": [
        "git",
        "source code",
        "merge request",
        "version control",
    ],

    "gitops": [
        "gitops",
        "kubernetes",
        "cluster",
    ],

    "ai": [
        "duo",
        "ai",
        "agentic",
    ],

    "customers": [
        "/customers/",
        "case study",
        "customer story",
    ],

    "enterprise": [
        "enterprise",
        "governance",
        "audit",
        "compliance",
        "sso",
        "saml",
    ],

    "migration": [
        "migrate",
        "github",
        "jenkins",
        "azure devops",
        "bitbucket",
    ],
}

# ─────────────────────────────────────────────────────────────
# PRIORITY RULES
# ─────────────────────────────────────────────────────────────

HIGH_VALUE = [
    "/pricing/",
    "/solutions/",
    "/customers/",
    "ultimate",
    "security",
    "compliance",
    "enterprise",
    "why gitlab",
    "gitlab duo",
]

LOW_VALUE = [
    "/api/",
    "/install/",
    "/runner/",
    "/yaml/",
    "/archives/",
    "troubleshooting",
]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────


def classify_page(text: str) -> list:
    text = text.lower()

    categories = []

    for category, rules in CATEGORY_RULES.items():
        for rule in rules:
            if rule.lower() in text:
                categories.append(category)
                break

    return sorted(list(set(categories)))



def score_page(text: str) -> int:
    text = text.lower()
    score = 0

    for p in HIGH_VALUE:
        if p.lower() in text:
            score += 3

    for p in LOW_VALUE:
        if p.lower() in text:
            score -= 2

    return score



def priority(score: int) -> str:
    if score >= 8:
        return "HIGH"
    elif score >= 4:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def run():

    with INPUT_FILE.open(encoding="utf-8") as fin, \
         OUTPUT_FILE.open("w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)

            combined = " ".join([
                row.get("url", ""),
                row.get("title", ""),
                row.get("h1", ""),
                row.get("meta_description", ""),
            ])

            categories = classify_page(combined)
            score = score_page(combined)
            rel = priority(score)

            # Ignore junk pages
            scrape = rel != "LOW"

            out = {
                **row,
                "categories": categories,
                "priority_score": score,
                "relevance": rel,
                "scrape": scrape,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Done. Classified URLs written to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
