RETRIEVAL_KEYWORDS = [
    "pricing",
    "security",
    "compare",
    "documentation",
    "feature",
]


class QueryRouter:

    def should_use_rag(
        self,
        query: str,
    ) -> bool:

        query_lower = query.lower()

        return any(
            keyword in query_lower
            for keyword in RETRIEVAL_KEYWORDS
        )