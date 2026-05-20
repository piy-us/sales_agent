MAX_CONTEXT_CHARS = 12000


class ContextBuilder:

    def build_context(
        self,
        docs: list,
    ) -> str:

        context_parts = []

        current_size = 0

        for doc in docs:

            chunk = f"""
SOURCE:
{doc["source"]}

CONTENT:
{doc["content"]}
"""

            if (
                current_size + len(chunk)
                > MAX_CONTEXT_CHARS
            ):
                break

            context_parts.append(chunk)

            current_size += len(chunk)

        return "\n\n".join(
            context_parts
        )