"""ChatML prompt templates for Qwen-style SQL generation."""

from __future__ import annotations

IM_END_TOKEN = "<|im_end|>"

_SYSTEM_MESSAGE = (
    "Generate a single SQLite SELECT query. "
    "No explanations."
)

_USER_TEMPLATE = """\
Given the following database schema:

{schema}

Question: {question}"""


def build_chatml_prompt(schema: str, question: str) -> str:
    """Builds a ChatML prompt optimized for Qwen2.5-0.5B.

    Args:
        schema: Database schema as CREATE TABLE statements.
        question: Natural language question.

    Returns:
        Full ChatML prompt ending with ``<|im_start|>assistant\\n``.
    """
    user_content = _USER_TEMPLATE.format(schema=schema, question=question)
    return (
        f"<|im_start|>system\n{_SYSTEM_MESSAGE}\n{IM_END_TOKEN}\n"
        f"<|im_start|>user\n{user_content}\n{IM_END_TOKEN}\n"
        "<|im_start|>assistant\n"
    )


def format_completion(sql: str) -> str:
    """Formats a SQL string as a ChatML assistant completion.

    Args:
        sql: Target SQL query.

    Returns:
        SQL string with trailing ``<|im_end|>`` token.
    """
    return sql.strip() + IM_END_TOKEN
