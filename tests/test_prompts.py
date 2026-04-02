"""Tests for ChatML prompt helpers."""

from __future__ import annotations

from sql_agent.dataset_utils.prompts import IM_END_TOKEN, build_chatml_prompt, format_completion


def test_build_chatml_prompt_includes_schema_and_question() -> None:
    """build_chatml_prompt embeds schema and question in ChatML."""
    schema = "CREATE TABLE users (id INT);"
    question = "How many users?"
    prompt = build_chatml_prompt(schema=schema, question=question)
    assert schema in prompt
    assert question in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_format_completion_appends_im_end_token() -> None:
    """format_completion appends the ChatML end token."""
    sql = "SELECT 1"
    formatted = format_completion(sql)
    assert formatted.endswith(IM_END_TOKEN)
    assert formatted.startswith(sql)
