"""Tests for shared SQL extraction from model geenrations."""

from __future__ import annotations

from sql_agent.generation import extract_sql_from_generation, strip_code_fences


def test_strip_code_fences() -> None:
    assert strip_code_fences("```sql\nSELECT 1\n```") == "SELECT 1"


def test_plain_sql_with_im_end() -> None:
    text = "<|im_start|>assistant\nSELECT * FROM users; <|im_end|>"
    assert extract_sql_from_generation(text) == "SELECT * FROM users"


def test_cuts_at_blank_line() -> None:
    text = "SELECT name FROM users\n\nSELECT garbage FROM users"
    assert extract_sql_from_generation(text) == "SELECT name FROM users"


def test_cuts_second_query_on_new_line() -> None:
    text = "SELECT a FROM t\nSELECT b FROM t"
    assert extract_sql_from_generation(text) == "SELECT a FROM t"


def test_cuts_at_non_ascii() -> None:
    text = "SELECT a FROM t \u4e2d\u6587 garbage"
    assert extract_sql_from_generation(text) == "SELECT a FROM t"


def test_keeps_first_statement_only() -> None:
    text = "SELECT a FROM t; DROP TABLE t"
    assert extract_sql_from_generation(text) == "SELECT a FROM t"

def test_rejects_non_select() -> None:
    assert extract_sql_from_generation("DROP TABLE users") == ""

def test_with_query_allowed() -> None:
    text = "WITH x AS (SELECT 1) SELECT * FROM x"
    assert extract_sql_from_generation(text) == text
