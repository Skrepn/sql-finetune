"""Tests for the SQL sandbox helpers."""

from __future__ import annotations

import re

from sql_agent.env.sql_sandbox import (
    SqlErrorType,
    SqlSandbox,
    normalize_result_rows,
)


def test_validate_sql_rejects_empty(sandbox: SqlSandbox) -> None:
    """validate_sql rejects empty or None SQL strings."""
    ok, reason = sandbox.validate_sql("")
    assert not ok
    assert "Empty" in reason

    ok, reason = sandbox.validate_sql(None)
    assert not ok
    assert "None" in reason


def test_validate_sql_rejects_forbidden_tokens(sandbox: SqlSandbox) -> None:
    """validate_sql blocks comment tokens."""
    ok, reason = sandbox.validate_sql("SELECT * FROM users -- comment")
    assert not ok
    assert "Forbidden token" in reason


def test_validate_sql_rejects_forbidden_keyword(sandbox: SqlSandbox) -> None:
    """validate_sql blocks write statements even if prefixed with SELECT."""
    ok, reason = sandbox.validate_sql("SELECT * FROM users; DROP TABLE users")
    assert not ok
    assert "Forbidden keyword" in reason or "Multiple statements" in reason

    ok, reason = sandbox.validate_sql("SELECT * FROM users WHERE 1=1")
    assert ok
    assert reason is None


def test_validate_sql_rejects_non_select(sandbox: SqlSandbox) -> None:
    """validate_sql rejects non-SELECT queries."""
    ok, reason = sandbox.validate_sql("UPDATE users SET age = 1")
    assert not ok
    assert "Only SELECT" in reason


def test_execute_success_returns_rows(sandbox: SqlSandbox) -> None:
    """execute returns rows and OK status for valid queries."""
    # Use a simple query with stable ordering.
    result = sandbox.execute("test_db", "SELECT name FROM users ORDER BY id")
    assert result.ok
    assert result.error_type == SqlErrorType.OK
    assert result.rows == (("Alice",), ("Bob",), ("Carol",))
    assert not result.truncated
    assert result.elapsed_ms >= 0


def test_execute_forbidden_returns_forbidden(sandbox: SqlSandbox) -> None:
    """execute returns forbidden for invalid SQL."""
    result = sandbox.execute("test_db", "DROP TABLE users")
    assert not result.ok
    assert result.error_type == SqlErrorType.FORBIDDEN


def test_execute_syntax_error_returns_syntax_error(
    sandbox: SqlSandbox,
) -> None:
    """execute returns syntax_error for malformed SQL."""
    result = sandbox.execute("test_db", "SELECT FROM users")
    assert not result.ok
    assert result.error_type == SqlErrorType.SYNTAX_ERROR


def test_normalize_result_rows_sort_and_nulls() -> None:
    """normalize_result_rows sorts rows and stringifies NULLs."""
    # Order should be normalized and NULL stringified.
    rows = [("b", None), ("a", 1)]
    normalized = normalize_result_rows(rows)
    assert normalized == (("a", "1"), ("b", "NULL"))


def test_validate_sql_allows_with_queries(sandbox: SqlSandbox) -> None:
    """validate_sql allows WITH queries."""
    ok, reason = sandbox.validate_sql(
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t"
    )
    assert ok
    assert reason is None
