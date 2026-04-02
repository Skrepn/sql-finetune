"""SQLite execution sandbox for SQL agents.

This module provides a defensive wrapper around SQLite execution suitable for
training and evaluating SQL-generation LLMs. All queries are executed in
read-only mode with timeout enforcement and forbidden-keyword filtering.
"""

from __future__ import annotations

import dataclasses
import enum
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import quote

__all__ = [
    "ExecutionResult",
    "SqlErrorType",
    "SqlSandbox",
    "normalize_result_rows",
]


class SqlErrorType(enum.Enum):
    """Categorization of SQL execution outcomes."""

    OK = "ok"
    FORBIDDEN = "forbidden"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    """Structured result of executing a SQL query in the sandbox.

    Attributes:
        ok: True if execution succeeded without errors.
        error_type: Categorized error type.
        error_message: Human-readable error description (empty on success).
        rows: Result rows as tuples. Empty on failure.
        truncated: True if more than ``max_rows`` rows were available.
        elapsed_ms: Wall-clock time spent on execution in milliseconds.
    """

    ok: bool
    error_type: SqlErrorType
    error_message: str
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    elapsed_ms: int


class SqlSandbox:
    """A safe, read-only SQLite execution environment."""

    # Keywords that should never appear in user SQL for a read-only sandbox.
    _FORBIDDEN_KEYWORDS = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "replace",
        "vacuum",
        "pragma",
        "attach",
        "detach",
        "reindex",
        "analyze",
        "explain",
        "begin",
        "commit",
        "rollback",
        "savepoint",
    )

    # Tokens usually used for comments.
    _FORBIDDEN_TOKENS = ("--", "/*", "*/")

    # Only queries that start with SELECT or WITH are allowed.
    _ALLOWED_PREFIX_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

    def __init__(
        self,
        database_root: str | Path,
        timeout_s: float = 2.0,
        max_rows: int = 200,
        progress_handler_steps: int = 10_000,
    ) -> None:
        """Initializes the sandbox.

        Args:
            database_root: Root directory that contains per-db folders,
                each holding a ``<db_id>.sqlite`` file.
            timeout_s: Timeout per query in seconds.
            max_rows: Maximum number of result rows to return.
            progress_handler_steps: SQLite VM instruction period for the
                progress handler callback.

        Raises:
            ValueError: If any numeric argument is non-positive.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if max_rows <= 0:
            raise ValueError("max_rows must be > 0")
        if progress_handler_steps <= 0:
            raise ValueError("progress_handler_steps must be > 0")

        self._database_root = Path(database_root)
        self._timeout_s = float(timeout_s)
        self._max_rows = int(max_rows)
        self._progress_handler_steps = int(progress_handler_steps)

    def get_db_path(self, db_id: str) -> Path:
        """Converts a database identifier to a SQLite file path.

        Args:
            db_id: Database identifier.

        Returns:
            Path to the SQLite database file.

        Raises:
            FileNotFoundError: If the expected SQLite file does not exist.
        """
        db_path = self._database_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        return db_path

    def validate_sql(self, sql: str) -> tuple[bool, Optional[str]]:
        """Validates that a SQL string is acceptable for read-only execution.

        Args:
            sql: SQL query text.

        Returns:
            A ``(is_valid, reason)`` tuple. If valid, reason is ``None``.
            If invalid, reason is a human-readable explanation.
        """
        if sql is None:
            return False, "SQL is None"

        sql_stripped = sql.strip()
        if not sql_stripped:
            return False, "Empty SQL"

        lowered = sql_stripped.lower()
        for tok in self._FORBIDDEN_TOKENS:
            if tok in lowered:
                return False, f"Forbidden token found: {tok}"

        if not self._ALLOWED_PREFIX_RE.match(sql_stripped):
            return False, "Only SELECT queries are allowed"

        for kw in self._FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", lowered):
                return False, f"Forbidden keyword found: {kw}"

        semi = sql_stripped.rstrip(";")
        if ";" in semi:
            return False, "Multiple statements are not allowed"

        return True, None

    def execute(self, db_id: str, sql: str) -> ExecutionResult:
        """Executes a SQL query safely against a read-only database.

        Args:
            db_id: Database identifier used to locate the SQLite file.
            sql: SQL query text.

        Returns:
            ``ExecutionResult`` with rows and diagnostics. On any failure
            the ``rows`` field is an empty tuple.
        """
        is_valid, reason = self.validate_sql(sql)
        if not is_valid:
            return ExecutionResult(
                ok=False,
                error_type=SqlErrorType.FORBIDDEN,
                error_message=reason or "Forbidden SQL",
                rows=(),
                truncated=False,
                elapsed_ms=0,
            )

        db_path = self.get_db_path(db_id)
        start = time.monotonic()

        try:
            with self._connect_read_only(db_path) as conn:
                # Enforce read-only at the connection level.
                try:
                    conn.execute("PRAGMA query_only = ON;")
                except sqlite3.DatabaseError:
                    pass

                # Timeout enforcement via progress handler.
                deadline = start + self._timeout_s

                def _progress_handler() -> int:
                    """Returns non-zero to interrupt execution on timeout."""
                    if time.monotonic() >= deadline:
                        return 1
                    return 0

                conn.set_progress_handler(
                    _progress_handler,
                    self._progress_handler_steps,
                )

                cur = conn.cursor()
                cur.execute(sql)

                # Fetch at most max_rows + 1 so we can detect truncation.
                fetched = cur.fetchmany(self._max_rows + 1)
                truncated = len(fetched) > self._max_rows
                if truncated:
                    fetched = fetched[: self._max_rows]

                rows = tuple(tuple(r) for r in fetched)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                # Disable progress handler explicitly.
                conn.set_progress_handler(None, 0)

                return ExecutionResult(
                    ok=True,
                    error_type=SqlErrorType.OK,
                    error_message="",
                    rows=rows,
                    truncated=truncated,
                    elapsed_ms=elapsed_ms,
                )

        except sqlite3.OperationalError as exc:
            msg = str(exc).strip()
            elapsed_ms = int((time.monotonic() - start) * 1000)

            if "interrupted" in msg.lower():
                return ExecutionResult(
                    ok=False,
                    error_type=SqlErrorType.TIMEOUT,
                    error_message=msg or "Query timeout",
                    rows=(),
                    truncated=False,
                    elapsed_ms=elapsed_ms,
                )

            if "syntax error" in msg.lower() or "near" in msg.lower():
                return ExecutionResult(
                    ok=False,
                    error_type=SqlErrorType.SYNTAX_ERROR,
                    error_message=msg,
                    rows=(),
                    truncated=False,
                    elapsed_ms=elapsed_ms,
                )

            return ExecutionResult(
                ok=False,
                error_type=SqlErrorType.RUNTIME_ERROR,
                error_message=msg,
                rows=(),
                truncated=False,
                elapsed_ms=elapsed_ms,
            )

        except sqlite3.ProgrammingError as exc:
            msg = str(exc).strip()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                ok=False,
                error_type=SqlErrorType.FORBIDDEN,
                error_message=msg,
                rows=(),
                truncated=False,
                elapsed_ms=elapsed_ms,
            )

        except Exception as exc:
            msg = str(exc).strip()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                ok=False,
                error_type=SqlErrorType.UNKNOWN_ERROR,
                error_message=msg or "Unknown error",
                rows=(),
                truncated=False,
                elapsed_ms=elapsed_ms,
            )

    def _connect_read_only(self, db_path: Path) -> sqlite3.Connection:
        """Opens a read-only connection to a SQLite database.

        Args:
            db_path: Path to the SQLite file.

        Returns:
            ``sqlite3.Connection`` opened in read-only mode.

        Raises:
            sqlite3.Error: If connection cannot be opened.
        """
        quoted = quote(db_path.as_posix(), safe="/")
        uri = f"file:{quoted}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)


def normalize_result_rows(
    rows: Sequence[Sequence[Any]],
) -> tuple[tuple[str, ...], ...]:
    """Normalizes result rows into a stable, comparable representation.

    Converts all values to strings (``None`` becomes ``"NULL"``), then
    sorts the rows. This is useful for execution-based evaluation where
    row ordering may not be guaranteed.

    Args:
        rows: Rows from a SQLite fetch operation.

    Returns:
        Normalized, sorted tuple of string-valued row tuples.
    """
    normalized: list[tuple[str, ...]] = []
    for row in rows:
        norm_row = tuple("NULL" if v is None else str(v) for v in row)
        normalized.append(norm_row)
    normalized.sort()
    return tuple(normalized)
