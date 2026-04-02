"""Shared test fixtures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sql_agent.env.sql_sandbox import SqlSandbox


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Creates a small SQLite database with two tables.

    Returns:
        Path to the database root directory.
    """
    db_id = "test_db"
    db_dir = tmp_path / db_id
    db_dir.mkdir()
    db_file = db_dir / f"{db_id}.sqlite"

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "CREATE TABLE users ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  age INTEGER"
            ")"
        )
        conn.executemany(
            "INSERT INTO users (id, name, age) VALUES (?, ?, ?)",
            [(1, "Alice", 30), (2, "Bob", 25), (3, "Carol", 35)],
        )
        conn.execute(
            "CREATE TABLE orders ("
            "  order_id INTEGER PRIMARY KEY,"
            "  user_id INTEGER,"
            "  amount REAL,"
            "  FOREIGN KEY (user_id) REFERENCES users(id)"
            ")"
        )
        conn.executemany(
            "INSERT INTO orders (order_id, user_id, amount) VALUES (?, ?, ?)",
            [(1, 1, 99.99), (2, 1, 49.50), (3, 2, 200.00)],
        )

    return tmp_path


@pytest.fixture()
def sandbox(tmp_db: Path) -> SqlSandbox:
    """Creates a sandbox pointed at the temp database."""
    return SqlSandbox(database_root=tmp_db, timeout_s=1.0, max_rows=100)


@pytest.fixture()
def sample_rollouts(tmp_path: Path) -> Path:
    """Creates a sample rollouts.jsonl for testing preferences.

    3 groups: big gap, small gap, only one valid candidate.
    """
    rows = [
        {"id": "ex-001", "db_id": "db1", "prompt": "p1",
         "candidate_id": 0, "candidate_sql": "SELECT count(*) FROM t",
         "reward": 1.0, "valid": True, "exec_ok": True},
        {"id": "ex-001", "db_id": "db1", "prompt": "p1",
         "candidate_id": 1, "candidate_sql": "SELECT * FROM t",
         "reward": -0.2, "valid": True, "exec_ok": False},
        {"id": "ex-002", "db_id": "db2", "prompt": "p2",
         "candidate_id": 0, "candidate_sql": "SELECT a FROM t",
         "reward": 1.0, "valid": True, "exec_ok": True},
        {"id": "ex-002", "db_id": "db2", "prompt": "p2",
         "candidate_id": 1, "candidate_sql": "SELECT b FROM t",
         "reward": 0.9, "valid": True, "exec_ok": True},
        {"id": "ex-003", "db_id": "db3", "prompt": "p3",
         "candidate_id": 0, "candidate_sql": "SELECT 1",
         "reward": 1.0, "valid": True, "exec_ok": True},
        {"id": "ex-003", "db_id": "db3", "prompt": "p3",
         "candidate_id": 1, "candidate_sql": "DROP TABLE",
         "reward": -1.0, "valid": False, "exec_ok": False},
    ]
    path = tmp_path / "rollouts.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return path
