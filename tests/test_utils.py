"""Tests for utility helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sql_agent.utils.config import load_config
from sql_agent.utils.io import read_jsonl, save_json, save_run_meta, write_jsonl_line
from sql_agent.utils.logging_setup import setup_logging
from sql_agent.utils.timestamp import timestamp


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    """load_config reads YAML config files."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("foo: 1\nbar: baz\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg["foo"] == 1
    assert cfg["bar"] == "baz"


def test_load_config_missing_raises(tmp_path: Path) -> None:
    """load_config raises FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """read_jsonl yields objects written by write_jsonl_line."""
    # Write two lines and make sure we read them back.
    path = tmp_path / "data.jsonl"
    with path.open("w", encoding="utf-8") as f:
        write_jsonl_line(f, {"a": 1})
        write_jsonl_line(f, {"b": 2})
    items = list(read_jsonl(path))
    assert items == [{"a": 1}, {"b": 2}]


def test_save_json_and_run_meta(tmp_path: Path) -> None:
    """save_json and save_run_meta create files with JSON content."""
    # Ensure both files are created and parse correctly.
    meta = {"run": 1}
    save_json(tmp_path / "out.json", meta)
    save_run_meta(tmp_path, {"status": "ok"})

    out = json.loads((tmp_path / "out.json").read_text("utf-8"))
    run_meta = json.loads((tmp_path / "run_meta.json").read_text("utf-8"))
    assert out == meta
    assert run_meta["status"] == "ok"


def test_setup_logging_creates_log_file(tmp_path: Path) -> None:
    """setup_logging creates the log file."""
    setup_logging(tmp_path, "test.log")
    assert (tmp_path / "test.log").exists()


def test_timestamp_format() -> None:
    """timestamp returns a YYYYMMDD_HHMMSS string."""
    ts = timestamp()
    assert re.match(r"^\d{8}_\d{6}$", ts)
