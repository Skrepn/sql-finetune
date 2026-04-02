"""Tests for rollout collection helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("transformers")
pytest.importorskip("peft")


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "collect_rollout.py"
    spec = importlib.util.spec_from_file_location(
        "collect_rollout", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_strip_code_fences() -> None:
    """_strip_code_fences removes wrapping fences."""
    mod = _load_module()
    text = "```sql\nSELECT 1\n```"
    assert mod._strip_code_fences(text) == "SELECT 1"


def test_extract_sql_from_generated() -> None:
    """_extract_sql_from_generated returns cleaned SQL."""
    mod = _load_module()
    text = "<|im_start|>assistant\nSELECT * FROM users; <|im_end|>"
    assert mod._extract_sql_from_generated(text) == "SELECT * FROM users"
