"""Tests for tokenizer loading utilities."""

from __future__ import annotations

import pytest


transformers = pytest.importorskip("transformers")

from sql_agent.models.tokenizer import load_tokenizer


class DummyTokenizer:
    """Minimal tokenizer stub for testing."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"


def test_load_tokenizer_sets_pad_token(monkeypatch) -> None:
    """load_tokenizer assigns pad_token when missing."""
    def _fake_from_pretrained(*_args, **_kwargs):
        return DummyTokenizer()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", _fake_from_pretrained
    )

    tok = load_tokenizer("dummy-model")
    assert tok.pad_token == tok.eos_token
