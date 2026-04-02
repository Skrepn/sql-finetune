"""Tests for preference building script helpers."""

from __future__ import annotations

import sys
import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


def _load_module():
    """Load the build_preferences module from the scripts directory.

    Uses importlib to load the module by file path since the scripts
    directory is not a Python package.

    Returns:
        The loaded build_preferences module.
    """
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "build_preferences.py"

    # Create a module specification from the file location
    spec = importlib.util.spec_from_file_location(
        "build_preferences", module_path
    )

    # Create a new module object from the specification
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader

    # Register the module in sys.modules so Python knows it exists
    sys.modules[spec.name] = module

    # Execute the module code.
    spec.loader.exec_module(module)
    return module


def test_ensure_im_end_appends_token() -> None:
    """_ensure_im_end appends the ChatML end token when missing."""
    mod = _load_module()
    assert mod._ensure_im_end("SELECT 1").endswith("<|im_end|>")
    assert mod._ensure_im_end("<|im_end|>") == "<|im_end|>"
    assert mod._ensure_im_end("") == "<|im_end|>"


def test_candidate_ok_filters_valid_and_exec() -> None:
    """_candidate_ok respects required validity flags."""
    mod = _load_module()
    cfg = mod.PreferenceBuilderConfig(require_valid_sql=True)
    assert mod._candidate_ok({"valid": True}, cfg)
    assert not mod._candidate_ok({"valid": False}, cfg)

    cfg = mod.PreferenceBuilderConfig(
        require_valid_sql=False,
        require_exec_ok=True,
    )
    assert mod._candidate_ok({"exec_ok": True}, cfg)
    assert not mod._candidate_ok({"exec_ok": False}, cfg)


def test_pick_pairs_respects_min_gap_and_ties() -> None:
    """_pick_pairs_for_group enforces min_reward_gap and allow_ties."""
    mod = _load_module()
    rows = [
        {"candidate_id": 0, "reward": 1.0, "valid": True, "exec_ok": True},
        {"candidate_id": 1, "reward": 0.9, "valid": True, "exec_ok": True},
        {"candidate_id": 2, "reward": 0.0, "valid": True, "exec_ok": True},
    ]

    cfg = mod.PreferenceBuilderConfig(min_reward_gap=0.5, allow_ties=False)
    pairs = mod._pick_pairs_for_group(rows, cfg)
    assert len(pairs) == 1
    assert pairs[0][2] >= 0.5

    cfg = mod.PreferenceBuilderConfig(min_reward_gap=0.5, allow_ties=True)
    pairs = mod._pick_pairs_for_group(
        [
            {"candidate_id": 0, "reward": 1.0, "valid": True, "exec_ok": True},
            {"candidate_id": 1, "reward": 1.0, "valid": True, "exec_ok": True},
        ],
        cfg,
    )
    assert len(pairs) == 1
    assert pairs[0][2] == 0.0


def test_validate_args_rejects_invalid_values() -> None:
    """_validate_args rejects negative or zero constraints."""
    mod = _load_module()
    args = Namespace(min_reward_gap=-1, max_pairs_per_example=1, max_examples=None)
    with pytest.raises(ValueError):
        mod._validate_args(args)

    args = Namespace(min_reward_gap=0.1, max_pairs_per_example=0, max_examples=None)
    with pytest.raises(ValueError):
        mod._validate_args(args)

    args = Namespace(min_reward_gap=0.1, max_pairs_per_example=1, max_examples=0)
    with pytest.raises(ValueError):
        mod._validate_args(args)
