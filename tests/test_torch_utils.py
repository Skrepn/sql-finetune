"""Tests for torch utility helpers."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sql_agent.utils.torch_utils import (
    build_quantization_config,
    resolve_dtype,
    resolve_sft_adapter_dir,
)


def test_resolve_dtype_accepts_valid_keys() -> None:
    """resolve_dtype maps known strings to torch dtypes."""
    assert resolve_dtype("float16") == torch.float16
    assert resolve_dtype("bfloat16") == torch.bfloat16
    assert resolve_dtype("float32") == torch.float32


def test_resolve_dtype_rejects_invalid_key() -> None:
    """resolve_dtype raises on unknown dtype keys."""
    with pytest.raises(ValueError):
        resolve_dtype("float64")


def test_build_quantization_config_has_4bit() -> None:
    """build_quantization_config returns a 4-bit quantization config."""
    cfg = build_quantization_config(torch.float16)
    assert cfg.load_in_4bit is True


def test_resolve_sft_adapter_dir_uses_override(tmp_path) -> None:
    """resolve_sft_adapter_dir prioritizes override paths."""
    override = tmp_path / "override"
    cfg = {"output_dir": "runs/sft"}
    assert resolve_sft_adapter_dir(cfg, str(override)) == override
