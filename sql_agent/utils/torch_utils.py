"""PyTorch helpers shared across training and tollout scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from transformers import BitsAndBytesConfig


DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype_key: str) -> torch.dtype:
    """Converts a string dtype name to the corresponding ``torch.dtype``.

    Args:
        dtype_key: One of ``"float16"``, ``"bfloat16"``, ``"float32"``.

    Returns:
        The matching ``torch.dtype``.

    Raises:
        ValueError: If ``dtype_key`` is not recognized.
    """
    if dtype_key not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype: '{dtype_key}'. "
            f"Supported values: {list(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[dtype_key]


def build_quantization_config(dtype: torch.dtype) -> BitsAndBytesConfig:
    """Creates a bitsandbytes 4-bit quantization config for QLoRA.

    Args:
        dtype: Compute dtype.

    Returns:
        Configured ``BitsAndBytesConfig``.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def resolve_sft_adapter_dir(
    sft_cfg: dict[str, Any],
    override: Optional[str] = None,
) -> Path:
    """Resolves the directory containing the saved SFT LoRA adapter.

    Args:
        sft_cfg: Loaded SFT configuration dictionary. Must contain
            ``"output_dir"`` if ``override`` is not provided.
        override: Optional CLI override path. When provided, takes
            precedence over the config.

    Returns:
        Path to the adapter directory.
    """
    if override:
        return Path(override)
    return Path(sft_cfg["output_dir"]) / "final"
