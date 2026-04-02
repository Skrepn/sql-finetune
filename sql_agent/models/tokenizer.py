"""Tokenizer loading utilities for the SQL agent pipeline."""

from __future__ import annotations

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_name: str,
    *,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerBase:
    """Loads and configures a tokenizer for SQL generation.

    Args:
        model_name: HuggingFace model name or local path.
        trust_remote_code: Whether to allow execution of custom
            tokenizer code from the model repository. Defaults to
            ``True`` because Qwen models require it.

    Returns:
        Configured tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "pad_token not set; using eos_token (%s) as pad_token.",
            tokenizer.eos_token,
        )

    return tokenizer
