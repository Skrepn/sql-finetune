"""Data preprocessing and collation for SQL agent training/evaluation."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from sql_agent.dataset_utils.prompts import build_chatml_prompt, format_completion


def collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Collates raw examples into tokenized tensors with label masking.

    Concatenates each prompt and completion, tokenizes the result, and
    masks prompt/padding tokens in the labels so that the loss is
    computed only on the SQL completion.

    Args:
        batch: A list of example dictionaries, each containing at
            least ``question``, ``sql``, and ``schema`` keys.
        tokenizer: HuggingFace tokenizer used for encoding.
        max_length: Maximum total sequence length (prompt + completion).

    Returns:
        A dictionary with ``input_ids``, ``attention_mask``, and
        ``labels`` tensors ready for model forward pass.
    """
    prompts: list[str] = []
    completions: list[str] = []

    for example in batch:
        prompt = build_chatml_prompt(
            schema=example["schema"],
            question=example["question"],
        )
        completion = format_completion(example["sql"])
        prompts.append(prompt)
        completions.append(completion)

    full_texts = [p + c for p, c in zip(prompts, completions)]

    tokenized = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = input_ids.clone()

    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=False,
            max_length=max_length,
        )["input_ids"]

        prompt_len = len(prompt_ids)

        labels[i, :prompt_len] = -100

        # Mask padding tokens.
        labels[i, attention_mask[i] == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
