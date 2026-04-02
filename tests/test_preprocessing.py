"""Tests for data preprocessing utilities."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from sql_agent.dataset_utils.preprocessing import collate_fn


class DummyTokenizer:
    """A minimal tokenizer for testing collation."""

    pad_token_id = 0

    def __call__(
        self,
        texts,
        *,
        padding=None,
        truncation=False,
        max_length=None,
        add_special_tokens=False,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        attention_mask = []

        for text in texts:
            tokens = [t for t in text.split(" ") if t]
            ids = list(range(1, len(tokens) + 1))
            if truncation and max_length is not None:
                ids = ids[:max_length]
            if padding == "max_length" and max_length is not None:
                pad_len = max_length - len(ids)
                if pad_len > 0:
                    ids = ids + [self.pad_token_id] * pad_len
            mask = [1 if i != self.pad_token_id else 0 for i in ids]
            input_ids.append(ids)
            attention_mask.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(
                    attention_mask, dtype=torch.long
                ),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collate_fn_masks_prompt_and_padding() -> None:
    """collate_fn masks prompt tokens and padding in labels."""
    # One example is enough to test masking behavior.
    tokenizer = DummyTokenizer()
    batch = [
        {
            "question": "q1",
            "sql": "SELECT 1",
            "schema": "CREATE TABLE t (id INT);",
        }
    ]
    out = collate_fn(batch, tokenizer, max_length=20)
    labels = out["labels"]
    attention_mask = out["attention_mask"]

    assert labels.shape == out["input_ids"].shape

    # All padding positions should be masked.
    assert torch.all(labels[attention_mask == 0] == -100)

    # At least one label should be unmasked (completion tokens).
    assert torch.any(labels[attention_mask == 1] != -100)
