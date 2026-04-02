"""Download Spider dataset from HuggingFace and save as JSON."""

from __future__ import annotations

import json
import os
from typing import Any

from datasets import load_dataset


def extract_examples(split_dataset: Any) -> list[dict[str, str]]:
    """Extracts and formats relevant fields from a raw dataset split.

    Args:
        split_dataset: A HuggingFace ``Dataset`` object.

    Returns:
        A list of dictionaries, each containing ``question``, ``sql``,
        and ``db_id`` keys.
    """
    examples: list[dict[str, str]] = []
    for item in split_dataset:
        examples.append({
            "question": item["question"],
            "sql": item["query"],
            "db_id": item["db_id"],
        })
    return examples


def download_spider(output_dir: str) -> None:
    """Downloads Spider dataset via HuggingFace and saves raw JSON files.

    Args:
        output_dir: Directory where the JSON files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset("spider")

    for split_name, split_dataset in dataset.items():
        output_path = os.path.join(output_dir, f"spider_{split_name}.json")
        examples = extract_examples(split_dataset)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Entry point for Spider dataset download."""
    download_spider("data/raw/spider")


if __name__ == "__main__":
    main()
