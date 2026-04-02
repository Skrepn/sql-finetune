"""File I/O helpers for JSONL and JSON used across the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yields parsed objects from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Yields:
        One parsed dictionary per non-empty line.

    Raises:
        ValueError: If a line contains invalid JSON.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {exc}"
                ) from exc


def write_jsonl_line(fp: Any, obj: dict[str, Any]) -> None:
    """Writes a single JSON object as one JSONL line.

    Args:
        fp: Writable file-like object.
        obj: Dictionary to serialize.
    """
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: dict[str, Any]) -> None:
    """Saves a dictionary as a JSON file.

    Args:
        path: Output file path.
        obj: JSON-serializable dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    """Saves run metadata to ``run_meta.json`` inside the run directory.

    Args:
        run_dir: Run output directory.
        meta: Metadata dictionary.
    """
    save_json(run_dir / "run_meta.json", meta)
