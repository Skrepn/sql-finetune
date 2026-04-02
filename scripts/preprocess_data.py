"""Preprocess Spider dataset and attach SQLite schemas to examples."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def load_json(path: str) -> list[dict[str, str]]:
    """Loads a JSON file containing a list of dictionaries.

    Args:
        path: Path to JSON file.

    Returns:
        List of dictionary examples.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], path: str) -> None:
    """Saves a list of dictionaries to a JSON file.

    Args:
        data: List of dictionaries.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_schema(db_path: Path) -> str:
    """Generates a CREATE TABLE representation of the database schema.

    Uses standard SQL DDL format that Qwen2.5 has seen extensively in
    pretraining. Includes column types, PRIMARY KEY markers, and
    FOREIGN KEY constraints.

    Args:
        db_path: Path to a SQLite file.

    Returns:
        Schema string as CREATE TABLE statements.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
            """
        )
        tables = [row[0] for row in cursor.fetchall()]

        ddl_blocks: list[str] = []

        for table in tables:
            # Get column definitions.
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            # table_info: (cid, name, type, notnull, default, pk)

            col_lines: list[str] = []
            for col in columns:
                col_name = col[1]
                col_type = col[2] or "TEXT"
                pk_flag = col[5]
                not_null = col[3]

                parts = [f"  {col_name} {col_type}"]
                if pk_flag:
                    parts.append("PRIMARY KEY")
                if not_null and not pk_flag:
                    parts.append("NOT NULL")
                col_lines.append(" ".join(parts))

            # Get foreign key constraints.
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            fks = cursor.fetchall()

            for fk in fks:
                fk_from = fk[3]
                fk_table = fk[2]
                fk_to = fk[4]
                col_lines.append(
                    f"  FOREIGN KEY ({fk_from}) REFERENCES {fk_table}({fk_to})"
                )

            body = ",\n".join(col_lines)
            ddl_blocks.append(f"CREATE TABLE {table} (\n{body}\n);")

    return "\n\n".join(ddl_blocks)


def add_schemas_to_json(
    dataset_path: Path,
    db_dir: Path,
    output_path: Path,
) -> None:
    """Enriches the dataset by adding a database schema to each example.

    Args:
        dataset_path: Path to the input JSON file.
        db_dir: Directory containing per-database SQLite folders.
        output_path: Path where the processed JSON will be saved.

    Raises:
        FileNotFoundError: If a referenced database file does not exist.
    """
    data = load_json(dataset_path)
    schema_cache: dict[str, str] = {}

    for example in data:
        db_id = example["db_id"]

        if db_id not in schema_cache:
            db_file = db_dir / db_id / f"{db_id}.sqlite"
            if not db_file.exists():
                raise FileNotFoundError(f"Database not found: {db_file}")
            schema_cache[db_id] = extract_schema(db_file)

        example["schema"] = schema_cache[db_id]

    save_json(data, output_path)


def main() -> None:
    """Preprocesses Spider train and validation splits."""
    db_folder = Path("data/raw/spider/database")
    add_schemas_to_json(
        Path("data/raw/spider/spider_train.json"),
        db_folder,
        Path("data/processed/sft_train.json"),
    )
    add_schemas_to_json(
        Path("data/raw/spider/spider_validation.json"),
        db_folder,
        Path("data/processed/sft_validation.json"),
    )


if __name__ == "__main__":
    main()
