"""Centralized logging configuration for all pipeline scripts."""

from __future__ import annotations

import logging
from pathlib import Path  


def setup_logging(run_dir: Path, log_filename: str) -> None:
    """Configuress logging to console and file.

    Args:
        run_dir: Output directory where the log file will be written.
        log_filename: Name of the log file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate log messages.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler.
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
