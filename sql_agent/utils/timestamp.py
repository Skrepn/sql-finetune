"""Timestamp helpers for generating filesystem-safe run identifiers."""

from __future__ import annotations

import datetime as dt


def timestamp() -> str:
    """Returns a filesystem-safe timestamp string.

    Format: ``YYYYMMDD_HHMMSS``

    Returns:
        Timestamp string safe for use in directory and file names.
    """
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")
