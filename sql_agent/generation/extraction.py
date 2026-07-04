"""SQL extraction from raw model generations."""

from __future__ import annotations

import re

__all__ = ["extract_sql_from_generation", "strip_code_fences"]

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

# Markers that signal the end of valid SQL.
CORRUPTION_MARKERS = ("<|", "<quote", "```", "\n\n", "\nSELECT", "\nWITH")

_ASSISTANT_PREFIXES = ("<|im_start|>assistant\n", "assistant\n")


def strip_code_fences(text: str) -> str:
    """Removes markdown code fences if present.

    Args:
        text: Decoded model output.

    Returns:
        Cleaned text without surrounding ``` fences.
    """
    t = text.strip()
    if "```" not in t:
        return t
    t = t.replace("```sql", "```")
    parts = t.split("```")
    blocks = [p.strip() for p in parts if p.strip()]
    if not blocks:
        return text.strip()
    return max(blocks, key=len)


def extract_sql_from_generation(
    generated_text: str,
    *,
    im_end_token: str = "<|im_end|>",
) -> str:
    """Extracts SQL from raw model-generated text.

    Applies multiple cleaning stages to isolate valid SQL:
    1. Strip code fences.
    2. Cut at ChatML end token.
    3. Strip echoed assistant prefix.
    4. Cut at first non-ASCII character.
    5. Cut at corruption markers.
    6. Cut at semicolon (take only the first statement).
    7. Validate that the result looks like SQL.

    Args:
        generated_text: Raw decoded text from the model.
        im_end_token: End-of-turn token to split on.

    Returns:
        Cleaned SQL string, or an empty string if no valid SQL found.
    """
    text = strip_code_fences(generated_text)

    if im_end_token in text:
        text = text.split(im_end_token, 1)[0]

    text = text.strip()

    for marker in _ASSISTANT_PREFIXES:
        if text.startswith(marker):
            text = text[len(marker):].strip()

    non_ascii_match = _NON_ASCII_RE.search(text)
    if non_ascii_match:
        text = text[: non_ascii_match.start()].strip()

    cut_positions = [text.find(m) for m in CORRUPTION_MARKERS if m in text]
    if cut_positions:
        text = text[: min(cut_positions)].strip()

    if ";" in text:
        text = text.split(";", 1)[0].strip()

    # Validate: must start with SELECT or WITH.
    lowered = text.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return ""

    return text.strip()
