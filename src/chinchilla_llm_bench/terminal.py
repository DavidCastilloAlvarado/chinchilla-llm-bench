"""Terminal-safe rendering helpers."""
from __future__ import annotations

import unicodedata


def safe_terminal_text(value: object) -> str:
    """Remove control and formatting characters from terminal-bound text."""
    return "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in str(value)
    )
