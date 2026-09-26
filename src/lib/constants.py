"""Shared domain constants for answer letters, cases, and data splits."""

from __future__ import annotations

# Canonical 4-option letters.
OPTION_LETTERS: list[str] = ["A", "B", "C", "D"]

# Letters for datasets with up to 10 options (e.g. MMLU-Pro); a question with
# fewer choices uses a prefix, and OPTION_LETTERS is exactly its first four.
EXTENDED_OPTION_LETTERS: list[str] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

CASES = ("positive", "negative")
SPLITS = ("train", "val", "test")
