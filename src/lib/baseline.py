"""Baseline CSV loader, baseline-status classification and the verified example pool.

Baseline CSV columns: original_index, question, choices (JSON list), prompt,
groundtruth, baseline_answer, correct, baseline_status, additional_fields (JSON dict).
``correct`` is exactly ``baseline_status == "correct"``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.lib.constants import EXTENDED_OPTION_LETTERS

BASELINE_STATUS_CORRECT = "correct"
BASELINE_STATUS_INCORRECT = "incorrect"
BASELINE_STATUS_INCONSISTENT = "inconsistent"
BASELINE_STATUS_UNANSWERED = "unanswered"
BASELINE_STATUSES = (
    BASELINE_STATUS_CORRECT,
    BASELINE_STATUS_INCORRECT,
    BASELINE_STATUS_INCONSISTENT,
    BASELINE_STATUS_UNANSWERED,
)


def threshold_vote(answers: list[str], threshold: int) -> tuple[str, int, float]:
    """Return ``(accepted_letter, top_votes, self_consistency)`` over parsed answers.

    Empty strings are ignored when tallying but count toward the ``self_consistency``
    denominator; ties break lexicographically; ``accepted_letter`` is ``""`` unless
    the top count reaches ``threshold``.
    """
    if not answers:
        return "", 0, 0.0
    counts = Counter(a for a in answers if a)
    if not counts:
        return "", 0, 0.0
    top = max(counts.values())
    winner = min(letter for letter, c in counts.items() if c == top)
    accepted = winner if top >= threshold else ""
    return accepted, top, top / len(answers)


def baseline_status_from_vote(
    accepted: str, answers: list[str], groundtruth: str
) -> str:
    """Map a :func:`threshold_vote` result to one of :data:`BASELINE_STATUSES`.

    ``accepted`` is the vote's accepted letter (``""`` when no option reached the
    threshold): correct / incorrect when accepted, else inconsistent when any
    sample parsed, else unanswered.
    """
    if accepted:
        return (
            BASELINE_STATUS_CORRECT
            if accepted == str(groundtruth).upper()
            else BASELINE_STATUS_INCORRECT
        )
    if any(a for a in answers):
        return BASELINE_STATUS_INCONSISTENT
    return BASELINE_STATUS_UNANSWERED


def encode_additional_fields(fields: dict | None) -> str:
    """Serialize an ``additional_fields`` dict as compact JSON; empty → ``""``."""
    if not fields:
        return ""
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def decode_additional_fields(value) -> dict:
    """Inverse of :func:`encode_additional_fields` (NaN / "" → ``{}``)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return {}


def collect_choice_widths(*dfs: pd.DataFrame) -> set[int]:
    """Distinct lengths of the non-empty ``choices`` lists across the frames.

    ``None`` / empty frames, frames without ``choices`` and empty cells are ignored.
    Raises ``ValueError`` on an undecoded (JSON-string) ``choices`` column.
    """
    widths: set[int] = set()
    for df in dfs:
        if df is None or len(df) == 0 or "choices" not in df.columns:
            continue
        for c in df["choices"]:
            if isinstance(c, str):
                raise ValueError(
                    "collect_choice_widths: 'choices' contains strings — the "
                    "column is not JSON-decoded. Load the CSV via load_baseline "
                    "/ load_sample_questions."
                )
            if c is not None and len(c) > 0:
                widths.add(len(c))
    return widths


def letters_for_width(n: int) -> list[str]:
    """The first ``n`` answer letters; ``ValueError`` when ``n`` is not in 1..len(EXTENDED_OPTION_LETTERS)."""
    if n <= 0:
        raise ValueError(f"letters_for_width: invalid option count {n}.")
    if n > len(EXTENDED_OPTION_LETTERS):
        raise ValueError(
            f"letters_for_width: {n} options per question exceeds the "
            f"supported letter set ({len(EXTENDED_OPTION_LETTERS)} letters)."
        )
    return list(EXTENDED_OPTION_LETTERS[:n])


def load_baseline(csv_path: str | Path) -> pd.DataFrame:
    """Load the baseline CSV, JSON-decoding ``choices`` and ``additional_fields``."""
    df = pd.read_csv(csv_path)
    if "choices" in df.columns:
        df["choices"] = df["choices"].apply(
            lambda c: json.loads(c) if isinstance(c, str) and c else []
        )
    if "additional_fields" not in df.columns:
        raise ValueError(
            f"{csv_path} has no 'additional_fields' column — every baseline CSV "
            "must carry it; regenerate the file with compute_baseline.py."
        )
    df["additional_fields"] = df["additional_fields"].apply(decode_additional_fields)
    return df


def build_verified_pool(
    baseline_df: pd.DataFrame,
    *,
    excluded_questions: set[str] | None = None,
) -> pd.DataFrame:
    """Correct baseline rows minus ``excluded_questions`` (so examples never leak
    the evaluated question), index reset."""
    df = baseline_df[baseline_df["correct"].astype(bool)].copy()
    if excluded_questions:
        df = df[~df["question"].isin(excluded_questions)]
    return df.reset_index(drop=True)


def load_sample_questions(csv_path: str | Path) -> pd.DataFrame:
    """Load the fixed example pool for the pool-dependent hints.

    Requires ``question``, ``choices`` (JSON list) and ``groundtruth`` columns.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in ("question", "choices", "groundtruth") if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing sample-question column(s) {missing}."
        )
    df["choices"] = df["choices"].apply(
        lambda c: json.loads(c) if isinstance(c, str) and c else []
    )
    return df


def load_exclusion_list(path: str | Path | None) -> set[int]:
    """Load a JSON array of ``original_index`` ints; ``None`` or a missing file → empty set."""
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    with open(p) as f:
        return {int(i) for i in json.load(f)}


def resolve_thinking(meta_path: Path) -> bool:
    """Whether a run had a reasoning block, from its sidecar's ``thinking`` (default on)."""
    if not meta_path.exists():
        return True
    with open(meta_path) as f:
        value = json.load(f).get("thinking")
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"off", "false", "0", "no"}
    return bool(value)


def resolve_option_letters(meta_path: Path) -> list[str] | None:
    """The run-wide answer-letter set the baseline was parsed with (the loader's static
    ``option_letters``), or None when the sidecar / dataset is unknown or the loader has none."""
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        dataset = json.load(f).get("dataset")
    name = dataset.get("name") if isinstance(dataset, dict) else dataset
    if not name:
        return None
    from src.lib.dataset import DATASET_REGISTRY

    loader = DATASET_REGISTRY.get(str(name).strip().lower())
    letters = getattr(loader, "option_letters", None) if loader else None
    return list(letters) if letters else None


def _row_choice_list(row: dict) -> list | None:
    raw = row.get("choices")
    if not raw:
        return None
    try:
        choices = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(choices, list) or not 1 <= len(choices) <= len(EXTENDED_OPTION_LETTERS):
        return None
    return choices


def row_choices(row: dict) -> list[str] | None:
    """The row's option texts from its ``choices`` JSON list (None when unusable)."""
    choices = _row_choice_list(row)
    return None if choices is None else [str(c) for c in choices]


def row_option_letters(row: dict) -> list[str] | None:
    """The answer-letter set of one row from its own ``choices`` width (None when unusable)."""
    choices = _row_choice_list(row)
    return None if choices is None else list(EXTENDED_OPTION_LETTERS[: len(choices)])
