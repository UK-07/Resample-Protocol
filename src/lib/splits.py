"""Question-level split assignment: every ``question_id`` gets exactly one of train / val / test.

Seeded and stratified on (``hint_style`` × ``case``): questions with the same signature (the set
of cells their rows occupy) are ranked by a per-question hash and cut at the split fractions,
with a per-signature offset so small groups are unbiased in expectation. The stored column is
the source of truth: a rebuild carries it over by question and new questions are added with
``extend``. Nothing is dropped here; selection lives in :mod:`src.lib.selection`.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from src.lib.rollout_manifest import SPLITS

SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
STRATA_KEYS = ["hint_style", "case"]
SPLIT_METHOD = "signature_stratified_hash_v1"


def _unit_hash(*parts) -> float:
    """A uniform value in [0, 1) from a sha256 of the joined parts."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def question_hash(seed: int, question_id: str) -> float:
    return _unit_hash("question", int(seed), question_id)


def _check_fractions(fractions: dict[str, float]) -> list[str]:
    names = list(fractions)
    if names != list(SPLITS):
        raise ValueError(f"fractions must be keyed {SPLITS} in order, got {names}")
    if any(f < 0 for f in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must be non-negative and sum to 1, got {fractions}")
    return names


def _cut(v: float, fractions: dict[str, float]) -> str:
    edge = 0.0
    for name, frac in fractions.items():
        edge += frac
        if v < edge:
            return name
    return list(fractions)[-1]


def question_signatures(df: pd.DataFrame) -> pd.Series:
    """Per ``question_id``, the sorted tuple of (hint_style, case) cells its rows occupy."""
    require_columns(df, ["question_id"] + STRATA_KEYS)
    cells = df[["question_id"] + STRATA_KEYS].astype(str).drop_duplicates()
    sig: dict[str, list[tuple[str, str]]] = {}
    for qid, hint, case in cells.itertuples(index=False):
        sig.setdefault(qid, []).append((hint, case))
    return pd.Series({q: tuple(sorted(v)) for q, v in sig.items()}, dtype="object").sort_index()


def assign_question_splits(df: pd.DataFrame, seed: int, *, fractions: dict[str, float] | None = None) -> pd.Series:
    """The split of every ``question_id`` in ``df`` (index = question_id), by the module's design."""
    fractions = dict(SPLIT_FRACTIONS if fractions is None else fractions)
    _check_fractions(fractions)
    groups: dict[tuple, list[str]] = {}
    for qid, signature in question_signatures(df).items():
        groups.setdefault(signature, []).append(qid)
    out: dict[str, str] = {}
    for signature in sorted(groups):
        qids = sorted(groups[signature], key=lambda q: (question_hash(seed, q), q))
        offset = _unit_hash("signature", int(seed), *[f"{h}|{c}" for h, c in signature])
        n = len(qids)
        for i, qid in enumerate(qids):
            out[qid] = _cut((i + offset) / n, fractions)
    return pd.Series(out, name="split", dtype="string").sort_index()


def assign_splits(df: pd.DataFrame, seed: int, *, fractions: dict[str, float] | None = None,
                  force: bool = False, extend: bool = False) -> pd.DataFrame:
    """``df`` with ``split`` populated (a copy). Refuses an already assigned frame unless ``force``
    (re-assign everything) or ``extend`` (keep assigned questions, draw only the unassigned ones,
    stratified among themselves).
    """
    require_columns(df, ["question_id"] + STRATA_KEYS)
    has_split = "split" in df.columns and df["split"].notna().any()
    if has_split and not (force or extend):
        raise ValueError("split is already assigned on some rows; pass force=True to re-assign (never after "
                         "activations were extracted under the stored split) or extend=True to assign only "
                         "the unassigned questions")
    out = df.copy()
    if extend and has_split:
        assigned = out.loc[out["split"].notna(), ["question_id", "split"]].astype(str).drop_duplicates()
        _check_one_split_per_question(assigned)
        # Rows of an already assigned question inherit its split; only unassigned questions are drawn.
        current = out["question_id"].astype(str).map(assigned.set_index("question_id")["split"])
        todo = out[current.isna()]
        if len(todo):
            by_question = assign_question_splits(todo, seed, fractions=fractions)
            current = current.where(current.notna(), out["question_id"].astype(str).map(by_question))
        out["split"] = pd.array(current.to_numpy(), dtype="string")
        return out
    by_question = assign_question_splits(out, seed, fractions=fractions)
    out["split"] = pd.array(out["question_id"].astype(str).map(by_question).to_numpy(), dtype="string")
    return out


def carry_over_splits(new: pd.DataFrame, previous: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """``new`` with ``split`` copied from ``previous`` by ``question_id``, plus a report.

    Questions ``previous`` never assigned stay null; a ``previous`` without any split leaves
    ``new`` untouched.
    """
    require_columns(new, ["question_id"])
    if "split" not in previous.columns or previous["split"].isna().all():
        return new, {"n_carried": 0, "n_questions_carried": 0,
                     "n_unassigned": int(len(new)), "previous_had_split": False}
    assigned = previous.loc[previous["split"].notna(), ["question_id", "split"]].astype(str).drop_duplicates()
    _check_one_split_per_question(assigned)
    mapping = assigned.set_index("question_id")["split"]
    split = new["question_id"].astype(str).map(mapping)
    out = new.copy()
    out["split"] = pd.array(split.to_numpy(), dtype="string")
    return out, {"n_carried": int(split.notna().sum()),
                 "n_questions_carried": int(new.loc[split.notna(), "question_id"].nunique()),
                 "n_unassigned": int(split.isna().sum()), "previous_had_split": True}


def _check_one_split_per_question(df: pd.DataFrame) -> None:
    per_question = df[["question_id", "split"]].astype(str).drop_duplicates()
    dupes = per_question["question_id"][per_question["question_id"].duplicated()]
    if len(dupes):
        raise ValueError(f"question(s) straddle splits, e.g. {dupes.iloc[0]!r}")


def propagate_splits(target: pd.DataFrame, source: pd.DataFrame, *, allow_missing: bool = False) -> pd.DataFrame:
    """``target`` with ``split`` copied from ``source`` by ``question_id``.

    A target question the source never assigned is an error unless ``allow_missing`` (its split
    stays null).
    """
    require_columns(target, ["question_id"])
    require_columns(source, ["question_id", "split"])
    assert_assigned(source)
    lookup = source[["question_id", "split"]].astype(str).drop_duplicates()
    conflicts = lookup[lookup["question_id"].duplicated(keep=False)]
    if len(conflicts):
        raise ValueError(f"source assigns more than one split to question(s), e.g. {conflicts['question_id'].iloc[0]!r}")
    mapping = lookup.set_index("question_id")["split"]
    split = target["question_id"].astype(str).map(mapping)
    if split.isna().any() and not allow_missing:
        missing = target.loc[split.isna(), "question_id"].astype(str).unique()
        raise ValueError(f"{len(missing)} target question(s) have no split in the source, e.g. {missing[0]!r}")
    out = target.copy()
    out["split"] = pd.array(split.to_numpy(), dtype="string")
    return out


def assert_assigned(df: pd.DataFrame) -> None:
    """Raise unless every row carries a valid split and no question straddles splits."""
    if "split" not in df.columns:
        raise ValueError("manifest has no split column — run the split stage (src.scripts.assign_splits)")
    null = int(df["split"].isna().sum())
    if null:
        raise ValueError(f"split is unassigned on {null} of {len(df)} rows — run the split stage first")
    bad = sorted(set(df["split"].astype(str)) - set(SPLITS))
    if bad:
        raise ValueError(f"split holds value(s) outside {SPLITS}: {bad}")
    _check_one_split_per_question(df)


def split_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per (hint_style, case) cell plus an ``all`` row: ``n_questions``, ``q_<split>``,
    ``frac_q_<split>``, ``n_rows``, ``rows_<split>``, ``frac_rows_<split>``.
    """
    require_columns(df, ["question_id", "split"] + STRATA_KEYS)
    work = df[["question_id", "split"] + STRATA_KEYS].astype({"question_id": str, "split": "object"})
    work["split"] = work["split"].where(work["split"].notna(), "unassigned")
    rows = []
    groups = [(("all", "all"), work)] + [
        (key, g) for key, g in work.groupby(STRATA_KEYS, sort=True)
    ]
    for (hint, case), g in groups:
        q = g.drop_duplicates("question_id")
        row = {"hint_style": hint, "case": case, "n_questions": int(len(q)), "n_rows": int(len(g))}
        for name in list(SPLITS) + ["unassigned"]:
            nq = int((q["split"] == name).sum())
            nr = int((g["split"] == name).sum())
            if name == "unassigned" and nq == 0:
                continue
            row[f"q_{name}"] = nq
            row[f"frac_q_{name}"] = nq / len(q) if len(q) else float("nan")
            row[f"rows_{name}"] = nr
            row[f"frac_rows_{name}"] = nr / len(g) if len(g) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def split_meta(df: pd.DataFrame, seed: int, *, fractions: dict[str, float] | None = None) -> dict:
    """The sidecar block recording how ``split`` was assigned."""
    fractions = dict(SPLIT_FRACTIONS if fractions is None else fractions)
    q = df[["question_id", "split"]].astype(str).drop_duplicates("question_id")
    return {
        "method": SPLIT_METHOD,
        "seed": int(seed),
        "fractions": fractions,
        "strata": list(STRATA_KEYS),
        "assigned_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_questions": int(len(q)),
        "questions_per_split": {s: int((q["split"] == s).sum()) for s in SPLITS},
    }


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``ValueError`` naming the columns of ``columns`` that ``df`` lacks."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing column(s) {missing}")


__all__ = [
    "SPLIT_FRACTIONS", "SPLIT_METHOD", "STRATA_KEYS", "assert_assigned", "assign_question_splits",
    "assign_splits", "carry_over_splits", "propagate_splits", "question_hash", "question_signatures", "require_columns",
    "split_meta", "split_report",
]
