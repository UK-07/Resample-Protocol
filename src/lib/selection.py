"""Named rollout predicates: the single selection path over the manifest.

Every set of rollouts extracted, trained on or evaluated is a named entry of
:data:`REGISTRY` mapping the per-rollout manifest (optionally with the
re-sampling columns of ``resample_manifest.parquet``) to a boolean mask through
column predicates only. ``case`` is never a training filter: the only case
filters are the ``<dataset>_positive_only`` entries (the robustness ablation)
and the explicit ``cases=`` argument of :func:`mask` / :func:`select` /
:func:`select_rows` / :func:`labels_for`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.lib.rollout_manifest import CASES, SPLITS

HINTED_ONCE = "hinted_once"
RESAMPLE_K4 = "resample_k4"
PROVENANCES = (HINTED_ONCE, RESAMPLE_K4)
PROVENANCE_COLUMN = "provenance"


def resample_provenance(k: int) -> str:
    """The provenance value of a re-roll from a k-fold re-sampling stage."""
    return f"resample_k{int(k)}"


BINARY_LABELS = (0, 1)
INCOHERENT_LABEL = -1

# Label column → encoding to the metrics convention (0 = the detection target);
# ``None`` means the column already holds 0/1.
LABEL_ENCODINGS: dict[str, dict | None] = {
    "judge_label_final": None,
    "contrast_a": {"used": 0, "ignored": 1},
    "label_verbalised_rest": {"rest": 0, "verbalised": 1},
    "label_used_ignored": {"used": 0, "ignored": 1},
}

CASE_FILTERS = ("both",) + tuple(CASES)

Mask = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Predicate:
    """One named selection: a boolean mask over the manifest, by column predicates only."""

    name: str
    description: str
    requires: tuple[str, ...]
    fn: Mask
    label_col: str | None
    filters_case: bool = False
    base: str | None = None


REGISTRY: dict[str, Predicate] = {}


def register(name: str, description: str, requires: tuple[str, ...], *, label_col: str | None,
             filters_case: bool = False, base: str | None = None) -> Callable[[Mask], Mask]:
    def deco(fn: Mask) -> Mask:
        if name in REGISTRY:
            raise ValueError(f"predicate {name!r} registered twice")
        REGISTRY[name] = Predicate(name, description, tuple(requires), fn, label_col, filters_case, base)
        return fn
    return deco


# Column helpers


def with_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with a ``provenance`` column: kept when present, else mapped from
    ``is_resample``, else every row is ``hinted_once``."""
    if PROVENANCE_COLUMN in df.columns:
        prov = df[PROVENANCE_COLUMN]
        bad = sorted(set(prov.dropna().astype(str)) - set(PROVENANCES))
        if bad:
            raise ValueError(f"provenance holds value(s) outside {PROVENANCES}: {bad}")
        return df
    if "is_resample" in df.columns:
        flag = df["is_resample"].fillna(False).astype(bool)
        prov = pd.Series(flag.map({True: RESAMPLE_K4, False: HINTED_ONCE}), index=df.index)
    else:
        prov = pd.Series(HINTED_ONCE, index=df.index)
    return df.assign(**{PROVENANCE_COLUMN: prov.astype("string")})


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool)


def _is(df: pd.DataFrame, col: str, value) -> pd.Series:
    return (df[col].astype(object) == value).fillna(False).astype(bool)


def _isin(df: pd.DataFrame, col: str, values) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").isin(list(values)).fillna(False).astype(bool)


def _clean(df: pd.DataFrame) -> pd.Series:
    return df["exclude_reason"].isna()


def _judged_binary(df: pd.DataFrame) -> pd.Series:
    return _isin(df, "judge_label_final", BINARY_LABELS)


# The named datasets

_CLEAN = ("exclude_reason",)
_RESAMPLE = (PROVENANCE_COLUMN, "reliance_label")


@register(
    "dataset_A",
    "contrast A (hint used vs ignored): k=4 re-rolls in contrast_a_set == train — used rows of "
    "robust_used questions, ignored rows of robust_ignored questions — with no exclude_reason",
    _RESAMPLE + ("contrast_a", "contrast_a_set") + _CLEAN, label_col="contrast_a",
)
def dataset_A(df: pd.DataFrame) -> pd.Series:
    return _is(df, PROVENANCE_COLUMN, RESAMPLE_K4) & _is(df, "contrast_a_set", "train") & _clean(df)


@register(
    "dataset_A_challenge",
    "contrast A challenge set (evaluation only): k=4 re-rolls in contrast_a_set == challenge — "
    "weak_used / mixed questions and the off-side rows of robust questions — with no exclude_reason",
    _RESAMPLE + ("contrast_a", "contrast_a_set") + _CLEAN, label_col="contrast_a",
)
def dataset_A_challenge(df: pd.DataFrame) -> pd.Series:
    return _is(df, PROVENANCE_COLUMN, RESAMPLE_K4) & _is(df, "contrast_a_set", "challenge") & _clean(df)


@register(
    "dataset_B",
    "contrast B (hint used and verbalised vs not): k=4 re-rolls of robust_used questions that "
    "switched to the target, judged 0/1, with no exclude_reason",
    _RESAMPLE + ("to_hint", "judge_label_final") + _CLEAN, label_col="judge_label_final",
)
def dataset_B(df: pd.DataFrame) -> pd.Series:
    return (
        _is(df, PROVENANCE_COLUMN, RESAMPLE_K4)
        & _is(df, "reliance_label", "robust_used")
        & _bool(df, "to_hint")
        & _judged_binary(df)
        & _clean(df)
    )


@register(
    "dataset_C",
    "hinted-once verbalisation set: the original single rollouts that switched to the target, "
    "judged 0/1, with no exclude_reason",
    (PROVENANCE_COLUMN, "to_hint", "judge_label_final") + _CLEAN, label_col="judge_label_final",
)
def dataset_C(df: pd.DataFrame) -> pd.Series:
    return _is(df, PROVENANCE_COLUMN, HINTED_ONCE) & _bool(df, "to_hint") & _judged_binary(df) & _clean(df)


@register(
    "dataset_C_incoherent",
    "exploratory, no label: the original single rollouts that switched to the target and got the "
    "judge's -1 (conclusion contradicts the tag) — exclude_reason == incoherent, so no earlier reason "
    "applies but noise flips are included",
    (PROVENANCE_COLUMN, "to_hint", "judge_label_final") + _CLEAN, label_col=None,
)
def dataset_C_incoherent(df: pd.DataFrame) -> pd.Series:
    return (
        _is(df, PROVENANCE_COLUMN, HINTED_ONCE)
        & _bool(df, "to_hint")
        & _isin(df, "judge_label_final", (INCOHERENT_LABEL,))
        & _is(df, "exclude_reason", "incoherent")
    )


# The label configurations (re-rolls only; label 0 = the detection target)

_dataset_B = REGISTRY["dataset_B"]
register(
    "verbalised_vs_unverbalised",
    "label configuration (a), hint used and verbalised vs unverbalised: k=4 re-rolls of robust_used "
    "questions that switched to the target (used rows), judged 0/1, with no exclude_reason — "
    "identical to dataset_B (its mask is reused, so the two cannot drift)",
    _dataset_B.requires, label_col=_dataset_B.label_col, base=_dataset_B.name,
)(_dataset_B.fn)


@register(
    "verbalised_vs_rest",
    "label configuration (b), verbalised vs rest: used rows (to-target re-rolls of robust_used questions) "
    "judged 0/1 plus ignored rows (modal-answer re-rolls of robust_ignored questions), with no "
    "exclude_reason; label_verbalised_rest — verbalised = used & 1, rest = used & 0 or ignored",
    (PROVENANCE_COLUMN, "label_verbalised_rest") + _CLEAN, label_col="label_verbalised_rest",
)
def verbalised_vs_rest(df: pd.DataFrame) -> pd.Series:
    return _is(df, PROVENANCE_COLUMN, RESAMPLE_K4) & df["label_verbalised_rest"].notna() & _clean(df)


@register(
    "used_vs_ignored",
    "label configuration (c), hint used vs ignored: used rows (to-target re-rolls of robust_used "
    "questions; the label ignores the verdict, unjudged rows included) plus ignored rows (modal-answer "
    "re-rolls of robust_ignored questions), with no exclude_reason — so a used row judged -1 is out "
    "through its exclude_reason == incoherent, not through the label; label_used_ignored — used = 0, ignored = 1",
    (PROVENANCE_COLUMN, "label_used_ignored") + _CLEAN, label_col="label_used_ignored",
)
def used_vs_ignored(df: pd.DataFrame) -> pd.Series:
    return _is(df, PROVENANCE_COLUMN, RESAMPLE_K4) & df["label_used_ignored"].notna() & _clean(df)


LABEL_CONFIGS = ("verbalised_vs_unverbalised", "verbalised_vs_rest", "used_vs_ignored")

TRAINING_DATASETS = ("dataset_A", "dataset_B", "dataset_C")


def _case_restricted(base_name: str, case: str) -> None:
    """Register ``<base>_<case>_only``: the base predicate ∧ ``case == <case>`` (robustness ablation)."""
    base = REGISTRY[base_name]
    name = f"{base_name}_{case}_only"

    @register(name, f"{base.description}; restricted to case == {case} (robustness ablation)",
              base.requires + ("case",), label_col=base.label_col, filters_case=True, base=base_name)
    def restricted(df: pd.DataFrame, _base=base, _case=case) -> pd.Series:
        return _base.fn(df) & _is(df, "case", _case)


for _name in TRAINING_DATASETS:
    _case_restricted(_name, "positive")


# Resolution


def get(name: str) -> Predicate:
    if name not in REGISTRY:
        raise KeyError(f"unknown predicate {name!r}; known: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def _check_cases(cases: str) -> str:
    if cases not in CASE_FILTERS:
        raise ValueError(f"unknown cases filter {cases!r}; choose from {CASE_FILTERS}")
    return cases


def mask(name: str, df: pd.DataFrame, *, split: str | None = None, cases: str = "both") -> pd.Series:
    """The boolean mask of predicate ``name`` over ``df`` (index-aligned).

    ``split`` restricts to that split and requires it assigned on every selected
    row; ``cases`` (one of :data:`CASE_FILTERS`) restricts to that case. Missing
    required columns are an error, never an empty selection.
    """
    pred = get(name)
    cases = _check_cases(cases)
    df = with_provenance(df)
    missing = [c for c in pred.requires if c not in df.columns]
    if "rollout_id" not in df.columns:
        missing.insert(0, "rollout_id")
    if cases != "both" and "case" not in df.columns:
        missing.append("case")
    if missing:
        raise ValueError(f"predicate {name!r} needs manifest column(s) {missing} — "
                         f"is this the right manifest (the re-roll datasets need resample_manifest.parquet)?")
    out = pred.fn(df).fillna(False).astype(bool)
    if cases != "both":
        out &= _is(df, "case", cases)
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
        if "split" not in df.columns or df.loc[out, "split"].isna().any():
            raise ValueError(f"split is not assigned on every row predicate {name!r} selects — "
                             "run the split stage (src.scripts.assign_splits) first")
        out &= _is(df, "split", split)
    out.name = name
    return out


def select(name: str, df: pd.DataFrame, *, split: str | None = None, cases: str = "both") -> list[str]:
    """The sorted ``rollout_id``\\s predicate ``name`` selects."""
    ids = df.loc[mask(name, df, split=split, cases=cases), "rollout_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"duplicate rollout_id(s) in the manifest, e.g. {ids[ids.duplicated()].iloc[0]!r}")
    return sorted(ids)


def select_rows(name: str, df: pd.DataFrame, *, split: str | None = None, cases: str = "both") -> pd.DataFrame:
    """The manifest rows predicate ``name`` selects (with ``provenance`` filled)."""
    df = with_provenance(df)
    return df[mask(name, df, split=split, cases=cases)].copy()


def encode_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    """``label_col`` as the metrics' 0/1 ints (0 = the detection target), via :data:`LABEL_ENCODINGS`."""
    if label_col not in LABEL_ENCODINGS:
        raise ValueError(f"no label encoding for {label_col!r}; known: {list(LABEL_ENCODINGS)}")
    encoding = LABEL_ENCODINGS[label_col]
    raw = df[label_col]
    values = pd.to_numeric(raw, errors="coerce") if encoding is None else raw.astype(object).map(encoding)
    if values.isna().any():
        bad = raw[values.isna()].astype(object).unique()[:5].tolist()
        raise ValueError(f"{label_col} holds unencodable value(s) {bad}")
    if not values.isin(BINARY_LABELS).all():
        raise ValueError(f"{label_col} must encode to {BINARY_LABELS}")
    return values.astype(int)


def labels_for(name: str, df: pd.DataFrame, *, split: str | None = None, cases: str = "both") -> pd.DataFrame:
    """``rollout_id`` + encoded ``label`` for every row predicate ``name`` selects."""
    pred = get(name)
    if pred.label_col is None:
        raise ValueError(f"predicate {name!r} carries no label column")
    rows = select_rows(name, df, split=split, cases=cases)
    return pd.DataFrame({"rollout_id": rows["rollout_id"].astype(str).to_numpy(),
                         "label": encode_labels(rows, pred.label_col).to_numpy()})


__all__ = [
    "BINARY_LABELS", "CASES", "CASE_FILTERS", "HINTED_ONCE", "INCOHERENT_LABEL", "LABEL_CONFIGS", "LABEL_ENCODINGS",
    "PROVENANCES", "PROVENANCE_COLUMN", "REGISTRY", "RESAMPLE_K4", "TRAINING_DATASETS", "Predicate",
    "encode_labels", "get", "labels_for", "mask", "register", "resample_provenance", "select",
    "select_rows", "with_provenance",
]
