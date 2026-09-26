"""Probe dataset specs, the balance policy, LOHO fold roles, the dataset fingerprint and the parquet I/O.

A probe dataset is a named predicate of :mod:`src.lib.selection` over the re-sampling manifest,
restricted by a :class:`DatasetSpec`, balanced, and written as one parquet per label configuration
(``<output_dir>/<name>.parquet`` + ``.meta.json`` + ``_report.json``). Rows are re-rolls only, every
``exclude_reason`` null, ``label`` 0 = the detection target. A fold holds out one hint style: other
styles follow their global ``split`` (train / val / test_id), the held-out style keeps its test rows
as test_ood. :func:`read_dataset` is the one reader every consumer goes through.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib import selection
from src.lib.baseline import letters_for_width
from src.lib.paths import REPO_ROOT, resolve_data_path
from src.lib.rollout_manifest import JUDGE_INPUT_COLUMNS, MANIFEST_COLUMNS, coerce_schema, meta_path
from src.lib.splits import assert_assigned, require_columns

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
BALANCE_METHODS = ("none", "ratio")
DEFAULT_BALANCE_GROUPS = ("subject_model", "run", "hint_style", "case")
DEFAULT_OUTPUT_DIR = "${DATA_ROOT}/probe_datasets"
DEFAULT_SEED = 42
FOLD_ROLES = ("train", "val", "test_id", "test_ood")
# The role of a row whose style is *not* held out, by its split; a held-out row keeps only ``test``.
_ID_ROLE = {"train": "train", "val": "val", "test": "test_id"}
_OOD_ROLE = {"test": "test_ood"}
LABELS = (0, 1)

# The restriction keys of a spec and the manifest column each filters.
RESTRICTIONS = (("subject_models", "subject_model"), ("runs", "run"), ("hint_styles", "hint_style"))
SPEC_KEYS = ("name", "manifest", "predicate", "subject_models", "runs", "hint_styles", "cases", "balance",
             "seed", "output_dir")
BALANCE_KEYS = ("method", "r", "groups")


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from the parts (same as ``activation_store.stable_seed``; copied so this
    module never imports torch)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BalanceSpec:
    """The class-balance policy: ``none``, or ``ratio`` with ``majority ≤ r × minority`` per group."""

    method: str = "none"
    r: float | None = None
    groups: tuple[str, ...] = DEFAULT_BALANCE_GROUPS

    def __post_init__(self):
        if self.method not in BALANCE_METHODS:
            raise ValueError(f"balance.method must be one of {BALANCE_METHODS}, got {self.method!r}")
        if self.method == "none":
            if self.r is not None:
                raise ValueError("balance.r must be null when balance.method is 'none'")
        else:
            if isinstance(self.r, bool) or not isinstance(self.r, (int, float)):
                raise ValueError("balance.r is required for balance.method 'ratio' (a number ≥ 1)")
            if not math.isfinite(self.r) or self.r < 1.0:
                raise ValueError(f"balance.r must be ≥ 1.0, got {self.r!r}")
            object.__setattr__(self, "r", float(self.r))
        groups = tuple(self.groups)
        if not groups or any(not isinstance(g, str) or not g for g in groups) or len(set(groups)) != len(groups):
            raise ValueError(f"balance.groups must be a non-empty list of distinct column names, got {self.groups!r}")
        object.__setattr__(self, "groups", groups)

    def to_dict(self) -> dict:
        return {"method": self.method, "r": self.r, "groups": list(self.groups)}


@dataclass(frozen=True)
class DatasetSpec:
    """One probe dataset: which rows of which manifest, under which label configuration.

    ``manifest`` and ``output_dir`` are kept as the raw strings (``${DATA_ROOT}``
    allowed) — resolution is the collector's job. ``subject_models`` / ``runs``
    / ``hint_styles`` null = every value present after selection.
    """

    name: str
    manifest: str
    predicate: str
    subject_models: tuple[str, ...] | None = None
    runs: tuple[str, ...] | None = None
    hint_styles: tuple[str, ...] | None = None
    cases: str = "both"
    balance: BalanceSpec = field(default_factory=BalanceSpec)
    seed: int = DEFAULT_SEED
    output_dir: str = DEFAULT_OUTPUT_DIR

    def __post_init__(self):
        if not isinstance(self.name, str) or not NAME_RE.match(self.name):
            raise ValueError(f"name must match {NAME_RE.pattern}, got {self.name!r}")
        for key in ("manifest", "output_dir"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty path string, got {value!r}")
        if not isinstance(self.predicate, str):
            raise ValueError(f"predicate must be a string, got {self.predicate!r}")
        try:
            pred = selection.get(self.predicate)
        except KeyError as e:
            raise ValueError(str(e)) from None
        if pred.label_col is None:
            raise ValueError(f"predicate {self.predicate!r} carries no label column and cannot be a probe dataset; "
                             f"choose one of {selection.LABEL_CONFIGS} (or any predicate with a label_col)")
        for key, _ in RESTRICTIONS:
            value = getattr(self, key)
            if value is None:
                continue
            if isinstance(value, str) or not all(isinstance(v, str) and v for v in value) or not len(value):
                raise ValueError(f"{key} must be null or a non-empty list of strings, got {value!r}")
            object.__setattr__(self, key, tuple(value))
        if self.cases not in selection.CASE_FILTERS:
            raise ValueError(f"cases must be one of {selection.CASE_FILTERS}, got {self.cases!r}")
        if not isinstance(self.balance, BalanceSpec):
            raise ValueError("balance must be a BalanceSpec")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")

    @property
    def label_col(self) -> str:
        return selection.get(self.predicate).label_col

    def to_dict(self) -> dict:
        return {
            "name": self.name, "manifest": self.manifest, "predicate": self.predicate,
            "subject_models": None if self.subject_models is None else list(self.subject_models),
            "runs": None if self.runs is None else list(self.runs),
            "hint_styles": None if self.hint_styles is None else list(self.hint_styles),
            "cases": self.cases, "balance": self.balance.to_dict(), "seed": self.seed, "output_dir": self.output_dir,
        }


def parse_balance(cfg) -> BalanceSpec:
    """``balance:`` as written in YAML: absent / ``none`` / ``{method: none}`` / ``{method: ratio, r: 2}``."""
    if cfg is None:
        return BalanceSpec()
    if isinstance(cfg, str):
        return BalanceSpec(method=cfg)
    if not isinstance(cfg, dict):
        raise ValueError(f"balance must be a string or a mapping, got {cfg!r}")
    unknown = sorted(set(cfg) - set(BALANCE_KEYS))
    if unknown:
        raise ValueError(f"unknown balance key(s) {unknown}; accepted: {list(BALANCE_KEYS)}")
    if "method" not in cfg:
        raise ValueError("balance mapping needs a 'method'")
    kwargs = {"method": cfg["method"], "r": cfg.get("r")}
    if cfg.get("groups") is not None:
        groups = cfg["groups"]
        if isinstance(groups, str) or not isinstance(groups, (list, tuple)):
            raise ValueError(f"balance.groups must be a list of column names, got {groups!r}")
        kwargs["groups"] = tuple(groups)
    return BalanceSpec(**kwargs)


def parse_spec(cfg: dict) -> DatasetSpec:
    """A :class:`DatasetSpec` from a config mapping; unknown keys, bad types and bad values raise."""
    if not isinstance(cfg, dict):
        raise ValueError(f"a dataset spec must be a mapping, got {type(cfg).__name__}")
    unknown = sorted(set(cfg) - set(SPEC_KEYS))
    if unknown:
        raise ValueError(f"unknown dataset spec key(s) {unknown}; accepted: {list(SPEC_KEYS)}")
    missing = [k for k in ("name", "manifest", "predicate") if cfg.get(k) is None]  # an explicit null too
    if missing:
        raise ValueError(f"dataset spec is missing required key(s) {missing}")
    # An explicit ``null`` means the default for every key (as ``balance: null`` does).
    kwargs = {k: cfg[k] for k in SPEC_KEYS if cfg.get(k) is not None and k != "balance"}
    kwargs["balance"] = parse_balance(cfg.get("balance"))
    return DatasetSpec(**kwargs)


# ---------------------------------------------------------------------------
# Row selection (through selection.select_rows) and restriction
# ---------------------------------------------------------------------------


def select_dataset_rows(manifest: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """The rows of ``spec`` — predicate (with the case filter), then the plain restrictions — with
    an Int8 ``label`` (0 = the detection target) and ``balance_kept = True`` on every row.

    Requires ``split`` assigned on every selected row and every row a re-roll.
    """
    label_col = spec.label_col
    selected = selection.select_rows(spec.predicate, manifest, cases=spec.cases)
    rows = selected
    for key, col in RESTRICTIONS:
        wanted = getattr(spec, key)
        if wanted is None:
            continue
        require_columns(selected, [col])
        available = sorted(selected[col].astype(str).unique())  # under the predicate, before any restriction
        unknown = sorted(set(wanted) - set(available))
        if unknown:
            raise ValueError(f"{key} names value(s) with no row under predicate {spec.predicate!r}: {unknown}; "
                             f"available: {available}")
        present = set(rows[col].astype(str).unique())  # under the earlier restrictions too
        absent = sorted(set(wanted) - present)
        if absent:
            earlier = [k for k, _ in RESTRICTIONS[: RESTRICTIONS.index((key, col))] if getattr(spec, k) is not None]
            raise ValueError(f"{key} names value(s) present under predicate {spec.predicate!r} but with no row "
                             f"once {earlier} restrict the selection: {absent} (a dataset that silently drops a "
                             f"requested {key} entry is a config error)")
        rows = rows[rows[col].astype(str).isin(wanted)]
    rows = rows.copy()
    if not len(rows):
        raise ValueError(f"dataset {spec.name!r}: predicate {spec.predicate!r} with its restrictions selects no row")
    try:
        assert_assigned(rows)
    except ValueError as e:
        raise ValueError(f"dataset {spec.name!r}: {e} — run the split stage (src.scripts.assign_splits) "
                         "on the manifest before building probe datasets") from None
    provenance = rows[selection.PROVENANCE_COLUMN].astype(str)
    originals = provenance.ne(selection.RESAMPLE_K4)
    if originals.any():
        raise ValueError(f"dataset {spec.name!r}: predicate {spec.predicate!r} admits {int(originals.sum())} "
                         f"row(s) that are not k=4 re-rolls (dataset rows are re-rolls only), e.g. "
                         f"{rows.loc[originals, 'rollout_id'].iloc[0]!r}")
    rows["label"] = pd.array(selection.encode_labels(rows, label_col).to_numpy(), dtype="Int8")
    rows["balance_kept"] = True
    return rows


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------


def _label_counts(g: pd.DataFrame) -> dict[str, int]:
    labels = g["label"].astype(int)
    return {str(lab): int((labels == lab).sum()) for lab in LABELS}


def apply_balance(rows: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, dict]:
    """``(kept rows, report)`` under ``spec.balance``, applied to every row regardless of ``split``.

    ``ratio``: per group, ``keep = min(n_maj, floor(r × n_min))`` majority rows are kept as a seeded
    permutation prefix over the group's rows sorted by ``rollout_id`` (nested in ``r``, row-order
    independent); the minority is never touched, a one-class group is kept whole and flagged.
    """
    bal = spec.balance
    require_columns(rows, ["rollout_id", "label"])
    if bal.method == "none":
        out = rows.copy()
        out["balance_kept"] = True
        return out, {"method": "none", "n_before": int(len(rows)), "n_after": int(len(rows)), "n_dropped": 0}
    require_columns(rows, list(bal.groups))
    if not rows.index.is_unique or rows["rollout_id"].duplicated().any():
        raise ValueError("apply_balance needs a unique index and unique rollout_id values (the draw is a "
                         "permutation over rows sorted by rollout_id, addressed by index)")
    order = rows.sort_values("rollout_id", kind="stable")
    kept = pd.Series(True, index=rows.index)
    groups_report: dict[str, dict] = {}
    n_missing = 0
    for key, g in order.groupby(list(bal.groups), sort=True, dropna=False):
        key = tuple(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        before = _label_counts(g)
        entry = {"group": dict(zip(bal.groups, key)), "before": before, "missing_class": False}
        if before["0"] == 0 or before["1"] == 0:
            entry["missing_class"] = True
            n_missing += 1
            entry["after"] = dict(before)
        else:
            maj_label = 0 if before["0"] > before["1"] else 1
            n_maj, n_min = max(before.values()), min(before.values())
            # +1e-9: r × n_min is exact in decimal (1.15 × 20 = 23) but not always in binary.
            keep = min(n_maj, int(math.floor(bal.r * n_min + 1e-9)))
            maj_idx = g.index[(g["label"].astype(int) == maj_label).to_numpy()]
            rng = np.random.default_rng(stable_seed(spec.seed, *key))
            perm = rng.permutation(len(maj_idx))
            kept.loc[maj_idx[perm[keep:]]] = False
            entry["after"] = _label_counts(g[kept.loc[g.index]])
            entry["majority_label"] = maj_label
            entry["n_dropped"] = int(n_maj - keep)
        groups_report["|".join(key)] = entry
    out = rows.copy()
    out["balance_kept"] = kept
    out = out[kept]
    report = {
        "method": bal.method, "r": bal.r, "groups": list(bal.groups), "seed": int(spec.seed),
        "n_before": int(len(rows)), "n_after": int(len(out)), "n_dropped": int((~kept).sum()),
        "labels_before": _label_counts(rows), "labels_after": _label_counts(out),
        "n_groups": len(groups_report), "n_groups_missing_class": n_missing, "per_group": groups_report,
    }
    return out, report


# ---------------------------------------------------------------------------
# Fold roles
# ---------------------------------------------------------------------------


def fold_roles(rows: pd.DataFrame, held_out_style: str) -> pd.Series:
    """The LOHO role of every row for the fold holding out ``held_out_style`` (index-aligned,
    string dtype, null = dropped): other styles follow their split (train / val / test_id), the
    held-out style keeps only its test rows (test_ood)."""
    require_columns(rows, ["split", "hint_style"])
    assert_assigned(rows)
    styles = rows["hint_style"].astype(str)
    if held_out_style not in set(styles):
        raise ValueError(f"held-out style {held_out_style!r} has no row; styles present: {sorted(set(styles))}")
    split = rows["split"].astype(str)
    held = styles == held_out_style
    role = split.map(_ID_ROLE).where(~held, split.map(_OOD_ROLE))
    return pd.Series(pd.array(role.to_numpy(dtype=object), dtype="string"), index=rows.index, name="fold_role")


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.astype(str).value_counts().sort_index().items()}


def _align_roles(rows: pd.DataFrame, roles: pd.Series) -> pd.Series:
    """``roles`` in ``rows``' index order — a role series is index-aligned, never positional."""
    if not rows.index.is_unique or not roles.index.is_unique:
        raise ValueError("rows and roles need unique indexes")
    if len(roles) != len(rows) or not roles.index.isin(rows.index).all():
        raise ValueError("roles is not aligned with rows (compute it with fold_roles on this frame)")
    return roles.reindex(rows.index)


def _role_mask(roles: pd.Series, role: str) -> pd.Series:
    return (roles == role).fillna(False).astype(bool)


def question_checks(rows: pd.DataFrame, roles: pd.Series) -> dict:
    """The question-level checks of one fold, from its ``roles`` (a :func:`fold_roles` series).

    ``question_consistency`` = ``train_val_vs_test_disjoint`` (no train/val question in a test_id/test_ood
    row) ∧ ``ood_questions_in_test_split`` (every test_ood row has ``split == "test"``).
    ``ood_questions_subset_of_id`` / ``n_ood_only_questions`` are informational only: the resample set is
    drawn per cell, so an OOD-only question is by design.
    """
    require_columns(rows, ["question_id", "split"])
    roles = _align_roles(rows, roles)
    questions = {role: set(rows.loc[_role_mask(roles, role), "question_id"].astype(str)) for role in FOLD_ROLES}
    trainval = questions["train"] | questions["val"]
    test = questions["test_id"] | questions["test_ood"]
    disjoint = not (trainval & test)
    ood_split = rows.loc[_role_mask(roles, "test_ood"), "split"].astype(str)
    ood_in_test = bool((ood_split == "test").all())
    return {
        "question_consistency": bool(disjoint and ood_in_test),
        "train_val_vs_test_disjoint": bool(disjoint),
        "ood_questions_in_test_split": ood_in_test,
        "ood_questions_subset_of_id": bool(questions["test_ood"] <= questions["test_id"]),
        "n_ood_only_questions": len(questions["test_ood"] - questions["test_id"]),
    }


def fold_report(rows: pd.DataFrame, hint_styles) -> dict:
    """Per held-out style → per role → counts (rows, questions, labels, per case, per subject model),
    the :func:`question_checks` and whether ``test_ood`` holds both labels."""
    require_columns(rows, ["question_id", "label", "case", "subject_model"])
    out: dict[str, dict] = {}
    for style in hint_styles:
        roles = fold_roles(rows, style)
        by_role: dict[str, dict] = {}
        for role in FOLD_ROLES:
            g = rows[_role_mask(roles, role)]
            by_role[role] = {
                "n_rows": int(len(g)), "n_questions": int(g["question_id"].astype(str).nunique()),
                "n_label_0": int((g["label"].astype(int) == 0).sum()), "n_label_1": int((g["label"].astype(int) == 1).sum()),
                "per_case": _counts(g["case"]), "per_subject_model": _counts(g["subject_model"]),
            }
        ood = by_role["test_ood"]
        out[str(style)] = {
            "roles": by_role,
            "n_dropped": int(roles.isna().sum()),
            **question_checks(rows, roles),
            "ood_has_both_labels": bool(ood["n_label_0"] > 0 and ood["n_label_1"] > 0),
        }
    return out


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def dataset_fingerprint(rows: pd.DataFrame) -> str:
    """sha256 over (``rollout_id``, ``label``, ``split``) of the rows sorted by ``rollout_id`` — the
    identity a collector or trainer checks against the parquet it reads."""
    require_columns(rows, ["rollout_id", "label", "split"])
    work = rows[["rollout_id", "label", "split"]].astype({"rollout_id": str}).sort_values("rollout_id", kind="stable")
    h = hashlib.sha256()
    for rid, label, split in work.itertuples(index=False):
        h.update(f"{rid}\x1f{int(label)}\x1f{split}\n".encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The dataset parquet contract and its paths
# ---------------------------------------------------------------------------

# The columns ``resample_relabel.py`` adds to the manifest (a copy, so this module never imports a
# script); the last two are the label columns.
RESAMPLE_EXTRA_COLUMNS = (
    "provenance", "is_resample", "sample_seed", "source_rollout_id", "role", "reliance_label",
    "contrast_a", "contrast_a_set", "paired_a", "contrast_b", "paired_b",
    "label_verbalised_rest", "label_used_ignored",
)
# The texts joined from the source CSVs (``choices`` / ``option_letters`` are JSON list strings).
TEXT_COLUMNS = ("prompt", "hinted_prompt", "reasoning", "rollout", "choices", "option_letters")
# The parquet's exact column order: manifest, resample extras, label, balance_kept, texts.
DATASET_COLUMNS = tuple(MANIFEST_COLUMNS) + RESAMPLE_EXTRA_COLUMNS + ("label", "balance_kept") + TEXT_COLUMNS
DATASET_SCHEMA_VERSION = 1
# A source CSV must carry the join keys and the four generation-side texts; ``choices`` /
# ``option_letters`` are optional there (the baseline fallback fills them).
SOURCE_KEY_COLUMNS = ("original_index", "hint_name")
SOURCE_TEXT_COLUMNS = ("prompt", "hinted_prompt", "reasoning", "rollout")
SOURCE_REQUIRED_COLUMNS = SOURCE_KEY_COLUMNS + SOURCE_TEXT_COLUMNS
assert set(JUDGE_INPUT_COLUMNS) <= set(SOURCE_REQUIRED_COLUMNS)
# Rows per CSV chunk of the text join (a 20 000-row chunk of a judged CSV is 0.25-1 GB of strings).
DEFAULT_CHUNK_ROWS = 20000
# Where a bare ``source_csv`` name and its recipe sidecar resolve, relative to the manifest's directory.
SOURCE_SUBDIR = "rollouts"
ITEMS_SUBDIR = "items"
_RESAMPLE_STEM_RE = re.compile(r"^(?P<stem>.+)_rs(?P<seed>\d+)(?P<judged>_judged)?$")


def dataset_paths(spec_or_name: DatasetSpec | str, output_dir: str | os.PathLike | None = None) -> tuple[Path, Path, Path]:
    """``(parquet, meta.json, report.json)`` of a dataset: ``<output_dir>/<name>.parquet`` and its sidecars.

    ``output_dir`` defaults to the spec's (``${DATA_ROOT}`` resolved through ``resolve_data_path``);
    a bare name needs it.
    """
    if isinstance(spec_or_name, DatasetSpec):
        name = spec_or_name.name
        output_dir = spec_or_name.output_dir if output_dir is None else output_dir
    else:
        name = str(spec_or_name)
        if output_dir is None:
            raise ValueError("dataset_paths needs output_dir when given a bare dataset name")
        if not NAME_RE.match(name):
            raise ValueError(f"dataset name must match {NAME_RE.pattern}, got {name!r}")
    parquet = resolve_data_path(output_dir) / f"{name}.parquet"
    return (parquet, *dataset_sidecars(parquet))


def dataset_sidecars(parquet: str | os.PathLike) -> tuple[Path, Path]:
    """``(meta.json, report.json)`` beside a dataset parquet — consumers derive them from the path they read."""
    parquet = Path(parquet)
    return meta_path(parquet), parquet.with_name(f"{parquet.stem}_report.json")


def resolve_source_csv(source_csv: str, source_dir: str | os.PathLike) -> Path:
    """The file a manifest ``source_csv`` names: an absolute value as is, a bare name under ``source_dir``."""
    if not isinstance(source_csv, str) or not source_csv.strip():
        raise ValueError(f"source_csv must be a non-empty string, got {source_csv!r}")
    path = Path(source_csv)
    return path if path.is_absolute() else Path(source_dir) / path


def baseline_csv_for(source_path: str | os.PathLike, *, items_dir: str | os.PathLike | None = None) -> Path | None:
    """The baseline CSV a re-roll source was generated from, from its recipe sidecars, else None.

    The items sidecar (``<items_dir>/<stem>_resample_items.meta.json``) is tried first, then the re-roll's
    own ``<name without _judged>.meta.json``; the first ``baseline_csv`` wins, returned resolved whether
    or not the file exists.
    """
    source_path = Path(source_path)
    items_dir = source_path.parent.parent / ITEMS_SUBDIR if items_dir is None else Path(items_dir)
    m = _RESAMPLE_STEM_RE.match(source_path.stem)
    candidates = []
    if m:
        candidates.append(Path(items_dir) / f"{m.group('stem')}_resample_items.meta.json")
    candidates.append(source_path.with_name(f"{source_path.stem.removesuffix('_judged')}.meta.json"))
    for sidecar in candidates:
        if not sidecar.exists():
            continue
        try:
            value = json.loads(sidecar.read_text()).get("baseline_csv")
        except (OSError, ValueError):
            continue
        if isinstance(value, str) and value.strip():
            try:
                return resolve_data_path(value)
            except ValueError:  # a relative value the resolver refuses — treat as "none named"
                continue
    return None


def _log(log, message: str) -> None:
    if log is not None:
        log(message)


def _first_ids(rows: pd.DataFrame, mask, n: int = 10) -> list[str]:
    return rows.loc[mask, "rollout_id"].astype(str).head(n).tolist()


def _option_letters_json(choices: str | None) -> str | None:
    """``["A", "B", ...]`` (JSON) for a JSON ``choices`` list, None when it is blank or unparseable."""
    if choices is None or (isinstance(choices, float) and math.isnan(choices)) or not str(choices).strip():
        return None
    try:
        parsed = json.loads(choices)
    except ValueError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    try:
        return json.dumps(letters_for_width(len(parsed)))
    except ValueError:
        return None


def _baseline_choices(baseline_csv: Path | None, log=None) -> pd.Series | None:
    """``choices`` (JSON string) by ``original_index`` from a baseline CSV; None when there is none to read."""
    if baseline_csv is None or not baseline_csv.exists():
        return None
    header = pd.read_csv(baseline_csv, nrows=0).columns
    if "original_index" not in header or "choices" not in header:
        _log(log, f"  baseline {baseline_csv.name}: no original_index/choices column — choices left null")
        return None
    base = pd.read_csv(baseline_csv, usecols=["original_index", "choices"], dtype=str, keep_default_na=False)
    base = base[base["original_index"].str.strip().ne("")].drop_duplicates("original_index")
    return pd.Series(base["choices"].to_numpy(), index=base["original_index"].astype(int).to_numpy())


def _check_source_seed(path: Path, sub: pd.DataFrame) -> None:
    """A re-roll row (non-null ``sample_seed``) must resolve to the per-seed CSV of *its* seed
    (``<stem>_rs<seed>[_judged].csv``) — the hinted-once CSV shares every (original_index, hint) key, so a
    wrong ``source_csv`` would join another generation's text without any other check firing."""
    if "sample_seed" not in sub.columns:
        return
    seeds = pd.to_numeric(sub["sample_seed"], errors="coerce").dropna().astype(int).unique()
    if not len(seeds):
        return
    m = _RESAMPLE_STEM_RE.match(path.stem)
    if m is None or int(m.group("seed")) not in seeds or len(seeds) != 1:
        raise ValueError(f"join_texts: {path.name} is not the per-seed re-roll CSV of the rows that name it "
                         f"(sample_seed {sorted(int(x) for x in seeds)}; expected a "
                         f"<stem>_rs<seed>[_judged].csv with that seed); e.g. {_first_ids(sub, slice(None))}")


def _source_key_index(original_index, hint_name) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([pd.Index(original_index).astype(int), pd.Index(hint_name).astype(str)])


def join_texts(rows: pd.DataFrame, *, source_dir: str | os.PathLike, chunk_rows: int = DEFAULT_CHUNK_ROWS,
               items_dir: str | os.PathLike | None = None, log=None, sources: list | None = None) -> pd.DataFrame:
    """``rows`` (order kept) with :data:`TEXT_COLUMNS` joined from the source CSVs.

    Rows are grouped by ``source_csv`` (resolved under ``source_dir``); each CSV is streamed ``chunk_rows``
    rows at a time and matched on ``(original_index, hint_name)`` = the row's ``(original_index,
    hint_style)``. Every dataset row must resolve to exactly one CSV row (a duplicate on either side, a
    missing row or a source without :data:`SOURCE_REQUIRED_COLUMNS` is an error). ``choices`` /
    ``option_letters`` fall back to the baseline CSV named by the source's recipe sidecar
    (:func:`baseline_csv_for`); a value that cannot be found is null. ``sources`` (a list) collects one
    entry per source CSV.
    """
    require_columns(rows, ["rollout_id", "source_csv", "original_index", "hint_style"])
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be ≥ 1, got {chunk_rows}")
    if not rows.index.is_unique or rows["rollout_id"].duplicated().any():
        raise ValueError("join_texts: rows need a unique index and unique rollout_id values")
    if rows["source_csv"].isna().any():
        raise ValueError(f"join_texts: rows without source_csv, e.g. {_first_ids(rows, rows['source_csv'].isna())}")
    texts = {col: pd.Series(pd.array([None] * len(rows), dtype=object), index=rows.index) for col in TEXT_COLUMNS}
    baseline_cache: dict[Path | None, pd.Series | None] = {}
    for source_csv, sub in rows.groupby(rows["source_csv"].astype(str), sort=True):
        path = resolve_source_csv(source_csv, source_dir)
        if not path.exists():
            raise FileNotFoundError(f"join_texts: source CSV {path} (source_csv {source_csv!r}) does not exist")
        _check_source_seed(path, sub)
        keys = _source_key_index(sub["original_index"], sub["hint_style"])
        if keys.duplicated().any():
            dup = keys[keys.duplicated()][0]
            raise ValueError(f"join_texts: {path.name}: several dataset rows share the key (original_index, hint) "
                             f"{dup} — a re-roll's source_csv must be its per-seed CSV; e.g. "
                             f"{_first_ids(sub, keys.duplicated())}")
        header = list(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in SOURCE_REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"join_texts: {path.name} lacks column(s) {missing} — its rows cannot be joined "
                             "(a source without prompt/reasoning was judged on degraded inputs)")
        csv_text_cols = [c for c in TEXT_COLUMNS if c in header]
        usecols = list(SOURCE_KEY_COLUMNS) + csv_text_cols
        _log(log, f"  {path.name}: joining {len(sub)} rows ({os.path.getsize(path) / 1e6:.0f} MB, chunks of {chunk_rows})")
        parts = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_rows, dtype=str, keep_default_na=False):
            idx = _source_key_index(chunk["original_index"], chunk["hint_name"])
            hit = idx.isin(keys)
            if hit.any():
                part = chunk.loc[hit, csv_text_cols].copy()
                part.index = idx[hit]
                parts.append(part)
        found = pd.concat(parts) if parts else pd.DataFrame(columns=csv_text_cols, index=keys[:0])
        if found.index.duplicated().any():
            dup = found.index[found.index.duplicated()][0]
            raise ValueError(f"join_texts: {path.name}: several CSV rows carry the key (original_index, hint) {dup}")
        present = keys.isin(found.index)
        if not present.all():
            n = int((~present).sum())
            raise ValueError(f"join_texts: {path.name}: {n} dataset row(s) have no CSV row with their "
                             f"(original_index, hint), e.g. {_first_ids(sub, ~present)}")
        aligned = found.reindex(keys)
        for col in csv_text_cols:
            texts[col].loc[sub.index] = aligned[col].to_numpy(dtype=object)
        baseline_csv = None
        if "choices" not in csv_text_cols:
            baseline_csv = baseline_csv_for(path, items_dir=items_dir)
            if baseline_csv not in baseline_cache:  # one read per baseline, not one per seed CSV
                baseline_cache[baseline_csv] = _baseline_choices(baseline_csv, log)
            choices = baseline_cache[baseline_csv]
            if choices is not None:
                got = choices.reindex(sub["original_index"].astype(int).to_numpy())
                texts["choices"].loc[sub.index] = got.to_numpy(dtype=object)
                n_missing = int(got.isna().sum())
                if n_missing:
                    _log(log, f"  {path.name}: {n_missing} row(s) without choices in {baseline_csv.name} — left null")
            else:
                _log(log, f"  {path.name}: no baseline CSV with choices found — choices/option_letters left null")
        if "option_letters" not in csv_text_cols:
            texts["option_letters"].loc[sub.index] = [_option_letters_json(c) for c in texts["choices"].loc[sub.index]]
        if sources is not None:
            stat = path.stat()
            sources.append({
                "source_csv": source_csv, "path": str(path), "size": int(stat.st_size),
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "n_rows_joined": int(len(sub)), "columns": list(csv_text_cols),
                "baseline_csv": None if baseline_csv is None else str(baseline_csv),
            })
    out = rows.copy()
    for col in TEXT_COLUMNS:
        series = texts[col]
        if col in ("choices", "option_letters"):
            series = series.where(~_blank(series), None)  # a blank cell is "no value", never "[]"
        out[col] = series.where(series.notna(), None).astype(object)
    return out


# ---------------------------------------------------------------------------
# Build: manifest → rows → texts → the dataset frame
# ---------------------------------------------------------------------------


def load_dataset_manifest(path: str | os.PathLike) -> tuple[pd.DataFrame, dict]:
    """The re-sampling manifest (``MANIFEST_COLUMNS`` coerced, the 13 extra columns required) and its
    ``.meta.json`` sidecar (``{}`` when absent)."""
    path = resolve_data_path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest {path} does not exist")
    raw = pd.read_parquet(path)
    missing = [c for c in RESAMPLE_EXTRA_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: not a re-sampling manifest — missing column(s) {missing} "
                         "(resample_relabel.py writes them)")
    base = coerce_schema(raw)
    extras = raw[list(RESAMPLE_EXTRA_COLUMNS)]
    manifest = pd.concat([base, extras], axis=1)
    sidecar = meta_path(path)
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return manifest, meta


def plan_dataset(spec: DatasetSpec, manifest: pd.DataFrame, manifest_meta: dict | None = None, *,
                 manifest_path: str | os.PathLike | None = None) -> tuple[pd.DataFrame, dict]:
    """Select, balance and report — everything but the text join (what ``--dry-run`` shows).

    Returns the kept rows (manifest + extra columns, ``label``, ``balance_kept``; sorted by
    ``rollout_id``) and the report: ``spec``, ``predicate``, ``label_col``, ``manifest`` (path,
    ``written_utc``, ``n_rows``, ``split`` block from the sidecar), ``n_selected``,
    ``n_after_balance``, ``labels_selected`` / ``labels_after_balance``, ``balance``, ``hint_styles``
    (sorted), ``fold_report``, ``fingerprint``, ``sources`` (empty until the join).
    """
    manifest_meta = manifest_meta or {}
    selected = select_dataset_rows(manifest, spec)
    kept, balance = apply_balance(selected, spec)
    kept = kept.sort_values("rollout_id", kind="stable").reset_index(drop=True)
    styles = sorted(kept["hint_style"].astype(str).unique())
    report = {
        "spec": spec.to_dict(), "predicate": spec.predicate, "label_col": spec.label_col,
        "manifest": {
            "path": None if manifest_path is None else str(manifest_path),
            "written_utc": manifest_meta.get("written_utc"), "n_rows": manifest_meta.get("n_rows"),
            "split": manifest_meta.get("split"),
        },
        "n_selected": int(len(selected)), "n_after_balance": int(len(kept)),
        "labels_selected": _label_counts(selected), "labels_after_balance": _label_counts(kept),
        "balance": balance, "hint_styles": styles, "fold_report": fold_report(kept, styles),
        "fingerprint": dataset_fingerprint(kept), "sources": [],
    }
    return kept, report


def build_dataset(spec: DatasetSpec, *, manifest: str | os.PathLike | None = None, log=None,
                  chunk_rows: int = DEFAULT_CHUNK_ROWS, source_dir: str | os.PathLike | None = None,
                  items_dir: str | os.PathLike | None = None) -> tuple[pd.DataFrame, dict]:
    """The dataset frame of ``spec`` and its report: read the manifest (``manifest`` overrides the spec's
    path), :func:`plan_dataset`, :func:`join_texts` (sources under ``source_dir``, default
    ``<manifest dir>/rollouts``; recipes under ``items_dir``, default ``<manifest dir>/items``), order
    the columns (:data:`DATASET_COLUMNS`), sort by ``rollout_id`` and validate."""
    manifest_path = resolve_data_path(spec.manifest if manifest is None else manifest)
    frame, meta = load_dataset_manifest(manifest_path)
    _log(log, f"manifest {manifest_path}: {len(frame)} rows (written {meta.get('written_utc')})")
    rows, report = plan_dataset(spec, frame, meta, manifest_path=manifest_path)
    _log(log, f"selected {report['n_selected']} rows → {report['n_after_balance']} after balance "
              f"({report['balance']['method']}); {len(report['hint_styles'])} styles")
    source_dir = manifest_path.parent / SOURCE_SUBDIR if source_dir is None else Path(source_dir)
    items_dir = manifest_path.parent / ITEMS_SUBDIR if items_dir is None else Path(items_dir)
    sources: list = []
    joined = join_texts(rows, source_dir=source_dir, chunk_rows=chunk_rows, items_dir=items_dir, log=log,
                        sources=sources)
    report["sources"] = sources
    out = joined[list(DATASET_COLUMNS)].sort_values("rollout_id", kind="stable").reset_index(drop=True)
    out["label"] = out["label"].astype("Int8")
    out["balance_kept"] = out["balance_kept"].astype(bool)
    validate_dataset(out)
    return out, report


# ---------------------------------------------------------------------------
# Validate, write, read
# ---------------------------------------------------------------------------


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(object).where(series.notna(), "").astype(str).str.strip().eq("")


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` on a frame that breaks the dataset contract: exactly :data:`DATASET_COLUMNS`
    in order, unique ``rollout_id``, ``label`` in {0, 1} everywhere, ``split`` assigned, ``balance_kept``
    all True, no blank ``reasoning`` / ``hint_style``, every row a re-roll."""
    if list(df.columns) != list(DATASET_COLUMNS):
        extra = sorted(set(df.columns) - set(DATASET_COLUMNS))
        missing = [c for c in DATASET_COLUMNS if c not in df.columns]
        raise ValueError(f"dataset columns differ from the contract (missing {missing}, extra {extra}, "
                         f"order {'ok' if not missing and not extra else 'n/a'}): got {list(df.columns)}")
    if not len(df):
        raise ValueError("dataset holds no row")
    dupes = df["rollout_id"][df["rollout_id"].duplicated()]
    if len(dupes):
        raise ValueError(f"{len(dupes)} duplicate rollout_id(s), e.g. {dupes.iloc[0]!r}")
    labels = pd.to_numeric(df["label"], errors="coerce")
    bad = (labels.isna() | ~labels.isin(LABELS)).fillna(True).astype(bool)
    if bad.any():
        raise ValueError(f"label must be 0 or 1 on every row; e.g. {_first_ids(df, bad)}")
    assert_assigned(df)
    kept = df["balance_kept"]
    not_kept = (kept.isna() | ~kept.fillna(False).astype(bool)).astype(bool)
    if not_kept.any():
        raise ValueError(f"balance_kept must be True on every written row; e.g. {_first_ids(df, not_kept)}")
    for col in ("reasoning", "hint_style"):
        blank = _blank(df[col])
        if blank.any():
            raise ValueError(f"{int(blank.sum())} row(s) with an empty {col}, e.g. {_first_ids(df, blank)}")
    provenance = df[selection.PROVENANCE_COLUMN].astype(object)
    originals = provenance.ne(selection.RESAMPLE_K4)
    if originals.any():
        raise ValueError(f"{int(originals.sum())} row(s) are not k=4 re-rolls, e.g. {_first_ids(df, originals)}")


def git_sha() -> str | None:
    """``git rev-parse HEAD`` of the repo, None when git or the checkout is unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def dataset_meta(df: pd.DataFrame, report: dict, spec: DatasetSpec, *, built_utc: str, sha: str | None) -> dict:
    """The ``.meta.json`` sidecar: spec, predicate, label column, manifest identity + split block,
    balance report, sources, styles, fingerprint, build time, git sha."""
    manifest = report.get("manifest") or {}
    return {
        "schema_version": DATASET_SCHEMA_VERSION, "name": spec.name, "spec": spec.to_dict(),
        "predicate": spec.predicate, "label_col": spec.label_col,
        "manifest_path": manifest.get("path"), "manifest_written_utc": manifest.get("written_utc"),
        "manifest_n_rows": manifest.get("n_rows"), "split": manifest.get("split"),
        "balance": report.get("balance"), "sources": report.get("sources", []),
        "hint_styles": sorted(df["hint_style"].astype(str).unique()),
        "n_rows": int(len(df)), "labels": _label_counts(df), "columns": list(DATASET_COLUMNS),
        "fingerprint": dataset_fingerprint(df), "built_utc": built_utc, "git_sha": sha,
    }


def umask_mode(mode: int = 0o666) -> int:
    """``mode`` masked by the process umask — what a plain ``open(..., "w")`` would create (``mkstemp`` gives 0600)."""
    current = os.umask(0)
    os.umask(current)
    return mode & ~current


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, umask_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_dataset(rows: pd.DataFrame, report: dict, spec: DatasetSpec, *, force: bool = False,
                  output_dir: str | os.PathLike | None = None) -> Path:
    """Write ``<output_dir>/<name>.parquet`` (pyarrow, zstd) + ``.meta.json`` + ``_report.json``
    atomically (temp file + rename each, parquet first) and return the parquet path. An existing
    parquet is refused unless ``force``. ``rows`` is validated first."""
    validate_dataset(rows)
    parquet, meta_file, report_file = dataset_paths(spec, output_dir)
    if parquet.exists() and not force:
        raise FileExistsError(f"{parquet} exists — pass force=True (--force) to overwrite it")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    built_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = dataset_meta(rows, report, spec, built_utc=built_utc, sha=git_sha())
    if meta["fingerprint"] != report.get("fingerprint", meta["fingerprint"]):
        raise ValueError("the rows' fingerprint differs from the report's — write the frame the report describes")
    fd, tmp = tempfile.mkstemp(dir=parquet.parent, prefix=f".{parquet.name}.", suffix=".tmp")
    os.close(fd)
    try:
        rows.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
        os.chmod(tmp, umask_mode())
        os.replace(tmp, parquet)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    _atomic_write_text(meta_file, json.dumps(meta, indent=2, default=_json_default) + "\n")
    _atomic_write_text(report_file, json.dumps({**report, "built_utc": built_utc, "git_sha": meta["git_sha"]},
                                               indent=2, default=_json_default) + "\n")
    return parquet


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_dataset(path: str | os.PathLike) -> tuple[pd.DataFrame, dict]:
    """``(frame, meta)`` of a written dataset — the one reader every consumer uses. Validates the
    contract (:func:`validate_dataset`), requires the ``.meta.json`` sidecar and checks its
    ``fingerprint`` / ``n_rows`` against the rows read."""
    parquet = resolve_data_path(path)
    if not parquet.exists():
        raise FileNotFoundError(f"dataset {parquet} does not exist")
    meta_file, _ = dataset_sidecars(parquet)
    if not meta_file.exists():
        raise FileNotFoundError(f"{parquet}: sidecar {meta_file.name} is missing — not a dataset written by write_dataset")
    df = pd.read_parquet(parquet)
    validate_dataset(df)
    meta = json.loads(meta_file.read_text())
    fingerprint = dataset_fingerprint(df)
    if meta.get("fingerprint") != fingerprint:
        raise ValueError(f"{parquet}: the sidecar's fingerprint {meta.get('fingerprint')!r} does not match the rows "
                         f"read ({fingerprint}) — the parquet or its sidecar was rewritten")
    if meta.get("n_rows") is not None and int(meta["n_rows"]) != len(df):
        raise ValueError(f"{parquet}: the sidecar records {meta['n_rows']} rows, the parquet holds {len(df)}")
    return df, meta


def spec_with(spec: DatasetSpec, **overrides) -> DatasetSpec:
    """A copy of ``spec`` with the given fields replaced (re-validated)."""
    return replace(spec, **overrides)


__all__ = [
    "BALANCE_METHODS", "DATASET_COLUMNS", "DATASET_SCHEMA_VERSION", "DEFAULT_BALANCE_GROUPS",
    "DEFAULT_CHUNK_ROWS", "DEFAULT_OUTPUT_DIR", "FOLD_ROLES", "ITEMS_SUBDIR", "RESAMPLE_EXTRA_COLUMNS",
    "RESTRICTIONS", "SOURCE_REQUIRED_COLUMNS", "SOURCE_SUBDIR", "SPEC_KEYS", "TEXT_COLUMNS",
    "BalanceSpec", "DatasetSpec", "apply_balance", "baseline_csv_for", "build_dataset", "dataset_fingerprint",
    "dataset_meta", "dataset_paths", "dataset_sidecars", "fold_report", "fold_roles", "git_sha", "join_texts",
    "load_dataset_manifest", "parse_balance", "parse_spec", "plan_dataset", "question_checks", "read_dataset",
    "resolve_source_csv", "select_dataset_rows", "spec_with", "stable_seed", "umask_mode", "validate_dataset",
    "write_dataset",
]
