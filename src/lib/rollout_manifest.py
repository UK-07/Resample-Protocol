"""The shared per-rollout manifest: one row per hinted rollout, keyed by ``rollout_id``.

A parquet file with exactly :data:`MANIFEST_COLUMNS` (in order, nullable pandas
dtypes) plus a ``.meta.json`` sidecar. ``rollout_id`` =
``<subject_model>:<run>:<hint_style>:<original_index>``; ``question_id`` =
``<dataset>:<dataset_split>:<original_index>`` is the cross-model question key.
:func:`pair_counts` is the groupby every summary number comes from.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.lib.hinted_rollouts import (
    compute_sensitivity_flags,
    row_reasoning,
    select_judgeable,
)
from src.lib.llm_judge_verb import JUDGE_ROLES
from src.lib.parsing import DEFAULT_DELIMITERS, ReasoningDelimiters, extract_cot
from src.lib.paths import resolve_data_path

MANIFEST_SCHEMA_VERSION = 2
DEFAULT_MANIFEST_PATH = "${DATA_ROOT}/hinted_rollouts/rollout_manifest.parquet"
DEFAULT_LABEL_OVERRIDES_PATH = "${DATA_ROOT}/hinted_rollouts/judge_label_overrides.csv"
OVERRIDE_COLUMNS = [
    "rollout_id", "judge_label_final", "judge_label_final_model", "judge_confidence_final",
    "judge_role_final", "hint_quote_final", "override_reason", "judged_utc",
]

CASES = ("positive", "negative")
SPLITS = ("train", "val", "test")
# Precedence order: a row gets the first reason that applies (question- and
# source-level reasons before row-level ones).
EXCLUDE_REASONS = (
    "baseline_relabelled", "judge_input_degraded",
    "truncated", "unanswered", "parse_fail", "incoherent", "noise_flip",
)
# The baseline status each case was selected on (select_case_rows).
CASE_BASELINE_STATUS = {"positive": "correct", "negative": "incorrect"}
# Judged-CSV columns the judge needs for an undegraded verdict.
JUDGE_INPUT_COLUMNS = ("prompt", "reasoning")
SMOKE_RUN_SUFFIX = "_smoke"
# A ``to_hint`` switch whose target already got this many no-hint votes is ``noise_flip``.
DEFAULT_NOISE_FLIP_MIN_VOTES = 1

# name -> nullable pandas dtype, in column order.
MANIFEST_COLUMNS: dict[str, str] = {
    "rollout_id": "string",
    "question_id": "string",
    "dataset": "string",
    "dataset_split": "string",
    "original_index": "Int64",
    "subject_model": "string",
    "subject_model_id": "string",
    "run": "string",
    "hint_style": "string",
    "case": "string",
    "target_option": "string",
    "groundtruth": "string",
    "baseline_modal_answer": "string",
    "baseline_stability": "Int8",
    "baseline_n_samples": "Int8",
    "baseline_hint_votes": "Int8",
    "model_answer": "string",
    "changed": "boolean",
    "to_hint": "boolean",
    "judge_model": "string",
    "judge_prompt": "string",
    "judge_label": "Int8",
    "judge_confidence": "Float64",
    "judge_role": "string",
    "hint_quote": "string",
    "judge_label_final": "Int8",
    "judge_label_final_model": "string",
    "trace_token_len": "Int64",
    "truncated": "boolean",
    "max_tokens_used": "Int64",
    "exclude_reason": "string",
    "split": "string",
    "source_csv": "string",
}

PAIR_KEYS = ["subject_model", "run"]
GROUP_KEYS = PAIR_KEYS + ["hint_style", "case"]
COUNT_COLUMNS = [
    "n_rollouts", "n_changed", "n_switched",
    "n_unfaithful", "n_faithful", "n_incoherent", "n_unjudged",
    "n_truncated", "n_unanswered",
]

# Any commit form parse_answer_from_response recognises: present but unparsed = parse_fail.
_ANSWER_MARKER_RE = re.compile(r"<answer>|\\boxed\{|[\"']answer[\"']\s*:", re.IGNORECASE)

# The judged-CSV columns a build reads.
JUDGED_READ_COLUMNS = [
    "original_index", "sample_type", "hint_name", "hinted_answer",
    "baseline_answer", "groundtruth", "rollout", "reasoning", "final_answer",
    "judge_model", "judge_label", "judge_confidence", "judge_role", "judge_hint_quote",
]


@dataclass(frozen=True)
class PairInfo:
    """What one judged CSV contributes to every row it yields."""

    subject_model: str
    subject_model_id: str | None
    run: str
    dataset: str
    dataset_split: str
    judge_prompt: str | None
    max_tokens: int | None
    source_csv: str
    baseline_n_samples: int | None = None


def is_smoke_run(run: str) -> bool:
    """Whether a run tag (or a judged-CSV stem) names a smoke test."""
    return str(run).endswith(SMOKE_RUN_SUFFIX) or f"{SMOKE_RUN_SUFFIX}_" in str(run)


def judge_input_degraded(columns) -> bool:
    """Whether a judged CSV with these columns was judged on degraded inputs."""
    return any(c not in set(columns) for c in JUDGE_INPUT_COLUMNS)


def make_rollout_id(subject_model: str, run: str, hint_style: str, original_index: int) -> str:
    return f"{subject_model}:{run}:{hint_style}:{int(original_index)}"


def make_question_id(dataset: str, dataset_split: str, original_index: int) -> str:
    return f"{dataset}:{dataset_split}:{int(original_index)}"


# The loader's default GPQA config: a baseline that omits ``config:`` was loaded from it.
_DEFAULT_GPQA_CONFIG = "gpqa_diamond"


def dataset_identity(baseline_meta: dict) -> tuple[str, str]:
    """(dataset, dataset_split) from a baseline ``.meta.json``.

    The split is the sidecar's resolved top-level ``split`` (else the
    ``dataset.params`` one on older sidecars), written ``<config>/<split>``
    when the loader took a config; subsetting knobs are not part of the identity.
    """
    block = baseline_meta.get("dataset") or {}
    name = str(block.get("name") or "")
    params = block.get("params") or {}
    split = str(baseline_meta.get("split") or params.get("split") or "")
    config = params.get("config") or params.get("subset")
    if not config and name == "gpqa":
        config = _DEFAULT_GPQA_CONFIG
    return name, f"{config}/{split}" if config else split


def answer_marker_present(rollout) -> bool:
    return isinstance(rollout, str) and _ANSWER_MARKER_RE.search(rollout) is not None


def _letter(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    return text or None


def _int_or_none(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    return int(value)


def _hint_votes(sample_answers, target: str | None) -> int | None:
    """Votes for ``target`` among a baseline row's JSON-encoded sample answers."""
    if target is None or not isinstance(sample_answers, str) or not sample_answers.strip():
        return None
    try:
        answers = json.loads(sample_answers)
    except json.JSONDecodeError:
        return None
    return sum(1 for a in answers if isinstance(a, str) and a.strip().upper() == target)


def load_baseline_votes(baseline_csv: Path) -> pd.DataFrame:
    """The per-question vote and status columns of a baseline CSV, indexed by ``original_index``.

    Columns the CSV lacks (greedy baselines, pre-status baselines) come back all-null.
    """
    header = pd.read_csv(baseline_csv, nrows=0).columns
    want = ["original_index", "baseline_answer", "n_top_votes", "sample_answers", "baseline_status"]
    usecols = [c for c in want if c in header]
    base = pd.read_csv(baseline_csv, usecols=usecols, dtype=str)
    for col in want:
        if col not in base.columns:
            base[col] = None
    base["original_index"] = base["original_index"].astype(int)
    return base.drop_duplicates("original_index").set_index("original_index")[want[1:]]


def derive_rollout_rows(
    chunk: pd.DataFrame,
    pair: PairInfo,
    *,
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
    baseline: pd.DataFrame | None = None,
    token_len: Callable[[list[str]], list[int]] | None = None,
    noise_flip_min_votes: int | None = DEFAULT_NOISE_FLIP_MIN_VOTES,
    degraded: bool = False,
) -> pd.DataFrame:
    """Manifest rows for one chunk of a judged hinted-rollouts CSV.

    ``chunk`` holds :data:`JUDGED_READ_COLUMNS` (missing columns read as null).
    ``baseline`` is :func:`load_baseline_votes`'s frame (None = no vote columns,
    no ``baseline_relabelled`` check); ``token_len`` maps traces to token counts
    (None = ``trace_token_len`` null); ``noise_flip_min_votes=None`` switches the
    ``noise_flip`` rule off; ``degraded`` marks a source judged without the
    judge-input columns. ``changed`` / ``to_hint`` follow the shared row contract.
    """
    df = chunk.copy()
    for col in JUDGED_READ_COLUMNS:
        if col not in df.columns:
            df[col] = None
    flagged = compute_sensitivity_flags(df)
    to_hint = df.index.isin(select_judgeable(df).index)

    idx = df["original_index"].astype(int)
    target = [_letter(v) for v in df["hinted_answer"]]
    labels = pd.to_numeric(df["judge_label"], errors="coerce")
    rollouts = df["rollout"].where(df["rollout"].notna(), "").astype(str)
    traces = [row_reasoning(row, delimiters) for _, row in df.iterrows()]

    relabelled = None
    if baseline is not None:
        joined = baseline.reindex(idx.to_numpy())
        stability = [_int_or_none(pd.to_numeric(v, errors="coerce")) for v in joined["n_top_votes"]]
        hint_votes = [_hint_votes(sa, t) for sa, t in zip(joined["sample_answers"], target)]
        # Checked on the baseline, not on this chunk's join: a chunk whose
        # questions are all missing from the baseline must still be flagged.
        if "baseline_status" in baseline.columns and baseline["baseline_status"].notna().any():
            relabelled = [
                (s if isinstance(s, str) else None) != CASE_BASELINE_STATUS.get(str(case))
                for s, case in zip(joined["baseline_status"], df["sample_type"])
            ]
    else:
        stability = [None] * len(df)
        hint_votes = [None] * len(df)

    out = pd.DataFrame({
        "rollout_id": [make_rollout_id(pair.subject_model, pair.run, h, i)
                       for h, i in zip(df["hint_name"].astype(str), idx)],
        "question_id": [make_question_id(pair.dataset, pair.dataset_split, i) for i in idx],
        "dataset": pair.dataset,
        "dataset_split": pair.dataset_split,
        "original_index": idx.to_numpy(),
        "subject_model": pair.subject_model,
        "subject_model_id": pair.subject_model_id,
        "run": pair.run,
        "hint_style": df["hint_name"].astype(str).to_numpy(),
        "case": df["sample_type"].astype(str).to_numpy(),
        "target_option": target,
        "groundtruth": [_letter(v) for v in df["groundtruth"]],
        "baseline_modal_answer": [_letter(v) for v in df["baseline_answer"]],
        "baseline_stability": stability,
        "baseline_n_samples": pair.baseline_n_samples,
        "baseline_hint_votes": hint_votes,
        "model_answer": [_letter(v) for v in df["final_answer"]],
        "changed": flagged["response_changed"].to_numpy(),
        "to_hint": to_hint,
        "judge_model": [v if isinstance(v, str) and v.strip() else None for v in df["judge_model"]],
        "judge_prompt": pair.judge_prompt,
        "judge_label": [_int_or_none(v) for v in labels],
        "judge_confidence": pd.to_numeric(df["judge_confidence"], errors="coerce").to_numpy(),
        "judge_role": [v if isinstance(v, str) and v in JUDGE_ROLES else None for v in df["judge_role"]],
        "hint_quote": [v if isinstance(v, str) and v.strip() else None for v in df["judge_hint_quote"]],
        "judge_label_final": [_int_or_none(v) for v in labels],
        "judge_label_final_model": [v if isinstance(v, str) and v.strip() else None for v in df["judge_model"]],
        "trace_token_len": token_len(traces) if token_len is not None else None,
        "truncated": [extract_cot(r, delimiters=delimiters) == "" if r.strip() else True for r in rollouts],
        "max_tokens_used": pair.max_tokens,
        "exclude_reason": None,
        "split": None,
        "source_csv": pair.source_csv,
    }, index=df.index)
    # Unjudged rows carry no judge identity, whatever the CSV column held.
    out.loc[out["judge_label"].isna(), ["judge_model", "judge_prompt"]] = None
    out.loc[out["judge_label_final"].isna(), "judge_label_final_model"] = None
    out["exclude_reason"] = exclude_reasons(
        out, has_marker=[answer_marker_present(r) for r in rollouts],
        noise_flip_min_votes=noise_flip_min_votes, relabelled=relabelled, degraded=degraded,
    )
    return coerce_schema(out.reset_index(drop=True))


def exclude_reasons(
    rows: pd.DataFrame,
    *,
    has_marker,
    noise_flip_min_votes: int | None = DEFAULT_NOISE_FLIP_MIN_VOTES,
    relabelled=None,
    degraded: bool = False,
) -> list[str | None]:
    """The ``exclude_reason`` of every row: the first applicable of :data:`EXCLUDE_REASONS`.

    ``relabelled`` (one flag per row; None = no baseline to check) drives
    ``baseline_relabelled``, ``degraded`` (source-level) ``judge_input_degraded``;
    ``truncated`` is checked before the parse reasons since a cut-off rollout may
    quote a commit form; ``incoherent`` reads ``judge_label_final`` when present,
    else ``judge_label``; ``noise_flip_min_votes=None`` never flags ``noise_flip``.
    """
    answered = rows["model_answer"].notna().to_numpy()
    truncated = rows["truncated"].fillna(False).astype(bool).to_numpy()
    label_col = "judge_label_final" if "judge_label_final" in rows.columns else "judge_label"
    labels = pd.to_numeric(rows[label_col], errors="coerce").to_numpy()
    votes = pd.to_numeric(rows["baseline_hint_votes"], errors="coerce").to_numpy()
    to_hint = rows["to_hint"].fillna(False).astype(bool).to_numpy()
    relabelled = [False] * len(rows) if relabelled is None else list(relabelled)
    out: list[str | None] = []
    for i in range(len(rows)):
        if relabelled[i]:
            out.append("baseline_relabelled")
        elif degraded:
            out.append("judge_input_degraded")
        elif truncated[i]:
            out.append("truncated")
        elif not answered[i]:
            out.append("parse_fail" if has_marker[i] else "unanswered")
        elif labels[i] == -1:
            out.append("incoherent")
        elif (noise_flip_min_votes is not None and to_hint[i] and not np.isnan(votes[i])
              and votes[i] >= noise_flip_min_votes):
            out.append("noise_flip")
        else:
            out.append(None)
    return out


def empty_manifest() -> pd.DataFrame:
    return coerce_schema(pd.DataFrame({c: [] for c in MANIFEST_COLUMNS}))


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with exactly :data:`MANIFEST_COLUMNS`, in order, in their dtypes.

    A v1 frame (no ``judge_label_final`` / ``judge_label_final_model``) gets
    them filled from ``judge_label`` / ``judge_model``.
    """
    if "judge_label_final" not in df.columns and "judge_label" in df.columns:
        df = df.assign(
            judge_label_final=df["judge_label"],
            judge_label_final_model=df["judge_model"] if "judge_model" in df.columns else None,
        )
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing column(s): {', '.join(missing)}")
    out = pd.DataFrame(index=df.index)
    for col, dtype in MANIFEST_COLUMNS.items():
        series = df[col]
        if dtype == "string":
            series = series.astype(object).where(series.notna(), None)
        out[col] = series.astype(dtype)
    return out


def validate_manifest(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` on a frame that breaks the manifest contract."""
    if list(df.columns) != list(MANIFEST_COLUMNS):
        raise ValueError(
            f"manifest columns differ from the schema: got {list(df.columns)}"
        )
    dupes = df["rollout_id"][df["rollout_id"].duplicated()]
    if len(dupes):
        raise ValueError(
            f"{len(dupes)} duplicate rollout_id(s), e.g. {dupes.iloc[0]!r}"
        )
    checks = {
        "case": CASES,
        "judge_role": JUDGE_ROLES,
        "exclude_reason": EXCLUDE_REASONS,
        "split": SPLITS,
    }
    for col, allowed in checks.items():
        bad = df[col].dropna()
        bad = bad[~bad.isin(allowed)]
        if len(bad):
            raise ValueError(f"{col} holds value(s) outside {allowed}: {sorted(set(bad))}")
    for col in ("judge_label", "judge_label_final"):
        labels = df[col].dropna()
        if len(labels) and not labels.isin([-1, 0, 1]).all():
            raise ValueError(f"{col} holds values outside {{-1, 0, 1}}")
    if (df["judge_label_final_model"].notna() & df["judge_label_final"].isna()).any():
        raise ValueError("judge_label_final_model is set on rows without a judge_label_final")
    if (df["to_hint"].fillna(False) & ~df["changed"].fillna(False)).any():
        raise ValueError("to_hint rows must also be changed rows")


def manifest_path(path: str | None = None) -> Path:
    return resolve_data_path(path or DEFAULT_MANIFEST_PATH)


def write_manifest(df: pd.DataFrame, path: Path, meta: dict) -> None:
    """Write the parquet and its ``.meta.json`` sidecar (validated first)."""
    df = coerce_schema(df)
    validate_manifest(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    sidecar = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "columns": list(MANIFEST_COLUMNS),
        **meta,
    }
    meta_path(path).write_text(json.dumps(sidecar, indent=2) + "\n")


def meta_path(path: Path) -> Path:
    return Path(path).with_suffix(".meta.json")


def read_manifest(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(manifest_path(str(path) if path else None))
    df = coerce_schema(df)
    validate_manifest(df)
    return df


def read_manifest_meta(path: str | Path | None = None) -> dict:
    p = meta_path(manifest_path(str(path) if path else None))
    return json.loads(p.read_text()) if p.exists() else {}


def read_label_overrides(path: str | Path | None = None) -> pd.DataFrame:
    """The label-overrides CSV (:data:`OVERRIDE_COLUMNS`); empty when the file is absent.

    Duplicate ids, labels outside {-1, 0, 1} or a blank model are errors.
    """
    p = resolve_data_path(str(path) if path else DEFAULT_LABEL_OVERRIDES_PATH)
    if not p.exists():
        return pd.DataFrame({c: [] for c in OVERRIDE_COLUMNS})
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    missing = [c for c in OVERRIDE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name}: missing override column(s) {missing}")
    df = df[OVERRIDE_COLUMNS].copy()
    dupes = df["rollout_id"][df["rollout_id"].duplicated()]
    if len(dupes):
        raise ValueError(f"{p.name}: duplicate rollout_id(s), e.g. {dupes.iloc[0]!r}")
    labels = pd.to_numeric(df["judge_label_final"], errors="coerce")
    if labels.isna().any() or not labels.isin([-1, 0, 1]).all():
        raise ValueError(f"{p.name}: judge_label_final must be -1, 0 or 1 on every row")
    df["judge_label_final"] = labels.astype(int)
    if (df["judge_label_final_model"].str.strip() == "").any():
        raise ValueError(f"{p.name}: judge_label_final_model is blank on some row")
    return df


def write_label_overrides(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write (validated) overrides; returns the path."""
    p = resolve_data_path(str(path) if path else DEFAULT_LABEL_OVERRIDES_PATH)
    out = pd.DataFrame({c: df[c] if c in df.columns else "" for c in OVERRIDE_COLUMNS})
    if out["rollout_id"].duplicated().any():
        raise ValueError("overrides hold duplicate rollout_id(s)")
    labels = pd.to_numeric(out["judge_label_final"], errors="coerce")
    if labels.isna().any() or not labels.isin([-1, 0, 1]).all():
        raise ValueError("judge_label_final must be -1, 0 or 1 on every override row")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return p


def apply_label_overrides(
    df: pd.DataFrame,
    overrides: pd.DataFrame,
    *,
    noise_flip_min_votes: int = DEFAULT_NOISE_FLIP_MIN_VOTES,
) -> tuple[pd.DataFrame, dict]:
    """``df`` with ``judge_label_final`` / ``judge_label_final_model`` set from ``overrides``.

    Unknown override ids are counted in the report (``n_unknown``), never applied.
    The overridden rows' ``exclude_reason`` is re-derived where it depends on the
    label (``incoherent`` → ``noise_flip`` → null).
    """
    out = df.copy()
    report = {"n_overrides": int(len(overrides)), "n_applied": 0, "n_unknown": 0, "models": []}
    if len(overrides) == 0:
        return out, report
    ids = overrides["rollout_id"].astype(str)
    position = pd.Series(np.arange(len(out)), index=out["rollout_id"].astype(str))
    hit = ids.isin(position.index).to_numpy()
    rows = position.loc[ids[hit]].to_numpy()
    labels = pd.to_numeric(overrides["judge_label_final"], errors="coerce").to_numpy()[hit]
    models = overrides["judge_label_final_model"].astype(str).to_numpy()[hit]
    final = out["judge_label_final"].astype("object").to_numpy(copy=True)
    final_model = out["judge_label_final_model"].astype("object").to_numpy(copy=True)
    final[rows] = labels.astype(int)
    final_model[rows] = models
    out["judge_label_final"] = pd.array(final, dtype="Int8")
    out["judge_label_final_model"] = pd.array(final_model, dtype="string")
    touched = out.iloc[rows]
    tail = touched[touched["exclude_reason"].isna() | touched["exclude_reason"].isin(["incoherent", "noise_flip"])]
    if len(tail):
        # These rows are answered and not truncated, so has_marker is never consulted.
        reasons = exclude_reasons(tail, has_marker=[False] * len(tail), noise_flip_min_votes=noise_flip_min_votes)
        out.loc[tail.index, "exclude_reason"] = pd.array(reasons, dtype="string")
    report.update(n_applied=int(hit.sum()), n_unknown=int((~hit).sum()), models=sorted(set(models.tolist())))
    return out, report


def pair_counts(df: pd.DataFrame, label_col: str = "judge_label") -> pd.DataFrame:
    """Per-(subject_model, run, hint_style, case) counts — the summary's numbers.

    Verdict counts are over ``to_hint`` rows only; ``n_unanswered`` includes the
    truncated rows; ``label_col`` picks the label column.
    """
    to_hint = df["to_hint"].fillna(False).astype(bool)
    labels = df[label_col].astype("float")
    counts = pd.DataFrame({
        "n_rollouts": 1,
        "n_changed": df["changed"].fillna(False).astype(int),
        "n_switched": to_hint.astype(int),
        "n_unfaithful": (to_hint & (labels == 0)).astype(int),
        "n_faithful": (to_hint & (labels == 1)).astype(int),
        "n_incoherent": (to_hint & (labels == -1)).astype(int),
        "n_unjudged": (to_hint & labels.isna()).astype(int),
        "n_truncated": df["truncated"].fillna(False).astype(bool).astype(int),
        "n_unanswered": df["model_answer"].isna().astype(int),
    }, index=df.index)
    for key in GROUP_KEYS:
        counts[key] = df[key].astype(str)
    return counts.groupby(GROUP_KEYS, sort=True)[COUNT_COLUMNS].sum().astype("int64")


def pair_info_dict(pair: PairInfo) -> dict:
    return asdict(pair)


__all__ = [
    "CASES", "CASE_BASELINE_STATUS", "COUNT_COLUMNS", "DEFAULT_LABEL_OVERRIDES_PATH",
    "DEFAULT_MANIFEST_PATH", "DEFAULT_NOISE_FLIP_MIN_VOTES", "EXCLUDE_REASONS", "GROUP_KEYS",
    "JUDGED_READ_COLUMNS", "JUDGE_INPUT_COLUMNS", "JUDGE_ROLES", "MANIFEST_COLUMNS",
    "MANIFEST_SCHEMA_VERSION", "OVERRIDE_COLUMNS", "PAIR_KEYS", "PairInfo", "SMOKE_RUN_SUFFIX",
    "SPLITS", "answer_marker_present", "apply_label_overrides", "coerce_schema",
    "dataset_identity", "derive_rollout_rows", "empty_manifest", "exclude_reasons",
    "is_smoke_run", "judge_input_degraded",
    "load_baseline_votes", "make_question_id", "make_rollout_id", "manifest_path", "meta_path",
    "pair_counts", "pair_info_dict", "read_label_overrides", "read_manifest", "read_manifest_meta",
    "validate_manifest", "write_label_overrides", "write_manifest",
]
