"""Shared code for the rate figures (rates under the single-sample protocol vs the clean resampled protocol):
rates_dumbbell, rates_rank_change, susceptibility_vs_unfaithfulness and table2_benchmark (each gather.py imports
this module and records its sha256 in data/gather_meta.json).

Reuse: the per-run stitching (manifest <-> binary-judge CSV <-> baseline CSV incl. the sample-0 re-parse
check <-> question_reliance.csv) and the SSP flags come from ``common.ssp_common``, imported unchanged. This
module adds only the rate flags (clean rows, SSP susceptibility), dataset_B and the cell tables.

    rows, meta = load_ssp_rows(only=None)  # one row per hinted-once rollout, both cases, SSP flags
    b    = dataset_B(load_resample())      # the resample manifest's dataset_B rows (selection.select_rows)
    cells = cell_table(rows, b, keys)      # every count + rate + Wilson CI per cell
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib import selection
from src.lib.resample import DROPPED_HINTS, SMOKE_RUN_MARKER, filter_manifest, wilson_interval
from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import ssp_common as sc

MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
STYLES = ["unethical_info", "metadata", "grader_hacking", "expert_opinion", "tool_output",
          "answer_key_artifact", "post_hoc", "consensus"]
CASES = ["positive", "negative"]
SMALL_N = 20                      # cells with a denominator below this are faded / flagged
CENSOR_TOKENS = 16384             # paper recipe budget (reported only; labels are NOT re-censored)


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_info(path: Path, *, sha: bool = False) -> dict:
    st = Path(path).stat()
    out = {"path": str(path), "size": st.st_size,
           "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    if sha:
        out["sha256"] = sc.sha256_file(path)
    return out


def shared_code_shas() -> dict:
    return {"rates_common.py": {"module": __name__, "sha256": sha256(Path(__file__).resolve())},
            "ssp_common.py": sc.lib_provenance(),
            "src/lib/selection.py": sc.code_deps_provenance(["src/lib/selection.py"])["src/lib/selection.py"]}


def input_infos(*, cueball: str | Path | None = None, sha: bool = False) -> dict:
    return {"manifest": file_info(P.manifest_path(cueball), sha=sha),
            "resample_manifest": file_info(P.resample_manifest_path(cueball), sha=sha),
            "question_reliance": file_info(P.reliance_path(cueball), sha=sha),
            "binary_judge_csvs": [file_info(p, sha=sha) for p in sorted(P.binary_dir(cueball).glob("*_binary_judged.csv"))
                                  if SMOKE_RUN_MARKER not in p.name]}


def _letters(s: pd.Series) -> pd.Series:
    out = s.astype(object).where(s.notna(), None)
    return out.map(lambda v: None if v is None or str(v).strip() in ("", "nan", "<NA>", "None")
                   else str(v).strip().upper())


# ---------------------------------------------------------------------------
# SSP: one row per hinted-once rollout, both manifest cases (stitching = ssp_common)
# ---------------------------------------------------------------------------

def load_ssp_rows(only: str | None = None, log=print, *, cueball: str | Path | None = None,
                  cache_dir: str | Path | None = sc.DEFAULT_CACHE, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """(rows, ssp_meta). The stitched + SSP-flagged rows of ssp_common.load_rows (cached parquet keyed on
    its sha and the inputs), plus the rate flags. ``only`` = runs whose stem contains it (debug); a subset
    never touches the cache."""
    df, meta = sc.load_rows(only, cache_dir=None if only else cache_dir, refresh=refresh, cueball=cueball)
    bad_models = sorted(set(df.subject_model.astype(str)) - set(MODELS))
    if bad_models:
        raise SystemExit(f"unexpected models in the binary-judged runs: {bad_models}")
    return add_rate_flags(df), meta


def add_rate_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Rate flags on top of ssp_common's (ssp_status, ssp_flip, ssp_unfaithful/faithful/label_missing,
    ssp_flip_truncated = truncated to-target rows EXCLUDED from the flips, robust_used, b0 = re-parsed sample 0,
    b0_stored = sample_answers[0])."""
    out = df.copy()
    for c in ("subject_model", "run", "dataset", "hint_style", "case"):
        out[c] = out[c].astype(str)
    tgt, h = _letters(out.target_option), _letters(out.model_answer)
    out["b0_reparse_mismatch"] = _letters(out.b0_stored).fillna("") != _letters(out.b0).fillna("")
    st = out.ssp_status.astype(str)
    out["b0_unanswered"] = st == "b0_unanswered"
    out["b0_is_target"] = st == "b0_is_target"
    out["wrong_to_wrong"] = st == "wrong_to_wrong"
    out["eligible_any"] = st == "eligible"
    truncated = out.truncated.fillna(False).astype(bool)
    # clean = drop truncated / unanswered / parse_fail rows (== model_answer parsed and not truncated;
    # equal to exclude_reason not in {truncated, unanswered, parse_fail} on the manifest, checked)
    out["clean"] = ~truncated & h.notna()
    out["eligible_clean"] = out.eligible_any & out.clean
    out["ssp_flip_clean"] = out.ssp_flip & out.clean              # SSP susceptibility numerator
    if (out.ssp_flip & ~out.clean).any():                         # truncated rollouts are never flips
        raise SystemExit("an SSP flip is not a clean row")
    if (out.ssp_flip & out.ssp_flip_truncated).any():
        raise SystemExit("ssp_flip overlaps ssp_flip_truncated")
    lab = pd.to_numeric(out.binary_judge_label, errors="coerce")
    out["binary_labeled"] = lab.isin([0, 1])
    out["to_hint_b"] = out.to_hint.fillna(False).astype(bool)
    out["robust_used_clean"] = out.robust_used.astype(bool) & out.clean
    out["to_hint_no_reliance_label"] = out.to_hint_b & out.reliance_label.isna()
    er = out.exclude_reason.astype(object)
    out["clean_vs_exclude_reason_mismatch"] = out.clean != ~er.isin(["truncated", "unanswered", "parse_fail"])
    return out


# ---------------------------------------------------------------------------
# RSP: dataset_B from the resample manifest
# ---------------------------------------------------------------------------

RESAMPLE_COLS = ["rollout_id", "source_rollout_id", "subject_model", "run", "dataset", "original_index",
                 "hint_style", "case", "provenance", "reliance_label", "to_hint", "judge_label_final",
                 "exclude_reason", "trace_token_len", "max_tokens_used", "sample_seed"]


def load_resample(only: str | None = None, *, cueball: str | Path | None = None) -> pd.DataFrame:
    r = pd.read_parquet(P.resample_manifest_path(cueball), columns=RESAMPLE_COLS)
    r = filter_manifest(r)
    r = r[r.subject_model.astype(str).isin(MODELS)]
    if only:
        stem_key = r.subject_model.astype(str) + "_" + r.run.astype(str)
        r = r[stem_key.str.contains(only, regex=False)]
    for c in ("subject_model", "run", "dataset", "hint_style", "case"):
        r[c] = r[c].astype(str)
    return r.copy()


def dataset_B(resample: pd.DataFrame) -> pd.DataFrame:
    """The RSP set: selection.py's dataset_B rows PLUS the v2-incoherent (-1) re-rolls that meet every other
    dataset_B condition. Those are admitted by re-using the dataset_B mask unchanged on a copy of exactly the
    rows with judge_label_final == -1 and exclude_reason == "incoherent" (the manifest's first-applicable
    exclusion, so no earlier reason — truncated / unanswered / parse_fail — applies), in which the verdict is
    set to 0 and the exclusion to null for the mask evaluation only. The stored -1 is kept.
    Flags: rsp_unfaithful (label 0), rsp_incoherent (-1, counted as unfaithful), rsp_faithful (1)."""
    b = selection.select_rows("dataset_B", resample).assign(in_dataset_B=True)
    lab_all = pd.to_numeric(resample.judge_label_final, errors="coerce")
    cand = resample[(lab_all == -1) & (resample.exclude_reason.astype(object) == "incoherent")].copy()
    if len(cand):
        probe = cand.copy()
        probe["judge_label_final"] = 0
        probe["exclude_reason"] = pd.Series([None] * len(probe), index=probe.index, dtype=object)
        keep = selection.mask("dataset_B", probe)
        inc = cand[keep.to_numpy()].assign(in_dataset_B=False)
        inc = selection.with_provenance(inc)
        b = pd.concat([b, inc], ignore_index=True)
    if b.rollout_id.duplicated().any():
        raise SystemExit("RSP set: duplicate rollout_id")
    lab = pd.to_numeric(b.judge_label_final, errors="coerce")
    b["rsp_unfaithful"] = lab == 0
    b["rsp_incoherent"] = lab == -1
    b["rsp_faithful"] = lab == 1
    assert (b.rsp_unfaithful | b.rsp_incoherent | b.rsp_faithful).all()
    b["over_budget"] = (pd.to_numeric(b.max_tokens_used, errors="coerce") > CENSOR_TOKENS) & (
        pd.to_numeric(b.trace_token_len, errors="coerce") > CENSOR_TOKENS)
    return b


def robust_reroll_exclusions(resample: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Reporting only: re-rolls of robust_used questions that are NOT in the RSP set, by reason."""
    rr = resample[(resample.provenance.astype(str) == "resample_k4")
                  & (resample.reliance_label.astype(object) == "robust_used")]
    rr = rr[~rr.rollout_id.isin(set(b.rollout_id))]
    lab = pd.to_numeric(rr.judge_label_final, errors="coerce").astype(float).to_numpy()
    to_hint = rr.to_hint.fillna(False).astype(bool).to_numpy()
    excl = rr.exclude_reason.astype(object).where(rr.exclude_reason.notna(), None).to_numpy()
    reason = np.where(~to_hint, "not_to_target",
             np.where(pd.notna(excl), np.array(["exclude_reason:" + str(e) for e in excl], dtype=object),
             np.where(lab == -1, "judge_-1", np.where(np.isnan(lab), "unjudged", "other"))))
    rr = rr.assign(reason=reason)
    return rr.groupby(["subject_model", "case", "reason"]).size().rename("n").reset_index()


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def _wilson(num, den):
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(num, den)]
    return [c[0] for c in ci], [c[1] for c in ci]


SSP_SUMS = {
    "n_hinted": "one", "n_clean": "clean", "n_b0_unanswered": "b0_unanswered", "n_b0_is_target": "b0_is_target",
    "n_wrong_to_wrong": "wrong_to_wrong", "n_eligible": "eligible_any", "n_eligible_clean": "eligible_clean",
    "n_ssp_flip": "ssp_flip", "n_truncated_to_target_excluded": "ssp_flip_truncated",
    "n_ssp_flip_clean": "ssp_flip_clean",
    "n_ssp_unfaithful": "ssp_unfaithful", "n_ssp_faithful": "ssp_faithful",
    "n_ssp_binary_missing": "ssp_label_missing",
    "n_binary_labeled_rows": "binary_labeled", "n_to_hint_modal": "to_hint_b", "n_robust_used": "robust_used",
    "n_robust_used_clean": "robust_used_clean", "n_to_hint_no_reliance_label": "to_hint_no_reliance_label",
    "n_b0_reparse_mismatch": "b0_reparse_mismatch", "n_clean_vs_exclude_reason_mismatch": "clean_vs_exclude_reason_mismatch",
}


def cell_table(rows: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Every count, rate and Wilson CI per ``keys`` (a subset of subject_model, dataset, hint_style, case).
    The grid is the full product of the key values present, so empty cells appear with zero counts and NaN
    rates."""
    r = rows.assign(one=1)
    agg = {k: (v, "sum") for k, v in SSP_SUMS.items()}
    s = r.groupby(keys).agg(**agg)
    bb = b.assign(one=1).groupby(keys).agg(n_rsp_rows=("one", "sum"), n_dataset_B=("in_dataset_B", "sum"),
                                           n_rsp_unfaithful=("rsp_unfaithful", "sum"),
                                           n_rsp_incoherent=("rsp_incoherent", "sum"),
                                           n_rsp_faithful=("rsp_faithful", "sum"),
                                           n_dataset_B_over_budget=("over_budget", "sum"),
                                           n_rsp_questions=("source_rollout_id", "nunique"))
    levels = {"subject_model": MODELS, "dataset": sorted(set(rows.dataset) | set(b.dataset)),
              "hint_style": STYLES, "case": CASES}
    idx = pd.MultiIndex.from_product([levels[k] for k in keys], names=keys) if len(keys) > 1 else \
        pd.Index(levels[keys[0]], name=keys[0])
    t = s.reindex(idx).join(bb.reindex(idx)).fillna(0).astype(int).reset_index()
    t["ssp_den"] = t.n_ssp_unfaithful + t.n_ssp_faithful
    t["ssp_rate"] = t.n_ssp_unfaithful / t.ssp_den.replace(0, np.nan)
    t["ssp_rate_lo"], t["ssp_rate_hi"] = _wilson(t.n_ssp_unfaithful, t.ssp_den)
    # RSP rate: (label 0 + label -1) / (0 + 1 + -1); -1 counted as unfaithful, reported separately
    t["rsp_num"] = t.n_rsp_unfaithful + t.n_rsp_incoherent
    t["rsp_den"] = t.n_rsp_unfaithful + t.n_rsp_incoherent + t.n_rsp_faithful
    assert (t.rsp_den == t.n_rsp_rows).all() and (t.n_dataset_B == t.n_rsp_unfaithful + t.n_rsp_faithful).all()
    t["rsp_rate"] = t.rsp_num / t.rsp_den.replace(0, np.nan)
    t["rsp_rate_lo"], t["rsp_rate_hi"] = _wilson(t.rsp_num, t.rsp_den)
    # companion only (not plotted): the strict dataset_B share of 0, -1 left out
    t["rsp_rate_excl_incoherent"] = t.n_rsp_unfaithful / t.n_dataset_B.replace(0, np.nan)
    t["robust_susc"] = t.n_robust_used_clean / t.n_clean.replace(0, np.nan)
    t["robust_susc_lo"], t["robust_susc_hi"] = _wilson(t.n_robust_used_clean, t.n_clean)
    t["ssp_susc"] = t.n_ssp_flip_clean / t.n_eligible_clean.replace(0, np.nan)
    t["ssp_susc_lo"], t["ssp_susc_hi"] = _wilson(t.n_ssp_flip_clean, t.n_eligible_clean)
    t["gap_rsp_minus_ssp"] = t.rsp_rate - t.ssp_rate
    for c, den in (("ssp_rate", "ssp_den"), ("rsp_rate", "rsp_den"), ("robust_susc", "n_clean"),
                   ("ssp_susc", "n_eligible_clean")):
        t[f"small_{c}"] = t[den] < SMALL_N
    return t


def exclusion_summary(rows: pd.DataFrame, b: pd.DataFrame, rr_excl: pd.DataFrame) -> dict:
    """Totals per model × case for NOTES.md / gather_meta.json."""
    out = {}
    tab = cell_table(rows, b, ["subject_model", "case"])
    keep = ["subject_model", "case", "n_hinted", "n_clean", "n_b0_unanswered", "n_b0_is_target",
            "n_wrong_to_wrong", "n_eligible", "n_eligible_clean", "n_ssp_flip", "n_truncated_to_target_excluded", "n_ssp_flip_clean", "n_ssp_unfaithful",
            "n_ssp_faithful", "n_ssp_binary_missing", "n_binary_labeled_rows", "n_robust_used",
            "n_robust_used_clean", "n_to_hint_no_reliance_label", "n_b0_reparse_mismatch", "n_dataset_B",
            "n_rsp_rows", "n_rsp_unfaithful", "n_rsp_incoherent", "n_dataset_B_over_budget"]
    out["per_model_case"] = tab[keep].to_dict("records")
    out["robust_used_rerolls_not_in_dataset_B"] = rr_excl.to_dict("records")
    return out


def write_exclusions_md(out: Path, rows: pd.DataFrame, b: pd.DataFrame, rr_excl: pd.DataFrame) -> None:
    """data/exclusions.md: the per model x case exclusion counts NOTES.md quotes."""
    t = cell_table(rows, b, ["subject_model", "case"])
    cols = ["subject_model", "case", "n_hinted", "n_b0_unanswered", "n_b0_is_target", "n_wrong_to_wrong",
            "n_eligible", "n_eligible_clean", "n_ssp_flip", "n_truncated_to_target_excluded", "n_ssp_binary_missing",
            "n_ssp_unfaithful", "n_ssp_faithful", "n_binary_labeled_rows", "n_clean", "n_robust_used",
            "n_robust_used_clean", "n_dataset_B", "n_rsp_rows", "n_rsp_unfaithful", "n_rsp_incoherent",
            "n_rsp_faithful", "n_dataset_B_over_budget"]
    tot = t[cols[2:]].sum()
    t = pd.concat([t[cols], pd.DataFrame([{"subject_model": "all", "case": "both", **tot.to_dict()}])])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    rr = rr_excl.pivot_table(index=["subject_model", "case"], columns="reason", values="n", fill_value=0)
    md += ["", "Re-rolls of robust_used questions NOT in the RSP set (dataset_B + admitted -1), by reason:", "",
           "| subject_model | case | " + " | ".join(rr.columns) + " |", "|---|---|" + "---|" * len(rr.columns)]
    md += [f"| {i[0]} | {i[1]} | " + " | ".join(str(int(v)) for v in r) + " |" for i, r in rr.iterrows()]
    (out / "exclusions.md").write_text("\n".join(md) + "\n")


def write_meta(out: Path, *, script: str, only, extra: dict, cueball: str | Path | None = None) -> None:
    meta = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "script": script,
            "only": only, "git_sha": sc.repo_git_sha(), "shared_code": shared_code_shas(),
            "cueball_dir": str(P.cueball_dir(cueball)), "inputs": input_infos(cueball=cueball, sha=True),
            "models": MODELS, "styles": STYLES, "cases": CASES, "small_n": SMALL_N,
            "dropped_hints": list(DROPPED_HINTS), "smoke_marker": SMOKE_RUN_MARKER, **extra}
    (out / "gather_meta.json").write_text(json.dumps(meta, indent=1, default=str))


def gather_common(out: Path, only: str | None, script: str, keysets: dict[str, list[str]],
                  write_rows: bool = False, log=print, *, cueball: str | Path | None = None,
                  cache_dir: str | Path | None = sc.DEFAULT_CACHE, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """The body every rate gather.py runs: stitch, dataset_B, one cells CSV per key set into ``out``.
    ``script`` is the calling gather's file path (its sha256 is recorded)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, ssp_meta = load_ssp_rows(only, log=log, cueball=cueball, cache_dir=cache_dir, refresh=refresh)
    res = load_resample(only, cueball=cueball)
    b = dataset_B(res)
    # consistency: every dataset_B row's source rollout is a hinted-once rollout of the same cell
    src = rows.set_index("rollout_id")
    joined = b[["source_rollout_id", "subject_model", "dataset", "hint_style", "case"]].join(
        src[["subject_model", "dataset", "hint_style", "case"]], on="source_rollout_id", rsuffix="_src")
    n_missing_src = int(joined.subject_model_src.isna().sum())
    n_cell_mismatch = int(sum((joined[c] != joined[f"{c}_src"]) & joined[f"{c}_src"].notna()
                              for c in ("subject_model", "dataset", "hint_style", "case")).astype(bool).sum())
    if n_missing_src or n_cell_mismatch:
        raise SystemExit(f"RSP rows vs SSP rows: {n_missing_src} sources missing, {n_cell_mismatch} cell mismatches")
    rr_excl = robust_reroll_exclusions(res, b)
    tabs = {}
    for name, keys in keysets.items():
        t = cell_table(rows, b, keys)
        t.to_csv(out / f"{name}.csv", index=False)
        tabs[name] = t
        log(f"wrote {out / name}.csv ({len(t)} cells)")
    pd.DataFrame(ssp_meta.get("join_checks", [])).to_csv(out / "join_checks.csv", index=False)
    rr_excl.to_csv(out / "robust_used_rerolls_not_in_dataset_B.csv", index=False)
    if write_rows:
        cols = ["rollout_id", "subject_model", "run", "dataset", "original_index", "hint_style", "case",
                "groundtruth", "target_option", "b0_stored", "b0", "model_answer", "truncated", "clean",
                "ssp_status", "eligible_clean", "ssp_flip", "ssp_flip_truncated", "ssp_flip_clean",
                "binary_judge_label", "ssp_unfaithful", "ssp_faithful", "ssp_label_missing", "to_hint",
                "rel_role", "reliance_label", "robust_used", "robust_used_clean", "exclude_reason",
                "baseline_stability"]
        rows[cols].to_csv(out / "ssp_rows.csv.gz", index=False)
        b[["rollout_id", "source_rollout_id", "subject_model", "run", "dataset", "hint_style", "case",
           "sample_seed", "judge_label_final", "in_dataset_B", "trace_token_len", "max_tokens_used",
           "over_budget"]].to_csv(out / "rsp_rows.csv.gz", index=False)
    jc = pd.DataFrame(ssp_meta.get("join_checks", []))
    checks_summary = {c: int(jc[c].sum()) for c in jc.columns if c != "stem"}
    write_meta(out, script=script, only=only, cueball=cueball, extra={
        "script_sha256": sha256(Path(script)),
        "keysets": keysets,
        "ssp_rows_meta": {k: v for k, v in ssp_meta.items() if k != "join_checks"},
        "join_checks_totals": checks_summary,
        "b0_reparse_mismatch_rows": int(rows.b0_reparse_mismatch.sum()),
        "clean_vs_exclude_reason_mismatch_rows": int(rows.clean_vs_exclude_reason_mismatch.sum()),
        "dataset_B_source_not_in_ssp_rows": n_missing_src,
        "dataset_B_cell_mismatch_vs_source": n_cell_mismatch,
        "n_rsp_rows": int(len(b)),
        "n_dataset_B_rows": int(b.in_dataset_B.sum()),
        "n_rsp_incoherent_admitted": int(b.rsp_incoherent.sum()),
        "n_dataset_B_over_16384_tokens": int(b.over_budget.sum()),
        "exclusions": exclusion_summary(rows, b, rr_excl),
    })
    write_exclusions_md(out, rows, b, rr_excl)
    log(f"RSP rows {len(b):,} (dataset_B {int(b.in_dataset_B.sum()):,}, -1 admitted {int(b.rsp_incoherent.sum()):,}); sources missing from SSP rows {n_missing_src}; cell mismatches {n_cell_mismatch}")
    return tabs
