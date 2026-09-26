"""Shared helpers for the alpha-implied vs measured-noise follow-up figures
(distractor_uniformity, alpha_bias_vs_stability, alpha_negative_case_check, noise_crosscheck).

Every figure is built from the per-rollout table ``rows.csv`` of the alpha_vs_measured_noise gather
(one row per hinted-once rollout of the 5 paper models, the sample-0 single-sample-protocol flags —
wrong→wrong dropped, truncated hinted rollouts never flips — and the re-sampling fields joined).
Its location is ``alpha_paths(...)``: an explicit ``data_dir``, else the env var ``ALPHA_DATA_DIR``, else
``paths.plot_dir("alpha_vs_measured_noise")/data``. Nothing here re-derives a join; it only re-aggregates that
table (restricted to the 5 paper models) and recomputes the flags from their raw columns as a consistency check.

Contents
    constants        the 5 models, the 8 styles, datasets, MIN_N (fade rule)
    alpha_paths()    where rows.csv / cells.csv / gather_meta.json of the alpha gather are
    load_rows()      read rows.csv (usecols), drop Qwen3.6-27B, re-derive and check every protocol flag
    cell_counts()    per-group counts + p, q, alpha, x = 1 - alpha, y_literal, y_strict (+ CIs)
    chi2_sf()        chi-square survival function (regularised upper incomplete gamma), no scipy
    pearson_uniform(), wald_heterogeneous_uniform()   the two distractor-uniformity tests
    selftest()       unit checks of chi2_sf and the tests (run by every gather.py; results go to gather_meta)
    file_info(), sha256_of(), lib_info()             provenance for gather_meta.json

Run: python -m src.scripts.visualizations.common.alpha_common [--mc]   (the self-test; --mc adds the
Monte-Carlo calibration check)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.lib.resample import ROBUST_MIN, wilson_interval
from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import ssp_common as sc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA_PLOT = "alpha_vs_measured_noise"
ALPHA_DATA_ENV = "ALPHA_DATA_DIR"      # debug override of the alpha gather's data/ directory

MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
EXCLUDED_MODELS = ["qwen3.6-27b"]
STYLES = ["unethical_info", "metadata", "grader_hacking", "expert_opinion", "tool_output",
          "answer_key_artifact", "post_hoc", "consensus"]
DATASETS = ["commonsense_qa", "medqa", "gpqa", "mmlu_pro"]
N_OPTIONS = {"commonsense_qa": 5, "medqa": 4, "gpqa": 4, "mmlu_pro": 10}
CASES = ["positive", "negative"]
STABILITY_BINS = [5, 6, 7, 8]          # manifest baseline_stability; the paper tree has no row below 5 (checked)
STABILITY_LABELS = {5: "≤5/8", 6: "6/8", 7: "7/8", 8: "8/8"}
LETTERS = "ABCDEFGHIJ"
MIN_N = 20                              # points / bars with n < MIN_N are drawn faded, n printed
FADE_ALPHA = 0.28
STABILITY_SENTENCE = ("Stability uses all 8 no-hint samples; it only bins the data and never defines an SSP "
                      "label, flip or case.")

ROW_COLS = [
    "rollout_id", "question_id", "subject_model", "run", "dataset", "original_index", "hint_style", "case",
    "n_options", "target_option", "groundtruth", "baseline_modal_answer", "baseline_stability", "b0",
    "model_answer", "truncated", "b0_answered", "b0_is_target", "wrong_to_wrong", "eligible", "truncated_eligible",
    "truncated_to_hint_excluded", "truncated_to_other_excluded", "hinted_answered", "p_to_hint",
    "q_to_other", "q_to_modal", "rel_reliance_label", "rel_k_n", "rel_k_to_hint_count", "rel_k_truncated",
    "pair", "pair_labeled", "pair_robust", "pair_strict_den", "pair_strict_zero",
]
BOOL_COLS = ["truncated", "b0_answered", "b0_is_target", "wrong_to_wrong", "eligible", "truncated_eligible",
             "truncated_to_hint_excluded", "truncated_to_other_excluded", "hinted_answered", "p_to_hint", "q_to_other",
             "q_to_modal", "pair", "pair_labeled", "pair_robust", "pair_strict_den", "pair_strict_zero"]
CELL_KEYS = ["subject_model", "run", "dataset", "hint_style", "case"]


# ---------------------------------------------------------------------------
# Paths of the alpha gather's outputs
# ---------------------------------------------------------------------------

class AlphaPaths(NamedTuple):
    data: Path          # the alpha_vs_measured_noise gather's data/ directory
    rows_csv: Path
    cells_csv: Path
    meta_json: Path
    overridden: bool    # data is not the default plot_dir(ALPHA_PLOT, cueball)/data (a debug subset is then allowed)


def alpha_paths(data_dir: str | Path | None = None, *, cueball: str | Path | None = None) -> AlphaPaths:
    """``data_dir``, else ``$ALPHA_DATA_DIR``, else ``paths.plot_dir("alpha_vs_measured_noise", cueball)/data``."""
    default = P.plot_dir(ALPHA_PLOT, cueball) / "data"
    raw = data_dir if data_dir is not None else os.environ.get(ALPHA_DATA_ENV)
    data = Path(raw) if raw else default
    return AlphaPaths(data, data / "rows.csv", data / "cells.csv", data / "gather_meta.json", data != default)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path: Path, *, sha: bool = False) -> dict:
    path = Path(path)
    st = path.stat()
    out = {"path": str(path), "size": st.st_size,
           "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    if sha:
        out["sha256"] = sha256_of(path)
    return out


def lib_info() -> dict:
    """Identity of this module (+ the repo git sha and the code deps' shas), recorded in every gather_meta.json."""
    return {"module": __name__, "sha256": sha256_of(Path(__file__).resolve()), "git_sha": sc.repo_git_sha(),
            "code_deps": sc.code_deps_provenance()}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), None)


def load_rows(only: str | None = None, *, data_dir: str | Path | None = None,
              cueball: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """rows.csv of the alpha_vs_measured_noise gather (5 paper models), every protocol flag re-derived from the raw
    columns and checked equal to the stored one. Returns (rows, report)."""
    ap = alpha_paths(data_dir, cueball=cueball)
    r = pd.read_csv(ap.rows_csv, usecols=ROW_COLS, dtype={"b0": str, "model_answer": str, "target_option": str,
                                                          "groundtruth": str, "baseline_modal_answer": str})
    rep: dict[str, object] = {"rows_csv_rows": int(len(r))}
    for c in BOOL_COLS:
        if r[c].dtype != bool:
            raise ValueError(f"rows.csv: {c} is not boolean")
    excl = r["subject_model"].isin(EXCLUDED_MODELS)
    rep["rows_excluded_qwen3.6-27b"] = int(excl.sum())
    r = r[~excl].copy()
    got = set(r["subject_model"])
    if not got <= set(MODELS) or (not ap.overridden and got != set(MODELS)):
        raise ValueError(f"models {sorted(got)} vs {MODELS} (a subset is allowed only with an overridden "
                         f"alpha data dir: --alpha-data / ${ALPHA_DATA_ENV})")
    rep["alpha_data_dir"] = str(ap.data)
    rep["alpha_data_overridden_for_debug"] = ap.overridden
    if set(r["hint_style"]) != set(STYLES):
        raise ValueError(f"styles {sorted(set(r['hint_style']))} != {STYLES}")
    if set(r["case"]) != set(CASES):
        raise ValueError("unexpected case values")
    bad_n = (r["n_options"] != r["dataset"].map(N_OPTIONS)).sum()
    if bad_n:
        raise ValueError(f"{bad_n} rows: n_options != registered width")
    stab = r["baseline_stability"]
    if stab.isna().any() or stab.min() < 5 or stab.max() > 8:
        raise ValueError(f"baseline_stability outside 5..8: {sorted(stab.dropna().unique())}")
    if r["run"].groupby(r["subject_model"] + "|" + r["dataset"]).nunique().max() != 1:
        raise ValueError("a (model, dataset) has more than one run")
    for c in ["b0", "model_answer", "target_option", "groundtruth", "baseline_modal_answer"]:
        r[c] = _norm(r[c])

    # re-derive the flags (same definitions as the alpha_vs_measured_noise gather)
    b0, h, t = r["b0"], r["model_answer"], r["target_option"]
    d = pd.DataFrame(index=r.index)
    d["b0_answered"] = b0.notna()
    d["b0_is_target"] = d["b0_answered"] & (b0 == t)
    d["wrong_to_wrong"] = (r["case"] == "positive") & d["b0_answered"] & ~d["b0_is_target"] & (b0 != r["groundtruth"])
    d["eligible"] = d["b0_answered"] & ~d["b0_is_target"] & ~d["wrong_to_wrong"]
    d["hinted_answered"] = h.notna()
    trunc = r["truncated"]
    d["truncated_eligible"] = d["eligible"] & trunc
    d["truncated_to_hint_excluded"] = d["truncated_eligible"] & d["hinted_answered"] & (h == t)
    d["truncated_to_other_excluded"] = d["truncated_eligible"] & d["hinted_answered"] & (h != t) & (h != b0)
    usable = d["hinted_answered"] & ~trunc
    d["p_to_hint"] = d["eligible"] & usable & (h == t)
    d["q_to_other"] = d["eligible"] & usable & (h != t) & (h != b0)
    d["q_to_modal"] = d["q_to_other"] & (h == r["baseline_modal_answer"])
    labeled = r["rel_reliance_label"].notna()
    d["pair"] = d["p_to_hint"]
    d["pair_labeled"] = d["pair"] & labeled
    d["pair_robust"] = d["pair_labeled"] & (r["rel_reliance_label"] == "robust_used")
    d["pair_strict_den"] = d["pair_labeled"] & (r["rel_k_truncated"].fillna(0) < r["rel_k_n"].fillna(0))
    d["pair_strict_zero"] = d["pair_strict_den"] & (r["rel_k_to_hint_count"] == 0)
    mism = {c: int((d[c] != r[c]).sum()) for c in d.columns}
    rep["flag_rederivation_mismatches"] = mism
    if any(mism.values()):
        raise ValueError(f"re-derived flags differ from rows.csv: {mism}")
    # robust_used <=> >= ROBUST_MIN of the re-rolls to target (stored labels, no re-censoring; checked only)
    rob_bad = int((d["pair_labeled"] & ((r["rel_k_to_hint_count"] >= ROBUST_MIN) != d["pair_robust"])).sum())
    rep["robust_label_vs_k_count_mismatch"] = rob_bad
    if rob_bad:
        raise ValueError(f"{rob_bad} pairs: robust_used label disagrees with k_to_hint_count >= {ROBUST_MIN}")
    # every answered letter is inside the run's letter set
    for c in ["b0", "model_answer", "target_option"]:
        v = r[c].notna()
        bad = int((v & ~pd.Series([str(x) in LETTERS[:n] for x, n in zip(r[c], r["n_options"])],
                                  index=r.index)).sum())
        if bad:
            raise ValueError(f"{bad} rows: {c} outside the run's letters")

    if only:   # debug subset: comma-separated substrings, a row is kept when model, run or dataset contains one
        keep = pd.Series(False, index=r.index)
        for tok in [t for t in only.split(",") if t]:
            keep |= (r["subject_model"].str.contains(tok, regex=False) | r["run"].str.contains(tok, regex=False)
                     | r["dataset"].str.contains(tok, regex=False))
        r = r[keep].copy()
        rep["only"] = only
    # per-row weights used by every pooled estimator
    n = r["n_options"].astype(float)
    r["w_implied"] = r["q_to_other"] / (n - 2)                     # q-count / (n - 2), pooled across datasets
    r["pred_base"] = np.where(r["eligible"], (8 - r["baseline_stability"]) / 8 / (n - 1), 0.0)
    rep["rows"] = int(len(r))
    rep["rows_per_model"] = {m: int(v) for m, v in r["subject_model"].value_counts().reindex(MODELS).fillna(0).items()}
    return r, rep


REPARSE_COL = "b0_stored_vs_reparsed_mismatch_questions"


def check_sample0_reparse(*, cueball: str | Path | None = None, cache_dir: str | Path | None = sc.DEFAULT_CACHE,
                          refresh: bool = False, join_checks: pd.DataFrame | None = None) -> dict:
    """The SSP stitch (``ssp_common``) re-parses every baseline's sample 0 with the repo parser; assert the stored
    sample-0 letter (the b0 of rows.csv) equals the re-parse for every source of the 5 models. The per-source
    counts come from ``join_checks`` (a frame with ``stem`` + :data:`REPARSE_COL`), by default the stitched
    rows' cached meta (``ssp_common.load_rows``; ``cache_dir`` / ``refresh`` are passed through)."""
    if join_checks is None:
        _, meta = sc.load_rows(None, cache_dir=cache_dir, refresh=refresh, cueball=cueball)
        jc = pd.DataFrame(meta["join_checks"])
        source = {"ssp_rows_cache": meta.get("cache_file"), "built_utc": meta.get("built_utc")}
    else:
        jc = join_checks
        source = {"join_checks": "caller-supplied"}
    col = REPARSE_COL if REPARSE_COL in jc.columns else "sample0_stored_vs_reparsed_mismatch_questions"
    jc = jc[jc["stem"].str.split("_").str[0].isin(MODELS)]
    per = dict(zip(jc["stem"], jc[col].astype(int)))
    models_seen = {s.split("_")[0] for s in per}
    if models_seen != set(MODELS):
        raise ValueError(f"join checks cover {sorted(models_seen)}, not all 5 models")
    if any(per.values()):
        raise ValueError(f"sample-0 stored vs re-parsed mismatches: {per}")
    return {"source": source, "n_sources": len(per), "total_mismatch_questions": int(sum(per.values()))}


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

COUNT_FLAGS = {
    "b0_answered": "n_b0_answered", "b0_is_target": "n_b0_is_target", "eligible": "n_eligible",
    "hinted_answered": "n_hinted_answered", "p_to_hint": "n_to_hint", "q_to_other": "n_to_other",
    "q_to_modal": "n_to_other_is_modal", "pair_labeled": "n_pairs_labeled", "pair_robust": "n_robust_used",
    "pair_strict_den": "n_pairs_strict", "pair_strict_zero": "n_zero_strict", "wrong_to_wrong": "n_wrong_to_wrong",
    "truncated_eligible": "n_truncated_eligible", "truncated_to_hint_excluded": "n_truncated_to_hint_excluded",
    "truncated_to_other_excluded": "n_truncated_to_other_excluded",
}


def cell_counts(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per group of `keys`: counts and the estimators.

    x_implied (= 1 − α, clipped to [0, 1]) = Σ q_to_other/(n−2) / Σ p_to_hint, i.e. the implied noise COUNTS are
    pooled (for a single-dataset cell this is exactly the alpha gather's q/((n−2)p)).
    y_literal = 1 − n_robust_used / n_pairs_labeled (Wilson CI);
    y_strict  = n_zero_strict / n_pairs_strict      (Wilson CI).
    baseline-predicted share = mean over eligible rows of (8 − stability)/8 · 1/(n − 1), divided by p.
    """
    w = rows[keys].copy()
    for f, name in COUNT_FLAGS.items():
        w[name] = rows[f].astype(int)
    w["implied_count"] = rows["w_implied"]
    w["pred_base_sum"] = rows["pred_base"]
    w["n_rollouts"] = 1
    w["n_questions_set"] = rows["question_id"]
    g = w.groupby(keys, sort=True)
    c = g[[*COUNT_FLAGS.values(), "implied_count", "pred_base_sum", "n_rollouts"]].sum()
    c["n_questions"] = g["n_questions_set"].nunique()
    c["n_options_set"] = rows.groupby(keys)["n_options"].agg(lambda s: ",".join(str(v) for v in sorted(set(s))))
    c["n_b0_unanswered"] = c["n_rollouts"] - c["n_b0_answered"]
    c["n_not_eligible"] = c["n_rollouts"] - c["n_eligible"]   # = b0 unanswered + b0 is target + wrong→wrong
    elig = c["n_eligible"].replace(0, np.nan)
    c["p"] = c["n_to_hint"] / elig
    c["q"] = c["n_to_other"] / elig
    hint = c["n_to_hint"].replace(0, np.nan)
    c["x_implied_raw"] = c["implied_count"] / hint                 # 1 − α_raw
    c["alpha_raw"] = 1 - c["x_implied_raw"]
    c["alpha_clipped"] = c["x_implied_raw"] > 1
    c["x_implied"] = c["x_implied_raw"].clip(upper=1.0)
    lab = c["n_pairs_labeled"].replace(0, np.nan)
    c["y_literal"] = 1 - c["n_robust_used"] / lab
    wl = [wilson_interval(int(k), int(n)) if n else (np.nan, np.nan)
          for k, n in zip(c["n_pairs_labeled"] - c["n_robust_used"], c["n_pairs_labeled"])]
    c["y_literal_lo"], c["y_literal_hi"] = [a for a, _ in wl], [b for _, b in wl]
    sd = c["n_pairs_strict"].replace(0, np.nan)
    c["y_strict"] = c["n_zero_strict"] / sd
    ws = [wilson_interval(int(k), int(n)) if n else (np.nan, np.nan)
          for k, n in zip(c["n_zero_strict"], c["n_pairs_strict"])]
    c["y_strict_lo"], c["y_strict_hi"] = [a for a, _ in ws], [b for _, b in ws]
    c["pred_base_mean"] = c["pred_base_sum"] / elig                # predicted P(noise lands on target)
    c["pred_base_share"] = c["pred_base_mean"] / c["p"]            # as a share of to-target flips
    c["faded_literal"] = c["n_pairs_labeled"] < MIN_N
    c["faded_strict"] = c["n_pairs_strict"] < MIN_N
    return c.reset_index()


def exclusion_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per model: where every hinted-once rollout goes under the sample-0 protocol (all counts)."""
    w = pd.DataFrame({
        "subject_model": rows["subject_model"],
        "n_rollouts": 1,
        "n_b0_unanswered": (~rows["b0_answered"]).astype(int),
        "n_b0_is_target": rows["b0_is_target"].astype(int),
        "n_wrong_to_wrong_dropped": rows["wrong_to_wrong"].astype(int),
        "n_eligible": rows["eligible"].astype(int),
        "n_eligible_hinted_unanswered": (rows["eligible"] & ~rows["hinted_answered"]).astype(int),
        "n_eligible_truncated": rows["truncated_eligible"].astype(int),
        "n_truncated_to_hint_excluded": rows["truncated_to_hint_excluded"].astype(int),
        "n_truncated_to_other_excluded": rows["truncated_to_other_excluded"].astype(int),
        "n_eligible_same_as_b0": (rows["eligible"] & rows["hinted_answered"] & ~rows["truncated"]
                                  & (rows["model_answer"] == rows["b0"])).astype(int),
        "n_to_hint": rows["p_to_hint"].astype(int),
        "n_to_other": rows["q_to_other"].astype(int),
        "n_to_other_is_modal": rows["q_to_modal"].astype(int),
        "n_pairs_labeled": rows["pair_labeled"].astype(int),
        "n_pairs_unlabeled": (rows["pair"] & ~rows["pair_labeled"]).astype(int),
        "n_pairs_all_rerolls_truncated": (rows["pair_labeled"] & ~rows["pair_strict_den"]).astype(int),
        "n_wrong_to_wrong_would_be_to_hint": (rows["wrong_to_wrong"] & rows["hinted_answered"] & ~rows["truncated"]
                                              & (rows["model_answer"] == rows["target_option"])).astype(int),
    })
    t = w.groupby("subject_model").sum().reindex(MODELS).dropna(how="all")
    t.loc["all"] = t.sum()
    return t.astype(int).reset_index()


def merge_alpha_x_ci(cells: pd.DataFrame, *, data_dir: str | Path | None = None,
                     cueball: str | Path | None = None) -> pd.DataFrame:
    """Attach the alpha gather's 95 % interval of x (Jeffreys draws, x_lo / x_hi) to (model, run, style, case) cells."""
    ap = alpha_paths(data_dir, cueball=cueball)
    v2 = pd.read_csv(ap.cells_csv, usecols=CELL_KEYS + ["x_lo", "x_hi"])
    out = cells.merge(v2, on=CELL_KEYS, how="left", validate="one_to_one")
    miss = int((out["x_lo"].isna() & out["x_implied"].notna()).sum())
    if miss:
        raise ValueError(f"{miss} cells with x but no alpha interval")
    return out


def check_against_alpha_cells(cells: pd.DataFrame, *, data_dir: str | Path | None = None,
                              cueball: str | Path | None = None) -> dict:
    """The per-cell (model × run × dataset × style × case) numbers must equal the alpha gather's cells.csv, both ways."""
    ap = alpha_paths(data_dir, cueball=cueball)
    ref = pd.read_csv(ap.cells_csv)
    if len(ref) != len(cells):
        raise ValueError(f"{len(cells)} cells here vs {len(ref)} in {ap.cells_csv}")
    j = cells.merge(ref, on=CELL_KEYS, how="left", suffixes=("", "_ref"), indicator=True)
    out = {"alpha_cells": str(ap.cells_csv), "cells": int(len(cells)),
           "cells_missing_in_alpha": int((j["_merge"] != "both").sum())}
    pairs = {"n_eligible": "n_eligible", "n_to_hint": "n_to_hint", "n_to_other": "n_to_other",
             "n_pairs_labeled": "n_pairs_labeled", "n_robust_used": "n_robust_used",
             "n_pairs_strict": "n_pairs_strict", "n_zero_strict": "n_zero_strict",
             "n_wrong_to_wrong": "n_wrong_to_wrong", "n_truncated_eligible": "n_truncated_eligible",
             "n_truncated_to_hint_excluded": "n_truncated_to_hint_excluded",
             "x_implied": "x_alpha_noise", "y_literal": "y_measured_noise", "y_strict": "y_strict"}
    for mine, theirs in pairs.items():
        a, b = j[mine].astype(float), j[theirs if theirs != mine else f"{mine}_ref"].astype(float)
        diff = (a - b).abs()
        both_nan = a.isna() & b.isna()
        out[f"max_abs_diff_{mine}"] = float(diff[~both_nan].max()) if (~both_nan).any() else 0.0
        out[f"nan_mismatch_{mine}"] = int((a.isna() != b.isna()).sum())
    bad = out["cells_missing_in_alpha"] or any(v > 1e-12 for k, v in out.items() if k.startswith("max_abs")) \
        or any(v for k, v in out.items() if k.startswith("nan_mismatch"))
    if bad:
        raise ValueError(f"cells disagree with the alpha gather's cells.csv: {out}")
    return out


# ---------------------------------------------------------------------------
# Chi-square survival function (no scipy)
# ---------------------------------------------------------------------------

_EPS = 1e-15
_FPMIN = 1e-300
_ITMAX = 10000


def gammainc_lower_reg(a: float, x: float) -> float:
    """P(a, x) by its power series (converges fast for x < a + 1)."""
    ap, s = a, 1.0 / a
    term = s
    for _ in range(_ITMAX):
        ap += 1.0
        term *= x / ap
        s += term
        if abs(term) < abs(s) * _EPS:
            break
    else:
        raise RuntimeError("gamma series did not converge")
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gammainc_upper_reg_cf(a: float, x: float) -> float:
    """Q(a, x) by its continued fraction, modified Lentz (converges fast for x >= a + 1)."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    else:
        raise RuntimeError("gamma continued fraction did not converge")
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) = Γ(a, x)/Γ(a)."""
    if a <= 0:
        raise ValueError("a must be > 0")
    if x < 0:
        raise ValueError("x must be >= 0")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, 1.0 - gammainc_lower_reg(a, x))
    return gammainc_upper_reg_cf(a, x)


def chi2_sf(x: float, df: float) -> float:
    """P(X > x) for X ~ chi-square(df) = Q(df/2, x/2)."""
    if df <= 0:
        raise ValueError("df must be > 0")
    if x <= 0:
        return 1.0
    return gammaincc(df / 2.0, x / 2.0)


# ---------------------------------------------------------------------------
# Distractor-uniformity tests
# ---------------------------------------------------------------------------

def pearson_uniform(counts: np.ndarray) -> tuple[float, int, float]:
    """Pearson chi-square of `counts` (k categories) against the uniform distribution; df = k − 1."""
    counts = np.asarray(counts, dtype=float)
    k, n = len(counts), counts.sum()
    e = n / k
    stat = float(((counts - e) ** 2 / e).sum())
    return stat, k - 1, chi2_sf(stat, k - 1)


def pearson_expected(observed, expected) -> tuple[float, int, float]:
    """Pearson chi-square of `observed` against given `expected` counts (same total); df = k − 1."""
    o = np.asarray(observed, dtype=float)
    e = np.asarray(expected, dtype=float)
    if abs(o.sum() - e.sum()) > 1e-9 * max(1.0, e.sum()):
        raise ValueError("observed and expected totals differ")
    stat = float(((o - e) ** 2 / e).sum())
    return stat, len(o) - 1, chi2_sf(stat, len(o) - 1)


def binom_sf_ge(k: int, m: int, p: float) -> float:
    """Exact P(X >= k), X ~ Binomial(m, p)."""
    if k <= 0:
        return 1.0
    return float(min(1.0, sum(math.comb(m, j) * p ** j * (1 - p) ** (m - j) for j in range(k, m + 1))))


def wald_heterogeneous_uniform(supports: np.ndarray, observed: np.ndarray) -> dict:
    """Test that every row picked uniformly inside its own support (rows independent).

    supports: [R, L] 0/1 matrix — row r's allowed categories (here: the n−2 letters other than target and b0).
    observed: [R, L] one-hot — the category row r picked.
    Under H0 the letter counts O = Σ_r o_r have mean E = Σ_r π_r and covariance Σ = Σ_r (diag π_r − π_r π_rᵀ),
    π_r = s_r / |s_r|. W = (O − E)ᵀ Σ⁺ (O − E) is asymptotically chi-square with rank(Σ) df. With a support shared
    by every row it equals Pearson's statistic over that support (df = |s| − 1)."""
    s = np.asarray(supports, dtype=float)
    o = np.asarray(observed, dtype=float)
    if (o * (1 - s)).any():
        raise ValueError("an observation falls outside its row's support")
    pi = s / s.sum(axis=1, keepdims=True)
    obs = o.sum(axis=0)
    exp = pi.sum(axis=0)
    cov = np.diag(exp) - pi.T @ pi
    keep = exp > 0
    cov_k = cov[np.ix_(keep, keep)]
    dvec = (obs - exp)[keep]
    tol = 1e-9 * max(1.0, float(np.abs(cov_k).max()))
    rank = int(np.linalg.matrix_rank(cov_k, tol=tol))
    stat = float(dvec @ np.linalg.pinv(cov_k, rcond=1e-10, hermitian=True) @ dvec)
    return {"stat": stat, "df": rank, "p": chi2_sf(stat, rank) if rank > 0 else float("nan"),
            "observed": obs, "expected": exp}


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (NaN kept)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest(monte_carlo: bool = False) -> dict:
    """Unit checks; raises on failure, returns the checked values (recorded in gather_meta.json)."""
    res: dict[str, object] = {}

    def close(a, b, rtol, name):
        if not (abs(a - b) <= rtol * max(abs(b), 1e-300)):
            raise AssertionError(f"{name}: {a!r} != {b!r} (rtol {rtol})")
        res[name] = {"got": a, "ref": b}

    # textbook critical values: chi2.sf(3.84, 1) = erfc(sqrt(1.92)) = 0.0500435...,
    # chi2.sf(11.07, 5) = erfc(sqrt(x/2)) + sqrt(2x/pi) e^(-x/2) (1 + x/3) = 0.0500096... (both ≈ 0.05)
    close(chi2_sf(3.84, 1), 0.05004352124870519, 1e-9, "sf(3.84,1)")
    close(chi2_sf(11.07, 5), 0.05000961862240547, 1e-9, "sf(11.07,5)")
    close(chi2_sf(3.841458820694124, 1), 0.05, 1e-9, "sf(q95,1)")
    close(chi2_sf(11.070497693516351, 5), 0.05, 1e-9, "sf(q95,5)")
    close(chi2_sf(16.918977604620448, 9), 0.05, 1e-9, "sf(q95,9)")
    close(chi2_sf(6.6348966010212145, 1), 0.01, 1e-9, "sf(q99,1)")
    # closed forms over a grid (both branches of the incomplete gamma, including far tails)
    for x in [1e-6, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 80.0, 200.0, 600.0]:
        close(chi2_sf(x, 1), math.erfc(math.sqrt(x / 2)), 1e-10, f"sf({x},1)=erfc")
        close(chi2_sf(x, 2), math.exp(-x / 2), 1e-10, f"sf({x},2)=exp")
        close(chi2_sf(x, 3), math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2), 1e-9,
              f"sf({x},3)")
        close(chi2_sf(x, 5), math.erfc(math.sqrt(x / 2))
              + math.sqrt(2 * x / math.pi) * math.exp(-x / 2) * (1 + x / 3), 1e-9, f"sf({x},5)")
        close(chi2_sf(x, 4), math.exp(-x / 2) * (1 + x / 2), 1e-10, f"sf({x},4)")
        close(chi2_sf(x, 8), math.exp(-x / 2) * sum((x / 2) ** k / math.factorial(k) for k in range(4)), 1e-10,
              f"sf({x},8)")
    if chi2_sf(0.0, 3) != 1.0:
        raise AssertionError("sf(0, k) != 1")
    # Wald statistic == Pearson when every row shares one support
    rng = np.random.default_rng(1)
    L = 10
    supp = np.zeros((300, L)); supp[:, 2:] = 1          # 8 allowed letters, same for every row
    picks = rng.integers(2, L, size=300)
    obs = np.zeros_like(supp); obs[np.arange(300), picks] = 1
    w = wald_heterogeneous_uniform(supp, obs)
    pe = pearson_uniform(obs.sum(axis=0)[2:])
    close(w["stat"], pe[0], 1e-9, "wald==pearson stat")
    if w["df"] != pe[1]:
        raise AssertionError(f"wald df {w['df']} != pearson df {pe[1]}")
    res["wald==pearson df"] = w["df"]
    # Pearson with explicit expected == uniform Pearson; exact binomial tail
    close(pearson_expected([12, 3], [7.5, 7.5])[0], pearson_uniform([12, 3])[0], 1e-12, "pearson_expected")
    close(binom_sf_ge(3, 4, 0.5), 5 / 16, 1e-12, "binom_sf_ge(3,4,.5)")
    # BH
    adj = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan, 0.5]))
    ref = np.array([0.04, 0.16 / 3, 0.16 / 3, np.nan, 0.5])   # m = 4: 0.04·4/3 = 0.0533, 0.03·4/2 = 0.06 → 0.0533
    if not np.allclose(adj, ref, equal_nan=True):
        raise AssertionError(f"BH {adj} != {ref}")
    res["bh"] = "ok"
    if monte_carlo:
        res["mc"] = monte_carlo_calibration()
    return res


def monte_carlo_calibration(n_sims: int = 2000, seed: int = 7) -> dict:
    """Size of both tests at 0.05 under H0 with heterogeneous supports (10 letters, rows drop 2 random
    letters, 40 rows per cell = 5·(n−2)); a well-calibrated test rejects ≈ 5 %."""
    rng = np.random.default_rng(seed)
    L, R = 10, 40
    rej_w = rej_p = 0
    for _ in range(n_sims):
        supp = np.ones((R, L))
        pos = np.zeros(L - 2)
        obs = np.zeros((R, L))
        for r in range(R):
            drop = rng.choice(L, size=2, replace=False)
            supp[r, drop] = 0
            allowed = np.flatnonzero(supp[r])
            j = rng.integers(0, L - 2)
            obs[r, allowed[j]] = 1
            pos[j] += 1
        rej_w += wald_heterogeneous_uniform(supp, obs)["p"] < 0.05
        rej_p += pearson_uniform(pos)[2] < 0.05
    out = {"n_sims": n_sims, "size_wald_letter": rej_w / n_sims, "size_pearson_position": rej_p / n_sims}
    for k in ("size_wald_letter", "size_pearson_position"):
        if not 0.03 <= out[k] <= 0.075:
            raise AssertionError(f"{k} = {out[k]} outside [0.03, 0.075]")
    out["regimes"] = monte_carlo_regimes(seed=seed + 1)
    return out


MC_REGIMES = [(4, 10), (4, 20), (5, 15), (5, 25)]   # (n options, switches per cell) at / near the qualifying rule


def monte_carlo_regimes(n_sims: int = 4000, seed: int = 8) -> list[dict]:
    """Size at 0.05 of the three tests under H0 in the regimes the real data qualify in (n = 4, 5; R near the
    5·(n−2) threshold). letter / position: R switches, each row drops 2 random letters and picks uniformly among
    the rest. majority: R switches in the pooled population, hits ~ Binomial(R, 1/(n−2)), Pearson df 1.
    Recorded, not tuned; a size well below 0.05 means the test is conservative there (low power). Fails only if a
    test is clearly anti-conservative (> 0.08)."""
    rng = np.random.default_rng(seed)
    res = []
    for n, R in MC_REGIMES:
        rej = {"letter": 0, "position": 0, "majority": 0}
        for _ in range(n_sims):
            supp = np.ones((R, n))
            obs = np.zeros((R, n))
            pos = np.zeros(n - 2)
            for r in range(R):
                supp[r, rng.choice(n, size=2, replace=False)] = 0
                allowed = np.flatnonzero(supp[r])
                j = rng.integers(0, n - 2)
                obs[r, allowed[j]] = 1
                pos[j] += 1
            w = wald_heterogeneous_uniform(supp, obs)
            rej["letter"] += w["p"] < 0.05
            rej["position"] += pearson_uniform(pos)[2] < 0.05
            k = rng.binomial(R, 1 / (n - 2))
            e = R / (n - 2)
            rej["majority"] += pearson_expected([k, R - k], [e, R - e])[2] < 0.05
        rec = {"n_options": n, "switches": R, "n_sims": n_sims,
               **{f"size_{t}": v / n_sims for t, v in rej.items()}}
        for t in rej:
            if rec[f"size_{t}"] > 0.08:
                raise AssertionError(f"{t} anti-conservative at n={n}, R={R}: {rec[f'size_{t}']}")
        res.append(rec)
    return res


if __name__ == "__main__":
    r = selftest(monte_carlo="--mc" in sys.argv)
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("sf(") or "q95" in k or k in
                      ("sf(3.84,1)", "sf(11.07,5)")}, indent=1, default=str))
    print(f"selftest ok ({len(r)} checks)")
