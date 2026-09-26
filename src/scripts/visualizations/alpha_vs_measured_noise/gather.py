"""Alpha-implied noise vs measured noise — data. One row per cell (model × run × style × case).

Rules: the 5 paper models only (Qwen3.6-27B dropped, counted); wrong→wrong rows dropped from the protocol
(manifest-positive rows whose sample 0 is answered, ≠ target and ≠ groundtruth are not eligible; counted as
``n_wrong_to_wrong``); a truncated hinted rollout is never a flip (neither to the target nor to another option)
but stays eligible, like a hinted rollout without a parsed answer (counted: ``n_truncated_eligible``,
``n_truncated_to_hint_excluded``, ``n_truncated_to_other_excluded``).

Reads only the paper tree (``--cueball-dir``) and writes into ``<out-dir>/data/``:

- ``rows.csv``   one row per hinted-once rollout (after the filters): the join keys, the single-sample-protocol
                 baseline (sample 0), the hinted answer, every flag counted into p / q, and the re-sampling
                 fields joined for the pair (role, reliance_label, k_to_hint_count, censored variant). Every cell
                 number can be recomputed from this file.
- ``cells.csv``  one row per cell (subject_model × run/dataset × hint_style × case): counts, n_options, p, q,
                 raw and clipped alpha, x = 1 − alpha, the measured noise y = 1 − robust_used share, CIs, the
                 modal-baseline comparison, the fade flags.
- ``gather_meta.json``  inputs (path, size, mtime, sha256), the repo git sha and the code deps' shas, filters,
                 rule constants, the checks run.

Definitions (NOTES.md has the reasoning):

  single-sample protocol: baseline answer b0 = sample_answers[0] of the no-hint baseline; the hinted answer
  h = the one hinted-once rollout's parsed answer (manifest model_answer). A rollout is *eligible* when b0 was
  parsed, b0 != target and the row is not wrong→wrong. Among eligible rollouts whose hinted rollout is not
  truncated:
      to_hint  = h == target            (h != b0 follows)
      to_other = h parsed, h != b0, h != target
  p = n_to_hint / n_eligible, q = n_to_other / n_eligible  (same denominator: the ratio q/p, and hence alpha,
  does not depend on it).
  alpha_raw = 1 − q / ((n − 2) p);  alpha = clip(alpha_raw, 0, 1);  x = 1 − alpha.  p == 0 → alpha undefined.
  y: over the to-target pairs of the protocol (eligible & to_hint), joined to resample/question_reliance.csv on
  rollout_id, pairs with a reliance_label:  y = 1 − n_robust_used / n_pairs_labeled.
  Strict companion: y_strict = share of pairs with 0 non-truncated re-rolls to the target, over pairs with >= 1
  non-truncated re-roll.

Run:
    python -m src.scripts.visualizations.alpha_vs_measured_noise.gather [--cueball-dir D] [--out-dir D]
        [--only SUBSTR[,SUBSTR]] [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/alpha_vs_measured_noise/data/). Heavy: reads every run's
baseline CSV (usecols). This gather keeps no row cache; the cache flags are accepted for a uniform CLI only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib.resample import (
    DROPPED_HINTS,
    N_OPTIONS_BY_DATASET,
    ROBUST_MIN,
    SMOKE_RUN_MARKER,
    filter_manifest,
    wilson_interval,
)
from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import ssp_common as sc

PLOT = "alpha_vs_measured_noise"
CODE_DEPS = ["src/lib/resample.py", f"src/scripts/visualizations/{PLOT}/gather.py"]

CELL_KEYS = ["subject_model", "run", "dataset", "hint_style", "case"]
ROW_KEYS = ["subject_model", "run", "original_index", "hint_style"]

# Fade rule: a cell is drawn at full strength when at least this many of its protocol to-target pairs carry a
# reliance label (the y denominator; it is <= n_to_hint, the p numerator). The strict figure uses the same rule
# on its own denominator (pairs with >= 1 non-truncated re-roll).
MIN_PAIRS = 20
MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
EXCLUDED_MODELS = ["qwen3.6-27b"]
# The paper recipe's generation budget; re-rolls generated under a larger budget are censored as truncated
# past this many CoT tokens in the censored variant.
CENSOR_TOKENS = 16384
N_BOOT = 2000
BOOT_SEED = 20260925

MANIFEST_COLS = [
    "rollout_id", "question_id", "dataset", "dataset_split", "original_index", "subject_model", "run",
    "hint_style", "case", "target_option", "groundtruth", "baseline_modal_answer", "baseline_stability",
    "model_answer", "changed", "to_hint", "truncated", "exclude_reason", "source_csv",
]
RELIANCE_COLS = ["rollout_id", "subject_model", "run", "original_index", "hint_style", "role",
                 "reliance_label", "k_n", "k_to_hint_count", "k_truncated"]


def load_baseline(path: Path, n_expected: int) -> pd.DataFrame:
    """original_index, b0 (sample 0 letter or None), n_options, modal baseline_answer."""
    b = pd.read_csv(path, usecols=["original_index", "choices", "sample_answers", "baseline_answer"],
                    dtype=str, keep_default_na=False)
    if b["original_index"].duplicated().any():
        raise ValueError(f"{path}: duplicate original_index")
    widths = set(b["choices"].map(lambda c: len(json.loads(c))))
    if widths != {n_expected}:
        raise ValueError(f"{path}: choice widths {sorted(widths)} != registered {n_expected}")
    samples = b["sample_answers"].map(lambda s: json.loads(s) if s.strip() else [])
    n_samples = set(samples.map(len))
    if n_samples != {8}:
        raise ValueError(f"{path}: sample_answers lengths {sorted(n_samples)} (expected 8)")
    # a JSON null / "" / non-string first sample is unanswered (never str(None) -> "NONE")
    b0 = samples.map(lambda xs: xs[0].strip().upper() if xs and isinstance(xs[0], str) and xs[0].strip() else None)
    return pd.DataFrame({
        "original_index": b["original_index"].astype(int),
        "b0": b0.astype(object),
        "baseline_answer_csv": b["baseline_answer"].replace("", None).astype(object),
        "n_options": n_expected,
    })


def censored_counts(resample_manifest: Path, rollout_ids: set[str]) -> pd.DataFrame:
    """Per selected rollout: k_to_hint recomputed from the re-rolls, raw and with re-rolls whose trace exceeds
    CENSOR_TOKENS under a budget above it treated as truncated."""
    cols = ["rollout_id", "source_rollout_id", "provenance", "to_hint", "trace_token_len", "max_tokens_used"]
    r = pd.read_parquet(resample_manifest, columns=cols)
    r = r[(r["provenance"] == "resample_k4") & r["source_rollout_id"].isin(rollout_ids)]
    to_hint = r["to_hint"].fillna(False).astype(bool)
    over = (pd.to_numeric(r["max_tokens_used"], errors="coerce") > CENSOR_TOKENS) & (
        pd.to_numeric(r["trace_token_len"], errors="coerce") > CENSOR_TOKENS)
    g = pd.DataFrame({"source_rollout_id": r["source_rollout_id"].astype(str),
                      "k_rr": 1, "k_to_hint_rr": to_hint.astype(int),
                      "k_to_hint_censored": (to_hint & ~over).astype(int),
                      "k_censored_to_hint": (to_hint & over).astype(int)})
    return g.groupby("source_rollout_id").sum()


def boot_x(n_elig: int, n_hint: int, n_other: int, n_opt: int, rng: np.random.Generator) -> tuple[float, float]:
    """95 % interval of x from independent Jeffreys posteriors, p ~ Beta(n_hint + ½, n_elig − n_hint + ½)
    and q ~ Beta(n_other + ½, n_elig − n_other + ½), x = min(1, q / ((n − 2) p)) per draw. Unlike a
    plug-in bootstrap it gives a non-zero upper bound when n_other = 0."""
    if n_elig == 0 or n_hint == 0:
        return (np.nan, np.nan)
    p = rng.beta(n_hint + 0.5, n_elig - n_hint + 0.5, size=N_BOOT)
    q = rng.beta(n_other + 0.5, n_elig - n_other + 0.5, size=N_BOOT)
    x = np.minimum(1.0, q / ((n_opt - 2) * p))
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR, help="the paper tree")
    ap.add_argument("--out-dir", default=None, help="default <cueball>/plots/alpha_vs_measured_noise")
    ap.add_argument("--only", default=None,
                    help="debug: comma-separated substrings; keep runs whose subject_model or run contains one")
    ap.add_argument("--no-cache", action="store_true", help="accepted for a uniform CLI (this gather keeps no cache)")
    ap.add_argument("--refresh-cache", action="store_true", help="accepted for a uniform CLI (no cache)")
    ap.add_argument("--cache-dir", default=None, help="accepted for a uniform CLI (no cache)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cueball = P.cueball_dir(args.cueball_dir)
    manifest = P.manifest_path(cueball)
    manifest_meta = P.manifest_meta_path(cueball)
    reliance = P.reliance_path(cueball)
    resample_manifest = P.resample_manifest_path(cueball)
    noise_model_cells = cueball / "resample" / "noise_model_cells.csv"
    if args.out_dir is None:
        # a subset (--only) never writes into the plot's real data/ folder
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if args.only else P.plot_dir(PLOT, cueball)
        if args.only:
            print(f"--only without --out-dir: writing to the debug dir {out_dir}", flush=True)
    else:
        out_dir = Path(args.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {}

    # ---- 1. manifest, filtered like the resample scripts (smoke runs, dropped styles) ----
    m_all = pd.read_parquet(manifest, columns=MANIFEST_COLS)
    m = filter_manifest(m_all)
    checks["manifest_rows"] = int(len(m_all))
    checks["manifest_rows_after_filter"] = int(len(m))
    excl_models = m["subject_model"].isin(EXCLUDED_MODELS)
    checks["manifest_rows_excluded_models"] = int(excl_models.sum())
    m = m[~excl_models]
    if set(m["subject_model"]) != set(MODELS):
        raise ValueError(f"models after the exclusion: {sorted(set(m['subject_model']))} != {MODELS}")
    if args.only:  # debug subset: comma-separated substrings of subject_model or run
        keep = pd.Series(False, index=m.index)
        for tok in [t for t in args.only.split(",") if t]:
            keep |= m["subject_model"].str.contains(tok, regex=False) | m["run"].str.contains(tok, regex=False)
        m = m[keep]
    m = m.copy()
    m["original_index"] = m["original_index"].astype(int)
    for c in ["subject_model", "run", "dataset", "hint_style", "case", "target_option", "groundtruth",
              "baseline_modal_answer", "model_answer", "rollout_id"]:
        m[c] = m[c].astype(object).where(m[c].notna(), None)
    if m.duplicated(ROW_KEYS).any():
        raise ValueError("manifest: duplicate (subject_model, run, original_index, hint_style)")
    # case is fixed by the cue direction: negative ⇔ target == groundtruth
    case_ok = ((m["case"] == "negative") == (m["target_option"] == m["groundtruth"])).all()
    if not case_ok:
        raise ValueError("case does not match target == groundtruth")
    checks["case_equals_target_is_groundtruth"] = True

    # ---- 2. baselines: sample 0 per (subject_model, run, original_index) ----
    meta = json.loads(manifest_meta.read_text())
    runs = set(zip(m["subject_model"], m["run"]))
    parts, baseline_files = [], []
    for src in meta["sources"]:
        key = (src["subject_model"], src["run"])
        if key not in runs:
            continue
        path = Path(src["baseline_csv"])
        if not str(path).startswith(str(cueball)):
            raise ValueError(f"{key}: baseline outside the paper tree {cueball}: {path}")
        n_opt = N_OPTIONS_BY_DATASET[src["dataset"]]
        b = load_baseline(path, n_opt)
        b["subject_model"], b["run"] = key
        parts.append(b)
        baseline_files.append({**sc.file_info(path, sha=True), "subject_model": key[0], "run": key[1],
                               "dataset": src["dataset"], "n_options": n_opt, "n_questions": int(len(b))})
        print(f"  baseline {path.name}: {len(b)} q, n_options {n_opt}", flush=True)
    missing_runs = runs - {(p_["subject_model"].iloc[0], p_["run"].iloc[0]) for p_ in parts}
    if missing_runs:
        raise ValueError(f"no baseline found for runs {sorted(missing_runs)}")
    base = pd.concat(parts, ignore_index=True)
    rows = m.merge(base, on=["subject_model", "run", "original_index"], how="left", validate="many_to_one",
                   indicator=True)
    n_unjoined = int((rows["_merge"] != "both").sum())
    if n_unjoined:
        raise ValueError(f"{n_unjoined} rollouts have no baseline row")
    rows = rows.drop(columns="_merge")
    modal_mismatch = int((rows["baseline_answer_csv"].fillna("") != rows["baseline_modal_answer"].fillna("")).sum())
    checks["baseline_modal_answer_mismatch_vs_csv"] = modal_mismatch
    if modal_mismatch:
        raise ValueError(f"{modal_mismatch} rows: manifest baseline_modal_answer != baseline CSV baseline_answer")

    # ---- 3. single-sample protocol flags ----
    b0, tgt, h = rows["b0"], rows["target_option"], rows["model_answer"]
    rows["b0_answered"] = b0.notna()
    rows["b0_is_target"] = rows["b0_answered"] & (b0 == tgt)
    rows["b0_is_modal"] = rows["b0_answered"] & (b0 == rows["baseline_modal_answer"])
    # wrong→wrong (manifest positive, sample 0 answered and wrong, and not the target — a b0 == target row
    # stays b0_is_target) is dropped from the protocol
    rows["wrong_to_wrong"] = ((rows["case"] == "positive") & rows["b0_answered"] & ~rows["b0_is_target"]
                              & (b0 != rows["groundtruth"]))
    rows["eligible"] = rows["b0_answered"] & ~rows["b0_is_target"] & ~rows["wrong_to_wrong"]
    rows["hinted_answered"] = h.notna()
    # a truncated hinted rollout is never a flip (it stays eligible, like an unparsed hinted answer)
    trunc = rows["truncated"].fillna(False).astype(bool)
    rows["truncated_eligible"] = rows["eligible"] & trunc
    rows["truncated_to_hint_excluded"] = rows["truncated_eligible"] & rows["hinted_answered"] & (h == tgt)
    rows["truncated_to_other_excluded"] = (rows["truncated_eligible"] & rows["hinted_answered"] & (h != tgt)
                                           & (h != b0))
    usable = rows["hinted_answered"] & ~trunc
    rows["p_to_hint"] = rows["eligible"] & usable & (h == tgt)
    rows["q_to_other"] = rows["eligible"] & usable & (h != tgt) & (h != b0)
    # q rows where a minority sample 0 (b0 != modal) is "left" for the modal answer
    rows["q_to_modal"] = rows["q_to_other"] & (h == rows["baseline_modal_answer"])
    rows["same_as_b0"] = rows["eligible"] & usable & (h == b0)
    # modal-baseline counterparts (the manifest's own definition; comparison only)
    rows["modal_to_hint"] = rows["to_hint"].fillna(False).astype(bool)
    rows["modal_off_target"] = rows["changed"].fillna(False).astype(bool) & ~rows["modal_to_hint"]
    # every protocol to-target pair is a modal to_hint pair (target is never the modal answer)
    not_subset = int((rows["p_to_hint"] & ~rows["modal_to_hint"]).sum())
    checks["protocol_to_hint_not_in_modal_to_hint"] = not_subset
    if not_subset:
        raise ValueError(f"{not_subset} protocol to-target rows are not manifest to_hint rows")

    # ---- 4. re-sampling: reliance labels (+ censored recount) joined on rollout_id ----
    rel = pd.read_csv(reliance, usecols=RELIANCE_COLS, dtype={"original_index": int})
    if rel["rollout_id"].duplicated().any():
        raise ValueError("question_reliance.csv: duplicate rollout_id")
    rows = rows.merge(rel.rename(columns={c: f"rel_{c}" for c in RELIANCE_COLS if c != "rollout_id"}),
                      on="rollout_id", how="left", validate="one_to_one")
    hit = rows["rel_role"].notna()
    key_mismatch = int((hit & ((rows["rel_subject_model"] != rows["subject_model"]) | (rows["rel_run"] != rows["run"])
                               | (rows["rel_original_index"] != rows["original_index"])
                               | (rows["rel_hint_style"] != rows["hint_style"]))).sum())
    checks["reliance_key_mismatch"] = key_mismatch
    if key_mismatch:
        raise ValueError(f"{key_mismatch} reliance rows disagree with the manifest on the join keys")
    rows = rows.drop(columns=["rel_subject_model", "rel_run", "rel_original_index", "rel_hint_style"])
    wrong_role = int((rows["p_to_hint"] & hit & (rows["rel_role"] != "used_candidate")).sum())
    checks["protocol_pairs_with_non_used_role"] = wrong_role
    if wrong_role:
        raise ValueError(f"{wrong_role} protocol to-target pairs are re-sampled as controls")

    cc = censored_counts(resample_manifest, set(rows.loc[hit, "rollout_id"]))
    rows = rows.merge(cc, left_on="rollout_id", right_index=True, how="left")
    k_disagree = int((hit & (rows["k_to_hint_rr"].fillna(-1) != rows["rel_k_to_hint_count"])).sum())
    checks["k_to_hint_recount_disagrees_with_question_reliance"] = k_disagree
    if k_disagree:
        raise ValueError(f"{k_disagree} pairs: re-roll recount of k_to_hint != question_reliance")
    labeled = rows["rel_reliance_label"].notna()
    rows["pair"] = rows["p_to_hint"]
    rows["pair_labeled"] = rows["pair"] & labeled
    rows["pair_robust"] = rows["pair_labeled"] & (rows["rel_reliance_label"] == "robust_used")
    rows["pair_weak"] = rows["pair_labeled"] & (rows["rel_reliance_label"] == "weak_used")
    rows["pair_zero"] = rows["pair_labeled"] & (rows["rel_k_to_hint_count"] == 0)
    rows["pair_any_trunc_reroll"] = rows["pair_labeled"] & (rows["rel_k_truncated"].fillna(0) > 0)
    # strict variant: 0 of the *non-truncated* re-rolls to the target, over pairs with >= 1 non-truncated re-roll
    rows["pair_strict_den"] = rows["pair_labeled"] & (rows["rel_k_truncated"].fillna(0) < rows["rel_k_n"].fillna(0))
    rows["pair_strict_zero"] = rows["pair_strict_den"] & (rows["rel_k_to_hint_count"] == 0)
    rows["pair_robust_censored"] = rows["pair_labeled"] & (rows["k_to_hint_censored"].fillna(0) >= ROBUST_MIN)
    # modal-protocol pairs (every manifest to_hint row = the re-sampled used candidates)
    rows["mpair_labeled"] = rows["modal_to_hint"] & labeled
    rows["mpair_robust"] = rows["mpair_labeled"] & (rows["rel_reliance_label"] == "robust_used")

    # ---- 5. cells ----
    flags = ["eligible", "b0_answered", "b0_is_target", "b0_is_modal", "wrong_to_wrong", "truncated_eligible",
             "truncated_to_hint_excluded", "truncated_to_other_excluded", "hinted_answered", "p_to_hint",
             "q_to_other", "q_to_modal", "same_as_b0", "modal_to_hint", "modal_off_target", "pair", "pair_labeled",
             "pair_robust", "pair_weak", "pair_zero", "pair_any_trunc_reroll", "pair_robust_censored",
             "pair_strict_den", "pair_strict_zero", "mpair_labeled", "mpair_robust"]
    work = rows[CELL_KEYS + flags + ["n_options"]].copy()
    work[flags] = work[flags].astype(int)
    agg = {f: "sum" for f in flags}
    agg["n_options"] = "first"
    cells = work.groupby(CELL_KEYS, sort=True).agg(agg)
    cells.insert(0, "n_rollouts", work.groupby(CELL_KEYS, sort=True).size())
    if (work.groupby(CELL_KEYS)["n_options"].nunique() != 1).any():
        raise ValueError("a cell mixes option widths")
    cells = cells.rename(columns={
        "eligible": "n_eligible", "b0_answered": "n_b0_answered", "b0_is_target": "n_b0_is_target",
        "wrong_to_wrong": "n_wrong_to_wrong", "truncated_eligible": "n_truncated_eligible",
        "truncated_to_hint_excluded": "n_truncated_to_hint_excluded",
        "truncated_to_other_excluded": "n_truncated_to_other_excluded",
        "b0_is_modal": "n_b0_is_modal", "hinted_answered": "n_hinted_answered", "p_to_hint": "n_to_hint",
        "q_to_other": "n_to_other", "q_to_modal": "n_to_other_is_modal",
        "pair_strict_den": "n_pairs_strict", "pair_strict_zero": "n_zero_strict", "same_as_b0": "n_same_as_b0", "modal_to_hint": "n_to_hint_modal",
        "modal_off_target": "n_off_target_modal", "pair": "n_pairs", "pair_labeled": "n_pairs_labeled",
        "pair_robust": "n_robust_used", "pair_weak": "n_weak_used", "pair_zero": "n_zero_of_4",
        "pair_any_trunc_reroll": "n_pairs_with_truncated_reroll", "pair_robust_censored": "n_robust_used_censored",
        "mpair_labeled": "n_pairs_labeled_modal", "mpair_robust": "n_robust_used_modal",
    })
    cells["n_b0_unanswered"] = cells["n_rollouts"] - cells["n_b0_answered"]
    cells["n_pairs_unlabeled"] = cells["n_pairs"] - cells["n_pairs_labeled"]
    nm2 = (cells["n_options"] - 2).astype(float)
    elig = cells["n_eligible"].replace(0, np.nan)
    cells["p"] = cells["n_to_hint"] / elig
    cells["q"] = cells["n_to_other"] / elig
    with np.errstate(divide="ignore", invalid="ignore"):
        cells["noise_share_raw"] = np.where(cells["n_to_hint"] > 0,
                                            cells["n_to_other"] / (nm2 * cells["n_to_hint"]), np.nan)
    cells["alpha_raw"] = 1 - cells["noise_share_raw"]
    cells["alpha"] = cells["alpha_raw"].clip(lower=0.0, upper=1.0)
    cells["alpha_clipped"] = cells["alpha_raw"] < 0
    cells["x_alpha_noise"] = 1 - cells["alpha"]
    rng = np.random.default_rng(BOOT_SEED)
    ci = [boot_x(int(e), int(t), int(o), int(n), rng)
          for e, t, o, n in zip(cells["n_eligible"], cells["n_to_hint"], cells["n_to_other"], cells["n_options"])]
    cells["x_lo"], cells["x_hi"] = [c[0] for c in ci], [c[1] for c in ci]

    lab = cells["n_pairs_labeled"].replace(0, np.nan)
    cells["robust_share"] = cells["n_robust_used"] / lab
    cells["y_measured_noise"] = 1 - cells["robust_share"]
    wil = [wilson_interval(int(r), int(n)) if n else (np.nan, np.nan)
           for r, n in zip(cells["n_robust_used"], cells["n_pairs_labeled"])]
    cells["y_lo"] = [1 - w[1] for w in wil]
    cells["y_hi"] = [1 - w[0] for w in wil]
    cells["y_measured_noise_censored"] = 1 - cells["n_robust_used_censored"] / lab
    cells["zero_of_4_share"] = cells["n_zero_of_4"] / lab
    sden = cells["n_pairs_strict"].replace(0, np.nan)
    cells["y_strict"] = cells["n_zero_strict"] / sden
    wil_s = [wilson_interval(int(z), int(n)) if n else (np.nan, np.nan)
             for z, n in zip(cells["n_zero_strict"], cells["n_pairs_strict"])]
    cells["y_strict_lo"] = [w[0] for w in wil_s]
    cells["y_strict_hi"] = [w[1] for w in wil_s]
    # modal-baseline comparison (the existing noise model's convention)
    with np.errstate(divide="ignore", invalid="ignore"):
        cells["x_modal"] = np.where(cells["n_to_hint_modal"] > 0,
                                    np.minimum(1.0, cells["n_off_target_modal"] / (nm2 * cells["n_to_hint_modal"])),
                                    np.nan)
    cells["y_modal"] = 1 - cells["n_robust_used_modal"] / cells["n_pairs_labeled_modal"].replace(0, np.nan)
    # n_pairs_labeled <= n_to_hint, so this single condition is the whole rule
    # every cell with a y is plotted; cells below MIN_PAIRS are drawn FADED with n printed
    cells["faded"] = cells["n_pairs_labeled"] < MIN_PAIRS
    cells["faded_strict"] = cells["n_pairs_strict"] < MIN_PAIRS

    # cross-check the modal x against resample/noise_model_cells.csv (noise_share_of_to_hint)
    nm = pd.read_csv(noise_model_cells)
    nm = nm.set_index(CELL_KEYS)["noise_share_of_to_hint"]
    joined = cells[["x_modal"]].join(nm, how="left")
    diff = (joined["x_modal"] - joined["noise_share_of_to_hint"].clip(upper=1.0)).abs()
    checks["noise_model_cells_missing"] = int(joined["noise_share_of_to_hint"].isna().sum() - joined["x_modal"].isna().sum())
    checks["x_modal_vs_noise_model_max_abs_diff"] = float(np.nanmax(diff)) if diff.notna().any() else None
    checks["noise_model_cells_note"] = ("a nonzero diff can come from a manifest rebuild after "
                                        "noise_model_cells.csv was written")

    cells = cells.reset_index()
    order = CELL_KEYS + [
        "n_options", "n_rollouts", "n_b0_answered", "n_b0_unanswered", "n_b0_is_target", "n_b0_is_modal",
        "n_wrong_to_wrong", "n_eligible", "n_truncated_eligible", "n_truncated_to_hint_excluded",
        "n_truncated_to_other_excluded", "n_hinted_answered", "n_to_hint", "n_to_other", "n_to_other_is_modal", "n_same_as_b0", "p", "q",
        "noise_share_raw", "alpha_raw", "alpha", "alpha_clipped", "x_alpha_noise", "x_lo", "x_hi",
        "n_pairs", "n_pairs_labeled", "n_pairs_unlabeled", "n_robust_used", "n_weak_used", "n_zero_of_4",
        "n_pairs_with_truncated_reroll", "robust_share", "y_measured_noise", "y_lo", "y_hi",
        "zero_of_4_share", "n_pairs_strict", "n_zero_strict", "y_strict", "y_strict_lo", "y_strict_hi",
        "n_robust_used_censored", "y_measured_noise_censored",
        "n_to_hint_modal", "n_off_target_modal", "x_modal", "n_pairs_labeled_modal", "n_robust_used_modal",
        "y_modal", "faded", "faded_strict",
    ]
    cells = cells[order]
    cells.to_csv(out / "cells.csv", index=False)

    row_cols = ["rollout_id", "question_id", "subject_model", "run", "dataset", "dataset_split", "original_index",
                "hint_style", "case", "n_options", "target_option", "groundtruth", "baseline_modal_answer",
                "baseline_stability", "b0", "model_answer", "truncated", "exclude_reason", "b0_answered",
                "b0_is_target", "wrong_to_wrong", "eligible", "truncated_eligible", "truncated_to_hint_excluded",
                "truncated_to_other_excluded", "hinted_answered", "p_to_hint", "q_to_other", "q_to_modal", "same_as_b0",
                "modal_to_hint", "modal_off_target", "rel_role", "rel_reliance_label", "rel_k_n",
                "rel_k_to_hint_count", "rel_k_truncated", "k_to_hint_censored", "k_censored_to_hint",
                "pair", "pair_labeled", "pair_robust", "pair_strict_den", "pair_strict_zero", "source_csv"]
    rows[row_cols].sort_values(ROW_KEYS).to_csv(out / "rows.csv", index=False)

    checks["n_rows_out"] = int(len(rows))
    checks["n_wrong_to_wrong_dropped"] = int(rows["wrong_to_wrong"].sum())
    checks["n_truncated_eligible"] = int(rows["truncated_eligible"].sum())
    checks["n_truncated_to_hint_excluded"] = int(rows["truncated_to_hint_excluded"].sum())
    checks["n_truncated_to_other_excluded"] = int(rows["truncated_to_other_excluded"].sum())
    checks["n_cells"] = int(len(cells))
    checks["n_cells_faded"] = int((cells["faded"] & cells["y_measured_noise"].notna()).sum())
    checks["n_cells_faded_strict"] = int((cells["faded_strict"] & cells["y_strict"].notna()).sum())
    checks["n_cells_alpha_clipped"] = int(cells["alpha_clipped"].sum())
    checks["n_cells_p_zero"] = int((cells["n_to_hint"] == 0).sum())
    checks["n_protocol_pairs_without_reliance_row"] = int((rows["pair"] & ~hit).sum())
    checks["n_protocol_pairs_with_null_label"] = int((rows["pair"] & hit & ~labeled).sum())
    checks["n_censored_to_hint_rerolls"] = int(rows["k_censored_to_hint"].fillna(0).sum())
    meta_out = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plot": PLOT,
        "script": f"src.scripts.visualizations.{PLOT}.gather",
        "git_sha": sc.repo_git_sha(),
        "code_deps": sc.code_deps_provenance(CODE_DEPS),
        "cueball_dir": str(cueball),
        "only": args.only,
        "inputs": {"manifest": sc.file_info(manifest, sha=True), "manifest_meta": sc.file_info(manifest_meta, sha=True),
                   "question_reliance": sc.file_info(reliance, sha=True),
                   "resample_manifest": sc.file_info(resample_manifest, sha=True),
                   "noise_model_cells": sc.file_info(noise_model_cells, sha=True), "baselines": baseline_files},
        "filters": {"dropped_hint_styles": list(DROPPED_HINTS), "smoke_run_marker": SMOKE_RUN_MARKER},
        "rules": "wrong→wrong dropped, truncated hinted rollouts never flip, 5 paper models",
        "models": MODELS, "excluded_models": EXCLUDED_MODELS,
        "constants": {"MIN_PAIRS": MIN_PAIRS, "ROBUST_MIN": ROBUST_MIN, "CENSOR_TOKENS": CENSOR_TOKENS,
                      "N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED},
        "tables": ["cells.csv", "rows.csv"],
        "checks": checks,
    }
    (out / "gather_meta.json").write_text(json.dumps(meta_out, indent=1, default=str) + "\n")
    print(json.dumps(checks, indent=1, default=str))
    print(f"{len(rows)} rows, {len(cells)} cells ({checks['n_cells_faded']} faded) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
