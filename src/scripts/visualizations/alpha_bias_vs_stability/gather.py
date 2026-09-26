"""Alpha bias vs baseline stability — data. Does alpha's error concentrate in unstable questions?

Input: the per-rollout ``rows.csv`` of the alpha_vs_measured_noise gather (wrong→wrong dropped, a truncated
hinted rollout never flips), the 5 paper models only, loaded through ``common.alpha_common`` (which re-derives
every protocol flag and checks it against the stored one). Per model × manifest case × stability bin (manifest
``baseline_stability`` = n_top_votes of the 8 no-hint samples; the tree has only 5..8, 5 read as "≤ 5/8"),
pooling every dataset and style inside the bin:

  implied   = Σ_rows q_to_other/(n − 2)  /  Σ_rows p_to_hint     (the implied-noise COUNTS pooled, then clipped
              at 1 — never an average of per-cell alphas, since n differs by dataset). = 1 − α of the bin.
  literal   = 1 − n_robust_used / n_pairs_labeled                 (y of alpha_vs_measured_noise)
  strict    = n_zero_strict / n_pairs_strict                      (pairs whose non-truncated re-rolls never hit
              the target, over pairs with >= 1 non-truncated re-roll)
  diff_literal = literal − implied,  diff_strict = strict − implied   (the plotted y)

95 % intervals of the differences: question-cluster bootstrap inside each (model, case, bin) — a question
(model-specific: one run per dataset) carries one stability value and one case, so it lives in exactly one
group (checked); its hint styles are resampled together. N_BOOT draws, fixed seed.

Tables in <out-dir>/data/:
  bins.csv                model × case × bin: every count, the three estimators, both differences + CIs, Wilson CIs
                          of the measured shares, faded flags (n < 20 pairs)
  cells.csv               model × dataset × style × case × bin: the same counts per cell (audit trail)
  exclusions_by_model.csv where every rollout goes (sample-0 protocol), per model
  gather_meta.json        inputs (+ sha256), the repo git sha + code shas, constants, self-test, checks

Run:
    python -m src.scripts.visualizations.alpha_bias_vs_stability.gather [--cueball-dir D] [--out-dir D]
        [--alpha-data D] [--only SUBSTR] [--no-cache | --refresh-cache] [--cache-dir D]
Default out-dir <cueball>/plots/alpha_bias_vs_stability/; --alpha-data = the alpha_vs_measured_noise gather's data/
(default <cueball>/plots/alpha_vs_measured_noise/data, or $ALPHA_DATA_DIR).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.visualizations.common import alpha_common as ac
from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import ssp_common as sc

PLOT = "alpha_bias_vs_stability"
N_BOOT = 2000
BOOT_SEED = 20260925
CHUNK = 200
BIN_KEYS = ["subject_model", "case", "stability_bin"]
Q_COLS = ["w_implied", "p_to_hint", "pair_labeled", "pair_robust", "pair_strict_den", "pair_strict_zero"]


def estimators(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """s: [..., 6] sums in Q_COLS order → (diff_literal, diff_strict), NaN where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.minimum(1.0, s[..., 0] / s[..., 1])
        literal = 1 - s[..., 3] / s[..., 2]
        strict = s[..., 5] / s[..., 4]
    return literal - implied, strict - implied


def bootstrap(qsums: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = []
    for key, g in qsums.groupby(BIN_KEYS, sort=True):
        arr = g[Q_COLS].to_numpy(dtype=float)
        nq = len(arr)
        dl, ds = [], []
        for start in range(0, N_BOOT, CHUNK):
            b = min(CHUNK, N_BOOT - start)
            wts = rng.multinomial(nq, np.full(nq, 1.0 / nq), size=b).astype(float)
            s = wts @ arr
            a, c = estimators(s)
            dl.append(a); ds.append(c)
        dl, ds = np.concatenate(dl), np.concatenate(ds)
        rec = dict(zip(BIN_KEYS, key))
        for name, v in (("diff_literal", dl), ("diff_strict", ds)):
            ok = np.isfinite(v)
            rec[f"{name}_lo"] = float(np.percentile(v[ok], 2.5)) if ok.sum() else np.nan
            rec[f"{name}_hi"] = float(np.percentile(v[ok], 97.5)) if ok.sum() else np.nan
            rec[f"{name}_boot_defined"] = int(ok.sum())
        out.append(rec)
    return pd.DataFrame(out)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None, help=f"default <cueball>/plots/{PLOT}")
    ap.add_argument("--alpha-data", default=None,
                    help="the alpha_vs_measured_noise gather's data/ dir (default: $ALPHA_DATA_DIR, else "
                         "<cueball>/plots/alpha_vs_measured_noise/data)")
    ap.add_argument("--only", default=None, help="debug subset: comma-separated model/run/dataset substrings")
    ap.add_argument("--no-cache", action="store_true", help="rebuild the stitched rows in memory, write no cache")
    ap.add_argument("--refresh-cache", action="store_true", help="rebuild the stitched rows and overwrite the cache")
    ap.add_argument("--cache-dir", default=None, help="default <cueball>/plots/_cache")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cueball = P.cueball_dir(args.cueball_dir)
    if args.out_dir is None:
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if args.only else P.plot_dir(PLOT, cueball)
        if args.only:
            sc.log(f"--only without --out-dir: writing to the debug dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else (Path(args.cache_dir) if args.cache_dir else P.cache_dir(cueball))
    ap = ac.alpha_paths(args.alpha_data, cueball=cueball)

    selftest = ac.selftest()
    rows, rep = ac.load_rows(args.only, data_dir=args.alpha_data, cueball=cueball)
    reparse = ac.check_sample0_reparse(cueball=cueball, cache_dir=cache, refresh=args.refresh_cache)
    alpha_check = ac.check_against_alpha_cells(ac.cell_counts(rows, ac.CELL_KEYS), data_dir=args.alpha_data,
                                               cueball=cueball)
    rows["stability_bin"] = rows["baseline_stability"].clip(lower=5).astype(int)

    # a question (per model) must sit in one bin and one case
    per_q = rows.groupby(["subject_model", "question_id"]).agg(nb=("stability_bin", "nunique"),
                                                             nc=("case", "nunique"))
    checks = {"questions_in_more_than_one_bin": int((per_q["nb"] > 1).sum()),
              "questions_in_more_than_one_case": int((per_q["nc"] > 1).sum())}
    if any(checks.values()):
        raise ValueError(f"cluster assumption violated: {checks}")

    bins = ac.cell_counts(rows, BIN_KEYS)
    bins = bins.rename(columns={"x_implied": "implied", "x_implied_raw": "implied_raw", "y_literal": "literal",
                                "y_literal_lo": "literal_lo", "y_literal_hi": "literal_hi", "y_strict": "strict",
                                "y_strict_lo": "strict_lo", "y_strict_hi": "strict_hi"})
    bins["diff_literal"] = bins["literal"] - bins["implied"]
    bins["diff_strict"] = bins["strict"] - bins["implied"]
    qs = rows.groupby(BIN_KEYS + ["question_id"], sort=True)[Q_COLS].sum().reset_index()
    # the bootstrap's point estimate must equal the table's
    tot = qs.groupby(BIN_KEYS)[Q_COLS].sum()
    pl, ps = estimators(tot.to_numpy(dtype=float))
    chk = bins.set_index(BIN_KEYS).loc[tot.index]
    checks["bootstrap_point_vs_table_max_abs_diff"] = float(np.nanmax(np.abs(np.r_[
        pl - chk["diff_literal"].to_numpy(), ps - chk["diff_strict"].to_numpy()])))
    if checks["bootstrap_point_vs_table_max_abs_diff"] > 1e-12:
        raise ValueError(f"bootstrap point estimate differs from the table: {checks}")
    ci = bootstrap(qs, np.random.default_rng(BOOT_SEED))
    bins = bins.merge(ci, on=BIN_KEYS, how="left", validate="one_to_one")
    bins["stability_label"] = bins["stability_bin"].map(ac.STABILITY_LABELS)
    order = BIN_KEYS + ["stability_label", "n_options_set", "n_questions", "n_rollouts", "n_b0_unanswered",
                        "n_b0_is_target", "n_eligible", "n_to_hint", "n_to_other", "n_to_other_is_modal",
                        "implied_count", "implied_raw", "alpha_clipped", "implied",
                        "n_pairs_labeled", "n_robust_used", "literal", "literal_lo", "literal_hi",
                        "n_pairs_strict", "n_zero_strict", "strict", "strict_lo", "strict_hi",
                        "diff_literal", "diff_literal_lo", "diff_literal_hi", "diff_literal_boot_defined",
                        "diff_strict", "diff_strict_lo", "diff_strict_hi", "diff_strict_boot_defined",
                        "faded_literal", "faded_strict", "n_wrong_to_wrong"]
    bins = bins[order]
    bins["case"] = pd.Categorical(bins["case"], ac.CASES)
    bins["subject_model"] = pd.Categorical(bins["subject_model"], ac.MODELS)
    bins = bins.sort_values(BIN_KEYS)
    bins.to_csv(out / "bins.csv", index=False)

    cells = ac.cell_counts(rows, ac.CELL_KEYS + ["stability_bin"])
    cells.to_csv(out / "cells.csv", index=False)
    ac.exclusion_table(rows).to_csv(out / "exclusions_by_model.csv", index=False)

    script = P.repo_path(f"src/scripts/visualizations/{PLOT}/gather.py")
    meta = {
        "written_utc": ac.now_utc(), "plot": PLOT, "script": f"src.scripts.visualizations.{PLOT}.gather",
        "script_sha256": ac.sha256_of(script) if script.exists() else None, "cueball_dir": str(cueball),
        "only": args.only, "alpha_common": ac.lib_info(),
        "inputs": {"rows_csv": ac.file_info(ap.rows_csv, sha=True), "alpha_cells": ac.file_info(ap.cells_csv, sha=True),
                   "alpha_gather_meta": ac.file_info(ap.meta_json, sha=True)},
        "constants": {"N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED, "MIN_N": ac.MIN_N, "bins": ac.STABILITY_LABELS,
                      "models": ac.MODELS, "excluded_models": ac.EXCLUDED_MODELS,
                      "stability_sentence": ac.STABILITY_SENTENCE},
        "rows_report": rep, "sample0_reparse_check": reparse, "alpha_cell_check": alpha_check, "checks": checks,
        "selftest": selftest,
    }
    (out / "gather_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    show = ["subject_model", "case", "stability_bin", "n_questions", "n_to_hint", "n_to_other", "implied",
            "n_pairs_labeled", "literal", "diff_literal", "diff_literal_lo", "diff_literal_hi", "n_pairs_strict",
            "strict", "diff_strict", "diff_strict_lo", "diff_strict_hi"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        print(bins[show].to_string(index=False))
    print(f"{len(bins)} bins, {len(cells)} cells → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
