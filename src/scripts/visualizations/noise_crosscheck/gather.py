"""Noise cross-check — data. Three estimators of the noise share of to-target flips per cell.

Input: the alpha_vs_measured_noise gather's ``rows.csv`` (wrong→wrong dropped, a truncated hinted rollout
never flips), the 5 paper models only, through ``common.alpha_common``.
Per cell (model × dataset × style × manifest case), all three as SHARES of the cell's to-target flips:
  measured      literal = 1 − robust_used share of the to-target pairs          (Wilson CI)
                strict  = share of pairs whose non-truncated re-rolls never hit the target (Wilson CI)
  alpha-implied 1 − α = q/((n−2)p), clipped at 1                                (the alpha gather's Jeffreys CI)
  baseline-predicted = mean over the cell's ELIGIBLE rollouts (sample 0 answered, != target, not wrong→wrong) of
                (8 − stability)/8 × 1/(n − 1), divided by the cell's observed p = n_to_hint / n_eligible.
                (8 − stability)/8 = share of the 8 no-hint samples off the modal answer; spread uniformly over
                the n − 1 other options, 1/(n − 1) of it lands on the target. Not clipped (can exceed 1; counted).
The x order of the figure (cells sorted by measured noise) is recorded per definition.

Tables in <out-dir>/data/ (default <cueball>/plots/noise_crosscheck/data/):
  cells.csv               every cell with >= 1 to-target flip: counts, the estimators + CIs, sort ranks,
                          faded flags (n < 20 pairs), pred_base > 1 flag
  summary_by_model.csv    per model × case: cells, medians of each estimator, median differences,
                          cells where baseline-predicted > 1
  exclusions_by_model.csv where every rollout goes (sample-0 protocol), per model
  gather_meta.json        inputs (+ sha256), the code provenance, constants, self-test, checks

Run:
    python -m src.scripts.visualizations.noise_crosscheck.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--alpha-data D] [--no-cache | --refresh-cache] [--cache-dir D]
--alpha-data is the alpha_vs_measured_noise gather's data/ directory (default <cueball>/plots/alpha_vs_measured_noise/data,
or $ALPHA_DATA_DIR); the cache flags concern the stitched-rows cache the sample-0 re-parse check reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.scripts.visualizations.common import alpha_common as ac
from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import ssp_common as sc

PLOT = "noise_crosscheck"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None, help=f"default {P.DEFAULT_CUEBALL_DIR}/plots/{PLOT}")
    ap.add_argument("--only", default=None, help="debug subset: comma-separated model/run/dataset substrings")
    ap.add_argument("--alpha-data", default=None, help="the alpha_vs_measured_noise gather's data/ directory")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)
    cueball = P.cueball_dir(args.cueball_dir)
    if args.out_dir is None:
        # a subset (--only) never writes into the plot's real data/ folder
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if args.only else P.plot_dir(PLOT, cueball)
    else:
        out_dir = Path(args.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else (Path(args.cache_dir) if args.cache_dir else P.cache_dir(cueball))
    alpha = ac.alpha_paths(args.alpha_data, cueball=cueball)

    selftest = ac.selftest()
    rows, rep = ac.load_rows(args.only, data_dir=alpha.data, cueball=cueball)
    if args.refresh_cache and cache is not None:
        sc.load_rows(None, cache_dir=cache, refresh=True, cueball=cueball)
    reparse = ac.check_sample0_reparse(cueball=cueball, cache_dir=cache)
    cells = ac.cell_counts(rows, ac.CELL_KEYS)
    alpha_check = ac.check_against_alpha_cells(cells, data_dir=alpha.data, cueball=cueball)
    cells = ac.merge_alpha_x_ci(cells, data_dir=alpha.data, cueball=cueball)
    n_all = len(cells)
    cells = cells[cells["n_to_hint"] > 0].copy()
    checks = {"n_cells": n_all, "n_cells_p_zero_dropped": int(n_all - len(cells))}
    # direct recomputation of the baseline predictor on one pass over the eligible rows
    e = rows[rows["eligible"]]
    pb = (e.assign(v=(8 - e["baseline_stability"]) / 8 / (e["n_options"] - 1))
          .groupby(ac.CELL_KEYS)["v"].mean().rename("pb_check"))
    j = cells.join(pb, on=ac.CELL_KEYS)
    checks["pred_base_mean_recompute_max_abs_diff"] = float((j["pred_base_mean"] - j["pb_check"]).abs().max())
    if checks["pred_base_mean_recompute_max_abs_diff"] > 1e-12:
        raise ValueError(f"baseline predictor recomputation differs: {checks}")
    cells["pred_base_gt_1"] = cells["pred_base_share"] > 1
    for d in ("literal", "strict"):
        # x order: within each model, ascending measured noise (ties: dataset, style, case order)
        cells[f"rank_{d}"] = (cells.sort_values([f"y_{d}", "dataset", "hint_style", "case"])
                              .groupby("subject_model").cumcount())
    cells["subject_model"] = pd.Categorical(cells["subject_model"], ac.MODELS)
    cells = cells.sort_values(["subject_model", "rank_literal"])
    cells.to_csv(out / "cells.csv", index=False)

    summ = []
    for (m, case), g in cells.groupby(["subject_model", "case"], observed=True):
        summ.append({"subject_model": m, "case": case, "n_cells": int(len(g)),
                     "n_cells_faded_literal": int(g["faded_literal"].sum()),
                     "median_measured_literal": float(g["y_literal"].median()),
                     "median_measured_strict": float(g["y_strict"].median()),
                     "median_alpha_implied": float(g["x_implied"].median()),
                     "median_baseline_predicted": float(g["pred_base_share"].median()),
                     "median_literal_minus_implied": float((g["y_literal"] - g["x_implied"]).median()),
                     "median_strict_minus_implied": float((g["y_strict"] - g["x_implied"]).median()),
                     "median_literal_minus_predicted": float((g["y_literal"] - g["pred_base_share"]).median()),
                     "median_strict_minus_predicted": float((g["y_strict"] - g["pred_base_share"]).median()),
                     "median_predicted_minus_implied": float((g["pred_base_share"] - g["x_implied"]).median()),
                     "n_cells_pred_base_gt_1": int(g["pred_base_gt_1"].sum()),
                     "n_cells_alpha_clipped": int(g["alpha_clipped"].sum())})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary_by_model.csv", index=False)
    ac.exclusion_table(rows).to_csv(out / "exclusions_by_model.csv", index=False)
    checks["n_cells_pred_base_gt_1"] = int(cells["pred_base_gt_1"].sum())
    script = P.repo_path(f"src/scripts/visualizations/{PLOT}/gather.py")
    meta = {
        "written_utc": ac.now_utc(), "plot": PLOT, "script": f"src.scripts.visualizations.{PLOT}.gather",
        "script_sha256": ac.sha256_of(script) if script.exists() else None, "cueball_dir": str(cueball),
        "only": args.only, "alpha_common": ac.lib_info(),
        "inputs": {"rows_csv": ac.file_info(alpha.rows_csv, sha=True), "alpha_cells": ac.file_info(alpha.cells_csv, sha=True),
                   "alpha_gather_meta": ac.file_info(alpha.meta_json, sha=True)},
        "constants": {"MIN_N": ac.MIN_N, "models": ac.MODELS, "excluded_models": ac.EXCLUDED_MODELS,
                      "stability_sentence": ac.STABILITY_SENTENCE},
        "rows_report": rep, "sample0_reparse_check": reparse, "alpha_cell_check": alpha_check, "checks": checks,
        "selftest": selftest,
    }
    (out / "gather_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        print(summ.to_string(index=False))
    print(json.dumps(checks))
    print(f"{len(cells)} cells → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
