"""Alpha negative-case check — data. The alpha-vs-measured comparison restricted to negative-case cells, where
the cue targets the correct answer: noise should favour the target there, so alpha (uniform noise over the n − 1
non-baseline options) should underestimate it most.

Input: the alpha_vs_measured_noise gather's ``data/rows.csv`` (wrong→wrong dropped, a truncated hinted rollout
never flips), the 5 paper models only, plus its ``cells.csv`` for x's 95 % interval (Jeffreys draws). Cells, x and
both y definitions are that gather's (re-aggregated from its rows and checked equal to its cells.csv):
  x         = 1 − α = q/((n−2)p), clipped at 1
  y_literal = 1 − robust_used share of the cell's to-target pairs            (Wilson CI)
  y_strict  = share of pairs whose non-truncated re-rolls never hit the target (Wilson CI)

Tables (in <out-dir>/data/):
  cells.csv                 negative-case cells (model × dataset × style): every count, x (+ CI), both y (+ CIs),
                            y − x, faded flags (n < 20 pairs)
  summary_by_model_case.csv per model × case (both cases, for the contrast): cells with pairs, median x,
                            median y, median y − x, cells with y > x, per y definition
  exclusions_by_model.csv   where every rollout goes (sample-0 protocol), per model
  gather_meta.json          inputs (+ sha256), the shared code's shas, the repo git sha, constants, self-test, checks

Run:
    python -m src.scripts.visualizations.alpha_negative_case_check.gather [--cueball-dir D] [--out-dir D]
        [--alpha-data D] [--only SUBSTR[,SUBSTR]] [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/alpha_negative_case_check/data/). --alpha-data is the
alpha_vs_measured_noise gather's data/ directory (default <cueball>/plots/alpha_vs_measured_noise/data/; the env var
ALPHA_DATA_DIR is the same override). --only keeps the rows whose model, run or dataset contains a substring
(debug; the cell check then needs an --alpha-data written with the same subset).
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

PLOT = "alpha_negative_case_check"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None, help=f"default <cueball>/plots/{PLOT}")
    ap.add_argument("--alpha-data", default=None, help="the alpha_vs_measured_noise gather's data/ directory")
    ap.add_argument("--only", default=None, help="debug subset: comma-separated model/run/dataset substrings")
    ap.add_argument("--no-cache", action="store_true", help="rebuild the stitched rows in memory, write no cache")
    ap.add_argument("--refresh-cache", action="store_true", help="rebuild the stitched rows and overwrite the cache")
    ap.add_argument("--cache-dir", default=None, help="default <cueball>/plots/_cache")
    args = ap.parse_args(argv)
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
    alpha = ac.alpha_paths(args.alpha_data, cueball=cueball)

    selftest = ac.selftest()
    rows, rep = ac.load_rows(args.only, data_dir=alpha.data, cueball=cueball)
    reparse = ac.check_sample0_reparse(cueball=cueball, cache_dir=cache, refresh=args.refresh_cache)
    cells = ac.cell_counts(rows, ac.CELL_KEYS)
    alpha_check = ac.check_against_alpha_cells(cells, data_dir=alpha.data, cueball=cueball)
    cells = ac.merge_alpha_x_ci(cells, data_dir=alpha.data, cueball=cueball)
    cells["diff_literal"] = cells["y_literal"] - cells["x_implied"]
    cells["diff_strict"] = cells["y_strict"] - cells["x_implied"]
    cells["has_pairs"] = cells["n_to_hint"] > 0

    summ = []
    for (m, case), g in cells.groupby(["subject_model", "case"]):
        g = g[g["has_pairs"]]
        rec = {"subject_model": m, "case": case, "n_cells_with_pairs": int(len(g)),
               "n_cells_alpha_clipped": int(g["alpha_clipped"].sum()), "median_x": float(g["x_implied"].median())}
        for d in ("literal", "strict"):
            ok = g[f"y_{d}"].notna()
            rec[f"n_cells_{d}"] = int(ok.sum())
            rec[f"n_cells_{d}_faded"] = int((ok & g[f"faded_{d}"]).sum())
            rec[f"median_y_{d}"] = float(g.loc[ok, f"y_{d}"].median())
            rec[f"median_diff_{d}"] = float(g.loc[ok, f"diff_{d}"].median())
            rec[f"n_cells_y_gt_x_{d}"] = int((g.loc[ok, f"diff_{d}"] > 0).sum())
            rec[f"n_cells_y_lt_x_{d}"] = int((g.loc[ok, f"diff_{d}"] < 0).sum())
            rec[f"n_cells_ci_disjoint_above_{d}"] = int((g.loc[ok, "x_hi"] < g.loc[ok, f"y_{d}_lo"]).sum())
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ["subject_model"] = pd.Categorical(summ["subject_model"], ac.MODELS)
    summ["case"] = pd.Categorical(summ["case"], ac.CASES)
    summ = summ.sort_values(["subject_model", "case"])
    summ.to_csv(out / "summary_by_model_case.csv", index=False)

    neg = cells[cells["case"] == "negative"].copy()
    neg.to_csv(out / "cells.csv", index=False)
    ac.exclusion_table(rows).to_csv(out / "exclusions_by_model.csv", index=False)
    checks = {"n_negative_cells": int(len(neg)), "n_negative_cells_p_zero": int((~neg["has_pairs"]).sum()),
              "n_negative_cells_alpha_clipped": int(neg["alpha_clipped"].sum())}
    script = P.repo_path(f"src/scripts/visualizations/{PLOT}/gather.py")
    meta = {
        "written_utc": ac.now_utc(), "plot": PLOT, "script": f"src.scripts.visualizations.{PLOT}.gather",
        "script_sha256": ac.sha256_of(script) if script.exists() else None, "git_sha": sc.repo_git_sha(),
        "cueball_dir": str(cueball), "only": args.only,
        "alpha_common": ac.lib_info(),
        "inputs": {"rows_csv": ac.file_info(alpha.rows_csv, sha=True),
                   "alpha_cells": ac.file_info(alpha.cells_csv, sha=True),
                   "alpha_gather_meta": ac.file_info(alpha.meta_json, sha=True)},
        "constants": {"MIN_N": ac.MIN_N, "models": ac.MODELS, "excluded_models": ac.EXCLUDED_MODELS},
        "rows_report": rep, "sample0_reparse_check": reparse, "alpha_cell_check": alpha_check, "checks": checks,
        "selftest": selftest,
    }
    (out / "gather_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        print(summ.to_string(index=False))
    print(json.dumps(checks))
    print(f"{len(neg)} negative cells → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
