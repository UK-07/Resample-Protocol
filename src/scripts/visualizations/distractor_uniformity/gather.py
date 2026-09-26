"""Distractor uniformity — data. Are third-option switches spread uniformly over the n − 2 distractors, as
alpha = 1 − q/((n−2)p) assumes?

Input: the alpha_vs_measured_noise gather's ``rows.csv`` (wrong→wrong dropped, truncated never flips), the 5
paper models only, plus its ``cells.csv`` as a consistency check. A third-option switch (the q numerator of alpha)
= an eligible hinted-once rollout (sample 0 answered, != target, not wrong→wrong) that is not truncated and whose
parsed hinted answer h is answered, != target and != sample 0. Its distractors = the run's n letters minus
{target, sample 0}: n − 2 of them, and h is one.

Three tests, all vs "each switch picks uniformly among its own n − 2 distractors":
  letter   — per cell (model × dataset × style × manifest case, both cases). Categories = option letters; rows have
             different distractor sets, so Wald W = (O − E)ᵀ Σ⁺ (O − E), E = Σ_r π_r, Σ = Σ_r (diag π_r − π_r π_rᵀ),
             df = rank Σ (exactly Pearson with df n − 3 when every row shares one set). Qualifies: >= 5·(n − 2)
             third-option switches. BH over the qualifying cells.
  position — per cell, same qualifying rule. Categories = rank of h among the row's distractors in letter order;
             Pearson, df n − 3. BH over the qualifying cells.
  majority — POOLED OVER MODELS: pooled cell = dataset × style × case. Population = the switches whose 8-sample
             modal (majority) no-hint answer is one of the row's distractors (sample 0 was a minority answer); the
             modal answer only NAMES a category (never a flip / label / case). Under uniformity P(h = modal) =
             1/(n − 2) for every row of a dataset, so rows of different models pool into one binomial. Pearson on
             {modal, other distractors}, df 1, plus the exact upper binomial tail. NEGATIVE only: once wrong→wrong
             is dropped a positive population is empty by construction (asserted). Qualifies: >= 5·(n − 2)
             switches in the pooled population (=> expected >= 5, asserted). BH over the qualifying pooled cells.
             Per-model population / hit counts are kept in the table.
Tests are computed for every cell with >= 1 switch (in the test's population); only qualifying cells enter the
figures, reject counts and BH.

Writes into <out-dir>/data/:
  cells.csv               per-model cells (every cell of the grid): counts, `qualifies`, letter / position tests
                          (stat, df, p, BH, reject flags), and the cell's majority-population counts (descriptive)
  majority_pooled.csv     pooled cells (dataset × style × case): population, hits, expected, Pearson + exact
                          binomial p, qualifies, BH, reject flags, per-model population / hit columns
  distractor_counts.csv   long: per-model cell × test (letter, position) × category → observed, expected
  summary.csv             per model × test (letter, position) and per dataset (majority): cells, qualifying,
                          rejecting at 0.05 / after BH, not-tested counts
  exclusions_by_model.csv where every rollout goes (sample-0 protocol), per model
  worked_example.json     the qualifying pooled cell with the most switches in the majority population (drawn), and
                          as `context_mmlu_pro_cell` the MMLU-Pro pooled cell with the most (context, not drawn)
  gather_meta.json        inputs (+ sha256), code provenance (alpha_common sha, repo git sha, code deps), constants,
                          self-test, checks

Run:
    python -m src.scripts.visualizations.distractor_uniformity.gather [--cueball-dir D] [--out-dir D]
        [--alpha-data D] [--only SUBSTR] [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/distractor_uniformity/data/). ``--alpha-data`` is the
alpha_vs_measured_noise gather's data/ directory (default <cueball>/plots/alpha_vs_measured_noise/data, or
``$ALPHA_DATA_DIR``); a ``--only`` subset needs an alpha data dir built with the same subset.
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

PLOT = "distractor_uniformity"
TESTS = ["letter", "position"]          # per-model cell tests; the majority test is pooled (majority_pooled.csv)
ALPHA_LEVEL = 0.05
WORKED_DATASET = "mmlu_pro"          # only for the context cell in worked_example.json
QRULE = {"letter": ">= 5*(n-2) third-option switches (per model cell)",
         "position": ">= 5*(n-2) third-option switches (per model cell)",
         "majority": "negative case, >= 5*(n-2) switches in the pooled (over models) majority population"}
POOL_KEYS = ["dataset", "hint_style", "case"]


def cell_tests(sub: pd.DataFrame, n: int) -> tuple[dict, list[dict]]:
    """Both tests for the third-option switches `sub` of one cell with n options."""
    letters = ac.LETTERS[:n]
    R = len(sub)
    supp = np.ones((R, n))
    obs = np.zeros((R, n))
    pos = np.zeros(n - 2)
    modal_in = modal_hit = 0
    for i, (t, b0, h, modal) in enumerate(zip(sub["target_option"], sub["b0"], sub["model_answer"],
                                              sub["baseline_modal_answer"])):
        supp[i, letters.index(t)] = 0
        supp[i, letters.index(b0)] = 0
        allowed = [L for L in letters if L not in (t, b0)]
        obs[i, letters.index(h)] = 1
        pos[allowed.index(h)] += 1
        if modal in allowed:
            modal_in += 1
            modal_hit += int(h == modal)
    w = ac.wald_heterogeneous_uniform(supp, obs)
    ps, pdf, pp = ac.pearson_uniform(pos)
    res = {"letter_stat": w["stat"], "letter_df": w["df"], "letter_p": w["p"],
           "position_stat": ps, "position_df": pdf, "position_p": pp,
           "n_third_modal_is_distractor": modal_in, "n_third_to_modal": modal_hit,
           "expected_third_to_modal": modal_in / (n - 2)}
    if modal_in:
        e_hit = modal_in / (n - 2)
        ms, mdf, mp = ac.pearson_expected([modal_hit, modal_in - modal_hit], [e_hit, modal_in - e_hit])
        res.update({"modal_stat": ms, "modal_df": mdf, "modal_p": mp,
                    "modal_min_expected": min(e_hit, modal_in - e_hit),
                    "modal_binom_p_upper": ac.binom_sf_ge(modal_hit, modal_in, 1 / (n - 2))})
    long = []
    for j, L in enumerate(letters):
        long.append({"test": "letter", "category": L, "category_order": j,
                     "observed": int(w["observed"][j]), "expected": float(w["expected"][j])})
    for j in range(n - 2):
        long.append({"test": "position", "category": f"d{j + 1}", "category_order": j,
                     "observed": int(pos[j]), "expected": R / (n - 2)})
    if modal_in:
        long.append({"test": "modal", "category": "modal", "category_order": 0,
                     "observed": int(modal_hit), "expected": modal_in / (n - 2)})
        long.append({"test": "modal", "category": "other", "category_order": 1,
                     "observed": int(modal_in - modal_hit), "expected": modal_in * (n - 3) / (n - 2)})
    return res, long


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR, help="the paper tree")
    ap.add_argument("--out-dir", default=None, help=f"default <cueball>/plots/{PLOT}; tables land in <out-dir>/data")
    ap.add_argument("--alpha-data", default=None,
                    help="the alpha_vs_measured_noise gather's data/ dir (default <cueball>/plots/alpha_vs_measured_noise"
                         "/data, or $ALPHA_DATA_DIR)")
    ap.add_argument("--only", default=None, help="debug subset: comma-separated model/run/dataset substrings")
    ap.add_argument("--no-cache", action="store_true", help="rebuild the stitched rows in memory, write no cache")
    ap.add_argument("--refresh-cache", action="store_true", help="rebuild the stitched rows and overwrite the cache")
    ap.add_argument("--cache-dir", default=None, help="default <cueball>/plots/_cache")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cueball = P.cueball_dir(args.cueball_dir)
    if args.out_dir is None:
        # a subset (--only) never writes into the plot's real data/ folder
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if args.only else P.plot_dir(PLOT, cueball)
        if args.only:
            print(f"--only without --out-dir: writing to the debug dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    # the stitched-rows cache backs the sample-0 re-parse check only; a subset never touches it
    if args.no_cache or args.only:
        cache: str | Path | None = None
    else:
        cache = Path(args.cache_dir) if args.cache_dir else P.cache_dir(cueball)
    alpha = ac.alpha_paths(args.alpha_data, cueball=cueball)

    selftest = ac.selftest(monte_carlo=True)
    rows, rep = ac.load_rows(args.only, data_dir=alpha.data, cueball=cueball)
    reparse = ac.check_sample0_reparse(cueball=cueball, cache_dir=cache, refresh=args.refresh_cache)
    base = ac.cell_counts(rows, ac.CELL_KEYS)
    alpha_check = ac.check_against_alpha_cells(base, data_dir=alpha.data, cueball=cueball)

    third = rows[rows["q_to_other"]]
    groups = {k: g for k, g in third.groupby(ac.CELL_KEYS, sort=False)}
    recs, longs = [], []
    for key, c in base.set_index(ac.CELL_KEYS).iterrows():
        n = int(c["n_options_set"])
        sub = groups.get(key, third.iloc[0:0])
        rec = dict(zip(ac.CELL_KEYS, key))
        rec.update({"n_options": n, "n_distractors": n - 2, "n_rollouts": int(c["n_rollouts"]),
                    "n_b0_unanswered": int(c["n_b0_unanswered"]), "n_b0_is_target": int(c["n_b0_is_target"]),
                    "n_eligible": int(c["n_eligible"]), "n_to_hint": int(c["n_to_hint"]),
                    "n_third": int(len(sub)), "threshold": 5 * (n - 2)})
        rec["qualifies"] = rec["n_third"] >= rec["threshold"]
        if len(sub):
            res, long = cell_tests(sub, n)
            rec.update(res)
            for x in long:
                longs.append({**dict(zip(ac.CELL_KEYS, key)), **x})
        recs.append(rec)
    cells = pd.DataFrame(recs)
    if int(cells["n_third"].sum()) != int(rows["q_to_other"].sum()):
        raise ValueError("third-option switches lost in the cell loop")
    m_pop = cells["n_third_modal_is_distractor"].fillna(0).astype(int)
    cells["n_modal_population"] = m_pop
    cells["n_modal_hits"] = cells["n_third_to_modal"].fillna(0).astype(int)
    pos_pop = int(m_pop[cells["case"] == "positive"].sum())
    if pos_pop:
        raise ValueError(f"{pos_pop} positive-case switches have the modal answer among the distractors")
    for t in TESTS:
        q = cells["qualifies"]
        cells[f"{t}_p_bh"] = np.nan
        cells.loc[q, f"{t}_p_bh"] = ac.benjamini_hochberg(cells.loc[q, f"{t}_p"].to_numpy())
        cells[f"{t}_reject"] = q & (cells[f"{t}_p"] < ALPHA_LEVEL)
        cells[f"{t}_reject_bh"] = q & (cells[f"{t}_p_bh"] < ALPHA_LEVEL)
    cells["faded"] = cells["n_third"] < ac.MIN_N
    drop = [c for c in cells.columns if c.startswith("modal_")] + ["n_third_modal_is_distractor", "n_third_to_modal",
                                                                    "expected_third_to_modal"]
    cells = cells.drop(columns=drop)   # per-model majority p-values are not part of the design (pooled instead)
    cells.to_csv(out / "cells.csv", index=False)
    pd.DataFrame(longs).to_csv(out / "distractor_counts.csv", index=False)

    # ---- majority test pooled over models: dataset × style × case (negative) ----
    neg = cells[cells["case"] == "negative"]
    pooled = (neg.groupby(POOL_KEYS, sort=True)
              .agg(n_options=("n_options", "first"), n_models=("subject_model", "nunique"),
                   n_third=("n_third", "sum"), n_population=("n_modal_population", "sum"),
                   n_hits=("n_modal_hits", "sum")).reset_index())
    if (neg.groupby(POOL_KEYS)["n_options"].nunique() != 1).any():
        raise ValueError("a pooled cell mixes option widths")
    pooled["threshold"] = 5 * (pooled["n_options"] - 2)
    pooled["expected_hits"] = pooled["n_population"] / (pooled["n_options"] - 2)
    stats = []
    for m, k, n in zip(pooled["n_population"], pooled["n_hits"], pooled["n_options"]):
        if m == 0:
            stats.append((np.nan, np.nan, np.nan, np.nan)); continue
        e = m / (n - 2)
        st, df, p = ac.pearson_expected([k, m - k], [e, m - e])
        stats.append((st, df, p, ac.binom_sf_ge(int(k), int(m), 1 / (n - 2))))
    pooled["majority_stat"], pooled["majority_df"], pooled["majority_p"], pooled["majority_binom_p_upper"] = zip(*stats)
    pooled["hit_share"] = pooled["n_hits"] / pooled["n_population"].replace(0, np.nan)
    pooled["uniform_share"] = 1 / (pooled["n_options"] - 2)
    pooled["qualifies"] = pooled["n_population"] >= pooled["threshold"]
    if (pooled["qualifies"] & (pooled["expected_hits"] < 5)).any():
        raise ValueError("a qualifying pooled cell has expected hits < 5")
    pooled["not_tested"] = (pooled["n_population"] > 0) & ~pooled["qualifies"]
    q = pooled["qualifies"]
    pooled["majority_p_bh"] = np.nan
    pooled.loc[q, "majority_p_bh"] = ac.benjamini_hochberg(pooled.loc[q, "majority_p"].to_numpy())
    pooled["majority_reject"] = q & (pooled["majority_p"] < ALPHA_LEVEL)
    pooled["majority_reject_bh"] = q & (pooled["majority_p_bh"] < ALPHA_LEVEL)
    pooled["faded"] = pooled["n_population"] < ac.MIN_N
    per_model = neg.pivot_table(index=POOL_KEYS, columns="subject_model",
                                values=["n_modal_population", "n_modal_hits"], aggfunc="sum", fill_value=0)
    per_model.columns = [f"{'pop' if a == 'n_modal_population' else 'hits'}__{m}" for a, m in per_model.columns]
    pooled = pooled.merge(per_model.reset_index(), on=POOL_KEYS, how="left")
    if int(pooled["n_population"].sum()) != int(m_pop.sum()):
        raise ValueError("majority population lost in pooling")
    pooled.to_csv(out / "majority_pooled.csv", index=False)

    summ = []
    for m in ac.MODELS + ["all"]:
        s = cells if m == "all" else cells[cells["subject_model"] == m]
        if s.empty:
            continue
        for t in TESTS:
            summ.append({"test": t, "group": m, "qualifying_rule": QRULE[t], "n_cells": int(len(s)),
                         "n_cells_with_switches": int((s["n_third"] > 0).sum()),
                         "n_switches_total": int(s["n_third"].sum()), "n_qualify": int(s["qualifies"].sum()),
                         "n_not_tested": int(((s["n_third"] > 0) & ~s["qualifies"]).sum()),
                         "n_reject_05": int(s[f"{t}_reject"].sum()), "n_reject_bh_05": int(s[f"{t}_reject_bh"].sum()),
                         "n_switches_in_qualifying": int(s.loc[s["qualifies"], "n_third"].sum())})
    for d in ac.DATASETS + ["all"]:
        s = pooled if d == "all" else pooled[pooled["dataset"] == d]
        if s.empty:
            continue
        summ.append({"test": "majority", "group": d, "qualifying_rule": QRULE["majority"], "n_cells": int(len(s)),
                     "n_cells_with_switches": int((s["n_population"] > 0).sum()),
                     "n_switches_total": int(s["n_population"].sum()), "n_qualify": int(s["qualifies"].sum()),
                     "n_not_tested": int(s["not_tested"].sum()),
                     "n_reject_05": int(s["majority_reject"].sum()), "n_reject_bh_05": int(s["majority_reject_bh"].sum()),
                     "n_switches_in_qualifying": int(s.loc[s["qualifies"], "n_population"].sum()),
                     "n_hits_total": int(s["n_hits"].sum()), "expected_hits_total": float(s["expected_hits"].sum())})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    ac.exclusion_table(rows).to_csv(out / "exclusions_by_model.csv", index=False)

    def as_dict(r):
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
    # worked example: the QUALIFYING pooled cell with the most population switches (ties: dataset, style); the
    # MMLU-Pro cell with the most population switches is kept as context only
    alt = pooled[pooled["qualifies"]].sort_values(["n_population", "dataset", "hint_style"],
                                                  ascending=[False, True, True])
    worked = None
    if len(alt):
        worked = as_dict(alt.iloc[0])
        worked["rule"] = ("the qualifying pooled cell (dataset × style × case, over models) with the most switches in "
                          "the majority population (ties: dataset, style)")
        worked["runner_up_n_population"] = int(alt["n_population"].iloc[1]) if len(alt) > 1 else None
        wm = pooled[(pooled["dataset"] == WORKED_DATASET) & (pooled["n_population"] > 0)].sort_values(
            ["n_population", "hint_style"], ascending=[False, True])
        if len(wm):
            worked["context_mmlu_pro_cell"] = as_dict(wm.iloc[0])
            worked["context_mmlu_pro_cell"]["rule"] = ("context only, not drawn: the MMLU-Pro pooled cell with the "
                                                       "most population switches")
    (out / "worked_example.json").write_text(json.dumps(worked, indent=1, default=str) + "\n")

    script = P.repo_path(f"src/scripts/visualizations/{PLOT}/gather.py")
    meta = {
        "written_utc": ac.now_utc(), "plot": PLOT, "script": f"src.scripts.visualizations.{PLOT}.gather",
        "script_sha256": sc.sha256_file(script) if script.exists() else None,
        "cueball_dir": str(cueball), "only": args.only,
        "alpha_common": ac.lib_info(),
        "inputs": {"rows_csv": ac.file_info(alpha.rows_csv, sha=True), "alpha_cells": ac.file_info(alpha.cells_csv, sha=True),
                   "alpha_gather_meta": ac.file_info(alpha.meta_json, sha=True)},
        "constants": {"ALPHA_LEVEL": ALPHA_LEVEL, "MIN_N": ac.MIN_N, "qualifying_rules": QRULE,
                      "models": ac.MODELS, "excluded_models": ac.EXCLUDED_MODELS},
        "rows_report": rep, "sample0_reparse_check": reparse, "alpha_cell_check": alpha_check, "selftest": selftest,
        "counts": {"n_cells": int(len(cells)), "n_cells_with_third": int((cells["n_third"] > 0).sum()),
                   **{f"n_qualify_{t}": int(cells["qualifies"].sum()) for t in TESTS},
                   "n_pooled_cells": int(len(pooled)), "n_pooled_qualify": int(pooled["qualifies"].sum()),
                   "n_pooled_not_tested": int(pooled["not_tested"].sum()),
                   "n_pooled_reject": int(pooled["majority_reject"].sum()),
                   "n_pooled_reject_bh": int(pooled["majority_reject_bh"].sum()),
                   **{f"n_reject_{t}": int(cells[f"{t}_reject"].sum()) for t in TESTS},
                   **{f"n_reject_bh_{t}": int(cells[f"{t}_reject_bh"].sum()) for t in TESTS}},
    }
    (out / "gather_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False))
        print(pooled[pooled["n_population"] > 0][POOL_KEYS + ["n_population", "threshold", "n_hits", "expected_hits",
                                                              "majority_p", "majority_binom_p_upper", "majority_p_bh",
                                                              "qualifies"]].to_string(index=False))
    print("worked example:", json.dumps({k: worked.get(k) for k in POOL_KEYS + [
        "n_population", "n_hits", "expected_hits", "majority_p", "majority_p_bh", "qualifies"]}
        if worked else None, default=str))
    print("context MMLU-Pro cell:", json.dumps({k: worked["context_mmlu_pro_cell"].get(k) for k in POOL_KEYS + [
        "n_population", "threshold", "n_hits", "expected_hits", "majority_binom_p_upper", "qualifies"]}
        if worked and worked.get("context_mmlu_pro_cell") else None, default=str))
    print(f"{len(cells)} cells → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
