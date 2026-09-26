"""Benchmark table — the LaTeX tables. Reads only data/cells_model_style.csv and data/cells_model_dataset_style.csv.

    table2_benchmark_main.csv / .tex       MAIN TEXT: model x style, pooled over datasets, one table per case
                                           (data/cells_model_style.csv)
    table2_benchmark_appendix.csv / .tex   APPENDIX: model x dataset x style, one longtable per case
                                           (data/cells_model_dataset_style.csv)
    Columns: robust susceptibility and RSP unfaithful rate (to-target re-rolls of robust_used questions, v2 judge,
    -1 counted as unfaithful) with Wilson 95 % CIs, the number of -1 re-rolls inside that rate, then the SSP
    susceptibility and SSP unfaithful rate (binary judge). A value whose denominator is < 20 carries a dagger;
    an undefined value (denominator 0) is an em dash. The CSVs carry every count (binary-judge rows, dataset_B
    rows, -1 rows, exclusions).

Needs in the LaTeX preamble: \\usepackage{booktabs} (main), \\usepackage{booktabs,longtable} (appendix).

Run:
    python -m src.scripts.visualizations.table2_benchmark.plot [--data D] [--out-dir D]
Tables: <out-dir>/table2_benchmark_{main,appendix}.{csv,tex} (default <cueball>/plots/table2_benchmark/); --data
defaults to <out-dir>/data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.paper_plots import CUE_NAMES, CUE_ORDER, DATASET_NAMES, DATASET_ORDER, MODEL_NAMES, MODEL_ORDER

PLOT = "table2_benchmark"
SMALL_N = 20
CASES = ["positive", "negative"]
MODELS = [m for m in MODEL_ORDER if m != "qwen3.6-27b"]
CASE_CAPTION = {
    "positive": "positive case (the cue points to a wrong option)",
    "negative": "negative case (the cue points to the correct option)",
}
CSV_COLS = [
    "subject_model", "dataset", "hint_style", "case",
    "robust_susc", "robust_susc_lo", "robust_susc_hi", "n_robust_used_clean", "n_clean",
    "rsp_rate", "rsp_rate_lo", "rsp_rate_hi", "rsp_num", "rsp_den", "n_rsp_unfaithful", "n_rsp_incoherent",
    "n_rsp_faithful", "n_dataset_B", "n_rsp_questions", "rsp_rate_excl_incoherent",
    "ssp_susc", "ssp_susc_lo", "ssp_susc_hi", "n_ssp_flip_clean", "n_eligible_clean",
    "ssp_rate", "ssp_rate_lo", "ssp_rate_hi", "n_ssp_unfaithful", "ssp_den", "n_binary_labeled_rows",
    "n_hinted", "n_b0_unanswered", "n_b0_is_target", "n_wrong_to_wrong", "n_eligible", "n_ssp_flip",
    "n_truncated_to_target_excluded", "n_ssp_binary_missing", "n_to_hint_modal", "n_robust_used",
    "n_to_hint_no_reliance_label", "n_dataset_B_over_budget",
    "small_robust_susc", "small_rsp_rate", "small_ssp_susc", "small_ssp_rate",
]


def cell(v, lo, hi, den) -> str:
    if den == 0 or pd.isna(v):
        return "---"
    s = f"{100 * v:.1f} {{\\scriptsize({100 * lo:.1f}--{100 * hi:.1f})}}"
    return s + ("$^\\dagger$" if den < SMALL_N else "")


HEAD = ("robust susc. & RSP unfaithful & $-1$ & SSP susc. & SSP unfaithful")


def row_cells(r) -> list[str]:
    return [cell(r.robust_susc, r.robust_susc_lo, r.robust_susc_hi, r.n_clean),
            cell(r.rsp_rate, r.rsp_rate_lo, r.rsp_rate_hi, r.rsp_den),
            str(int(r.n_rsp_incoherent)),
            cell(r.ssp_susc, r.ssp_susc_lo, r.ssp_susc_hi, r.n_eligible_clean),
            cell(r.ssp_rate, r.ssp_rate_lo, r.ssp_rate_hi, r.ssp_den)]


def caption(case: str, scope: str) -> str:
    return (f"\\caption{{Benchmark ({scope}), {CASE_CAPTION[case]}. Primary columns: robust susceptibility "
            "(share of clean hinted rollouts whose question follows the cue in $\\geq$3 of 4 re-rolls) and the RSP "
            "unfaithful rate (to-target re-rolls of those questions, v2 role-based judge; incoherent verdicts "
            "($-1$) count as unfaithful, their number is given in the $-1$ column). Secondary columns: the "
            "single-sample protocol (sample-0 baseline, binary judge). SSP and RSP rates differ in protocol "
            "and judge. Values in \\%, Wilson 95\\% CI in parentheses (the RSP interval ignores the clustering "
            "of up to 4 re-rolls per question); $^\\dagger$: denominator $<20$; ---: no data.}")


def header(n_key: int, keys: str) -> list[str]:
    ncol = n_key + 5
    return [
        f" {'& ' * n_key}\\multicolumn{{3}}{{c}}{{resampled (primary)}} & \\multicolumn{{2}}{{c}}{{single-sample (secondary)}} \\\\",
        f"\\cmidrule(lr){{{n_key + 1}-{n_key + 3}}}\\cmidrule(lr){{{n_key + 4}-{ncol}}}",
        f"{keys} & {HEAD} \\\\",
    ]


def table_main(t: pd.DataFrame, case: str) -> list[str]:
    t = t[t.case == case]
    L = [f"% ---- {case} case (main) ----", "\\begin{table}[t]", "\\centering", "\\scriptsize",
         "\\setlength{\\tabcolsep}{3pt}", caption(case, "cue styles pooled over datasets"),
         f"\\label{{tab:benchmark_main_{case}}}", "\\begin{tabular}{ll rrrrr}", "\\toprule"]
    L += header(2, "Model & Cue") + ["\\midrule"]
    first = True
    for m in MODELS:
        tm = t[t.subject_model == m].set_index("hint_style")
        if tm.empty:
            continue
        if not first:
            L.append("\\midrule")
        first = False
        for k, s_ in enumerate([s_ for s_ in CUE_ORDER if s_ in tm.index]):
            r = tm.loc[s_]
            L.append(" & ".join([MODEL_NAMES[m] if k == 0 else "", CUE_NAMES[s_]] + row_cells(r)) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def table_appendix(t: pd.DataFrame, case: str) -> list[str]:
    t = t[t.case == case]
    hdr = header(3, "Model & Dataset & Cue")
    L = [f"% ---- {case} case (appendix) ----", "\\begin{scriptsize}", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{longtable}{lll rrrrr}",
         caption(case, "per dataset") + f"\\label{{tab:benchmark_appendix_{case}}}\\\\",
         "\\toprule", *hdr, "\\midrule", "\\endfirsthead",
         "\\toprule", hdr[-1], "\\midrule", "\\endhead",
         "\\bottomrule", "\\endfoot"]
    first_model = True
    for m in MODELS:
        tm = t[t.subject_model == m]
        if tm.empty:
            continue
        if not first_model:
            L.append("\\midrule")
        first_model = False
        first_ds = True
        for d in [d for d in DATASET_ORDER if d in set(tm.dataset)] + sorted(set(tm.dataset) - set(DATASET_ORDER)):
            td = tm[tm.dataset == d].set_index("hint_style")
            if not first_ds:
                L.append("\\cmidrule(l){2-8}")
            for k, s_ in enumerate([s_ for s_ in CUE_ORDER if s_ in td.index]):
                r = td.loc[s_]
                L.append(" & ".join([MODEL_NAMES[m] if (first_ds and k == 0) else "",
                                     DATASET_NAMES.get(d, d) if k == 0 else "", CUE_NAMES[s_]] + row_cells(r))
                         + " \\\\")
            first_ds = False
    L += ["\\end{longtable}", "\\end{scriptsize}", ""]
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else P.plot_dir(PLOT)
    data = Path(a.data) if a.data else out / "data"
    out.mkdir(parents=True, exist_ok=True)
    order = {"subject_model": MODELS, "dataset": DATASET_ORDER, "hint_style": CUE_ORDER, "case": CASES}
    for scope, src, build, pkgs in (
            ("main", "cells_model_style.csv", table_main, "booktabs"),
            ("appendix", "cells_model_dataset_style.csv", table_appendix, "booktabs,longtable")):
        t = pd.read_csv(data / src)
        t = t[t.n_hinted > 0]           # model x dataset pairs outside the grid have no rows at all
        keys = [k for k in order if k in t.columns]
        t = t.sort_values(keys, key=lambda c: c.map({v: i for i, v in enumerate(order[c.name])}))
        t[keys + [c for c in CSV_COLS if c not in order]].to_csv(out / f"table2_benchmark_{scope}.csv", index=False)
        lines = [f"% Table 2 ({scope}, section 5). Generated by src.scripts.visualizations.table2_benchmark.plot from data/{src}.",
                 f"% Requires \\usepackage{{{pkgs}}}. SSP and RSP rates use DIFFERENT judges (binary vs v2).", ""]
        for case in CASES:
            lines += build(t, case)
        (out / f"table2_benchmark_{scope}.tex").write_text("\n".join(lines))
        n_flag = {c: int(t[f"small_{c}"].sum()) for c in ("robust_susc", "rsp_rate", "ssp_susc", "ssp_rate")}
        print(f"wrote {out}/table2_benchmark_{scope}.csv ({len(t)} rows) and .tex; cells with n < {SMALL_N}: {n_flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
