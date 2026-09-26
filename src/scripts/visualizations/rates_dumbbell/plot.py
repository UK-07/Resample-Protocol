"""Rates dumbbell — plot. Reads only data/cells_model_dataset.csv and data/cells_model_dataset_style.csv.

    rates_dumbbell.{png,pdf}                     main: SSP vs RSP unfaithful rate per model x dataset
                                                 (styles pooled), one panel per manifest case, rows sorted
                                                 by gap (RSP - SSP, largest first) inside each panel
    rates_dumbbell_by_style_<case>.{png,pdf}     appendix: the same per hint style (8 panels), one figure
                                                 per case

SSP rate = binary-judge unfaithful / (unfaithful + faithful) over SSP flips (sample-0 baseline).
RSP rate = (v2 label 0 + label -1) / (0 + 1 + -1) over the to-target k=4 re-rolls of robust_used questions
(dataset_B + its -1 rows). The right margin gives the gap (pp) and, in brackets, the -1 re-rolls.
Whiskers: Wilson 95 %. Endpoints whose denominator is < 20 are faded and carry their n.

Run:
    python -m src.scripts.visualizations.rates_dumbbell.plot [--data D] [--out-dir D]
Figures: <out-dir>/rates_dumbbell*.{png,pdf} (default <cueball>/plots/rates_dumbbell/); --data defaults to
<out-dir>/data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.scripts.visualizations.common import paths as P  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    AXIS, CUE_NAMES, CUE_ORDER, DATASET_ORDER, DATASET_SHORT, FULL_W, GRID, INK2, MODEL_ORDER, MODEL_SHORT, MUTED,
    SLOTS, SURFACE, apply_style, pct_axis,
)

PLOT = "rates_dumbbell"
SMALL_N = 20
FADE = 0.3
CASES = ["positive", "negative"]
CASE_TITLE = {"positive": "positive case (cue → wrong option)",
              "negative": "negative case (cue → correct option)"}
SSP_C, RSP_C = MUTED, SLOTS[7]
MODELS = [m for m in MODEL_ORDER if m != "qwen3.6-27b"]
JUDGE_NOTE = ("SSP: one hinted rollout vs sample-0 baseline, binary judge.  RSP: to-target k=4 re-rolls of "
              "robust_used questions, v2 role-based judge, −1 counted as unfaithful.  Right column: gap = RSP − SSP (pp) "
              "[−1 re-rolls in the RSP rate].\nNemotron-Nano-9B re-rolls used a 24,576-token budget (16,384 elsewhere).")


def sort_cells(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["_m"] = t.subject_model.map({m: i for i, m in enumerate(MODELS)})
    t["_d"] = t.dataset.map({d: i for i, d in enumerate(DATASET_ORDER)})
    t["_g"] = t.gap_rsp_minus_ssp.fillna(-np.inf)
    return t.sort_values(["_g", "_m", "_d"], ascending=[False, True, True])


def dumbbell(ax, t: pd.DataFrame, *, xmax: float, label_size: float = 6.5, marker: float = 3.6,
             n_size: float = 5.0) -> None:
    """One dumbbell per row of ``t`` (already ordered top to bottom)."""
    t = t.reset_index(drop=True)
    y = np.arange(len(t))
    off = 0.24
    for i, r in t.iterrows():
        ssp_ok, rsp_ok = not pd.isna(r.ssp_rate), not pd.isna(r.rsp_rate)
        if ssp_ok and rsp_ok:
            ax.plot([r.ssp_rate, r.rsp_rate], [i, i], color=AXIS, linewidth=1.1, zorder=1,
                    alpha=FADE if (r.ssp_den < SMALL_N or r.rsp_den < SMALL_N) else 1)
        for ok, rate, lo, hi, den, col, dy, face in (
                (ssp_ok, r.ssp_rate, r.ssp_rate_lo, r.ssp_rate_hi, r.ssp_den, SSP_C, -off, SURFACE),
                (rsp_ok, r.rsp_rate, r.rsp_rate_lo, r.rsp_rate_hi, r.rsp_den, RSP_C, off, RSP_C)):
            if not ok:
                ax.text(0.006 * xmax, i + dy, f"{'SSP' if col == SSP_C else 'RSP'} n=0", fontsize=n_size,
                        color=MUTED, va="center", ha="left")
                continue
            a = FADE if den < SMALL_N else 1.0
            ax.hlines(i + dy, lo, hi, color=col, linewidth=0.6, alpha=a, zorder=2)
            ax.plot(rate, i, "o", ms=marker, mfc=face, mec=col, mew=0.9, alpha=a, zorder=3)
            if den < SMALL_N:
                if hi > 0.8 * xmax:     # near the right edge: put n left of the whisker (gap column is right)
                    ax.text(lo - 0.01 * xmax, i + dy, f"n={int(den)}", fontsize=n_size, color=MUTED,
                            va="center", ha="right")
                else:
                    ax.text(hi + 0.006 * xmax, i + dy, f"n={int(den)}", fontsize=n_size, color=MUTED,
                            va="center", ha="left")
    labels = [f"{MODEL_SHORT[m]} · {DATASET_SHORT.get(d, d)}" for m, d in zip(t.subject_model, t.dataset)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_size)
    for tick, m in zip(ax.get_yticklabels(), t.subject_model):
        tick.set_color(INK2)
    ax.set_ylim(len(t) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    pct_axis(ax, "x", top=xmax)
    step = 0.5 if xmax > 0.6 else 0.2 if xmax > 0.3 else 0.1       # few ticks: panels are narrow
    ax.set_xticks(np.arange(0, xmax + 1e-9, step))
    # gap (RSP - SSP) in pp [-1 re-rolls] as a column INSIDE the axes, right of the data range, so it can
    # never run into the neighbouring panel's tick labels
    gap_x = xmax * 1.04
    ax.set_xlim(0, xmax * 1.42)
    ax.spines["bottom"].set_bounds(0, xmax)
    ax.axvline(xmax * 1.015, color=GRID, lw=0.6)
    for i, r in t.iterrows():
        if not pd.isna(r.gap_rsp_minus_ssp):
            faded = r.ssp_den < SMALL_N or r.rsp_den < SMALL_N
            ax.text(gap_x, i, f"{100 * r.gap_rsp_minus_ssp:+.1f} [{int(r.n_rsp_incoherent)}]",
                    fontsize=n_size, color=MUTED if faded else INK2, va="center", ha="left")


def legend(fig, y: float) -> None:
    h = [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=SSP_C, mew=0.9, ms=4,
                label="SSP rate (binary judge)"),
         Line2D([], [], marker="o", ls="", mfc=RSP_C, mec=RSP_C, ms=4, label="RSP rate (re-rolls, v2 judge)"),
         Line2D([], [], color=INK2, lw=0.6, label="Wilson 95% CI"),
         Line2D([], [], marker="o", ls="", mfc=MUTED, mec=MUTED, alpha=FADE, ms=4, label=f"n < {SMALL_N} (faded)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, y), fontsize=6.5, handletextpad=0.3,
               columnspacing=1.2)


def xmax_of(t: pd.DataFrame) -> float:
    hi = np.nanmax(np.r_[t.ssp_rate_hi.to_numpy(float), t.rsp_rate_hi.to_numpy(float), [0.05]])
    return float(min(1.0, np.ceil((hi * 1.08) * 20) / 20))


def main_figure(t: pd.DataFrame, out: Path) -> None:
    n_rows = max(len(t[t.case == c]) for c in CASES)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 0.9 + 0.155 * n_rows))
    xmax = xmax_of(t)
    for ax, case in zip(axes, CASES):
        dumbbell(ax, sort_cells(t[t.case == case]), xmax=xmax)
        ax.set_title(CASE_TITLE[case], fontsize=7.5)
        ax.set_xlabel("unfaithful rate", fontsize=7)
    fig.subplots_adjust(left=0.17, right=0.91, top=0.93, bottom=0.5 / (0.9 + 0.155 * n_rows) + 0.03,
                        wspace=0.85)
    legend(fig, 0.0)
    fig.text(0.5, -0.035, JUDGE_NOTE, ha="center", fontsize=5.3, color=MUTED)
    save(fig, out)


def style_figure(t: pd.DataFrame, case: str, out: Path) -> None:
    t = t[t.case == case]
    styles = [s for s in CUE_ORDER if s in set(t.hint_style)]
    n_rows = max(len(t[t.hint_style == s]) for s in styles)
    ncol = 2
    nrow = int(np.ceil(len(styles) / ncol))
    ph = 0.35 + 0.1 * n_rows
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_W, 0.55 + nrow * ph), squeeze=False)
    xmax = xmax_of(t)
    for k, ax in enumerate(axes.flat):
        if k >= len(styles):
            ax.set_visible(False)
            continue
        s = styles[k]
        dumbbell(ax, sort_cells(t[t.hint_style == s]), xmax=xmax, label_size=5.3, marker=2.8, n_size=4.5)
        ax.set_title(CUE_NAMES.get(s, s), fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
    fig.suptitle(f"Per cue style — {CASE_TITLE[case]}", x=0.01, y=0.995, ha="left", va="top", fontsize=7.5)
    fig.subplots_adjust(left=0.17, right=0.9, top=1 - 0.55 / (0.55 + nrow * ph),
                        bottom=0.45 / (0.55 + nrow * ph), wspace=0.85, hspace=0.3)
    legend(fig, 0.0)
    fig.text(0.5, -0.012, JUDGE_NOTE, ha="center", fontsize=5.3, color=MUTED)
    save(fig, out)


def save(fig, stem: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else P.plot_dir(PLOT)
    data = Path(a.data) if a.data else out / "data"
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    md = pd.read_csv(data / "cells_model_dataset.csv")
    mds = pd.read_csv(data / "cells_model_dataset_style.csv")
    # a cell with no hinted rollouts at all (model x dataset not in the grid) is not a row
    md = md[md.n_hinted > 0]
    mds = mds[mds.n_hinted > 0]
    main_figure(md, out / "rates_dumbbell")
    for case in CASES:
        style_figure(mds, case, out / f"rates_dumbbell_by_style_{case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
