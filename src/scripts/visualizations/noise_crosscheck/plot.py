"""Noise cross-check — figures. Reads ONLY data/cells.csv (written by gather.py).

  noise_crosscheck_literal.{png,pdf}  one panel per model, cells sorted by the literal measured noise
                                      (1 − robust_used share);
  noise_crosscheck_strict.{png,pdf}   the same with the strict measured noise (never re-flips).
In both: per cell three markers, all as shares of to-target flips: measured (with its Wilson interval),
α-implied 1 − α, baseline-predicted mean[(8 − stability)/8 · 1/(n − 1)] / p. Filled = positive case,
hollow = negative. Cells with n < 20 pairs faded, n printed beside the measured marker. Values above 1 (any
estimator) are drawn at the top edge as ⇑. Style from paper_plots.

Run:
    python -m src.scripts.visualizations.noise_crosscheck.plot [--data D] [--out-dir D]
Figures: <out-dir>/noise_crosscheck_{literal,strict}.{png,pdf} (default <cueball>/plots/noise_crosscheck/);
--data defaults to <out-dir>/data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

from src.scripts.visualizations.common import alpha_common as ac
from src.scripts.visualizations.common import section1_plot as SP
from src.scripts.visualizations.common.section1_plot import plt
from src.scripts.visualizations.paper_plots import (
    FULL_W, INK2, MODEL_NAMES, MUTED, SLOTS, SURFACE, apply_style, pct_axis, tidy,
)

PLOT = "noise_crosscheck"
YTOP = 1.0
OVER_MARKER = r"$\Uparrow$"   # a value > 100 % drawn at the top edge (distinct from the ▲ baseline-predicted marker)
EST = {"measured": (SLOTS[0], "o", "measured (re-sampling)"),
       "implied": (SLOTS[7], "D", "α-implied  1 − α"),
       "predicted": (SLOTS[2], "^", "baseline-predicted")}
DEFS = [("literal", "measured = literal (1 − robust_used share)", "n_pairs_labeled", "faded_literal"),
        ("strict", "measured = strict (never re-flips)", "n_pairs_strict", "faded_strict")]


def dot(ax, x, y, est, positive, faded, size=9):
    col, mk, _ = EST[est]
    over = y > YTOP
    y = min(y, YTOP)
    ax.scatter(x, y, s=size * (2.2 if over else 1), marker=OVER_MARKER if over else mk, facecolor=col if positive else SURFACE,
               edgecolor=col, linewidth=0.6, alpha=ac.FADE_ALPHA if faded else 0.9, zorder=3, clip_on=False)


def one_figure(c: pd.DataFrame, models: list[str], d: str, title: str, ncol: str, fcol: str, out: Path) -> None:
    fig, axes = plt.subplots(len(models), 1, figsize=(FULL_W, 1.2 * len(models) + 0.7), squeeze=False,
                             sharey=True)
    for i, m in enumerate(models):
        ax = axes[i, 0]
        t = c[(c["subject_model"] == m) & c[f"y_{d}"].notna()].sort_values(f"rank_{d}")
        for x, (_, r) in enumerate(t.iterrows()):
            pos, faded = r["case"] == "positive", bool(r[fcol])
            a = ac.FADE_ALPHA if faded else 0.7
            ax.vlines(x, r[f"y_{d}_lo"], r[f"y_{d}_hi"], color=EST["measured"][0], linewidth=0.5, alpha=a * 0.6,
                      zorder=2)
            dot(ax, x, r[f"y_{d}"], "measured", pos, faded, size=11)
            dot(ax, x, r["x_implied"], "implied", pos, faded, size=7)
            dot(ax, x, r["pred_base_share"], "predicted", pos, faded, size=8)
            if faded:
                ax.annotate(f"n={int(r[ncol])}", (x, min(r[f"y_{d}"], YTOP)), xytext=(-3, 0),
                            textcoords="offset points", ha="right", fontsize=4.6, color=MUTED, va="center")
        ax.set_xlim(-1, max(len(t), 1))
        ax.set_ylim(-0.02, YTOP + 0.04)
        ax.set_xticks([])
        pct_axis(ax, "y")
        tidy(ax, grid_axis="y")
        n_f = int(t[fcol].sum())
        ax.text(0.01, 0.97, f"{len(t)} cells" + (f", {n_f} faded (n < {ac.MIN_N})" if n_f else ""),
                transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=MUTED)
        if i == 0:
            ax.set_title(title, fontsize=7.4)
        ax.set_ylabel(MODEL_NAMES[m], fontsize=7)
        if i == len(models) - 1:
            ax.set_xlabel("cells (dataset × style × case), sorted by measured noise", fontsize=6.8)
    h = [Line2D([], [], marker=mk, linestyle="none", color=col, markeredgecolor=col, markersize=4.5, label=lab)
         for col, mk, lab in EST.values()]
    h += [Line2D([], [], marker="o", linestyle="none", color=INK2, markersize=4.5, label="positive case"),
          Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=4.5,
                 label="negative case"),
          Line2D([], [], marker=OVER_MARKER, linestyle="none", color=MUTED, markersize=6,
                 label="> 100 %, drawn at the top")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.2, columnspacing=1.0,
               handletextpad=0.3, title="all three as shares of the cell's to-target flips", title_fontsize=6.2)
    fig.supylabel("noise share of to-target flips", fontsize=7.4, color=INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"noise_crosscheck_{d}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main(argv=None) -> int:
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    apply_style()
    c = pd.read_csv(a.data / "cells.csv")
    models = [m for m in ac.MODELS if m in set(c["subject_model"])]
    for d, title, ncol, fcol in DEFS:
        one_figure(c, models, d, title, ncol, fcol, a.out_dir)
    print(f"figures → {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
