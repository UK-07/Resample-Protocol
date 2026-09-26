"""Alpha negative-case check — plot. Reads ONLY data/cells.csv (written by gather.py).

alpha_negative_case_check.{png,pdf}: negative-case cells (model × dataset × style), x = 1 − α, y = measured
noise; top row: literal y (1 − robust_used share), bottom row: strict y (never re-flips); left: all cells, right:
zoom on 1 − α ≤ 10 % (as the alpha_vs_measured_noise plot); colour = model, marker = dataset; light 95 %
intervals on both axes (x: the alpha gather's Jeffreys interval, y: Wilson); dashed y = x. Cells with n < 20
pairs faded with n printed; α < 0 (clipped, x = 1) hollow. The left panels share one x range (the full x range
of the negative cells; x need not equal y's scale). Style from paper_plots.

Run:
    python -m src.scripts.visualizations.alpha_negative_case_check.plot [--data D] [--out-dir D]
Figure: <out-dir>/alpha_negative_case_check.{png,pdf} (default <cueball>/plots/alpha_negative_case_check/);
--data defaults to <out-dir>/data.
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from src.scripts.visualizations.common import alpha_common as ac  # noqa: E402
from src.scripts.visualizations.common.section1_plot import plot_args  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    DATASET_NAMES, DATASET_ORDER, FULL_W, INK2, MODEL_COLOR, MODEL_NAMES, MUTED, SURFACE, apply_style, pct_axis,
    tidy,
)

PLOT = "alpha_negative_case_check"
ZOOM_X = 0.10                           # as in alpha_vs_measured_noise
DATASET_MARKER = dict(zip(DATASET_ORDER, ["o", "s", "^", "D"]))
DEFS = [("literal", "literal: 1 − robust_used share", "n_pairs_labeled"),
        ("strict", "strict: never re-flips (0 of the non-truncated re-rolls)", "n_pairs_strict")]


def draw(ax, t: pd.DataFrame, d: str, ncol: str, xmax: float, ymax: float) -> None:
    top = min(xmax, ymax)
    ax.plot([0, top], [0, top], color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    for _, r in t.iterrows():
        faded = int(r[ncol]) < ac.MIN_N
        a = ac.FADE_ALPHA if faded else 0.85
        col = MODEL_COLOR[r["subject_model"]]
        ax.errorbar(r["x_implied"], r[f"y_{d}"],
                    xerr=[[max(0, r["x_implied"] - r["x_lo"])], [max(0, r["x_hi"] - r["x_implied"])]],
                    yerr=[[max(0, r[f"y_{d}"] - r[f"y_{d}_lo"])], [max(0, r[f"y_{d}_hi"] - r[f"y_{d}"])]],
                    fmt="none", ecolor=col, elinewidth=0.4, alpha=0.25 * a, zorder=2)
        hollow = bool(r["alpha_clipped"])
        ax.scatter(r["x_implied"], r[f"y_{d}"], s=15, marker=DATASET_MARKER[r["dataset"]],
                   facecolor=SURFACE if hollow else col, edgecolor=col if hollow else SURFACE,
                   linewidth=0.8 if hollow else 0.4, alpha=a, zorder=3, clip_on=False)
        if faded:
            ax.annotate(f"n={int(r[ncol])}", (r["x_implied"], r[f"y_{d}"]), xytext=(3, 0),
                        textcoords="offset points", fontsize=4.8, color=MUTED, va="center")
    ax.set_xlim(-0.01 * xmax, xmax)
    ax.set_ylim(-0.01 * ymax, ymax)
    pct_axis(ax, "x"); pct_axis(ax, "y")
    tidy(ax, grid_axis="both")


def ceil_to(v: float, step: float) -> float:
    return max(step, math.ceil(v / step) * step)


def main(argv=None) -> int:
    a = plot_args(PLOT, argv, doc=__doc__)
    out = a.out_dir
    apply_style()
    c = pd.read_csv(a.data / "cells.csv")
    c = c[c["has_pairs"].astype(bool)]
    models = [m for m in ac.MODELS if m in set(c["subject_model"])]
    xmax = ceil_to(float(np.nanmax(c["x_hi"])) * 1.03, 0.02)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.3), gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    for i, (d, title, ncol) in enumerate(DEFS):
        t = c[c[f"y_{d}"].notna()]
        ymax = ceil_to(float(np.nanmax(t[f"y_{d}_hi"])) * 1.03, 0.05)
        n_above = int((t[f"y_{d}"] > t["x_implied"]).sum())
        for j, xlim in enumerate((xmax, ZOOM_X)):
            ax = axes[i, j]
            sub = t if j == 0 else t[t["x_implied"] <= ZOOM_X * 1.001]
            draw(ax, sub, d, ncol, xlim, ymax)
            if j == 0:
                ax.set_title(f"{title}\nall negative cells ({n_above} of {len(t)} above y = x)", fontsize=7.2)
                ax.add_patch(Rectangle((-0.01 * xmax, -0.01 * ymax), ZOOM_X + 0.01 * xmax, ymax * 1.02, fill=False,
                                       edgecolor=MUTED, linewidth=0.6, linestyle=":", zorder=1))
                ax.set_ylabel("measured noise share of\nto-target flips (re-sampling)")
            else:
                ax.set_title(f"zoom: 1 − α ≤ {100 * ZOOM_X:.0f}%  ({len(sub)} of {len(t)} cells)", fontsize=7.2)
                ax.xaxis.set_major_locator(MultipleLocator(0.02))
            ax.set_xlabel("1 − α  (noise share α assumes)")
    h = [Line2D([], [], marker="o", linestyle="none", color=MODEL_COLOR[m], markeredgecolor=SURFACE, markersize=5,
                label=MODEL_NAMES[m]) for m in models]
    h += [Line2D([], [], marker=DATASET_MARKER[ds], linestyle="none", color=INK2, markersize=4,
                 label=DATASET_NAMES[ds]) for ds in DATASET_ORDER if ds in set(c["dataset"])]
    if c["alpha_clipped"].any():
        h.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK2,
                        markersize=5, label="α < 0, clipped (x = 1)"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.2,
               columnspacing=0.9, handletextpad=0.3,
               title=f"negative case (cue points at the correct answer); dashed: y = x; "
                     f"faded: fewer than {ac.MIN_N} pairs", title_fontsize=6.2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{PLOT}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"figures → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
