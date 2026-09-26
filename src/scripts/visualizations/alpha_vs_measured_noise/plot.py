"""Alpha-implied noise (x) vs the noise re-sampling measures (y) — plot. Reads ONLY data/cells.csv.

Figures (in <out-dir>):
    alpha_vs_measured_noise.{png,pdf}             y = 1 − robust_used share. Left panel: the full range on equal
                                                  linear axes (y = x a true diagonal); right panel: zoom on
                                                  x ∈ [0, ZOOM_X] with the full y range.
    alpha_vs_measured_noise_strict.{png,pdf}      companion: y = share of pairs whose non-truncated re-rolls
                                                  never reach the target (0 of k), same layout.
    alpha_vs_measured_noise_by_dataset.{png,pdf}  secondary: the main figure's points faceted by dataset, with
                                                  the 95 % intervals of both coordinates.
Style (fonts, palette, model order/names, case colours) comes from ``paper_plots`` so the figure matches the
rest of the paper.

Run:
    python -m src.scripts.visualizations.alpha_vs_measured_noise.plot [--data D] [--out-dir D]
Figures: <out-dir>/alpha_vs_measured_noise[_strict|_by_dataset].{png,pdf} (default
<cueball>/plots/alpha_vs_measured_noise/); --data defaults to <out-dir>/data.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from src.scripts.visualizations.common.section1_plot import plot_args  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    CASE_COLOR, CASE_ORDER, DATASET_NAMES, DATASET_ORDER, FULL_W, INK2, MODEL_NAMES, MODEL_ORDER,
    MUTED, SURFACE, apply_style, pct_axis, tidy,
)

PLOT = "alpha_vs_measured_noise"
MODEL_MARKER = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v", "P"]))
SIZE_KEY = [25, 100, 400, 1000]
ZOOM_X = 0.10
MIN_PAIRS = 20                        # must match gather.py (the faded flags are computed there)
FADE_ALPHA = 0.28                     # cells with < MIN_PAIRS pairs: drawn faded, n printed
PAD_LO, PAD_HI = 0.02, 0.03          # axis padding so points on 0 / at the limit are not cut

# (y column, lo, hi, n column for size, faded flag, y label, output stem)
SPEC = ("y_measured_noise", "y_lo", "y_hi", "n_pairs_labeled", "faded",
        "1 − robust_used share\n(noise share re-sampling finds)", "alpha_vs_measured_noise")
STRICT = ("y_strict", "y_strict_lo", "y_strict_hi", "n_pairs_strict", "faded_strict",
          "share of pairs never re-flipping\n(0 of the non-truncated re-rolls to target)",
          "alpha_vs_measured_noise_strict")


def size_of(n) -> np.ndarray:
    """Marker area (pt²) from the number of re-rolled to-target pairs (area ∝ √n)."""
    return 3.0 + 2.2 * np.sqrt(np.asarray(n, dtype=float))


def upper(values) -> float:
    top = float(np.nanmax(np.asarray(values, dtype=float)))
    return min(1.0, math.ceil(top * 1.05 * 10) / 10)


def draw_points(ax, df: pd.DataFrame, ycol: str, ncol: str, fcol: str, *, err: tuple[str, str] | None = None,
                scale: float = 1.0) -> None:
    for m in MODEL_ORDER:
        for case in CASE_ORDER:
            t = df[(df.subject_model == m) & (df.case == case)]
            if t.empty:
                continue
            if err is not None:
                lo, hi = err
                ax.errorbar(t.x_alpha_noise, t[ycol],
                            xerr=[(t.x_alpha_noise - t.x_lo).clip(lower=0), (t.x_hi - t.x_alpha_noise).clip(lower=0)],
                            yerr=[(t[ycol] - t[lo]).clip(lower=0), (t[hi] - t[ycol]).clip(lower=0)],
                            fmt="none", ecolor=CASE_COLOR[case], elinewidth=0.4, alpha=0.35, zorder=2)
            clipped = t.alpha_clipped.astype(bool)
            faded = t[fcol].astype(bool)
            for sub, hollow, fd in ((t[~clipped & ~faded], False, False), (t[clipped & ~faded], True, False),
                                    (t[~clipped & faded], False, True), (t[clipped & faded], True, True)):
                if sub.empty:
                    continue
                ax.scatter(sub.x_alpha_noise, sub[ycol], s=size_of(sub[ncol]) * scale,
                           marker=MODEL_MARKER[m], facecolor=SURFACE if hollow else CASE_COLOR[case],
                           edgecolor=CASE_COLOR[case] if hollow else SURFACE, linewidth=0.9 if hollow else 0.5,
                           alpha=FADE_ALPHA if fd else 0.8, zorder=3, clip_on=False)
                if fd:
                    for xv, yv, n in zip(sub.x_alpha_noise, sub[ycol], sub[ncol]):
                        if ax.get_xlim()[0] <= xv <= ax.get_xlim()[1]:
                            ax.annotate(f"n={int(n)}", (xv, yv), xytext=(3, 0), textcoords="offset points",
                                        fontsize=4.6, color=MUTED, va="center")


def frame(ax, xmax: float, ymax: float, *, equal: bool) -> None:
    top = max(xmax, ymax)
    ax.plot([0, top], [0, top], color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1, clip_on=True)
    ax.set_xlim(-PAD_LO * xmax, xmax * (1 + PAD_HI))
    ax.set_ylim(-PAD_LO * ymax, ymax * (1 + PAD_HI))
    if equal:
        ax.set_aspect("equal", adjustable="box")
    pct_axis(ax, "x"); pct_axis(ax, "y")
    tidy(ax, grid_axis="both")


def legend_handles(df: pd.DataFrame, ncol: str) -> tuple[list, list, list]:
    h_case = [Line2D([], [], marker="o", linestyle="none", color=CASE_COLOR[c], markeredgecolor=SURFACE,
                     markersize=5, label=f"{c} case") for c in CASE_ORDER]
    if df.alpha_clipped.any():
        h_case.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE,
                             markeredgecolor=INK2, markersize=5, label="α < 0, clipped to 0 (x = 1)"))
    h_model = [Line2D([], [], marker=MODEL_MARKER[m], linestyle="none", color=INK2, markersize=4.5,
                      label=MODEL_NAMES[m]) for m in MODEL_ORDER if m in set(df.subject_model)]
    h_size = [plt.scatter([], [], s=size_of(n), marker="o", facecolor=MUTED, edgecolor=SURFACE, label=f"{n}")
              for n in SIZE_KEY]
    return h_case, h_model, h_size


def two_panel(cells: pd.DataFrame, spec: tuple, out_dir: Path) -> tuple[int, int]:
    ycol, lo, hi, ncol, flag, ylabel, stem = spec
    df = cells[cells[ycol].notna() & cells.x_alpha_noise.notna()].copy()
    lim = upper(np.r_[df.x_alpha_noise, df[ycol]])
    fig, (ax, az) = plt.subplots(1, 2, figsize=(FULL_W, 3.35), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})
    frame(ax, lim, lim, equal=True)
    draw_points(ax, df, ycol, ncol, flag, scale=0.8)
    ax.add_patch(Rectangle((-PAD_LO * lim, -PAD_LO * lim), ZOOM_X + PAD_LO * lim, lim * (1 + PAD_LO + PAD_HI),
                           fill=False, edgecolor=MUTED, linewidth=0.6, linestyle=":", zorder=1))
    ax.set_title("all cells (equal axes)")
    ax.set_xlabel("1 − α  (noise share α assumes)")
    ax.set_ylabel(ylabel)
    frame(az, ZOOM_X, lim, equal=False)
    az.set_box_aspect(1)
    az.xaxis.set_major_locator(MultipleLocator(0.02))
    draw_points(az, df[df.x_alpha_noise <= ZOOM_X * (1 + PAD_HI)], ycol, ncol, flag, scale=0.8)
    n_out = int((df.x_alpha_noise > ZOOM_X * (1 + PAD_HI)).sum())
    az.set_title(f"zoom: 1 − α ≤ {100 * ZOOM_X:.0f}%  ({len(df) - n_out} of {len(df)} cells)")
    az.set_xlabel("1 − α  (zoom)")
    h_case, h_model, h_size = legend_handles(df, ncol)
    fig.legend(handles=h_case + h_model, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5,
               columnspacing=0.9, handletextpad=0.3, fontsize=6.2)
    fig.legend(handles=h_size, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=len(SIZE_KEY),
               columnspacing=1.0, handletextpad=0.2, fontsize=6.2,
               title=f"to-target pairs re-rolled (marker area); all {len(df)} cells with a y shown, "
                     f"{int(df[flag].astype(bool).sum())} with < {MIN_PAIRS} pairs faded (n printed); dashed: y = x",
               title_fontsize=6.2)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return len(df), len(cells)


def by_dataset(cells: pd.DataFrame, out_dir: Path) -> None:
    ycol, lo, hi, ncol, flag, _, _ = SPEC
    df = cells[cells[ycol].notna() & cells.x_alpha_noise.notna()].copy()
    lim = upper(np.r_[df.x_alpha_noise, df[ycol], df.x_hi.fillna(0), df[hi].fillna(0)])
    datasets = [d for d in DATASET_ORDER if d in set(df.dataset)]
    ncols = 2
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_W, 2.55 * nrows + 0.6), squeeze=False)
    for ax, ds in zip(axes.flat, datasets):
        t = df[df.dataset == ds]
        frame(ax, lim, lim, equal=True)
        draw_points(ax, t, ycol, ncol, flag, err=(lo, hi), scale=0.7)
        ax.set_title(f"{DATASET_NAMES.get(ds, ds)} · {int(t.n_options.iloc[0])} options · {len(t)} cells")
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("1 − α (α-implied noise)")
    for ax in axes[:, 0]:
        ax.set_ylabel("1 − robust_used share")
    h_case, h_model, _ = legend_handles(df, ncol)
    fig.legend(handles=h_case + h_model, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
               columnspacing=0.9, handletextpad=0.3, fontsize=6.2,
               title=f"cells with < {MIN_PAIRS} pairs faded (n printed)", title_fontsize=6.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"alpha_vs_measured_noise_by_dataset.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main(argv=None) -> int:
    a = plot_args(PLOT, argv, doc=__doc__)
    apply_style()
    cells = pd.read_csv(a.data / "cells.csv")
    for spec in (SPEC, STRICT):
        n_plot, n_all = two_panel(cells, spec, a.out_dir)
        print(f"{spec[-1]}: {n_plot}/{n_all} cells")
    by_dataset(cells, a.out_dir)
    print(f"figures → {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
