"""Plot helpers shared by the survival figures' plot.py scripts. Style comes from paper_plots.

Rules: Wilson 95 % whiskers; a bar whose denominator is < 20 is drawn faded (kept) and every bar carries its n.

Run (any survival plot): python -m src.scripts.visualizations.<plot>.plot [--data D] [--out-dir D]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.scripts.visualizations.common import paths as P  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402,F401
    CASE_COLOR, CUE_ABBREV, CUE_NAMES, CUE_ORDER, DATASET_NAMES, DATASET_ORDER, DATASET_SHORT, FULL_W, INK, INK2,
    MODEL_COLOR, MODEL_NAMES, MODEL_ORDER, MODEL_SHORT, MUTED, ORDINAL_3, SLOTS, SURFACE, apply_style, pct_axis, tidy,
)

SMALL_N = 20
FADE_ALPHA = 0.35
N_FONT = 4.8
MODELS = [m for m in MODEL_ORDER if m != "qwen3.6-27b"]   # the five binary-judged models, paper order


def plot_args(plot_name: str, argv: list[str] | None = None, *, doc: str | None = None) -> argparse.Namespace:
    """The plot-side CLI: ``--out-dir`` (default ``paths.plot_dir(plot_name)``, the figures' folder) and
    ``--data`` (default ``<out-dir>/data``, the gather's tables). Both come back as ``Path``."""
    ap = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else plot_name)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    a.out_dir = Path(a.out_dir) if a.out_dir else P.plot_dir(plot_name)
    a.data = Path(a.data) if a.data else a.out_dir / "data"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    return a


def fmt_n(n) -> str:
    return f"{int(n):,}"


def grouped_bars(ax, xkeys, series, get, *, group_w=0.8, n_rotation=90, n_labels=True, ymax=1.0, n_backing=False):
    """series: list of (key, color, hatch|None); get(xkey, skey) -> (rate, lo, hi, n) or None.
    Draws one bar per (x, series) with a Wilson whisker; faded when n < SMALL_N; n printed above the whisker.
    n_backing: put the n labels on a small white box (drawn above reference lines, so a line never crosses them).
    Returns True when a faded bar was drawn."""
    x = np.arange(len(xkeys))
    bw = group_w / len(series)
    faded_any = False
    for i, (skey, color, hatch) in enumerate(series):
        for xi, xk in zip(x - group_w / 2 + bw * (i + 0.5), xkeys):
            v = get(xk, skey)
            if v is None:
                continue
            r, lo, hi, n = v
            if n == 0 or pd.isna(r):
                if n_labels:
                    ax.text(xi, 0.01, "n=0", rotation=n_rotation, ha="center", va="bottom", fontsize=N_FONT,
                            color=MUTED)
                continue
            faded = n < SMALL_N
            faded_any |= bool(faded)
            ax.bar(xi, r, width=bw * 0.88, color=color, alpha=FADE_ALPHA if faded else 1.0, hatch=hatch,
                   edgecolor=SURFACE if hatch else "none", linewidth=0, zorder=2)
            ax.vlines(xi, lo, hi, color=INK, linewidth=0.55, zorder=3)
            if n_labels:
                ax.text(xi, min(hi, ymax) + 0.012 * ymax, fmt_n(n), rotation=n_rotation, ha="center", va="bottom",
                        fontsize=N_FONT, color=INK2 if not faded else MUTED, zorder=6, clip_on=False,
                        bbox=dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none")
                        if n_backing else None)
    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(xkeys) - 0.5)
    return faded_any


def table_getter(t: pd.DataFrame, xcol: str, scol: str, rate="rate", lo="ci_lo", hi="ci_hi", n="denominator"):
    idx = {(r[xcol], r[scol]): (r[rate], r[lo], r[hi], r[n]) for _, r in t.iterrows()}
    return lambda xk, sk: idx.get((xk, sk))


def faded_swatch(color: str, label: str = f"n < {SMALL_N} (faded)") -> Patch:
    """A legend swatch that looks like a faded bar (a series colour at the fade alpha)."""
    return Patch(facecolor=color, alpha=FADE_ALPHA, edgecolor="none", label=label)


def legend_handles(series_labels, faded_any: bool, extra=None):
    """series_labels: list of (label, color, hatch). The faded swatch uses the first series' colour."""
    h = [Patch(facecolor=c, edgecolor=SURFACE if ha else "none", hatch=ha, linewidth=0, label=l)
         for l, c, ha in series_labels]
    h += extra or []
    if faded_any:
        h.append(faded_swatch(series_labels[0][1]))
    return h


def save(fig, stem: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(Path(stem).with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.02, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")
