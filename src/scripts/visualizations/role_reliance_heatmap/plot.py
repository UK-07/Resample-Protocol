"""Role x reliance heatmap — plot.

Reads ONLY data/{role_share_by_model[_case].csv, column_summary[_by_case].csv,
exclusions_by_model[_case]_column.csv (footer = drawn columns only), gather_meta.json}. One panel per model:
rows = v2 roles + "incoherent (−1)" (a v2 −1 rollout goes in that row whatever role it carries), columns = the
pair's reliance label (robust_used, weak_used, mixed = used-candidate pairs at 0/4 only, robust_ignored); cell =
share of the column's kept rollouts in that row, with the count n printed. Colour uses a square-root scale (the
`credited` row dominates; small shares stay visible). robust_ignored is greyed and labelled "not judged
(control)". Columns whose total n < 20 are drawn faded.

Figures (in <out-dir>, default <cueball>/plots/role_reliance_heatmap/):
  role_reliance_heatmap.{png,pdf}            both cases pooled
  role_reliance_heatmap_positive.{png,pdf}   manifest case positive only
  role_reliance_heatmap_negative.{png,pdf}   manifest case negative only

Run:
    python -m src.scripts.visualizations.role_reliance_heatmap.plot [--data D] [--out-dir D]
--data defaults to <out-dir>/data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

from src.scripts.visualizations.common import paths as P  # noqa: E402
from src.scripts.visualizations.common import section3_claims as s3  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    FULL_W, INK, INK2, MODEL_NAMES, MUTED, ROLE_NAMES, SEQ_CMAP, SURFACE, apply_style,
)

PLOT = "role_reliance_heatmap"
COL_NAMES = {"robust_used": "robust\nused", "weak_used": "weak\nused", "mixed": "mixed\n(0/4)",
             "mixed_control": "mixed\n(control)", "robust_ignored": "robust\nignored"}
GREY = "#e9e8e4"
CAT_NAMES = {**ROLE_NAMES, "incoherent": "incoherent (−1)"}


def fmt_n(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 10000 else f"{n:,}"


def fmt_pct(v: float) -> str:
    if v == 0:
        return "0%"
    if v < 0.001:
        return "<0.1%"
    if v < 0.10:
        return f"{100 * v:.1f}%"
    return f"{100 * v:.0f}%"


def draw_panel(ax, t: pd.DataFrame, summ: pd.DataFrame, cols: list[str], norm, show_y: bool, title: str):
    arr = np.full((len(s3.CATEGORIES), len(cols)), np.nan)
    ns = np.zeros_like(arr)
    tot = {}
    for j, c in enumerate(cols):
        tc = t[t.column_key == c].set_index("category")
        tot[c] = int(tc.n_column.iloc[0]) if len(tc) else 0
        for i, r in enumerate(s3.CATEGORIES):
            if r in tc.index:
                arr[i, j] = tc.loc[r, "rate"]
                ns[i, j] = tc.loc[r, "n"]
    ax.imshow(np.where(np.isnan(arr), np.nan, arr), cmap=SEQ_CMAP, norm=norm, aspect="auto")
    for j, c in enumerate(cols):
        faded = tot[c] < s3.SMALL_N
        if c == "robust_ignored" or tot[c] == 0:
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(s3.CATEGORIES), color=GREY, zorder=2, linewidth=0))
            np_ = summ.loc[summ.column_key == c, "n_pairs"]
            npairs = int(np_.iloc[0]) if len(np_) else 0
            ax.text(j, (len(s3.CATEGORIES) - 1) / 2, f"not judged\n(control)\n{fmt_n(npairs)} pairs", ha="center",
                    va="center", fontsize=5.5, color=MUTED, rotation=90, zorder=3)
            continue
        if faded:
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(s3.CATEGORIES), color=SURFACE, alpha=0.6, zorder=2,
                                       linewidth=0))
        for i in range(len(s3.CATEGORIES)):
            v = arr[i, j]
            if np.isnan(v):
                continue
            dark = norm(v) > 0.55
            ax.text(j, i, f"{fmt_pct(v)}\n{fmt_n(int(ns[i, j]))}", ha="center", va="center", fontsize=5,
                    color=SURFACE if dark and not faded else INK, zorder=3, linespacing=0.95)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{COL_NAMES[c]}\nn={fmt_n(tot[c])}" if c != "robust_ignored" else COL_NAMES[c]
                        for c in cols], fontsize=5.5)
    ax.set_yticks(range(len(s3.CATEGORIES)))
    ax.set_yticklabels([CAT_NAMES[r] for r in s3.CATEGORIES] if show_y else [], fontsize=6.5)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(s3.CATEGORIES), 1), minor=True)
    ax.grid(True, which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=7.5)


def draw(t: pd.DataFrame, summ: pd.DataFrame, excl: pd.DataFrame, cols: list[str], subtitle: str,
         out_stem: Path) -> None:
    models = [m for m in s3.MODELS if m in set(t.subject_model)]
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 5.4))
    axes = axes.ravel()
    for k, m in enumerate(models):
        draw_panel(axes[k], t[t.subject_model == m], summ[summ.subject_model == m], cols, norm, k % 3 == 0,
                   MODEL_NAMES[m])
    for ax in axes[len(models):]:
        ax.axis("off")
    # colour bar in the free panel
    cax = fig.add_axes([0.70, 0.30, 0.012, 0.16])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQ_CMAP)
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.01, 0.1, 0.25, 0.5, 1])
    cb.ax.set_yticklabels(["0", "1%", "10%", "25%", "50%", "100%"], fontsize=5.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("share of column (√ scale)", fontsize=6, color=INK2)
    fig.text(0.70, 0.20, subtitle + "\ncell: share of column, n rollouts\ncolumns with n < 20 faded\n"
             "incoherent = v2 verdict −1 (any role)",
             fontsize=5.5, color=INK2, va="top")
    fig.text(0.01, 0.005, "drawn columns — " + s3.exclusion_footer(excl, MODEL_NAMES), fontsize=4.6, color=MUTED,
             wrap=True)
    fig.subplots_adjust(left=0.155, right=0.99, top=0.95, bottom=0.145, wspace=0.08, hspace=0.55)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"wrote {out_stem}.png/.pdf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None, help="the gather's tables (default <out-dir>/data)")
    ap.add_argument("--out-dir", default=None, help=f"figures folder (default <cueball>/plots/{PLOT})")
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else P.plot_dir(PLOT)
    data = Path(a.data) if a.data else out / "data"
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    summ = pd.read_csv(data / "column_summary.csv")
    drawn = ["robust_used", "weak_used", "mixed"]    # footer counts only the drawn columns (whole pool: tables)
    excl = pd.read_csv(data / "exclusions_by_model_column.csv")
    excl = excl[excl.column_key.isin(drawn)]
    json.loads((data / "gather_meta.json").read_text())
    cols = ["robust_used", "weak_used", "mixed", "robust_ignored"]   # mixed = used-candidate 0/4 only
    pooled = pd.read_csv(data / "role_share_by_model.csv")
    draw(pooled, summ, excl, cols, "both cases pooled;\noriginals + re-rolls", out / PLOT)
    by_case = pd.read_csv(data / "role_share_by_model_case.csv")
    summ_c = pd.read_csv(data / "column_summary_by_case.csv")
    excl_c = pd.read_csv(data / "exclusions_by_model_case_column.csv")
    excl_c = excl_c[excl_c.column_key.isin(drawn)]
    for case in ("positive", "negative"):
        draw(by_case[by_case.case == case], summ_c[summ_c.case == case], excl_c[excl_c.case == case], cols,
             f"manifest case: {case};\noriginals + re-rolls", out / f"{PLOT}_{case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
