"""SSP vs RSP false rejections — plot.

Reads ONLY data/rates_by_{model,model_style,model_case,model_stability}.csv and counts_by_model.csv.
Every bar/cell = share of SSP false rejections (SSP flip whose original rollout's v2 role is rejected or
verification_only, or whose v2 verdict is −1) whose (question, hint) pair is robust_used; Wilson 95 % whiskers;
n < 20 faded.

Figures (in <out-dir>):
  ssp_vs_rsp_false_rejections.{png,pdf}              one bar per model (+ grey tick: robust_used share over ALL
                                                     kept SSP flips, as a reference)
  ssp_vs_rsp_false_rejections_by_style.{png,pdf}     style × model heatmap (primary slice), share and k/n per cell
  ssp_vs_rsp_false_rejections_by_case.{png,pdf}      model on x, bars positive | negative (manifest case)
  ssp_vs_rsp_false_rejections_by_stability.{png,pdf} model on x, bars per stability bin (binning only)

Run:
    python -m src.scripts.visualizations.ssp_vs_rsp_false_rejections.plot [--data D] [--out-dir D]
Defaults: --out-dir <cueball>/plots/ssp_vs_rsp_false_rejections, --data <out-dir>/data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.scripts.visualizations.common import section3_claims as s3  # noqa: E402
from src.scripts.visualizations.common.section1_plot import plot_args  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    CASE_COLOR, CUE_NAMES, CUE_ORDER, FULL_W, INK, INK2, MODEL_COLOR, MODEL_NAMES, MODEL_SHORT, MUTED,
    ORDINAL_5, SEQ_CMAP, SURFACE, STABILITY_NAMES, apply_style, pct_axis, tidy,
)

PLOT = "ssp_vs_rsp_false_rejections"
YLAB = "SSP false rejections\nconfirmed robust_used"
STAB_COLOR = dict(zip([5, 6, 7, 8], ORDINAL_5[3::-1]))    # 5/8 light → 8/8 dark


def save(fig, stem: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def footer(counts: pd.DataFrame) -> str:
    parts = []
    for m in s3.MODELS:
        r = counts[counts.subject_model == m]
        if r.empty:
            continue
        r = r.iloc[0]
        cov = r.n_kept / r.n_ssp_flips if r.n_ssp_flips else float("nan")
        s = (f"{MODEL_SHORT[m]}: {100 * cov:.0f}% of flips have a v2 role or −1; −1 {int(r.n_cat_incoherent):,} (counted "
             f"as false rejections), excl. no role {int(r.n_excl_missing_role):,}")
        if int(r.n_excl_unjudged):
            s += f", unjudged {int(r.n_excl_unjudged):,}"
        if int(r.get("n_excl_minus1_sensitivity", 0)):
            s += f", −1 excl. (sensitivity) {int(r.n_excl_minus1_sensitivity):,}"
        s += f", truncated {int(r.n_truncated_to_target_excluded):,}"
        parts.append(s)
    return " | ".join(parts)


def bar(ax, x, r, lo, hi, n, color, width, **kw):
    if pd.isna(r):
        return
    faded = n < s3.SMALL_N
    ax.bar(x, r, width=width, color=color, alpha=0.35 if faded else 1.0, linewidth=0, zorder=2, **kw)
    ax.vlines(x, lo, hi, color=INK, linewidth=0.6, zorder=3)


def fig_model(r: pd.DataFrame, counts: pd.DataFrame, out: Path) -> None:
    models = [m for m in s3.MODELS if m in set(r.subject_model)]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
    for i, m in enumerate(models):
        t = r[r.subject_model == m].iloc[0]
        bar(ax, i, t.rate, t.ci_lo, t.ci_hi, t.n_fr_with_reliance, MODEL_COLOR[m], 0.62)
        ax.hlines(t.ref_all_flips_rate, i - 0.36, i + 0.36, color=MUTED, linewidth=1.0, linestyle=(0, (2, 1)),
                  zorder=4)
        top = t.ci_hi if not pd.isna(t.ci_hi) else 0
        ax.text(i, top + 0.02, f"{int(t.n_fr_robust_used)}/{int(t.n_fr_with_reliance)}", ha="center",
                va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], fontsize=7)
    ax.set_ylim(0, 1.08)
    pct_axis(ax, "y")
    ax.set_ylabel(YLAB)
    tidy(ax, grid_axis="y")
    ax.legend(handles=[plt.Line2D([], [], color=MUTED, linestyle=(0, (2, 1)), linewidth=1.0,
                                  label="reference: robust_used share over all kept SSP flips")],
              loc="upper left", bbox_to_anchor=(0, 1.12), fontsize=6)
    fig.text(0.01, 0.005, footer(counts) + "; faded: n < 20", fontsize=4.6, color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.89, bottom=0.24)
    save(fig, out / PLOT)


def fig_style(r: pd.DataFrame, counts: pd.DataFrame, out: Path) -> None:
    models = [m for m in s3.MODELS if m in set(r.subject_model)]
    styles = [s for s in CUE_ORDER if s in set(r.hint_style)]
    fig, ax = plt.subplots(figsize=(FULL_W, 3.6))
    arr = np.full((len(styles), len(models)), np.nan)
    for i, s in enumerate(styles):
        for j, m in enumerate(models):
            t = r[(r.subject_model == m) & (r.hint_style == s)]
            if len(t) and t.n_fr_with_reliance.iloc[0] > 0:
                arr[i, j] = t.rate.iloc[0]
    ax.imshow(arr, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i, s in enumerate(styles):
        for j, m in enumerate(models):
            t = r[(r.subject_model == m) & (r.hint_style == s)]
            n = int(t.n_fr_with_reliance.iloc[0]) if len(t) else 0
            k = int(t.n_fr_robust_used.iloc[0]) if len(t) else 0
            if n == 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f3f2ef", zorder=2, linewidth=0))
                ax.text(j, i, "n=0", ha="center", va="center", fontsize=5.5, color=MUTED, zorder=3)
                continue
            faded = n < s3.SMALL_N
            if faded:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=SURFACE, alpha=0.6, zorder=2,
                                           linewidth=0))
            v = arr[i, j]
            ax.text(j, i, f"{100 * v:.0f}%\n{k}/{n}", ha="center", va="center", fontsize=5.5, zorder=3,
                    color=SURFACE if (v > 0.55 and not faded) else INK, linespacing=0.95)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=7)
    ax.set_yticks(range(len(styles)))
    ax.set_yticklabels([CUE_NAMES[s] for s in styles], fontsize=7)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(styles), 1), minor=True)
    ax.grid(True, which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("share of SSP false rejections confirmed robust_used (k/n); faded: n < 20", fontsize=7,
                 color=INK2)
    fig.text(0.01, 0.005, footer(counts), fontsize=4.6, color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.92, bottom=0.2)
    save(fig, out / f"{PLOT}_by_style")


def fig_grouped(r: pd.DataFrame, counts: pd.DataFrame, key: str, levels: list, colors: dict, names: dict,
                title: str, stem: str, out: Path) -> None:
    models = [m for m in s3.MODELS if m in set(r.subject_model)]
    fig, ax = plt.subplots(figsize=(FULL_W, 3.1))
    gw = 0.8
    bw = gw / len(levels)
    for i, m in enumerate(models):
        for j, lv in enumerate(levels):
            t = r[(r.subject_model == m) & (r[key] == lv)]
            if t.empty or t.n_fr_with_reliance.iloc[0] == 0:
                continue
            t = t.iloc[0]
            x = i - gw / 2 + bw * (j + 0.5)
            bar(ax, x, t.rate, t.ci_lo, t.ci_hi, t.n_fr_with_reliance, colors[lv], bw * 0.88)
            ax.text(x, (t.ci_hi if not pd.isna(t.ci_hi) else 0) + 0.015, f"{int(t.n_fr_with_reliance)}",
                    ha="center", va="bottom", fontsize=4.8, color=INK2, rotation=90)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], fontsize=7)
    ax.set_ylim(0, 1.15)
    pct_axis(ax, "y")
    ax.set_ylabel(YLAB)
    tidy(ax, grid_axis="y")
    ax.legend(handles=[Patch(color=colors[lv], label=names[lv]) for lv in levels], loc="upper left",
              bbox_to_anchor=(0, 1.13), ncol=len(levels), fontsize=6.5)
    fig.text(0.01, 0.005, f"{title}. " + footer(counts) + "; n above bars; faded: n < 20", fontsize=4.6,
             color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.87, bottom=0.26)
    save(fig, out / stem)


def main(argv=None) -> int:
    a = plot_args(PLOT, argv, doc=__doc__)
    data, out = a.data, a.out_dir
    apply_style()
    counts = pd.read_csv(data / "counts_by_model.csv")
    fig_model(pd.read_csv(data / "rates_by_model.csv"), counts, out)
    fig_style(pd.read_csv(data / "rates_by_model_style.csv"), counts, out)
    fig_grouped(pd.read_csv(data / "rates_by_model_case.csv"), counts, "case", ["positive", "negative"],
                CASE_COLOR, {"positive": "positive case", "negative": "negative case"}, "by manifest case",
                f"{PLOT}_by_case", out)
    rs = pd.read_csv(data / "rates_by_model_stability.csv")
    rs["stability_bin"] = rs.stability_bin.astype(int)
    fig_grouped(rs, counts, "stability_bin", [5, 6, 7, 8], STAB_COLOR,
                {b: f"stability {STABILITY_NAMES[b]}" for b in [5, 6, 7, 8]},
                "stability = 8 no-hint samples; bins only", f"{PLOT}_by_stability", out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
