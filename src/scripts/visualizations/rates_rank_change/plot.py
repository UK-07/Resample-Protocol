"""Rates rank change — plot. Reads only data/cells_model_style.csv and data/cells_model_dataset.csv.

    rates_rank_change_styles.{png,pdf}   bump chart: rank of each cue style by unfaithful rate under SSP (left
                                         axis) vs RSP (right axis); rows = manifest case, one panel per model;
                                         rates pooled over the 4 datasets (data/cells_model_style.csv)
    rates_rank_change_models.{png,pdf}   the same for the 5 models ranked within each dataset (styles pooled);
                                         rows = case, one panel per dataset (data/cells_model_dataset.csv)
    ranks_styles.csv, ranks_models.csv (+ *_left_out.csv)   the ranks drawn, Kendall tau-b per panel (next to the figures)

Rank 1 = highest unfaithful rate. Ranked items = those with BOTH denominators >= 20 (the rest are left out and
listed in ranks_*_left_out.csv next to the figures); tied rates share the average rank and one label ("A=B x%").
Kendall tau-b per panel. SSP: binary judge; RSP: robust_used re-rolls, v2 judge, -1 counted as unfaithful.

Run:
    python -m src.scripts.visualizations.rates_rank_change.plot [--data D] [--out-dir D]
Figures: <out-dir>/rates_rank_change_*.{png,pdf} (default <cueball>/plots/rates_rank_change/); --data defaults to
<out-dir>/data. The ranks CSVs go next to the figures, never into data/.
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
    AXIS, CUE_ABBREV, CUE_NAMES, CUE_ORDER, DATASET_NAMES, DATASET_ORDER, FULL_W, INK2, MODEL_COLOR,
    MODEL_NAMES, MODEL_ORDER, MODEL_SHORT, MUTED, SLOTS, apply_style,
)

PLOT = "rates_rank_change"
SMALL_N = 20
CASES = ["positive", "negative"]
MODELS = [m for m in MODEL_ORDER if m != "qwen3.6-27b"]
STYLE_COLOR = dict(zip(CUE_ORDER, SLOTS))
DASH = {"qwen3.5-9b": (0, (3, 1.5))}        # Qwen3.5-9B vs Gemma-4-12B greens: dashed line


def rank_desc(values: pd.Series) -> pd.Series:
    """Rank 1 = highest rate; tied rates share the average rank."""
    return values.rank(ascending=False, method="average")


def kendall_tau_b(a, b) -> float:
    """Kendall tau-b (tie-corrected) between two rank vectors over the same items."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2:
        return float("nan")
    conc = ties_a = ties_b = 0.0
    n0 = n * (n - 1) / 2
    for i in range(n):
        for j in range(i + 1, n):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            conc += da * db
            ties_a += da == 0
            ties_b += db == 0
    den = np.sqrt((n0 - ties_a) * (n0 - ties_b))
    return float(conc / den) if den > 0 else float("nan")


def rank_table(t: pd.DataFrame, item: str, panel: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (case, panel): average ranks of ``item`` among those with BOTH denominators >= 20.
    Returns (ranks, left_out) — left_out lists every item not ranked and why."""
    out, dropped = [], []
    for (case, p), g in t.groupby(["case", panel]):
        small = (g.ssp_den < SMALL_N) | (g.rsp_den < SMALL_N)
        for _, x in g[small].iterrows():
            dropped.append({"case": case, panel: p, item: x[item], "ssp_den": int(x.ssp_den),
                            "rsp_den": int(x.rsp_den), "reason": f"n < {SMALL_N}"})
        g = g[~small].set_index(item)
        if g.empty:
            continue
        r = g.assign(rank_ssp=rank_desc(g.ssp_rate), rank_rsp=rank_desc(g.rsp_rate))
        r["kendall_tau_b_panel"] = kendall_tau_b(r.rank_ssp, r.rank_rsp)
        r["n_items_ranked"] = len(r)
        out.append(r.reset_index())
    cols = ["case", panel, item, "ssp_rate", "ssp_den", "rsp_rate", "rsp_den", "n_rsp_incoherent", "rank_ssp",
            "rank_rsp", "n_items_ranked", "kendall_tau_b_panel"]
    return pd.concat(out, ignore_index=True)[cols], pd.DataFrame(dropped)


def fmt(v: float) -> str:
    """Enough decimals that distinct small rates stay distinct (sub-1 % rates get two)."""
    if v < 0.01:
        return f"{100 * v:.2f}%"
    return f"{100 * v:.1f}%" if v < 0.1 else f"{100 * v:.0f}%"


NEMO_NOTE = "Nemotron-Nano-9B re-rolls used a 24,576-token budget (16,384 elsewhere)."


def bump(ax, r: pd.DataFrame, item: str, color: dict, label: dict, n_max: int, lab_size: float,
         dash: dict | None = None) -> None:
    for _, x in r.iterrows():
        c = color[x[item]]
        ax.plot([0, 1], [x.rank_ssp, x.rank_rsp], color=c, lw=1.3, ls=(dash or {}).get(x[item], "-"),
                solid_capstyle="round", zorder=2)
        ax.plot([0, 1], [x.rank_ssp, x.rank_rsp], "o", ms=3.2, color=c, zorder=3)
    # left side: names only; right side: rate only (one label per occupied rank; tied items share it)
    for rk, g in r.groupby("rank_ssp"):
        names = [label[v] for v in g[item]]
        c = color[g[item].iloc[0]] if len(g) == 1 else INK2
        joined = "\n".join("=".join(names[i:i + 2]) for i in range(0, len(names), 2))  # wrap ties, 2 per line
        ax.text(-0.08, rk, joined, ha="right", va="center", fontsize=lab_size, color=c, linespacing=0.9)
    for rk, g in r.groupby("rank_rsp"):
        c = color[g[item].iloc[0]] if len(g) == 1 else INK2
        ax.text(1.08, rk, fmt(g["rsp_rate"].iloc[0]), ha="left", va="center", fontsize=lab_size, color=c)
    for rk, g in r.groupby("rank_ssp"):
        c = color[g[item].iloc[0]] if len(g) == 1 else INK2
        ax.text(0.0, rk - 0.32, fmt(g["ssp_rate"].iloc[0]), ha="center", va="bottom", fontsize=lab_size * 0.85,
                color=c, alpha=0.8)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(n_max + 0.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SSP", "RSP"], fontsize=6.5)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    for x0 in (0, 1):
        ax.axvline(x0, color=AXIS, lw=0.5, zorder=1)


def styles_figure(rk: pd.DataFrame, out: Path) -> None:
    models = [m for m in MODELS if m in set(rk.subject_model)]
    fig, axes = plt.subplots(2, len(models), figsize=(FULL_W + 0.6, 4.6), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.9, bottom=0.14, wspace=0.95, hspace=0.5)
    for i, case in enumerate(CASES):
        for j, m in enumerate(models):
            ax = axes[i, j]
            r = rk[(rk.case == case) & (rk.subject_model == m)]
            if r.empty:
                ax.set_visible(False)
                continue
            bump(ax, r, "hint_style", STYLE_COLOR, CUE_ABBREV, 8, 4.6)
            tau = r.kendall_tau_b_panel.iloc[0]
            ax.set_title(f"{MODEL_SHORT[m]}\nτ = {tau:+.2f}", fontsize=6.3, loc="center")
    for i, case in enumerate(CASES):
        pos = axes[i, 0].get_position()
        fig.text(0.005, pos.y0 + pos.height / 2, f"{case} case", rotation=90, va="center", ha="left",
                 fontsize=7.5, color=INK2)
    h = [Line2D([], [], color=STYLE_COLOR[s], lw=1.3, label=f"{CUE_ABBREV[s]} = {CUE_NAMES[s]}") for s in CUE_ORDER]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=5.6, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.4, columnspacing=0.9)
    fig.text(0.5, -0.045, "Rank 1 = highest unfaithful rate (datasets pooled). SSP: binary judge on SSP flips; "
             "RSP: robust_used re-rolls, v2 judge (−1 = unfaithful). "
             f"Tied rates share the average rank; items with n < {SMALL_N} on either side are not ranked. "
             "τ = Kendall τ-b.\nLeft: SSP rank (small number = SSP rate); right: RSP rate. Ranks among sub-1% rates "
             "are noisy. " + NEMO_NOTE, ha="center", fontsize=5.3, color=MUTED)
    save(fig, out)


def models_figure(rk: pd.DataFrame, out: Path) -> None:
    """Rows = case x dataset-pair, 2 dataset panels per row (4 x 2 grid): wide enough for the model names."""
    dsets = [d for d in DATASET_ORDER if d in set(rk.dataset)]
    ncol = 2
    nrow_per_case = int(np.ceil(len(dsets) / ncol))
    fig, axes = plt.subplots(2 * nrow_per_case, ncol, figsize=(FULL_W, 1.45 * 2 * nrow_per_case + 0.7),
                             squeeze=False)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.12, wspace=1.1, hspace=0.7)
    for i, case in enumerate(CASES):
        for k in range(nrow_per_case * ncol):
            ax = axes[i * nrow_per_case + k // ncol, k % ncol]
            if k >= len(dsets):
                ax.set_visible(False)
                continue
            d = dsets[k]
            r = rk[(rk.case == case) & (rk.dataset == d)]
            if r.empty:
                ax.set_visible(False)
                continue
            bump(ax, r, "subject_model", MODEL_COLOR, MODEL_SHORT, len(MODELS), 5.8, dash=DASH)
            tau = r.kendall_tau_b_panel.iloc[0]
            ax.set_title(f"{DATASET_NAMES[d]} · {case}   τ = {tau:+.2f}", fontsize=6.5, loc="center")
        top, bot = axes[i * nrow_per_case, 0].get_position(), axes[(i + 1) * nrow_per_case - 1, 0].get_position()
        fig.text(0.005, (top.y1 + bot.y0) / 2, f"{case} case", rotation=90, va="center", ha="left", fontsize=7.5,
                 color=INK2)
    h = [Line2D([], [], color=MODEL_COLOR[m], lw=1.3, ls=DASH.get(m, "-"), label=MODEL_NAMES[m]) for m in MODELS]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=5.6, bbox_to_anchor=(0.5, 0.02),
               handlelength=1.8, columnspacing=0.9)
    fig.text(0.5, 0.0, "Rank 1 = highest unfaithful rate (styles pooled). SSP: binary judge on SSP flips; "
             "RSP: robust_used re-rolls, v2 judge (−1 = unfaithful).\n"
             f"Tied rates share the average rank; items with n < {SMALL_N} on either side are not ranked. "
             "τ = Kendall τ-b. Ranks among sub-1% rates are noisy. " + NEMO_NOTE,
             ha="center", fontsize=5.3, color=MUTED)
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
    ms = pd.read_csv(data / "cells_model_style.csv")
    md = pd.read_csv(data / "cells_model_dataset.csv")
    rs, ds = rank_table(ms, "hint_style", "subject_model")
    rm, dm = rank_table(md, "subject_model", "dataset")
    # derived tables go next to the figures (plot.py never writes into data/)
    rs.to_csv(out / "ranks_styles.csv", index=False)
    rm.to_csv(out / "ranks_models.csv", index=False)
    ds.to_csv(out / "ranks_styles_left_out.csv", index=False)
    dm.to_csv(out / "ranks_models_left_out.csv", index=False)
    print(f"left out (n < {SMALL_N}): {len(ds)} model x style x case items, {len(dm)} dataset x model x case items")
    styles_figure(rs, out / "rates_rank_change_styles")
    models_figure(rm, out / "rates_rank_change_models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
