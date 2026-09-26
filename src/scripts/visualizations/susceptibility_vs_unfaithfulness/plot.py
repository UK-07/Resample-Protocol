"""Susceptibility vs unfaithfulness — plot. Reads only data/cells_model_style.csv.

    susceptibility_vs_unfaithfulness.{png,pdf}   one point per model x cue style (datasets pooled), one panel per
                                                 manifest case. x = robust susceptibility = clean hinted rollouts
                                                 whose (question, hint) is robust_used / clean hinted rollouts;
                                                 y = RSP unfaithful rate ((v2 0 + -1) / all to-target re-rolls of
                                                 robust_used questions; -1 counted as unfaithful).
    spearman.csv                                 Spearman rho per case (all points; points with both n >= 20),
                                                 next to the figure
    dropped_points.csv                           model x style x case cells not drawn (n_clean == 0 or rsp_den == 0)

Colour = model, marker = cue style; Wilson 95 % whiskers on both axes; a point with either denominator < 20
is faded and carries n (x n / y n).

Run:
    python -m src.scripts.visualizations.susceptibility_vs_unfaithfulness.plot [--data D] [--out-dir D]
Figure: <out-dir>/susceptibility_vs_unfaithfulness.{png,pdf} (default
<cueball>/plots/susceptibility_vs_unfaithfulness/); --data defaults to <out-dir>/data.
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
    CUE_NAMES, CUE_ORDER, FULL_W, GRID, INK, MODEL_COLOR, MODEL_NAMES, MODEL_ORDER, MUTED, apply_style, pct_axis,
)

PLOT = "susceptibility_vs_unfaithfulness"
SMALL_N = 20
FADE = 0.3
CASES = ["positive", "negative"]
MODELS = [m for m in MODEL_ORDER if m != "qwen3.6-27b"]
MARKER = dict(zip(CUE_ORDER, ["o", "s", "^", "v", "D", "P", "X", "*"]))
EDGE = {"qwen3.5-9b": INK}          # Qwen3.5-9B vs Gemma-4-12B greens: dark marker edge


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank())[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else P.plot_dir(PLOT)
    data = Path(a.data) if a.data else out / "data"
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    t = pd.read_csv(data / "cells_model_style.csv")
    drop = (t.n_clean == 0) | (t.rsp_den == 0)
    t[drop][["subject_model", "hint_style", "case", "n_clean", "rsp_den"]].to_csv(out / "dropped_points.csv", index=False)
    print(f"points not drawn (n_clean == 0 or rsp_den == 0): {int(drop.sum())}")
    t = t[~drop]
    t["small"] = (t.n_clean < SMALL_N) | (t.rsp_den < SMALL_N)

    rho = []
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9), sharey=True)
    for ax, case in zip(axes, CASES):
        g = t[t.case == case]
        for _, r in g.iterrows():
            al = FADE if r.small else 1.0
            c = MODEL_COLOR[r.subject_model]
            ax.errorbar(r.robust_susc, r.rsp_rate,
                        xerr=[[max(0.0, r.robust_susc - r.robust_susc_lo)], [max(0.0, r.robust_susc_hi - r.robust_susc)]],
                        yerr=[[max(0.0, r.rsp_rate - r.rsp_rate_lo)], [max(0.0, r.rsp_rate_hi - r.rsp_rate)]],
                        fmt="none", ecolor=c, elinewidth=0.5, alpha=al * 0.8, zorder=1)
            ax.plot(r.robust_susc, r.rsp_rate, MARKER[r.hint_style], ms=4.2, color=c, alpha=al,
                    mec=EDGE.get(r.subject_model, c), mew=0.5, zorder=2)
            if r.small:
                ax.annotate(f"n={int(r.n_clean)}/{int(r.rsp_den)}", (r.robust_susc, r.rsp_rate), fontsize=4.5,
                            color=MUTED, xytext=(3, 2), textcoords="offset points")
        big = g[~g.small]
        rho_all, rho_big = spearman(g.robust_susc, g.rsp_rate), spearman(big.robust_susc, big.rsp_rate)
        rho.append({"case": case, "n_points": len(g), "spearman_rho": rho_all,
                    "n_points_n_ge_20": len(big), "spearman_rho_n_ge_20": rho_big})
        ax.set_title(f"{case} case   Spearman ρ = {rho_all:+.2f} (n≥20: {rho_big:+.2f})", fontsize=7)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        pct_axis(ax, "x", top=1.0)
        pct_axis(ax, "y", top=1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("robust susceptibility", fontsize=6.5)
    axes[0].set_ylabel("RSP unfaithful rate", fontsize=6.5)
    pd.DataFrame(rho).to_csv(out / "spearman.csv", index=False)

    hm = [Line2D([], [], marker="o", ls="", color=MODEL_COLOR[m], mec=EDGE.get(m, MODEL_COLOR[m]), mew=0.5, ms=4,
                 label=MODEL_NAMES[m]) for m in MODELS]
    hs = [Line2D([], [], marker=MARKER[s], ls="", color=MUTED, ms=4, label=CUE_NAMES[s]) for s in CUE_ORDER]
    hs.append(Line2D([], [], marker="o", ls="", color=MUTED, alpha=FADE, ms=4, label=f"n < {SMALL_N} (x n / y n)"))
    fig.legend(handles=hm, loc="lower center", ncol=5, fontsize=5.8, bbox_to_anchor=(0.5, -0.02))
    fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=5.8, bbox_to_anchor=(0.5, -0.12))
    fig.text(0.5, -0.19, "One point per model × cue style, datasets pooled. Robust susceptibility = clean hinted rollouts "
             "whose question is robust_used / clean hinted rollouts.\nRSP rate = share unfaithful among the to-target re-rolls of "
             "those questions (v2 role-based judge, incoherent −1 counted as unfaithful). Whiskers: Wilson 95%.\n"
             "Nemotron-Nano-9B re-rolls used a 24,576-token budget (16,384 elsewhere).", ha="center", va="top", fontsize=5.3, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.2, wspace=0.18)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{PLOT}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/{PLOT}.png/.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
