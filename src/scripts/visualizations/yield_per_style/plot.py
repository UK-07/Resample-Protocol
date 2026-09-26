"""Yield per cue style — plot. Reads ONLY data/yield_by_model_style.csv (written by gather.py).

Grouped bars, cue style on x (paper order), one bar per model (paper colours; Qwen3.5-9B hatched), y = clean
unfaithful examples (re-rolls meeting dataset_B's conditions with label 0 or −1) per 1,000 original hinted
rollouts, 95 % bootstrap whiskers. Bars whose denominator is below 20 hinted rollouts are drawn faded (none are
expected).

Run:
    python -m src.scripts.visualizations.yield_per_style.plot [--data D] [--out-dir D]
Figure: <out-dir>/yield_per_style.{png,pdf} (default <cueball>/plots/yield_per_style/).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.scripts.visualizations.common.section1_plot import plot_args  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    CUE_NAMES, CUE_ORDER, FULL_W, INK2, MODEL_COLOR, MODEL_SHORT, SURFACE, apply_style, tidy,
)

PLOT = "yield_per_style"
MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
HATCH = {"qwen3.5-9b": "////"}
SMALL_N = 20


def main(argv=None) -> int:
    a = plot_args(PLOT, argv, doc=__doc__)
    apply_style()
    t = pd.read_csv(a.data / "yield_by_model_style.csv").set_index(["subject_model", "hint_style"])
    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
    x = np.arange(len(CUE_ORDER))
    w = 0.8 / len(MODELS)
    for i, m in enumerate(MODELS):
        for j, s in enumerate(CUE_ORDER):
            if (m, s) not in t.index:
                continue
            r = t.loc[(m, s)]
            xi = x[j] - 0.4 + (i + 0.5) * w
            faded = r.hinted_rollouts < SMALL_N
            ax.bar(xi, r.yield_per_1000, width=w * 0.92, color=MODEL_COLOR[m], hatch=HATCH.get(m),
                   edgecolor=SURFACE, linewidth=0.5, alpha=0.35 if faded else 1.0, zorder=2)
            ax.plot([xi, xi], [r.ci_lo, r.ci_hi], color=INK2, lw=0.6, zorder=3)
            if faded:
                ax.annotate(f"n={int(r.hinted_rollouts)}", (xi, r.ci_hi), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=5.5, color=INK2)
    ax.set_xticks(x, [CUE_NAMES[s] for s in CUE_ORDER], fontsize=7, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("clean unfaithful examples\nper 1,000 hinted rollouts")
    ax.set_ylim(0, None)
    tidy(ax, grid_axis="y")
    handles = [Patch(facecolor=MODEL_COLOR[m], hatch=HATCH.get(m), edgecolor=SURFACE, label=MODEL_SHORT[m])
               for m in MODELS]
    ax.legend(handles=handles, loc="upper left", ncol=1, fontsize=6.5)
    for ext in ("png", "pdf"):
        fig.savefig(a.out_dir / f"{PLOT}.{ext}", bbox_inches="tight")
    print(f"wrote {a.out_dir}/{PLOT}.png/.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
