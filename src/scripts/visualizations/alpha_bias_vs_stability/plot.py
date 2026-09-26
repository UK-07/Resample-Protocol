"""Alpha bias vs baseline stability — plot. Reads ONLY data/bins.csv (written by gather.py).

  alpha_bias_vs_stability.{png,pdf}   y = measured − α-implied noise (share of to-target flips), x = stability bin
                                      (≤5/8 … 8/8); rows = manifest case (positive, negative); columns = the two
                                      measured-noise definitions side by side (literal: 1 − robust_used share |
                                      strict: never re-flips); one line per model with 95 % question-cluster
                                      bootstrap intervals. Points with n < 20 pairs faded, n printed.
Style from paper_plots.

Run:
    python -m src.scripts.visualizations.alpha_bias_vs_stability.plot [--data D] [--out-dir D]
Figure: <out-dir>/alpha_bias_vs_stability.{png,pdf} (default <cueball>/plots/alpha_bias_vs_stability/).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.scripts.visualizations.common import alpha_common as ac  # noqa: E402
from src.scripts.visualizations.common.section1_plot import plot_args  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    AXIS, FULL_W, MODEL_COLOR, MODEL_NAMES, MUTED, SURFACE, apply_style, pct_axis, tidy,
)

PLOT = "alpha_bias_vs_stability"
DEFS = [("literal", "literal: 1 − robust_used share", "n_pairs_labeled", "faded_literal"),
        ("strict", "strict: never re-flips (0 of the\nnon-truncated re-rolls)", "n_pairs_strict", "faded_strict")]


def main(argv=None) -> int:
    a = plot_args(PLOT, argv, doc=__doc__)
    out = a.out_dir
    apply_style()
    b = pd.read_csv(a.data / "bins.csv")
    models = [m for m in ac.MODELS if m in set(b["subject_model"])]
    offs = dict(zip(models, np.linspace(-0.18, 0.18, len(models))))
    lo = np.nanmin(np.r_[b["diff_literal_lo"], b["diff_strict_lo"]])
    hi = np.nanmax(np.r_[b["diff_literal_hi"], b["diff_strict_hi"]])
    pad = 0.05 * (hi - lo)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.1), sharex=True, sharey=True)
    for i, case in enumerate(ac.CASES):
        for j, (d, title, ncol, fcol) in enumerate(DEFS):
            ax = axes[i, j]
            ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
            for m in models:
                t = b[(b["subject_model"] == m) & (b["case"] == case)].sort_values("stability_bin")
                if t.empty:
                    continue
                x = t["stability_bin"].to_numpy() + offs[m]
                y = t[f"diff_{d}"].to_numpy()
                ax.plot(x, y, color=MODEL_COLOR[m], linewidth=1.0, zorder=2)
                for xi, yi, l, h, n, f in zip(x, y, t[f"diff_{d}_lo"], t[f"diff_{d}_hi"], t[ncol], t[fcol]):
                    a_ = ac.FADE_ALPHA if f else 1.0
                    ax.vlines(xi, l, h, color=MODEL_COLOR[m], linewidth=0.8, alpha=a_, zorder=2)
                    ax.scatter(xi, yi, s=14, color=MODEL_COLOR[m], edgecolor=SURFACE, linewidth=0.4, alpha=a_,
                               zorder=3)
                    if f:
                        ax.annotate(f"n={int(n)}", (xi, yi), xytext=(3, 0), textcoords="offset points",
                                    fontsize=4.8, color=MUTED, va="center")
            ax.set_ylim(lo - pad, hi + pad)
            pct_axis(ax, "y")
            tidy(ax, grid_axis="y")
            if i == 0:
                ax.set_title(title, fontsize=7.4)
            if j == 0:
                ax.set_ylabel(f"{case} case\nmeasured − α-implied noise")
            ax.set_xticks(ac.STABILITY_BINS)
            ax.set_xticklabels([ac.STABILITY_LABELS[k] for k in ac.STABILITY_BINS])
    fig.supxlabel("baseline stability (no-hint samples agreeing with the modal answer; bins the data only)",
                  fontsize=7.4, color=axes[0, 0].xaxis.label.get_color(), y=0.035)
    h = [Line2D([], [], color=MODEL_COLOR[m], marker="o", markersize=3.5, linewidth=1.0, label=MODEL_NAMES[m])
         for m in models]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(models), fontsize=6.4,
               columnspacing=1.0, handletextpad=0.4,
               title=f"95 % question-cluster bootstrap intervals; faded: fewer than {ac.MIN_N} to-target pairs",
               title_fontsize=6.2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{PLOT}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"figures → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
