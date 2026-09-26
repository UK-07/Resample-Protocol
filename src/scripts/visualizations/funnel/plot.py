"""Yield funnel and token cost — plot. Reads ONLY data/{funnel_counts.csv, cost_by_model.csv, cost_summary.csv}.

    funnel.{png,pdf}            (a) pairs surviving each stage of the chain, log scale, one line per model
                                (both manifest cases pooled); (b) charged tokens per model by component
                                (generation OUTPUT tokens of the funnel's questions / SSP-flip pairs, and SERVED
                                judge calls, prompt + completion), with charged tokens per clean pair at the bar end.
    funnel_by_case.{png,pdf}    panel (a) drawn separately for manifest-positive and manifest-negative pairs.

Run:
    python -m src.scripts.visualizations.funnel.plot [--data D] [--out-dir D]
Figures: <out-dir>/funnel.{png,pdf}, <out-dir>/funnel_by_case.{png,pdf} (default <cueball>/plots/funnel/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.scripts.visualizations.common.section1_plot import plot_args  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    FULL_W, INK2, MODEL_COLOR, MODEL_SHORT, ORDINAL_5, SURFACE, apply_style, tidy,
)

PLOT = "funnel"
MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
STAGES = ["hinted_pairs", "ssp_flip", "robust_used", "binary_verdict_0", "clean"]
STAGE_NAMES = {"hinted_pairs": "hinted\npairs", "ssp_flip": "SSP\nflip", "robust_used": "robust\nused",
               "binary_verdict_0": "binary\nlabel 0", "clean": "clean"}
MARKERS = {"qwen3.5-9b": "s"}        # Qwen3.5-9B / Gemma-4-12B colours are close: a second encoding
# component → (legend label, fill, hatch)
COMPONENTS = [
    ("baseline_sample0", "baseline, sample 0 (SSP)", ORDINAL_5[0], None),
    ("baseline_samples_1to7", "baseline, samples 1–7 (RSP only)", ORDINAL_5[1], None),
    ("hinted_rollouts", "hinted rollouts", ORDINAL_5[2], None),
    ("rerolls", "k=4 re-rolls (SSP-flip pairs)", ORDINAL_5[3], None),
    ("judge_binary_orig", "judge: binary, SSP flips", "#c3c2b7", "////"),
    ("judge_v2_reroll", "judge: v2, their re-rolls", "#52514e", "////"),
]


def counts_panel(ax, counts: pd.DataFrame, case_group: str, title: str) -> None:
    c = counts[counts.case_group == case_group].set_index("subject_model")
    x = np.arange(len(STAGES))
    for m in MODELS:
        if m not in c.index:
            continue
        y = np.array([c.loc[m, f"n_{s}"] for s in STAGES], dtype=float)
        ax.plot(x, np.where(y > 0, y, np.nan), color=MODEL_COLOR[m], lw=1.4, marker=MARKERS.get(m, "o"), ms=3.5,
                label=f"{MODEL_SHORT[m]}: clean {int(c.loc[m, 'n_clean']):,} (−1: {int(c.loc[m, 'n_clean_v2_label_-1'])})",
                zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x, [STAGE_NAMES[s] for s in STAGES], fontsize=6.5)
    ax.set_xlim(-0.3, len(STAGES) - 0.7)
    ax.set_ylabel("question × style pairs")
    ax.set_title(title)
    tidy(ax, grid_axis="y")


def cost_panel(ax, cost: pd.DataFrame, summary: pd.DataFrame, xlabel: str) -> None:
    ch = cost[(cost.scope == "charged") & (cost.case_group == "all") & cost.in_figure].copy()
    ch["t"] = ch.output_tokens + ch.judge_prompt_tokens_served.fillna(0)   # generation: output only
    c = ch.pivot(index="subject_model", columns="component", values="t")
    y = np.arange(len(MODELS))[::-1]
    left = np.zeros(len(MODELS))
    for comp, _, color, hatch in COMPONENTS:
        v = np.array([c.loc[m, comp] if m in c.index else 0 for m in MODELS], dtype=float) / 1e6
        ax.barh(y, v, left=left, height=0.62, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=0.8)
        left += v
    s = summary.set_index("subject_model")
    for yi, m, tot in zip(y, MODELS, left):
        if m in s.index and pd.notna(s.loc[m, "charged_tokens_per_clean_pair"]):
            ax.annotate(f"{s.loc[m, 'charged_tokens_per_clean_pair'] / 1e6:.2f}M / clean pair", (tot, yi),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [MODEL_SHORT[m] for m in MODELS])
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(left.max(), 1e-9) * 1.4)
    tidy(ax, grid_axis="x")


def legend_components(ax):
    handles = [Patch(facecolor=col, hatch=h, edgecolor=SURFACE, label=lab) for _, lab, col, h in COMPONENTS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=5.8,
              handlelength=1.4)


def save(fig, out: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / stem}.png/.pdf")


def main(argv=None) -> int:
    args = plot_args(PLOT, argv, doc=__doc__)
    data, out = args.data, args.out_dir
    apply_style()
    counts = pd.read_csv(data / "funnel_counts.csv")
    cost = pd.read_csv(data / "cost_by_model.csv")
    summary = pd.read_csv(data / "cost_summary.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.9), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.42})
    counts_panel(a, counts, "all", "(a) pairs remaining at each stage")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=5.8, handlelength=1.5)
    cost_panel(b, cost, summary, "tokens (millions): generation output + judge calls")
    b.set_title("(b) token cost by component")
    legend_components(b)
    save(fig, out, "funnel")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, cg in zip(axes, ("positive", "negative")):
        counts_panel(ax, counts, cg, f"manifest-{cg} pairs")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=5.8, handlelength=1.5)
    save(fig, out, "funnel_by_case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
