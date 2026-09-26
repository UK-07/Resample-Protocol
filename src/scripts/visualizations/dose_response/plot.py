"""Dose-response — plot.

Reads ONLY data/{dose_by_model.csv, dose_by_model_case.csv, exclusions.csv}.
x = re-rolls on target (0–4) of the rollout's (question, hint) pair; y = share of hint-mentioning rollouts (every
category but `none`) that make a non-reliance claim (v2 role rejected or verification_only, or v2 verdict −1 =
incoherent); one line per model with Wilson 95 % whiskers; points with a denominator < 20 drawn faded.
Per-rollout weighting (original + re-rolls).

Figures:
  dose_response.{png,pdf}                  scope used_pairs (primary), both cases pooled
  dose_response_by_case.{png,pdf}          scope used_pairs, panels positive | negative

Run:
    python -m src.scripts.visualizations.dose_response.plot [--data D] [--out-dir D]
Figures: <out-dir>/dose_response[_by_case].{png,pdf} (default <cueball>/plots/dose_response/); --data defaults to
<out-dir>/data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.scripts.visualizations.common import section1_plot as SP  # noqa: E402
from src.scripts.visualizations.common import section3_claims as s3  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    FULL_W, INK2, MODEL_COLOR, MODEL_NAMES, MUTED, apply_style, pct_axis, tidy,
)

PLOT = "dose_response"
KS = [0, 1, 2, 3, 4]
OFFSET = 0.06     # horizontal dodge between models so whiskers do not overlap


def draw_lines(ax, t: pd.DataFrame, models: list[str]) -> None:
    for i, m in enumerate(models):
        d = t[t.subject_model == m].set_index("k").reindex(KS)
        x = np.array(KS) + (i - (len(models) - 1) / 2) * OFFSET
        ok = d.rate.notna().to_numpy()
        ax.plot(x[ok], d.rate[ok], color=MODEL_COLOR[m], linewidth=1.1, zorder=2, label=MODEL_NAMES[m])
        for xi, (r, lo, hi, n) in zip(x, d[["rate", "ci_lo", "ci_hi", "denominator"]].itertuples(index=False)):
            if pd.isna(r):
                continue
            alpha = 0.35 if n < s3.SMALL_N else 1.0
            ax.vlines(xi, lo, hi, color=MODEL_COLOR[m], linewidth=0.7, alpha=alpha, zorder=2)
            ax.plot(xi, r, "o", ms=3, color=MODEL_COLOR[m], alpha=alpha, zorder=3, markeredgewidth=0)
    ax.set_xticks(KS)
    ax.set_xticklabels([f"{k}/4" for k in KS])
    ax.set_xlim(-0.35, 4.35)
    ax.set_xlabel("re-rolls on target (of 4)")
    pct_axis(ax, "y")
    tidy(ax, grid_axis="y")


def n_note(t: pd.DataFrame, models: list[str]) -> str:
    parts = []
    for m in models:
        d = t[t.subject_model == m]
        parts.append(f"{MODEL_NAMES[m]} n={int(d.denominator.sum()):,} ({int(d.numerator.sum()):,} claims)")
    return "mentions (incl. −1) — " + "; ".join(parts)


def footer(excl: pd.DataFrame, scope: str) -> str:
    e = excl[(excl.scope == scope) & (excl.grouping == "model")]
    return s3.exclusion_footer(e, MODEL_NAMES)


def save(fig, stem: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def main(argv=None) -> int:
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    data, out = a.data, a.out_dir
    apply_style()
    by_model = pd.read_csv(data / "dose_by_model.csv")
    by_case = pd.read_csv(data / "dose_by_model_case.csv")
    excl = pd.read_csv(data / "exclusions.csv")
    ylabel = "non-reliance claims / mentions\n(rejected + verification only + −1)"

    for scope, stem, sub in (("used_pairs", "dose_response", "pairs whose original switched; both cases"),):
        t = by_model[by_model.scope == scope]
        models = [m for m in s3.MODELS if m in set(t.subject_model)]
        fig, ax = plt.subplots(figsize=(FULL_W, 3.2))
        draw_lines(ax, t, models)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_title(sub, fontsize=7, color=INK2, pad=2)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=6.5,
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=1.2)
        fig.text(0.01, 0.005, n_note(t, models) + "\n" + footer(excl, scope) + "; faded points: n < 20",
                 fontsize=4.8, color=MUTED, wrap=True, va="bottom", linespacing=1.4)
        fig.subplots_adjust(left=0.11, right=0.98, top=0.87, bottom=0.32)
        save(fig, out / stem)

    t = by_case[by_case.scope == "used_pairs"]
    models = [m for m in s3.MODELS if m in set(t.subject_model)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7), sharey=True)
    for ax, case in zip(axes, ("positive", "negative")):
        draw_lines(ax, t[t.case == case], models)
        ax.set_title(f"manifest case: {case}", fontsize=7.5)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=1.2)
    e = excl[(excl.scope == "used_pairs") & (excl.grouping == "model")]
    fig.text(0.01, 0.005, s3.exclusion_footer(e, MODEL_NAMES) + " (both cases); faded points: n < 20",
             fontsize=4.8, color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.28, wspace=0.08)
    save(fig, out / "dose_response_by_case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
