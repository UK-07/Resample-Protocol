#!/usr/bin/env python3
"""Crosschecks with clustered uncertainty and explicit non-causal labels.

The baseline-only prediction exactly follows Resample-Protocol's
src/scripts/visualizations/common/alpha_common.py::load_rows/cell_counts:
sum over ALL SSP-eligible originals of (8-stability)/8/(n_options-1), divided
by the number of clean SSP target flips. Eligible truncated/unanswered cued
draws stay in this predictor numerator; they cannot contribute target flips.

Changes from the old public plot are deliberate: alpha is not statistically
clipped; every quantity has the same question-cluster bootstrap; and strict
non-recurrence uses k_hit==0 over ALL SSP flips, matching the revised paper's
primary missing-as-miss analysis. The old plot excluded all-four-truncated
pairs from this strict denominator (163 pairs here). Missing-outcome bounds
are reported separately in the paper. Plot-only tails beyond 100% receive an
upward arrow; no underlying point estimate or confidence interval is clipped.

The package owns the bootstrap implementation. No input bundle is modified.
"""
from . import common as rc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

MODELS = rc.MODELS
NAMES = rc.MODEL_SHORT
KEYS = ["subject_model", "dataset", "hint_style", "case"]
FONT = 7.5
STYLE = {"font.family": "DejaVu Sans", "font.size": FONT,
                    "axes.labelsize": FONT, "axes.titlesize": 8,
                    "xtick.labelsize": FONT, "ytick.labelsize": FONT,
                    "legend.fontsize": FONT, "axes.spines.top": False,
                    "axes.spines.right": False, "axes.linewidth": .6,
                    "xtick.major.width": .5, "ytick.major.width": .5,
                    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                    "pdf.fonttype": 42, "ps.fonttype": 42,
                    "savefig.facecolor": "white"}


def tables(paths):
    df = rc.load_pairs(paths.results)
    bs = rc.ClusterBootstrap(df.qkey)
    assert bs.Q == 3982 and bs.fingerprint == "1a8b5e9a029dc3ce"
    # Re-derive and verify the public source's SSP eligibility and clean-flip
    # predicates before relying on the archived eligibility flags.
    b0, h, t = df.b0, df.model_answer, df.target_option
    wrong_to_wrong = (df.case == "positive") & b0.notna() & (b0 != t) & (b0 != df.groundtruth)
    eligible = b0.notna() & (b0 != t) & ~wrong_to_wrong
    clean_flip = eligible & h.notna() & ~df.truncated & (h == t)
    assert (eligible == df.eligible).all()
    assert (clean_flip == df.ssp_flip).all()
    e = df[eligible].copy()
    numerator = (8-e.baseline_stability.astype(float))/8/(e.n_options.astype(float)-1)
    pred = rc.plain(rc.ratio_table(bs, e, numerator, e.ssp_flip.astype(float), KEYS, "baseline_prediction"))
    # Ratio cancellation is checked independently using the original formula.
    direct = e.assign(_v=numerator).groupby(KEYS).agg(v=("_v", "mean"), p=("ssp_flip", "mean"))
    assert np.allclose(pred.set_index(KEYS).baseline_prediction.sort_index(), (direct.v/direct.p).sort_index())
    a = pd.read_csv(paths.results / "s1_persistence/alpha_vs_measured_noise.csv")
    a = a[a.grouping == "+".join(KEYS)].copy()
    c = a.merge(pred[KEYS + ["baseline_prediction", "baseline_prediction_ci_lo", "baseline_prediction_ci_hi"]], on=KEYS, validate="one_to_one")
    assert len(c) == 320
    assert (c.groupby("subject_model").size() == 64).all()
    print("Baseline prediction: 320 cells; direct-formula equality verified;")
    print(f"bootstrap {bs.n_boot} draws, seed {bs.seed}, {bs.Q} canonical questions, fingerprint {bs.fingerprint}.")
    print(f"Prediction point range: {100*c.baseline_prediction.min():.3f}% to {100*c.baseline_prediction.max():.3f}%.")
    print(f"Prediction CI tails >100%: {(c.baseline_prediction_ci_hi>1).sum()}; alpha CI tails >100%: {(c.one_minus_alpha_ci_hi>1).sum()}.")
    c.to_csv(paths.derived / "noise_crosscheck_clusterCI.csv", index=False)
    return c


def draw(c, paths, strict=False):
    field = "nonrecurring_share_missing_as_miss" if strict else "nonpersistent_share"
    metric_name = "Non-recurring" if strict else "Non-persistent"
    metrics = [(field, "#0072B2", "o", metric_name),
               ("one_minus_alpha", "#8E44AD", "D", r"$1-\alpha$"),
               ("baseline_prediction", "#D55E00", "s", "Baseline prediction")]
    fig, axes = plt.subplots(5, 1, figsize=(5.5, 7.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.975, bottom=.165, top=.97, hspace=.26)
    for ax, model in zip(axes, MODELS):
        d = c[c.subject_model == model].sort_values([field, "dataset", "hint_style", "case"])
        for j, (_, r) in enumerate(d.iterrows(), 1):
            pos = r.case == "positive"
            fade = bool(r.small_n_lt_20_flips)
            for offset, (f, color, marker, _) in zip([-.2, 0, .2], metrics):
                x = j + offset
                point, lo, hi = 100*np.array([r[f], r[f+"_ci_lo"], r[f+"_ci_hi"]])
                a = .25 if fade else .78
                ax.plot([x, x], [lo, hi], color=color, lw=.45, alpha=a*.65, zorder=1)
                if hi > 100:
                    ax.plot(x, 101, marker="^", color=color, ms=2.7, alpha=a, clip_on=False, zorder=3)
                ax.plot(x, min(point, 100), marker=marker, ms=2.7, markeredgewidth=.55,
                        markerfacecolor=color if pos else "white", markeredgecolor=color,
                        alpha=a, zorder=3)
        ax.set(xlim=(.25, 64.75), ylim=(-4, 105), yticks=[0, 50, 100])
        ax.grid(axis="y", color=".88", lw=.4, zorder=0)
        ax.text(.015, .91, NAMES[model], transform=ax.transAxes, ha="left", va="top", fontsize=8)
    axes[-1].set_xticks([1, 16, 32, 48, 64])
    axes[-1].set_xlabel(f"Cell rank within model (sorted by {metric_name.lower()} share)")
    fig.supylabel("Share of SSP flips, %", x=.018, fontsize=7.5)
    hs = [Line2D([], [], marker=mk, linestyle="none", markerfacecolor=col, markeredgecolor=col,
                 markersize=3.5, label=lab) for _, col, mk, lab in metrics]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(.5, .063), ncol=3,
               frameon=False, handletextpad=.4, columnspacing=1)
    fig.text(.5, .030, "Filled: positive; hollow: negative; faded: n < 20; arrows: CI extends above 100%",
             ha="center", fontsize=7.5)
    filename = "fig_noise_crosscheck_strict.pdf" if strict else "fig_noise_crosscheck_literal.pdf"
    fig.savefig(paths.figures / filename, metadata={"Creator": "build_revision_crosscheck.py",
                "Subject": "Question-cluster 95% intervals; baseline formula verified against public alpha_common.py",
                "CreationDate": None, "ModDate": None})
    plt.close(fig)


def build(paths):
    """Write both crosscheck PDFs and their complete numerical source table."""
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.derived.mkdir(parents=True, exist_ok=True)
    c = tables(paths)
    with plt.rc_context(STYLE):
        draw(c, paths)
        draw(c, paths, strict=True)
    print("Rendered both full-width crosscheck PDFs.")
