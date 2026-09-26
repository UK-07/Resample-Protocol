"""Render the revised alpha and exclusive-role figures from prepared tables.

All figure inputs are explicit in ``build(paths)``. Missing stability slices are
recomputed with the package's dataset-stratified question-cluster bootstrap,
then saved to ``paths.derived``. The supplied bundle is never modified.
"""
from . import common as rc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
NAMES = dict(zip(MODELS, ["Nemotron", "Qwen3-8B", "Qwen3.5-9B", "OLMo-3-7B", "Gemma-4-12B"]))
COLORS = dict(zip(MODELS, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]))
CATEGORIES = ["coherent_rejected", "coherent_verification_only", "incoherent"]
CAT_LABELS = {"coherent_rejected": "Coherent\nrejected", "coherent_verification_only": "Coherent\nverification only", "incoherent": "Incoherent"}
CUES = ["expert_opinion", "unethical_info", "tool_output", "consensus", "metadata", "answer_key_artifact", "grader_hacking", "post_hoc"]
CUE_LABELS = ["Expert opinion", "Unethical info.", "Tool output", "Consensus", "Metadata", "Answer key", "Grader hacking", "Post hoc"]
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


def save(fig, name, paths):
    """Keep physical dimensions fixed; cropping would alter final font size."""
    fig.savefig(paths.figures / name, metadata={"Creator": "build_revision_alpha_roles.py",
                "Subject": "Question-cluster bootstrap intervals from revision_analysis, 2026-09-26",
                "CreationDate": None, "ModDate": None})
    plt.close(fig)


def model_legend(fig, y=.025, ncol=3, markersize=3.5):
    hs = [Line2D([], [], marker="o", linestyle="none", color=COLORS[m], markersize=markersize, label=NAMES[m]) for m in MODELS]
    return fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(.5, y), ncol=ncol,
                      frameon=False, handletextpad=.35, columnspacing=.85,
                      borderaxespad=0, labelspacing=.4)


def alpha_panel(ax, data, metric, xmax=100, ylabel=True, title=None):
    ycol = "nonpersistent_share" if metric == "literal" else "nonrecurring_share_missing_as_miss"
    # Intervals are drawn without statistical clipping. Axes limits can conceal
    # an interval tail; explicit right-pointing arrows identify those tails.
    for m in MODELS:
        d = data[data.subject_model == m]
        for _, r in d.iterrows():
            x, y = 100 * r.one_minus_alpha, 100 * r[ycol]
            if not np.isfinite(x + y) or x > xmax:
                continue
            lo, hi = 100 * r.one_minus_alpha_ci_lo, 100 * r.one_minus_alpha_ci_hi
            yl, yh = 100 * r[ycol + "_ci_lo"], 100 * r[ycol + "_ci_hi"]
            a = .18 if r.small_n_lt_20_flips else .42
            ax.plot([lo, hi], [y, y], color=COLORS[m], lw=.45, alpha=a, zorder=1)
            ax.plot([x, x], [yl, yh], color=COLORS[m], lw=.45, alpha=a, zorder=1)
            if hi > xmax:
                ax.plot(xmax - .007*xmax, y, marker=">", ms=3, color=COLORS[m], alpha=a + .25, clip_on=False, zorder=2)
            ax.plot(x, y, "o", ms=2.6, markeredgewidth=0, color=COLORS[m],
                    alpha=.27 if r.small_n_lt_20_flips else .75, zorder=3)
    ax.plot([0, min(xmax, 100)], [0, min(xmax, 100)], ls="--", lw=.65, color=".3", zorder=0)
    ax.set(xlim=(-.025*xmax, xmax), ylim=(-2.5, 103), xlabel=r"$1-\alpha$, %")
    ax.set_xticks([0, xmax/2, xmax])
    ax.set_yticks([0, 25, 50, 75, 100])
    if ylabel:
        ax.set_ylabel("Non-persistent pairs, %" if metric == "literal" else "Non-recurring pairs, %")
    if title:
        ax.set_title(title, pad=5)
    ax.grid(axis="both", lw=.35, color=".91", zorder=-5)


def main_alpha(cells, paths):
    for metric, name in [("literal", "fig_alpha_vs_measured_noise.pdf"), ("strict", "fig_alpha_vs_measured_noise_strict.pdf")]:
        fig, ax = plt.subplots(figsize=(2.695, 2.25))
        fig.subplots_adjust(left=.20, right=.94, bottom=.40, top=.97)
        alpha_panel(ax, cells, metric)
        model_legend(fig, y=.012, ncol=2)
        save(fig, name, paths)


def appendix_alpha(cells, paths):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.05))
    fig.subplots_adjust(left=.095, right=.975, bottom=.19, top=.955, wspace=.31, hspace=.48)
    for ax, (dataset, title) in zip(axes.flat, [("commonsense_qa", "CommonsenseQA"), ("medqa", "MedQA"), ("gpqa", "GPQA"), ("mmlu_pro", "MMLU-Pro")]):
        # Full 0-200% x-range retains the five unbounded ratio intervals.
        alpha_panel(ax, cells[cells.dataset == dataset], "literal", xmax=200, title=title)
    model_legend(fig, y=.025)
    save(fig, "fig_alpha_vs_measured_noise_by_dataset.pdf", paths)

    neg = cells[cells.case == "negative"]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.05))
    fig.subplots_adjust(left=.095, right=.975, bottom=.19, top=.955, wspace=.31, hspace=.48)
    for row, metric in enumerate(["literal", "strict"]):
        for col, xmax in enumerate([200, 10]):
            alpha_panel(axes[row, col], neg, metric, xmax=xmax, title="Full range" if col == 0 else r"Zoom: $1-\alpha\leq10\%$")
    model_legend(fig, y=.025)
    save(fig, "fig_alpha_negative_case_check.pdf", paths)

    data = pd.read_csv(paths.results / "plot_tables_followup/fig_alpha_bias_vs_stability_clusterCI.csv")
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.75), sharex=True)
    fig.subplots_adjust(left=.105, right=.975, bottom=.19, top=.945, wspace=.31, hspace=.43)
    for row, case in enumerate(["positive", "negative"]):
        for col, metric in enumerate(["lit", "strict"]):
            ax = axes[row, col]
            field = "measured_minus_implied_" + metric
            for i, m in enumerate(MODELS):
                d = data[(data.subject_model == m) & (data.case == case)].sort_values("bin")
                x = d.bin.to_numpy(float) + (i-2)*.06
                y = 100*d[field].to_numpy()
                ax.errorbar(x, y, yerr=np.vstack([y-100*d[field+"_ci_lo"], 100*d[field+"_ci_hi"]-y]),
                            fmt="o-", color=COLORS[m], ms=3, lw=.65, elinewidth=.65, capsize=1.5)
            ax.axhline(0, color=".4", lw=.7, ls="--")
            ax.set(xlim=(4.7, 8.3), xticks=[5, 6, 7, 8], xticklabels=["5/8", "6/8", "7/8", "8/8"])
            ax.grid(axis="y", lw=.35, color=".9")
            ax.set_title(("Positive" if row == 0 else "Negative") + " case", pad=4)
            ax.set_ylabel(("Non-persistent" if col == 0 else "Non-recurring") + r" $-(1-\alpha)$, pp")
            if row == 1:
                ax.set_xlabel("Baseline stability")
    model_legend(fig, y=.024)
    save(fig, "fig_alpha_bias_vs_stability.pdf", paths)


def main_roles(roles, paths):
    pooled = roles[roles.grouping == "pooled"].set_index("category")
    fig, ax = plt.subplots(figsize=(2.695, 2.12))
    fig.subplots_adjust(left=.20, right=.97, bottom=.40, top=.96)
    colors = ["#0072B2", "#009E73", "#D55E00"]
    for i, (cat, color) in enumerate(zip(CATEGORIES, colors)):
        r = pooled.loc[cat]
        y, lo, hi = 100*np.array([r.persist3, r.persist3_ci_lo, r.persist3_ci_hi])
        ax.errorbar(i, y, yerr=[[y-lo], [hi-y]], fmt="o", color=color, ms=4,
                    elinewidth=1.2, capsize=3, zorder=3)
        ax.annotate(f"{y:.1f}%", (i, hi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5)
    ref = 100*pooled.loc[CATEGORIES[0], "ref_all_flips_persist3"]
    ax.axhline(ref, color=".53", lw=.8, ls="--", zorder=1)
    ax.set(ylim=(0, 102), xlim=(-.48, 2.48), yticks=[0, 25, 50, 75, 100], xticks=[0, 1, 2])
    labels = [f"{CAT_LABELS[cat]}\n(n = {int(pooled.loc[cat, 'n'])})" for cat in CATEGORIES]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Persistent (≥3/4), %", labelpad=2)
    ax.grid(axis="y", lw=.35, color=".91")
    fig.legend([Line2D([], [], color=".53", lw=.8, ls="--")], [f"All SSP flips: {ref:.1f}%"],
               loc="lower center", bbox_to_anchor=(.5, .025), frameon=False)
    save(fig, "fig_false_rejections.pdf", paths)


def role_slices(data, slice_col, values, labels, name, paths, title=None):
    """Three exclusive categories, matched all-flip reference for every mark."""
    h = 5.2 if len(values) == 8 else 3.9
    fig, axes = plt.subplots(1, 3, figsize=(5.5, h), sharey=True)
    fig.subplots_adjust(left=.20 if len(values) == 8 else .15, right=.98,
                        bottom=.24 if len(values) == 8 else .28, top=.91, wspace=.17)
    for ax, cat in zip(axes, CATEGORIES):
        d = data[data.category == cat]
        for j, value in enumerate(values):
            for i, m in enumerate(MODELS):
                rows = d[(d[slice_col] == value) & (d.subject_model == m)]
                if len(rows) == 0:
                    continue
                assert len(rows) == 1
                r = rows.iloc[0]
                n = r.get("n", r.get("den"))
                y = j + (i-2)*.135
                x, lo, hi, ref = 100*np.array([r.persist3, r.persist3_ci_lo, r.persist3_ci_hi, r.ref_all_flips_persist3])
                a = .27 if n < 20 else 1
                ax.plot(ref, y, marker="|", color=".45", ms=5, mew=.8, zorder=1)
                ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt="o", color=COLORS[m], alpha=a,
                            ms=2.8, elinewidth=.65, capsize=1.3, zorder=3)
        ax.set(xlim=(-3, 103), xticks=[0, 50, 100], xlabel="Persistent, %", title=CAT_LABELS[cat])
        ax.set_yticks(range(len(values)), labels)
        ax.set_ylim(len(values)-.45, -.6)
        ax.grid(axis="x", color=".9", lw=.4)
    model_legend(fig, y=.075)
    fig.text(.5, .025, "Grey tick: matched all-flip rate; faded: n < 20", ha="center", fontsize=7.5)
    save(fig, name, paths)


def role_by_case(roles, paths):
    data = roles[roles.grouping == "subject_model+case"]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 3.9), sharey=True)
    fig.subplots_adjust(left=.165, right=.98, bottom=.27, top=.91, wspace=.18)
    for ax, cat in zip(axes, CATEGORIES):
        d = data[data.category == cat]
        for j, m in enumerate(MODELS):
            for offset, case, marker in [(-.13, "positive", "o"), (.13, "negative", "^")]:
                rows = d[(d.subject_model == m) & (d.case == case)]
                if len(rows) == 0:
                    continue
                r = rows.iloc[0]
                x, lo, hi, ref = 100*np.array([r.persist3, r.persist3_ci_lo, r.persist3_ci_hi, r.ref_all_flips_persist3])
                a = .27 if r.n < 20 else 1
                ax.plot(ref, j+offset, marker="|", color=".45", ms=5, mew=.8, zorder=1)
                ax.errorbar(x, j+offset, xerr=[[x-lo], [hi-x]], fmt=marker, color=COLORS[m],
                            alpha=a, ms=3.3, elinewidth=.7, capsize=1.5, zorder=3)
        ax.set(xlim=(-3, 103), xticks=[0, 50, 100], xlabel="Persistent, %", title=CAT_LABELS[cat])
        ax.set_yticks(range(5), [NAMES[m] for m in MODELS])
        ax.set_ylim(4.5, -.5)
        ax.grid(axis="x", color=".9", lw=.4)
    fig.legend([Line2D([], [], marker="o", color=".2", linestyle="none", ms=3.5), Line2D([], [], marker="^", color=".2", linestyle="none", ms=3.5)],
               ["Positive case", "Negative case"], loc="lower center", bbox_to_anchor=(.5, .105), ncol=2, frameon=False)
    fig.text(.5, .035, "Grey tick: matched all-flip rate; faded: n < 20", ha="center", fontsize=7.5)
    save(fig, "fig_false_rejections_by_case.pdf", paths)


def roles_by_stability(paths):
    df = rc.load_pairs(paths.results)
    bs = rc.ClusterBootstrap(df.qkey)
    assert bs.fingerprint == "1a8b5e9a029dc3ce"
    f = df[df.ssp_flip].copy()
    f["bin"] = f.baseline_stability.astype(int)
    f["category"] = np.where(f.role_label == -1, "incoherent",
                  np.where(f.role_label.isna(), "unjudged",
                  np.where(f.role_orig.isna(), "missing_role", "coherent_" + f.role_orig.astype(object).fillna("").astype(str))))
    keys = ["subject_model", "bin"]
    ref = rc.ratio_table(bs, f, f.persist3.astype(float), np.ones(len(f)), keys, "persist3")
    ref = rc.plain(ref)[keys + ["persist3"]].rename(columns={"persist3": "ref_all_flips_persist3"})
    rows = []
    for cat in CATEGORIES:
        d = f[f.category == cat]
        t = rc.plain(rc.ratio_table(bs, d, d.persist3.astype(float), np.ones(len(d)), keys, "persist3"))
        t = t.merge(ref, on=keys, validate="many_to_one")
        t["category"] = cat
        t["bin"] = t["bin"].astype(int)
        rows.append(t)
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(paths.derived / "role_category_persistence_by_stability.csv", index=False)
    role_slices(data, "bin", [5, 6, 7, 8], ["5/8", "6/8", "7/8", "8/8"], "fig_false_rejections_by_stability.pdf", paths)


def build(paths):
    """Write nine PDFs and the recomputed stability-role table."""
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.derived.mkdir(parents=True, exist_ok=True)
    alpha = pd.read_csv(paths.results / "s1_persistence/alpha_vs_measured_noise.csv")
    cells = alpha[alpha.grouping == "subject_model+dataset+hint_style+case"].copy()
    roles = pd.read_csv(paths.results / "s2_roles/role_category_persistence.csv")
    assert len(cells) == 320 and (cells.one_minus_alpha <= 1).all()
    assert len(cells[cells.one_minus_alpha_ci_hi > 1]) == 5
    with plt.rc_context(STYLE):
        main_alpha(cells, paths)
        appendix_alpha(cells, paths)
        main_roles(roles, paths)
        cue_table = pd.read_csv(paths.results / "plot_tables_followup/fig_false_rejections_by_style_role_category_clusterCI.csv")
        role_slices(cue_table, "hint_style", CUES, CUE_LABELS, "fig_false_rejections_by_style.pdf", paths)
        role_by_case(roles, paths)
        roles_by_stability(paths)
    print("Rendered nine alpha and exclusive-role PDFs.")
