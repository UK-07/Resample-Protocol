#!/usr/bin/env python3
"""Render the twelve survival, recurrence-composition, and yield PDF assets.

Input tables in ``paths.results`` preserve the published populations and contain
precomputed 10,000-replicate question-cluster percentile intervals (seed 42,
stratified by dataset). Intervals are plotted directly, not reconstructed from
aggregate counts. Recurrence compositions count the four scheduled draws, so
missing/truncated outcomes remain non-hits in this published view; the manuscript
reports missingness sensitivity separately. Only ``build(paths)`` reads/writes
files. It never imports or executes analysis code supplied with the data bundle.

This is the repository port of the September 26 manuscript renderer. Main width
is 3.52 inches, appendix width 5.5 inches; render at native size for >=7 pt type.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

RESULTS: Path
FIGURES: Path
MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
MODEL_NAMES = ["Nemotron-Nano-9B-v2", "Qwen3-8B", "Qwen3.5-9B", "OLMo-3-7B-Think", "Gemma-4-12B"]
MODEL_SHORT = ["Nemotron", "Qwen3-8B", "Qwen3.5-9B", "OLMo-3-7B", "Gemma-4-12B"]
MODEL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
CASES = ["positive", "negative"]
CASE_COLORS = {"positive": "#0072B2", "negative": "#E69F00"}
CUES = ["expert_opinion", "consensus", "metadata", "grader_hacking", "tool_output", "answer_key_artifact", "unethical_info", "post_hoc"]
CUE_NAMES = ["Expert opinion", "Consensus", "Metadata", "Grader hacking", "Tool output", "Answer-key artifact", "Unethical info", "Post hoc"]
DATASETS = ["commonsense_qa", "medqa", "gpqa", "mmlu_pro"]
DATASET_NAMES = ["CommonsenseQA", "MedQA", "GPQA", "MMLU-Pro"]

STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
}


def read(relative: str) -> list[dict]:
    with (RESULTS / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else float("nan")


def single(rows: list[dict], **keys) -> dict | None:
    found = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    if len(found) > 1:
        raise ValueError(f"Nonunique selection: {keys}")
    return found[0] if found else None


def finish(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"Title": name, "Creator": "build_revision_survival.py", "CreationDate": None})
    plt.close(fig)


def clean(ax, *, grid: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(length=2.5, width=0.6, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#e3e3e3", linewidth=0.5)


def point_interval(ax, row: dict, y: float, color: str, *, marker="o", metric="survival", lower="survival_ci_lo", upper="survival_ci_hi", factor=100) -> None:
    value, low, high = [number(row, k) * factor for k in (metric, lower, upper)]
    if not math.isfinite(value):
        return
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError(f"Missing or inverted interval in {row}")
    alpha = 0.35 if number(row, "den") < 20 else 1
    # Draw the supplied percentile endpoints directly. A percentile interval
    # need not contain the original point estimate in a small sample.
    ax.plot([low, high], [y, y], color=color, alpha=alpha, linewidth=1.05)
    ax.plot([low, low], [y - .065, y + .065], color=color, alpha=alpha, linewidth=.7)
    ax.plot([high, high], [y - .065, y + .065], color=color, alpha=alpha, linewidth=.7)
    ax.plot(value, y, marker=marker, color=color, alpha=alpha, markersize=3.2, linestyle="none")


def case_legend(fig, y=.015):
    fig.legend([Line2D([], [], marker="o", color=CASE_COLORS[c], linewidth=1, markersize=3.3) for c in CASES], ["Positive case", "Negative case"], loc="lower center", bbox_to_anchor=(.5, y), ncol=2, frameon=False)


def stability():
    rows = read("plot_tables/fig_survival_vs_stability_positive_clusterCI.csv")
    fig, ax = plt.subplots(figsize=(3.52, 2.10))
    fig.subplots_adjust(left=.14, right=.985, bottom=.38, top=.97)
    for i, (model, color) in enumerate(zip(MODELS, MODEL_COLORS)):
        selected = sorted([r for r in rows if r["subject_model"] == model], key=lambda r: int(r["stability_bin"]))
        xs = np.array([int(r["stability_bin"]) for r in selected]) + (i - 2) * .045
        ys = np.array([number(r, "survival") * 100 for r in selected])
        ax.plot(xs, ys, color=color, alpha=.75, linewidth=.65)
        for x, y, row in zip(xs, ys, selected):
            low, high = number(row, "survival_ci_lo") * 100, number(row, "survival_ci_hi") * 100
            alpha = .30 if number(row, "den") < 20 else 1
            ax.plot([x, x], [low, high], color=color, linewidth=.75, alpha=alpha)
            ax.plot(x, y, "o", color=color, markersize=2.8, alpha=alpha)
    ax.set(xlim=(4.78, 8.22), ylim=(0, 103), xticks=[5, 6, 7, 8], xticklabels=["5/8", "6/8", "7/8", "8/8"], yticks=[0, 25, 50, 75, 100])
    ax.set_ylabel("Survival (%)", fontsize=7.5)
    ax.set_xlabel("Baseline stability", fontsize=7.5, labelpad=2)
    ax.tick_params(labelsize=7)
    clean(ax, grid="y")
    fig.legend([Line2D([], [], color=c, marker="o", markersize=2.8) for c in MODEL_COLORS], MODEL_SHORT, loc="lower center", bbox_to_anchor=(.5, .005), ncol=2, frameon=False, fontsize=7, columnspacing=1.1, handlelength=1.4, labelspacing=.3)
    finish(fig, "fig_survival_vs_stability")


def model_facets(n_categories: int, *, tall=False):
    height = 6.6 if tall else 4.35
    fig, axes = plt.subplots(3, 2, figsize=(5.5, height))
    fig.subplots_adjust(left=.225, right=.978, bottom=.105 if tall else .17, top=.955, wspace=1.0, hspace=.50 if tall else .73)
    axes[-1, -1].set_visible(False)
    for i, ax in enumerate(axes.flat):
        if i >= len(MODELS):
            break
        ax.set_title(MODEL_NAMES[i], loc="left", pad=5)
        clean(ax)
        ax.set_ylim(n_categories - .55, -.45)
    return fig, list(axes.flat)


def survival_facet(filename: str, dimension: str, levels: list[str], labels: list[str]):
    rows = read(f"plot_tables/{filename}_clusterCI.csv")
    fig, axes = model_facets(len(levels), tall=len(levels) > 4)
    for model, ax in zip(MODELS, axes):
        for j, category in enumerate(levels):
            for case, offset in zip(CASES, [-.15, .15]):
                row = single(rows, subject_model=model, **{dimension: category}, case=case)
                if row:
                    point_interval(ax, row, j + offset, CASE_COLORS[case])
        ax.set(xlim=(-2, 102), xticks=[0, 50, 100], yticks=np.arange(len(levels)), yticklabels=labels)
        ax.set_xlabel("Survival (%)", labelpad=3)
    case_legend(fig, .01)
    finish(fig, filename)


def case_slice():
    rows = read("plot_tables/fig_survival_slice_case_clusterCI.csv")
    fig, ax = plt.subplots(figsize=(5.5, 2.25))
    fig.subplots_adjust(left=.28, right=.975, bottom=.30, top=.96)
    for j, model in enumerate(MODELS):
        for case, offset in zip(CASES, [-.12, .12]):
            point_interval(ax, single(rows, subject_model=model, case=case), j + offset, CASE_COLORS[case])
    clean(ax)
    ax.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], yticks=np.arange(5), yticklabels=MODEL_NAMES, ylim=(4.45, -.45))
    ax.set_xlabel("Survival (%)")
    case_legend(fig, -.005)
    finish(fig, "fig_survival_slice_case")


def subgroup_slice(filename: str, dimension: str, levels: list[str], labels: list[str]):
    rows = read(f"plot_tables_followup/{filename}_clusterCI.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    fig.subplots_adjust(left=.26, right=.975, bottom=.31, top=.85, wspace=.26)
    for ax, case in zip(axes, CASES):
        for j, model in enumerate(MODELS):
            for category, offset, marker in zip(levels, [-.13, .13], ["o", "^"]):
                row = single(rows, subject_model=model, case=case, **{dimension: category})
                if row:
                    point_interval(ax, row, j + offset, CASE_COLORS[case], marker=marker)
        clean(ax)
        ax.set(title=f"{case.capitalize()} case", xlim=(-2, 102), xticks=[0, 50, 100], yticks=np.arange(5), ylim=(4.45, -.45))
        ax.set_xlabel("Survival (%)")
    axes[0].set_yticklabels(MODEL_NAMES)
    axes[1].set_yticklabels([])
    fig.legend([Line2D([], [], color="#555555", marker=m, markersize=3.5) for m in ["o", "^"]], labels, loc="lower center", bbox_to_anchor=(.5, .005), ncol=2, frameon=False)
    finish(fig, filename)


def composition(population: str, case: str):
    rows = read(f"plot_tables/fig_reliance_composition_{population}.csv")
    colors = ["#0072B2", "#56B4E9", "#D9D9D9"]
    fig, axes = model_facets(8, tall=True)
    for model, ax in zip(MODELS, axes):
        for j, cue in enumerate(CUES):
            row = single(rows, subject_model=model, hint_style=cue, case=case)
            if not row:
                ax.text(50, j, "No observations", ha="center", va="center", fontsize=7.5, color="#777777")
                continue
            n = number(row, "n")
            components = [number(row, k) for k in ["robust", "weak", "none"]]
            if sum(components) != n:
                raise ValueError(f"Composition counts do not sum: {row}")
            start = 0
            for value, color in zip(components, colors):
                percent = 100 * value / n
                ax.barh(j, percent, left=start, height=.63, color=color, alpha=.4 if n < 20 else 1, edgecolor="white", linewidth=.35)
                start += percent
            ax.text(104, j, f"{n:,.0f}", ha="left", va="center", fontsize=7.5, color="#555555", clip_on=False)
        ax.set(xlim=(0, 100), xticks=[0, 50, 100], yticks=np.arange(8), yticklabels=CUE_NAMES)
        ax.set_xlabel("Share (%)", labelpad=3)
        ax.text(104, -.65, "n", ha="left", va="center", fontsize=7.5, color="#555555", clip_on=False)
    fig.subplots_adjust(right=.92, wspace=1.26)
    fig.legend([Patch(facecolor=c) for c in colors], ["Persistent (3-4)", "Partial recurrence (1-2)", "No recurrence (0)"], loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False, fontsize=7.5, columnspacing=.9, handlelength=1)
    suffix = "_negative" if case == "negative" else ""
    finish(fig, f"fig_reliance_composition_{population}{suffix}")


def reroll_distribution():
    rows = read("s1_persistence/persistence_counts_and_survival.csv")
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 4.8))
    fig.subplots_adjust(left=.105, right=.98, bottom=.16, top=.95, hspace=.64, wspace=.25)
    axes[-1, -1].set_visible(False)
    for model, name, ax in zip(MODELS, MODEL_NAMES, axes.flat):
        for case, offset in zip(CASES, [-.19, .19]):
            row = single(rows, population="ssp_flips", grouping="subject_model+case", subject_model=model, case=case)
            counts = [number(row, f"k{i}") for i in range(5)]
            if sum(counts) != number(row, "n"):
                raise ValueError(f"k counts do not sum: {row}")
            ax.bar(np.arange(5) + offset, np.array(counts) / number(row, "n") * 100, width=.36, color=CASE_COLORS[case])
        ax.set(title=name, ylim=(0, 100), yticks=[0, 50, 100], xticks=np.arange(5))
        ax.set_xlabel("Target re-rolls (of 4)", labelpad=2)
        ax.set_ylabel("Share (%)", labelpad=3)
        clean(ax, grid="y")
    case_legend(fig, .005)
    finish(fig, "fig_reroll_distribution")


def yield_by_style():
    rows = read("plot_tables_followup/fig_yield_per_style_clusterCI.csv")
    fig, axes = model_facets(8, tall=True)
    for model, color, ax in zip(MODELS, MODEL_COLORS, axes):
        for j, cue in enumerate(CUES):
            row = single(rows, subject_model=model, hint_style=cue)
            if row:
                point_interval(ax, row, j, color, metric="yield_per_1000", lower="ci_lo", upper="ci_hi", factor=1)
        ax.set(xlim=(0, 800), xticks=[0, 400, 800], yticks=np.arange(8), yticklabels=CUE_NAMES)
        ax.set_xlabel("Yield per 1,000 cued rollouts", labelpad=3, fontsize=7.5)
    finish(fig, "fig_yield_per_style")


def configure(paths) -> None:
    """Set explicit paths for shared rendering helpers; do not alter rcParams."""
    global RESULTS, FIGURES
    RESULTS, FIGURES = Path(paths.results), Path(paths.figures)
    FIGURES.mkdir(parents=True, exist_ok=True)


def build(paths) -> list[Path]:
    """Write the published figures using the prepared tables and explicit paths."""
    configure(paths)
    with plt.rc_context(STYLE):
        _build()
    return [FIGURES / (name + ".pdf") for name in FIGURE_NAMES]


FIGURE_NAMES = (
    "fig_survival_vs_stability", "fig_survival_by_style", "fig_survival_by_dataset",
    "fig_survival_slice_case", "fig_survival_slice_cue_family", "fig_survival_slice_post_hoc",
    "fig_reliance_composition_ssp_unfaithful", "fig_reliance_composition_ssp_unfaithful_negative",
    "fig_reliance_composition_ssp_flips", "fig_reliance_composition_ssp_flips_negative",
    "fig_reroll_distribution", "fig_yield_per_style",
)


def _build():
    stability()
    survival_facet("fig_survival_by_style", "hint_style", CUES, CUE_NAMES)
    survival_facet("fig_survival_by_dataset", "dataset", DATASETS, DATASET_NAMES)
    case_slice()
    subgroup_slice("fig_survival_slice_cue_family", "cue_family", ["social", "artifact"], ["Social cues", "Artifact cues"])
    subgroup_slice("fig_survival_slice_post_hoc", "post_hoc_slice", ["post_hoc", "other_cues"], ["Post hoc", "Other cues"])
    for population in ["ssp_unfaithful", "ssp_flips"]:
        for case in CASES:
            composition(population, case)
    reroll_distribution()
    yield_by_style()
