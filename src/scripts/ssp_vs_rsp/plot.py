#!/usr/bin/env python3
"""Figures of the SSP-vs-RSP probe comparison.

``auroc__<task>__<test>__<variant>.{png,pdf}``: AUROC (y) against the share of the CoT the probe sees (x),
one panel per model, six paired lines — attention / linear (last prefix token) / TF-IDF (prefix text), each
trained on ssp-all (dashed, hollow markers) and on RSP-survivors (solid, filled markers); bands: 95%
question-clustered bootstrap intervals. ``delta__<task>.{png,pdf}``: the model-mean paired difference
RSP − SSP per probe, one panel per (test set, variant).

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.plot
"""

from __future__ import annotations

import argparse
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.lib import ssp_vs_rsp as X  # noqa: E402
from src.lib.paths import resolve_data_path  # noqa: E402

ROOT = "${DATA_ROOT}/ssp_vs_rsp"
# Reference palette, categorical slots 1-3 (validated all-pairs); identity also carried by marker shape.
PROBE_STYLE = {"attention": ("#2a78d6", "o", "Attention probe"),
               "linear": ("#eb6834", "s", "Linear probe (last prefix token)"),
               "tfidf": ("#1baf7a", "^", "TF-IDF (prefix text)")}
ARM_STYLE = {"rsp_survivors": ("-", True, "RSP-survivors"), "ssp_all": ("--", False, "ssp-all")}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_LABEL = {"qwen3-8b": "Qwen3-8B", "nemotron-nano-9b-v2": "Nemotron-Nano-9B-v2",
               "olmo3-7b-think": "OLMo-3-7B-Think", "gemma4-12b-it": "Gemma-4-12B"}
TASK_LABEL = {"unfaithful_vs_ignored": "unfaithful vs cue-ignored", "unfaithful_vs_faithful": "unfaithful vs faithful"}
TEST_LABEL = {"ssp_all": "test set: ssp-all", "rsp_survivors": "test set: RSP-survivors",
              "gold": "test set: gold (12 fresh re-rolls)"}
VARIANT_LABEL = {"posthoc_capped": "all cue styles, post_hoc capped to 1/8", "no_posthoc": "all cue styles except post_hoc"}


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def legend_handles():
    h = [Line2D([], [], color=c, marker=m, linestyle="-", linewidth=2, markersize=6, label=lab)
         for c, m, lab in PROBE_STYLE.values()]
    h += [Line2D([], [], color=MUTED, linestyle=ls, linewidth=2, marker="o", markersize=6,
                 markerfacecolor=MUTED if filled else "white", label=lab)
          for ls, filled, lab in ARM_STYLE.values()]
    return h


def auroc_figure(df, task, test, variant, models, out):
    sub = df[(df["task"] == task) & (df["test"] == test) & (df["variant"] == variant)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.4), sharey=True, constrained_layout=True)
    axes = [axes] if len(models) == 1 else list(axes)
    lo_all = max(0.0, min(0.45, float(sub["lo"].min()) - 0.02))
    for ax, model in zip(axes, models):
        m = sub[sub["model"] == model]
        style_axes(ax)
        ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
        for probe, (color, marker, _) in PROBE_STYLE.items():
            for arm, (ls, filled, _) in ARM_STYLE.items():
                g = m[(m["probe"] == probe) & (m["arm"] == arm)].sort_values("fraction")
                if g.empty:
                    continue
                x = g["fraction"] * 100
                ax.fill_between(x, g["lo"], g["hi"], color=color, alpha=0.07, linewidth=0)
                ax.plot(x, g["auroc"], color=color, linestyle=ls, linewidth=2, marker=marker, markersize=5,
                        markerfacecolor=color if filled else "white", markeredgecolor=color, markeredgewidth=1.3)
        n = m.drop_duplicates(["probe", "fraction"])[["n_pos", "n_neg"]].max() if len(m) else None
        title = MODEL_LABEL.get(model, model)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        if n is not None and len(m):
            ax.text(0.98, 0.03, f"test n = {int(n['n_pos'])} unf. / {int(n['n_neg'])} neg.", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks([10, 30, 50, 70, 100])
        ax.set_xlim(5, 103)
        ax.set_ylim(lo_all, 1.0)
        ax.set_xlabel("CoT seen by the probe (%)", fontsize=8, color=MUTED)
    axes[0].set_ylabel("AUROC (detecting unfaithful)", fontsize=8, color=MUTED)
    fig.suptitle(f"{TASK_LABEL[task]}  ·  {TEST_LABEL[test]}  ·  {VARIANT_LABEL[variant]}", fontsize=10,
                 color=TEXT, x=0.01, ha="left")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"auroc__{task}__{test}__{variant}.{ext}", dpi=200)
    plt.close(fig)


def delta_figure(dm, task, out):
    sub = dm[dm["task"] == task]
    if sub.empty:
        return
    tests = [t for t in X.TEST_SETS if t in set(sub["test"])]
    fig, axes = plt.subplots(len(X.VARIANTS), len(tests), figsize=(3.4 * len(tests), 5.6), sharey=True,
                             sharex=True, constrained_layout=True, squeeze=False)
    for (i, variant), (j, test) in itertools.product(enumerate(X.VARIANTS), enumerate(tests)):
        ax = axes[i][j]
        style_axes(ax)
        ax.axhline(0, color=MUTED, linewidth=0.9)
        g = sub[(sub["test"] == test) & (sub["variant"] == variant)]
        for probe, (color, marker, _) in PROBE_STYLE.items():
            p = g[g["probe"] == probe].sort_values("fraction")
            if p.empty:
                continue
            x = p["fraction"] * 100
            ax.fill_between(x, p["lo"], p["hi"], color=color, alpha=0.10, linewidth=0)
            ax.plot(x, p["delta"], color=color, linewidth=2, marker=marker, markersize=5)
        ax.set_title(f"{TEST_LABEL[test]}\n{VARIANT_LABEL[variant]}", fontsize=8, color=TEXT, loc="left")
        ax.set_xticks([10, 30, 50, 70, 100])
        if i == len(X.VARIANTS) - 1:
            ax.set_xlabel("CoT seen by the probe (%)", fontsize=8, color=MUTED)
        if j == 0:
            ax.set_ylabel("ΔAUROC, RSP-survivors − ssp-all\n(mean over models)", fontsize=8, color=MUTED)
    fig.suptitle(f"{TASK_LABEL[task]}: paired AUROC gain from training on RSP-survivors", fontsize=10, color=TEXT,
                 x=0.01, ha="left")
    fig.legend(handles=legend_handles()[:3], loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"delta__{task}.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(X.MODELS))
    ap.add_argument("--results", default=f"{ROOT}/results", help="results directory (default %(default)s)")
    ap.add_argument("--figures", default=f"{ROOT}/figures", help="figures directory (default %(default)s)")
    a = ap.parse_args(argv)
    res = resolve_data_path(a.results)
    df = pd.read_csv(res / "auroc.csv")
    models = [m for m in a.models if m in set(df["model"])]
    out = resolve_data_path(a.figures)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for task, test, variant in itertools.product(X.TASKS, X.TEST_SETS, X.VARIANTS):
        auroc_figure(df, task, test, variant, models, out)
    dm = pd.read_csv(res / "delta_model_mean.csv")
    for task in X.TASKS:
        delta_figure(dm, task, out)
    print(f"figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
