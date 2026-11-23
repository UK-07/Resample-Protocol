#!/usr/bin/env python3
"""Summary figures of the SSP-vs-RSP comparison.

``precision__<variant>.{png,pdf}``: label precision on the gold pairs — per label, the share of test
rollouts whose 12 fresh re-rolls confirm the switch (≥ 9/12 on the cued answer), for every ssp-all row, the
RSP-survivors and the rows RSP drops; bars per model and pooled; 95% question-clustered bootstrap intervals.

``sized__<variant>.{png,pdf}``: the model-mean, fraction-averaged paired difference RSP-survivors − ssp-all
per probe and test set, against the full ssp-all training set and against ssp-all subsampled to the
RSP-survivors size; 95% intervals from results/summary_fracavg.csv and results/sized_fracavg.csv.

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.plot_summary
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.lib import ssp_vs_rsp as X  # noqa: E402
from src.lib.paths import resolve_data_path  # noqa: E402
from src.scripts.ssp_vs_rsp.plot import MUTED, PROBE_STYLE, TASK_LABEL, TEXT, VARIANT_LABEL, style_axes  # noqa: E402

CLASS_LABEL = {0: "Unfaithful (used the cue, did not say so)", 1: "Faithful (used the cue, said so)"}
GROUP_STYLE = {  # (face, edge, hatch, label)
    "ssp_all": ("#b9b7b0", "#b9b7b0", None, "ssp-all (every rollout)"),
    "rsp_survivors": ("#2a78d6", "#2a78d6", None, "RSP-survivors (kept by re-sampling)"),
    "dropped": ("white", "#8a8882", "////", "dropped by re-sampling"),
}
TICK_LABEL = {"qwen3-8b": "Qwen3-8B", "nemotron-nano-9b-v2": "Nemotron-\nNano-9B-v2", "olmo3-7b-think": "OLMo-3-\n7B-Think",
              "gemma4-12b-it": "Gemma-4-\n12B"}
TEST_SHORT = {"ssp_all": "ssp-all", "rsp_survivors": "RSP-survivors", "gold": "gold\n(12 fresh re-rolls)"}
N_BOOT = 2000


def clustered_share(conf: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float]:
    """Share of True in ``conf`` and its 95% interval, resampling ``clusters`` (questions) with replacement."""
    point = float(conf.mean())
    codes, inv = np.unique(clusters, return_inverse=True)
    k = len(codes)
    hits = np.bincount(inv, weights=conf.astype(float), minlength=k)
    sizes = np.bincount(inv, minlength=k).astype(float)
    draws = np.random.default_rng(X.SEED).integers(0, k, size=(N_BOOT, k))
    shares = hits[draws].sum(1) / sizes[draws].sum(1)
    lo, hi = np.percentile(shares, [2.5, 97.5])
    return point, float(lo), float(hi)


def precision_table(models, variant) -> pd.DataFrame:
    rows = []
    frames = []
    for model in models:
        uni = X.load_universe(model, gold=True).assign(model=model)
        g = uni[X.membership(uni, "ssp_all", variant) & uni["in_gold"]].copy()
        g["kept"] = X.membership(uni, "rsp_survivors", variant)[g.index]
        g["cluster"] = model + "|" + g["question_id"]
        frames.append(g)
    allg = pd.concat(frames, ignore_index=True)
    for model in [*models, "pooled"]:
        g = allg if model == "pooled" else allg[allg["model"].eq(model)]
        for cls in CLASS_LABEL:
            c = g[g[X.LABEL].eq(cls)]
            for group, sub in (("ssp_all", c), ("rsp_survivors", c[c["kept"]]), ("dropped", c[~c["kept"]])):
                p, lo, hi = clustered_share(sub["gold_used"].to_numpy(), sub["cluster"].to_numpy())
                rows.append({"model": model, "cls": cls, "group": group, "n": len(sub), "confirmed": p,
                             "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def precision_figure(tab: pd.DataFrame, variant, models, out) -> None:
    classes = list(CLASS_LABEL)
    groups = list(GROUP_STYLE)
    cols = [*models, "pooled"]
    fig, axes = plt.subplots(1, len(classes), figsize=(6.2 * len(classes), 3.9), sharey=True, constrained_layout=True)
    axes = [axes] if len(classes) == 1 else list(axes)
    width = 0.26
    for ax, cls in zip(axes, classes):
        style_axes(ax)
        for i, model in enumerate(cols):
            x0 = i + (0.35 if model == "pooled" else 0)
            for j, group in enumerate(groups):
                r = tab[(tab["model"] == model) & (tab["cls"] == cls) & (tab["group"] == group)]
                if r.empty:
                    continue
                r = r.iloc[0]
                face, edge, hatch, _ = GROUP_STYLE[group]
                x = x0 + (j - 1) * width
                ax.bar(x, r["confirmed"] * 100, width=width * 0.92, color=face, edgecolor=edge, hatch=hatch,
                       linewidth=0.9)
                ax.errorbar(x, r["confirmed"] * 100, yerr=[[100 * (r["confirmed"] - r["lo"])],
                                                           [100 * (r["hi"] - r["confirmed"])]],
                            fmt="none", ecolor=TEXT, elinewidth=0.9, capsize=2)
                if model == "pooled":
                    ax.text(x, r["hi"] * 100 + 2, f"{r['confirmed'] * 100:.0f}%", ha="center", va="bottom",
                            fontsize=8, color=TEXT, fontweight="bold" if group == "rsp_survivors" else None)
        ax.set_xticks([i + (0.35 if m == "pooled" else 0) for i, m in enumerate(cols)])
        ax.set_xticklabels([TICK_LABEL.get(m, "All models\n(pooled)") for m in cols], fontsize=8, color=TEXT)
        ax.axvline(len(models) - 0.5 + 0.17, color="#d6d5cf", linewidth=0.8)
        ax.set_ylim(0, 108)
        ax.set_title(CLASS_LABEL[cls], fontsize=10, color=TEXT, loc="left")
        p = tab[(tab["model"] == "pooled") & (tab["cls"] == cls)].set_index("group")
        ax.text(0.0, -0.2, "n pooled = " + " · ".join(f"{GROUP_STYLE[g][3].split(' (')[0]} {int(p.loc[g, 'n'])}"
                                                      for g in groups if g in p.index),
                transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    axes[0].set_ylabel("rollouts confirmed as cue-driven (%)\n≥ 9 of 12 fresh re-rolls on the cued answer",
                       fontsize=8, color=MUTED)
    fig.suptitle(f"Label precision on the gold test pairs  ·  {TASK_LABEL}  ·  {VARIANT_LABEL[variant]}",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    handles = [Patch(facecolor=f, edgecolor=e, hatch=h, label=lab) for f, e, h, lab in GROUP_STYLE.values()]
    handles.append(Line2D([], [], color=TEXT, linewidth=0.9, marker="|", markersize=6,
                          label="95% interval (questions resampled)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"precision__{variant}.{ext}", dpi=200)
    plt.close(fig)


def train_size_ratio(models, variant) -> tuple[float, float]:
    ratios = []
    for model in models:
        masks, _ = X.split_masks(X.load_universe(model))
        ratios.append(masks[variant, "rsp_survivors"][0].sum() / masks[variant, "ssp_all"][0].sum())
    return min(ratios), max(ratios)


def sized_figure(variant, models, res, out) -> None:
    full, sized = (d[d["variant"].eq(variant)]
                   for d in (pd.read_csv(res / "summary_fracavg.csv"), pd.read_csv(res / "sized_fracavg.csv")))
    tests = X.TEST_SETS
    lo_r, hi_r = train_size_ratio(models, variant)
    fig, axes = plt.subplots(1, len(PROBE_STYLE), figsize=(3.6 * len(PROBE_STYLE), 3.6), sharey=True,
                             constrained_layout=True)
    comparisons = {"full": (full, "o", True, f"vs ssp-all, full size (RSP-survivors has {lo_r:.0%}–{hi_r:.0%} "
                                              "of its training rows)"),
                   "sized": (sized, "D", False, "vs ssp-all subsampled to the RSP-survivors size")}
    span = []
    for ax, (probe, (color, _, plabel)) in zip(axes, PROBE_STYLE.items()):
        style_axes(ax)
        ax.axhline(0, color=MUTED, linewidth=0.9)
        for k, (name, (d, marker, filled, _)) in enumerate(comparisons.items()):
            for i, test in enumerate(tests):
                r = d[(d["probe"] == probe) & (d["test"] == test)].iloc[0]
                x = i + (k - 0.5) * 0.28
                ax.errorbar(x, r["delta"], yerr=[[r["delta"] - r["delta_lo"]], [r["delta_hi"] - r["delta"]]],
                            fmt=marker, color=color, markersize=7, elinewidth=1.6, capsize=3,
                            markerfacecolor=color if filled else "white", markeredgewidth=1.6)
                span += [r["delta_lo"], r["delta_hi"]]
        ax.set_xticks(range(len(tests)))
        ax.set_xticklabels([TEST_SHORT[t] for t in tests], fontsize=8, color=TEXT)
        ax.set_xlim(-0.6, len(tests) - 0.4)
        ax.set_title(plabel, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("test set", fontsize=8, color=MUTED)
    pad = 0.005
    axes[0].set_ylim(min(span) - pad, max(span) + pad)
    axes[0].set_ylabel(f"ΔAUROC, RSP-survivors − ssp-all\n(mean over {len(models)} models and 10 CoT fractions)", fontsize=8,
                       color=MUTED)
    fig.suptitle(f"{TASK_LABEL}  ·  does training on RSP-survivors help?  ·  {VARIANT_LABEL[variant]}",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    handles = [Line2D([], [], color=MUTED, marker=m, linestyle="none", markersize=7,
                      markerfacecolor=MUTED if f else "white", markeredgewidth=1.6, label=lab)
               for _, m, f, lab in comparisons.values()]
    handles.append(Line2D([], [], color=MUTED, linewidth=1.6, marker="|", markersize=7, label="95% interval"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sized__{variant}.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(X.MODELS))
    ap.add_argument("--results", default=f"{X.ROOT}/results", help="results directory (default %(default)s)")
    ap.add_argument("--figures", default=f"{X.ROOT}/figures", help="figures directory (default %(default)s)")
    a = ap.parse_args(argv)
    out = resolve_data_path(a.figures)
    res = resolve_data_path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for variant in X.VARIANTS:
        tab = precision_table(a.models, variant)
        tab.to_csv(res / f"precision__{variant}.csv", index=False)
        precision_figure(tab, variant, a.models, out)
        sized_figure(variant, a.models, res, out)
    print(f"figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
