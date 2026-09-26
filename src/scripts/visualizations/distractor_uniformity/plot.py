"""Distractor uniformity — figures. Reads ONLY the gather's data/ (cells.csv, majority_pooled.csv,
worked_example.json) and writes into <out-dir>:

  distractor_uniformity_letter.{png,pdf}    per-cell (model × dataset × style × case, both cases) Wald p-value of
                                            "third-option switches are uniform over the row's distractors" by
                                            option letter; one strip per model; qualifying cells only (>= 5·(n−2)
                                            third-option switches).
  distractor_uniformity_position.{png,pdf}  the same by rank among the distractors (Pearson, df n − 3).
  distractor_uniformity_majority.{png,pdf}  the majority-answer test POOLED OVER MODELS: one point per negative
                                            dataset × style cell with >= 5·(n−2) switches whose majority no-hint
                                            answer is a distractor; one strip per dataset; style abbreviation printed.
  distractor_uniformity_example.{png,pdf}   worked example: the qualifying pooled cell with the most switches in the
                                            majority population, observed vs uniform (majority answer / other
                                            distractors), Pearson and exact binomial p.
All: dashed p = 0.05; "rejected / qualifying" and the Benjamini–Hochberg count per strip; y limited to the data
range; points with n < 20 faded with n printed; p below P_FLOOR drawn at the floor (▼). No literal / strict y pair
(no re-sampling quantity). Style from src/scripts/visualizations/paper_plots.py.

Run:
    python -m src.scripts.visualizations.distractor_uniformity.plot [--data D] [--out-dir D]
Figures: <out-dir>/distractor_uniformity_{letter,position,majority,example}.{png,pdf}
(default <cueball>/plots/distractor_uniformity/; --data defaults to <out-dir>/data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.scripts.visualizations.common import alpha_common as ac  # noqa: E402
from src.scripts.visualizations.common import section1_plot as SP  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    CASE_COLOR, CASE_ORDER, CUE_ABBREV, CUE_NAMES, DATASET_NAMES, DATASET_ORDER, DATASET_SHORT, FULL_W, HALF_W,
    INK2, MODEL_SHORT, MUTED, SURFACE, apply_style, tidy,
)

PLOT = "distractor_uniformity"
DATASET_MARKER = dict(zip(DATASET_ORDER, ["o", "s", "^", "D"]))
P_FLOOR = 1e-12
TITLES = {"letter": "by option letter (Wald, df = rank Σ)",
          "position": "by rank among the distractors (Pearson, df = n − 3)",
          "majority": "majority no-hint answer vs the other distractors\n(pooled over models; Pearson, df = 1)"}


def p_axis(ax, pvals) -> float:
    """Log p axis limited to the data range (lowest decade at or below min p, never below P_FLOOR). Returns ymin."""
    ax.set_yscale("log")
    ax.axhline(0.05, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    pv = np.asarray(pvals, dtype=float)
    pmin = float(np.nanmin(pv)) if np.isfinite(pv).any() else 0.01
    ymin = max(P_FLOOR / 3, min(0.01, 10 ** np.floor(np.log10(max(pmin, P_FLOOR))) / 3))
    ax.set_ylim(ymin, 12)
    return ymin


def point(ax, x, p, n, *, color, marker, label=None, side: int = 1) -> None:
    """side = +1 puts the label right of the point, −1 left (alternated by the caller to avoid overlaps)."""
    p = max(float(p), P_FLOOR)
    faded = int(n) < ac.MIN_N
    ax.scatter(x, p, s=16, marker="v" if p == P_FLOOR else marker, facecolor=color, edgecolor=SURFACE,
               linewidth=0.4, alpha=ac.FADE_ALPHA if faded else 0.85, zorder=3, clip_on=False)
    text = " ".join(t for t in (label, f"n={int(n)}" if faded else None) if t)
    if text:
        ax.annotate(text, (x, p), xytext=(3 * side, 0), textcoords="offset points", fontsize=4.4,
                    color=MUTED if faded else INK2, va="center", ha="left" if side > 0 else "right")


def counts_label(p, p_bh) -> str:
    p, p_bh = np.asarray(p, dtype=float), np.asarray(p_bh, dtype=float)
    return f"{int((p < 0.05).sum())}/{len(p)}\nBH {int((p_bh < 0.05).sum())}"


def per_model_strip(cells: pd.DataFrame, t: str, out: Path) -> None:
    q = cells[cells["qualifies"].astype(bool) & cells[f"{t}_p"].notna()]
    models = [m for m in ac.MODELS if m in set(cells["subject_model"])]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.75, 3.1))
    p_axis(ax, q[f"{t}_p"])
    for i, m in enumerate(models):
        # deterministic spread: cells sorted by p, alternately left / right of the model's axis position, labels
        # pointing outwards, so neighbouring "n=" labels do not overlap
        s = q[q["subject_model"] == m].sort_values(f"{t}_p")
        k = len(s)
        for j, (_, r) in enumerate(s.iterrows()):
            side = -1 if j % 2 == 0 else 1
            dx = side * (0.06 + 0.3 * (j // 2) / max(1, (k + 1) // 2))
            point(ax, i + dx, r[f"{t}_p"], r["n_third"], color=CASE_COLOR[r["case"]],
                  marker=DATASET_MARKER[r["dataset"]], side=side)
        ax.text(i, 2.0, counts_label(s[f"{t}_p"], s[f"{t}_p_bh"]), ha="center", va="bottom", fontsize=5.8,
                color=INK2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], rotation=45, ha="right", fontsize=6.5)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_title(TITLES[t], fontsize=7.2)
    h = [Line2D([], [], marker="o", linestyle="none", color=CASE_COLOR[c], markeredgecolor=SURFACE, markersize=5,
                label=f"{c} case") for c in CASE_ORDER]
    h += [Line2D([], [], marker=DATASET_MARKER[d], linestyle="none", color=INK2, markersize=4,
                 label=DATASET_NAMES[d]) for d in DATASET_ORDER if d in set(q["dataset"])]
    n_nt = int(((cells["n_third"] > 0) & ~cells["qualifies"].astype(bool)).sum())
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.2, columnspacing=0.9,
               handletextpad=0.3,
               title=f"{len(q)} qualifying model × dataset × style × case cells (≥ 5·(n−2) third-option switches; "
                     f"{n_nt} with fewer not tested)\nnumbers: p < 0.05 / cells, and after Benjamini–Hochberg; "
                     f"dashed: p = 0.05; faded: n < {ac.MIN_N} (n printed)", title_fontsize=6.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"distractor_uniformity_{t}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def majority_strip(pooled: pd.DataFrame, out: Path) -> None:
    q = pooled[pooled["qualifies"].astype(bool)]
    datasets = [d for d in DATASET_ORDER if d in set(pooled["dataset"])]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 2.9))
    p_axis(ax, q["majority_p"])
    for i, d in enumerate(datasets):
        s = q[q["dataset"] == d].sort_values("majority_p")
        for j, (_, r) in enumerate(s.iterrows()):
            side = -1 if j % 2 == 0 else 1
            dx = side * 0.08
            point(ax, i + dx, r["majority_p"], r["n_population"], color=CASE_COLOR["negative"],
                  marker=DATASET_MARKER[d], label=CUE_ABBREV.get(r["hint_style"], r["hint_style"]), side=side)
        ax.text(i, 2.0, counts_label(s["majority_p"], s["majority_p_bh"]), ha="center", va="bottom", fontsize=5.8,
                color=INK2)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([f"{DATASET_SHORT[d]}\n{int(pooled.loc[pooled['dataset'] == d, 'not_tested'].sum())} not tested"
                        for d in datasets], fontsize=6.4)
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_title(TITLES["majority"], fontsize=7.2)
    legend = ", ".join(f"{a} {CUE_NAMES[k]}" for k, a in CUE_ABBREV.items() if k in set(q["hint_style"]))
    fig.text(0.5, -0.02, f"{len(q)} qualifying negative dataset × style cells (≥ 5·(n−2) switches whose majority "
                         f"answer is a distractor, models pooled); positive case untestable by construction\n"
                         f"numbers: p < 0.05 / cells, and after Benjamini–Hochberg; dashed: p = 0.05; "
                         f"faded: n < {ac.MIN_N} (n printed)\n"
                         f"{legend}", ha="center", va="top", fontsize=5.8, color=INK2, wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"distractor_uniformity_majority.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def example(worked: dict, out: Path) -> None:
    m, k, n = int(worked["n_population"]), int(worked["n_hits"]), int(worked["n_options"])
    e = m / (n - 2)
    fig, ax = plt.subplots(figsize=(HALF_W * 1.25, 2.3))
    x = np.arange(2)
    ax.bar(x, [k, m - k], width=0.62, color=CASE_COLOR["negative"], linewidth=0, label="observed", zorder=2)
    ax.scatter(x, [e, m - e], marker="_", s=120, color=INK2, linewidth=1.8, zorder=3, label="uniform expectation")
    ax.set_xticks(x)
    ax.set_xticklabels(["majority no-hint\nanswer", "other\ndistractors"], fontsize=6.5)
    ax.set_ylabel("switches (models pooled)")
    tidy(ax, grid_axis="y")
    ax.legend(loc="best", fontsize=6, handlelength=1.2)
    if not bool(worked["qualifies"]):
        raise ValueError("the worked example must be a qualifying cell")
    ax.set_title(f"Pearson p = {float(worked['majority_p']):.2g} (BH {float(worked['majority_p_bh']):.2g}), "
                 f"exact binomial p = {float(worked['majority_binom_p_upper']):.2g}\nuniform share 1/{n - 2}",
                 fontsize=6.8, color=INK2)
    models = [c.split("__", 1)[1] for c in worked if c.startswith("pop__") and worked[c]]
    fig.suptitle(f"{DATASET_NAMES[worked['dataset']]} · {CUE_NAMES.get(worked['hint_style'], worked['hint_style'])} · "
                 f"negative case · {m} switches with the majority answer among the distractors, "
                 f"{len(models)} model(s) pooled", fontsize=6.8, x=0.01, ha="left", y=1.16, wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"distractor_uniformity_example.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main(argv=None) -> int:
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    data, out = a.data, a.out_dir
    apply_style()
    cells = pd.read_csv(data / "cells.csv")
    for t in ("letter", "position"):
        per_model_strip(cells, t, out)
    majority_strip(pd.read_csv(data / "majority_pooled.csv"), out)
    worked = json.loads((data / "worked_example.json").read_text())
    if worked:
        example(worked, out)
    print(f"figures → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
