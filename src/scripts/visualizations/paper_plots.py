#!/usr/bin/env python3
"""The paper's figures, from the CUEBALL manifests (GPU-free, no API key).

Reads ``hinted_rollouts/rollout_manifest.parquet``, ``resample/resample_manifest.parquet`` and
``resample/question_reliance.csv`` under ``--cueball-dir`` and writes every figure variant (``F1a`` …
``F9a``) into ``--out`` as PDF + PNG, plus ``figures.md`` with one caption draft per variant.
A rollout **switched** when its parsed answer is the cued option (``to_hint``); the unfaithful
rate is unfaithful / (unfaithful + faithful) over judged switched rollouts, with 95 % Wilson
intervals. A cell whose share of label-less rollouts exceeds 10 % is marked ``†``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.lib.paths import resolve_data_path  # noqa: E402
from src.lib.resample import wilson_interval  # noqa: E402

DEFAULT_CUEBALL_DIR = "${DATA_ROOT}/cueball"

# Entities, orders, names

MODEL_ORDER = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "qwen3.6-27b", "olmo3-7b-think", "gemma4-12b-it"]
MODEL_NAMES = {
    "nemotron-nano-9b-v2": "Nemotron-Nano-9B", "qwen3-8b": "Qwen3-8B", "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.6-27b": "Qwen3.6-27B", "olmo3-7b-think": "Olmo-3-7B-Think", "gemma4-12b-it": "Gemma-4-12B",
}
DATASET_ORDER = ["commonsense_qa", "medqa", "gpqa", "mmlu_pro"]
DATASET_NAMES = {"commonsense_qa": "CommonsenseQA", "medqa": "MedQA", "gpqa": "GPQA-Ext.", "mmlu_pro": "MMLU-Pro"}
DATASET_SHORT = {"commonsense_qa": "CSQA", "medqa": "MedQA", "gpqa": "GPQA", "mmlu_pro": "MMLU-Pro"}
MODEL_SHORT = {"nemotron-nano-9b-v2": "Nemotron-9B", "qwen3-8b": "Qwen3-8B", "qwen3.5-9b": "Qwen3.5-9B",
               "qwen3.6-27b": "Qwen3.6-27B", "olmo3-7b-think": "Olmo-3-7B", "gemma4-12b-it": "Gemma-4-12B"}
CUE_ORDER = ["expert_opinion", "unethical_info", "tool_output", "consensus", "metadata",
             "answer_key_artifact", "grader_hacking", "post_hoc"]
CUE_NAMES = {
    "expert_opinion": "expert opinion", "unethical_info": "unethical info", "tool_output": "tool output",
    "consensus": "consensus", "metadata": "metadata", "answer_key_artifact": "answer key",
    "grader_hacking": "grader code", "post_hoc": "post-hoc",
}
CUE_ABBREV = {"expert_opinion": "EO", "unethical_info": "UI", "tool_output": "TO", "consensus": "CS",
              "metadata": "MD", "answer_key_artifact": "AK", "grader_hacking": "GC", "post_hoc": "PH"}
ROLE_ORDER = ["credited", "verification_only", "rejected", "neutral", "none"]
ROLE_NAMES = {"credited": "credited", "verification_only": "verification only", "rejected": "rejected",
              "neutral": "neutral mention", "none": "no mention"}
RELIANCE_ORDER = ["robust_used", "weak_used", "mixed"]
RELIANCE_NAMES = {"robust_used": "robust (≥3/4 re-rolls follow the cue)", "weak_used": "weak (1–2/4)",
                  "mixed": "mixed / not all judged"}
STABILITY_BINS = [8, 7, 6, 5]
STABILITY_NAMES = {8: "8/8", 7: "7/8", 6: "6/8", 5: "≤5/8"}
EXCLUDED_FLAG_THRESHOLD = 0.10

# Palette: fixed slot order (the colour-blind-safety mechanism), never re-sorted by value

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLOR = dict(zip(MODEL_ORDER, SLOTS))
CASE_ORDER = ["positive", "negative"]
CASE_COLOR = {"positive": SLOTS[0], "negative": SLOTS[1]}
LABEL_COLOR = {"unfaithful": SLOTS[7], "faithful": SLOTS[0], "unchanged": "#898781"}
ROLE_COLOR = dict(zip(ROLE_ORDER, [SLOTS[0], SLOTS[2], SLOTS[6], SLOTS[3], SLOTS[7]]))
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("cueball_blue", SEQ_STEPS)
ORDINAL_3 = ["#1c5cab", "#6da7ec", "#cde2fb"]        # dark → light, one hue
ORDINAL_5 = ["#0d366b", "#256abf", "#5598e7", "#9ec5f4", "#cde2fb"]
INK, INK2, MUTED, GRID, AXIS, EMPH_GRAY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#c3c2b7"
SURFACE = "#ffffff"

FULL_W, HALF_W = 5.5, 2.7   # ICLR text width in inches, and a half


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": INK2, "text.color": INK,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 4,
    })


# Data: the manifests, reduced to the columns and derived flags the figures use


def baseline_accuracy(cueball_dir: Path) -> pd.DataFrame:
    """(subject_model, dataset, accuracy) from every ``baselines/*_baseline.meta.json`` (smoke runs skipped)."""
    import json
    rows = []
    for meta in sorted((cueball_dir / "baselines").glob("*_baseline.meta.json")):
        stem = meta.name.removesuffix("_baseline.meta.json")
        if "_smoke" in stem:
            continue
        model, _, tag = stem.partition("_")
        tag = tag.replace("-", "_")
        dataset = next((ds for ds in DATASET_ORDER if tag.startswith(ds)), None)
        d = json.loads(meta.read_text())
        if dataset and d.get("accuracy") is not None:
            rows.append({"subject_model": model, "dataset": dataset, "accuracy": float(d["accuracy"])})
    return pd.DataFrame(rows, columns=["subject_model", "dataset", "accuracy"])


@dataclass
class Data:
    once: pd.DataFrame            # hinted-once rollouts (rollout_manifest.parquet)
    rerolls: pd.DataFrame         # k=4 re-rolls only (resample_manifest.parquet, provenance == resample_k4)
    reliance: pd.DataFrame        # question_reliance.csv, one row per selected (question, cue)
    flagged: set = field(default_factory=set)   # (model, dataset) cells above the excluded-share threshold
    accuracy: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["subject_model", "dataset", "accuracy"]))


def _final_label(df: pd.DataFrame) -> pd.Series:
    col = "judge_label_final" if "judge_label_final" in df.columns else "judge_label"
    return pd.to_numeric(df[col], errors="coerce")


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """The per-row flags every figure counts: switched, verdict, judged, unanswered."""
    out = df.copy()
    label = _final_label(out)
    out["switched"] = out["to_hint"].fillna(False).astype(bool)
    out["unfaithful"] = out["switched"] & (label == 0)
    out["faithful"] = out["switched"] & (label == 1)
    out["incoherent"] = out["switched"] & (label == -1)
    out["judged"] = out["unfaithful"] | out["faithful"]
    out["unjudged"] = out["switched"] & label.isna()
    out["unanswered"] = out["model_answer"].isna()
    out["is_truncated"] = out["truncated"].fillna(False).astype(bool)
    out["changed_flag"] = out["changed"].fillna(False).astype(bool)
    out["unchanged"] = ~out["changed_flag"] & ~out["unanswered"]
    return out


def load(cueball_dir: Path) -> Data:
    once = add_flags(pd.read_parquet(cueball_dir / "hinted_rollouts" / "rollout_manifest.parquet"))
    once = once[once["subject_model"].isin(MODEL_ORDER)]
    res = pd.read_parquet(cueball_dir / "resample" / "resample_manifest.parquet")
    rerolls = add_flags(res[res["provenance"] == "resample_k4"])
    rerolls = rerolls[rerolls["subject_model"].isin(MODEL_ORDER)]
    reliance = pd.read_csv(cueball_dir / "resample" / "question_reliance.csv")
    reliance = reliance[reliance["subject_model"].isin(MODEL_ORDER)]
    cells = once.groupby(["subject_model", "dataset"]).agg(
        n=("rollout_id", "size"), unanswered=("unanswered", "sum"), incoherent=("incoherent", "sum"),
        unjudged=("unjudged", "sum"))
    share = (cells.unanswered + cells.incoherent + cells.unjudged) / cells.n
    flagged = set(share[share > EXCLUDED_FLAG_THRESHOLD].index)
    return Data(once=once, rerolls=rerolls, reliance=reliance, flagged=flagged, accuracy=baseline_accuracy(cueball_dir))


# Rates

def rate_table(df: pd.DataFrame, by: list[str], num: str = "unfaithful", den: str = "judged") -> pd.DataFrame:
    """``num / den`` per group with Wilson 95 % bounds; ``n`` is the denominator."""
    g = df.groupby(by, observed=True).agg(k=(num, "sum"), n=(den, "sum")).reset_index()
    g["rate"] = g.k / g.n.replace(0, np.nan)
    ci = [wilson_interval(int(k), int(n)) if n else (np.nan, np.nan) for k, n in zip(g.k, g.n)]
    g["lo"] = [c[0] for c in ci]; g["hi"] = [c[1] for c in ci]
    return g


def ordered(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    return df[df[col].isin(order)].assign(**{col: pd.Categorical(df[col], order, ordered=True)}).sort_values(col)


def pct(x: float, digits: int = 0) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.{digits}f}%"


# Drawing helpers

def figure(width: float, height: float, **kw):
    return plt.subplots(figsize=(width, height), **kw)


def tidy(ax, *, grid_axis: str = "x", zero_line: bool = False) -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if zero_line:
        ax.axhline(0, color=AXIS, linewidth=0.6)


def bars_h(ax, values, ci_lo=None, ci_hi=None, *, labels, colors, thickness=0.62, value_labels=True, unit="%"):
    """Horizontal bars with a CI whisker and a value at the tip; returns the y positions."""
    y = np.arange(len(values))
    ax.barh(y, values, height=thickness, color=colors, linewidth=0)
    if ci_lo is not None:
        ax.hlines(y, ci_lo, ci_hi, color=INK, linewidth=0.8)
    if value_labels:
        ends = ci_hi if ci_hi is not None else values
        xmax = float(np.nanmax(np.asarray(ends, dtype=float)))
        for yi, v, end in zip(y, values, ends):
            if not pd.isna(v):
                ax.text((v if pd.isna(end) else end) + 0.015 * xmax, yi,
                        f"{100 * v:.1f}%" if unit == "%" else f"{v:.2f}", va="center", ha="left",
                        fontsize=7, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    return y


def pct_axis(ax, axis: str = "x", top: float | None = None) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    if top is not None:
        (ax.set_xlim if axis == "x" else ax.set_ylim)(0, top)
        step = 0.25 if top > 0.9 else 0.2 if top > 0.6 else 0.1 if top > 0.3 else 0.05 if top > 0.15 else 0.02
        a.set_major_locator(MultipleLocator(step))


def heatmap(ax, values: pd.DataFrame, *, cmap=SEQ_CMAP, vmin=0.0, vmax=None, fmt=lambda v: pct(v),
            annotate=True, flags: pd.DataFrame | None = None, cbar_label: str | None = None, fig=None,
            row_names=None, col_names=None, text_threshold=0.55, norm=None, cbar_format=None):
    """A cell grid with one sequential hue (or a caller's ``norm``, e.g. diverging) and an optional † flag per cell."""
    arr = values.to_numpy(dtype=float)
    vmax = np.nanmax(arr) if vmax is None else vmax
    if norm is not None:
        im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(values.shape[1])); ax.set_yticks(range(values.shape[0]))
    ax.set_xticklabels(col_names or list(values.columns), rotation=0)
    ax.set_yticklabels(row_names or list(values.index))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # the 2 px surface gap between cells
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(True, which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v):
                    continue
                frac = float(norm(v)) if norm is not None else ((v - vmin) / (vmax - vmin) if vmax > vmin else 0)
                color = SURFACE if frac > text_threshold else INK
                flag = " †" if flags is not None and bool(flags.iloc[i, j]) else ""
                ax.text(j, i, fmt(v) + flag, ha="center", va="center", fontsize=7, color=color)
    if cbar_label and fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=7)
        cb.set_label(cbar_label, fontsize=7, color=INK2)
        cb.ax.yaxis.set_major_formatter(cbar_format or matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    return im


def place_labels(ax, points: list[tuple[float, float, str]], *, xspan: float, yspan: float, fontsize=6):
    """Point labels that move to the next slot (UR, LR, UL, LL) when the default would overlap a placed one."""
    placed: list[tuple[float, float]] = []
    slots = [(3, 3, "left", "bottom"), (3, -3, "left", "top"), (-3, 3, "right", "bottom"), (-3, -3, "right", "top")]
    for x, y, text in sorted(points, key=lambda t: (t[0], t[1])):
        for dx, dy, ha, va in slots:
            lx, ly = x / xspan + dx / 60, y / yspan + dy / 60      # rough label centre in axes fractions
            if all(abs(lx - px) > 0.11 or abs(ly - py) > 0.085 for px, py in placed):
                break
        placed.append((lx, ly))
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize, color=INK2)


def rounded_bar(ax, x: float, height: float, width: float, color: str, *, radius_frac: float = 0.16, zorder=2, y0: float = 0.0):
    """A vertical bar with a rounded top and a square base, in data units.

    Call after the axes' limits are final: the corner radius uses the current data-to-inch aspect."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    if pd.isna(height) or height <= 0:
        return
    bbox = ax.get_window_extent()
    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    aspect = ((yb - ya) / bbox.height) / ((xb - xa) / bbox.width)   # data-y per data-x for equal pixels
    r = min(width * radius_frac, height / (2 * aspect))   # never rounder than the bar is tall
    ax.add_patch(FancyBboxPatch((x - width / 2, y0), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=aspect, facecolor=color, edgecolor="none", zorder=zorder))
    base = min(height, r * aspect)
    ax.add_patch(Rectangle((x - width / 2, y0), width, base, facecolor=color, edgecolor="none", zorder=zorder))


def lightness_ramp(hex_color: str, n: int, *, lo: float = 0.30, hi: float = 0.86) -> list[str]:
    """``n`` steps of one hue from dark to light (HLS lightness ``lo`` → ``hi``)."""
    import colorsys
    from matplotlib.colors import to_hex, to_rgb
    h, l, sat = colorsys.rgb_to_hls(*to_rgb(hex_color))
    out = []
    for i in range(n):
        li = lo + (hi - lo) * i / max(n - 1, 1)
        si = sat * (1 - 0.25 * i / max(n - 1, 1))       # a touch less saturated toward the light end
        out.append(to_hex(colorsys.hls_to_rgb(h, li, si)))
    return out


# The figure registry

@dataclass
class Variant:
    figure: str
    variant: str
    caption: str
    fig: plt.Figure

    @property
    def name(self) -> str:
        return f"{self.figure}{self.variant}"


REGISTRY: dict[str, tuple] = {}


def register(figure: str, title: str):
    def deco(fn):
        REGISTRY[figure] = (title, fn)
        return fn
    return deco


def model_rates(d: Data, by_extra: list[str] | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = d.once if df is None else df
    r = rate_table(df, ["subject_model"] + (by_extra or []))
    return ordered(r, "subject_model", MODEL_ORDER)


# ---- F1 · headline ----------------------------------------------------------

@register("F1", "Headline: unfaithful rate per model")
def fig_f1(d: Data) -> list[Variant]:
    out = []
    r = model_rates(d)
    names = [MODEL_NAMES[m] for m in r.subject_model]
    colors = [MODEL_COLOR[m] for m in r.subject_model]

    # a · bars, one per model, pooled over the four datasets and eight cues
    fig, ax = figure(HALF_W + 0.6, 1.9)
    bars_h(ax, r.rate.to_numpy(), r.lo.to_numpy(), r.hi.to_numpy(), labels=names, colors=colors)
    pct_axis(ax, "x", top=min(1, r.hi.max() * 1.35))
    ax.set_xlabel("unfaithful rate among switched rollouts")
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    out.append(Variant("F1", "a", "Unfaithful rate per model, pooled over the four datasets and eight cue "
               "styles: the share of rollouts that switched to the cued option whose reasoning never "
               "credits the cue (95 % Wilson intervals; n = judged switched rollouts, "
               + ", ".join(f"{MODEL_NAMES[m]} {int(n):,}" for m, n in zip(r.subject_model, r.n)) + ").", fig))

    # b · two panels: susceptibility (switch rate over answered rollouts) and concealment (unfaithful rate)
    g = d.once.groupby("subject_model").agg(sw=("switched", "sum"), ans=("unanswered", lambda x: int((~x).sum())))
    g = g.reindex(MODEL_ORDER)
    sw = pd.DataFrame({"subject_model": g.index, "k": g.sw, "n": g.ans})
    sw["rate"] = sw.k / sw.n
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(sw.k, sw.n)]
    sw["lo"], sw["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    fig, axes = figure(FULL_W, 2.0, ncols=2, sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(MODEL_ORDER))
    for ax, tab, title, xlabel in [
        (axes[0], sw, "Susceptibility", "switch rate (of answered rollouts)"),
        (axes[1], r.set_index("subject_model").reindex(MODEL_ORDER).reset_index(), "Concealment",
         "unfaithful rate (of judged switches)"),
    ]:
        ax.hlines(y, tab.lo, tab.hi, color=INK, linewidth=0.8, zorder=2)
        ax.scatter(tab.rate, y, s=28, color=[MODEL_COLOR[m] for m in tab.subject_model], zorder=3,
                   edgecolor=SURFACE, linewidth=1)
        for yi, v, hi in zip(y, tab.rate, tab.hi):
            ax.text(hi + 0.012, yi, pct(v, 1), ha="left", va="center", fontsize=6.5, color=INK2)
        ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=7)
        pct_axis(ax, "x", top=min(1.0, float(tab.hi.max()) * 1.25 + 0.05))
        tidy(ax, grid_axis="x")
    axes[0].set_yticks(y); axes[0].set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); axes[0].invert_yaxis()
    fig.tight_layout()
    out.append(Variant("F1", "b", "Two separate questions per model: how often the model follows the cue "
               "(left: switched to the cued option, share of answered rollouts) and how often, having "
               "followed it, its reasoning hides that (right: unfaithful rate). The two are not aligned: "
               "the most susceptible models are among the least likely to verbalize. 95 % Wilson intervals.", fig))

    # c · the rate under three denominators (single switches; single switches minus noise flips; robust re-rolls)
    once = d.once
    clean = once[once["exclude_reason"].isna() | (once["exclude_reason"] != "noise_flip")]
    rob = d.rerolls[d.rerolls["contrast_b"].fillna(False).astype(bool)]
    tabs = [("all single switches", model_rates(d), "o"),
            ("single switches, noise flips removed", model_rates(d, df=clean), "s"),
            ("robust switches (re-rolls, ≥3/4 follow the cue)", model_rates(d, df=rob), "D")]
    fig, ax = figure(FULL_W, 2.1)
    offs = [-0.22, 0, 0.22]
    for (label, tab, marker), off in zip(tabs, offs):
        tab = tab.set_index("subject_model").reindex(MODEL_ORDER)
        ax.hlines(y + off, tab.lo, tab.hi, color=INK, linewidth=0.7, zorder=2)
        ax.scatter(tab.rate, y + off, s=22, marker=marker, color=[MODEL_COLOR[m] for m in MODEL_ORDER],
                   edgecolor=SURFACE, linewidth=0.8, zorder=3, label=label)
    ax.set_yticks(y); ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); ax.invert_yaxis()
    pct_axis(ax, "x", top=0.4)
    ax.set_xlabel("unfaithful rate")
    tidy(ax, grid_axis="x")
    handles = [plt.Line2D([], [], marker=m, color=INK2, linestyle="none", markersize=4.5, label=l)
               for (l, _, m) in tabs]
    ax.legend(handles=handles, loc="upper right", handletextpad=0.4)
    fig.tight_layout()
    out.append(Variant("F1", "c", "The unfaithful rate is stable across three denominators: every switched "
               "rollout (circles), switched rollouts whose cued option the model never voted for without "
               "a cue (squares; removes the switches that can be sampling noise), and the k=4 re-rolls of "
               "questions where at least three of four re-rolls follow the cue (diamonds). 95 % Wilson "
               "intervals.", fig))

    # d / e / f · vertical forms of (a), models sorted by rate
    rs = r.sort_values("rate", ascending=False).reset_index(drop=True)
    xs = np.arange(len(rs)); top = min(1.0, float(rs.hi.max()) * 1.3)
    labels = [f"{MODEL_SHORT[m]}\n$n$ = {int(n):,}" for m, n in zip(rs.subject_model, rs.n)]
    per_ds = rate_table(d.once, ["subject_model", "dataset"])

    def frame(ax):
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_xlim(-0.6, len(rs) - 0.4)
        pct_axis(ax, "y", top=top)
        ax.set_ylabel("unfaithful rate")
        ax.spines["bottom"].set_color(AXIS)
        tidy(ax, grid_axis="y")

    # d · rounded bars, interval whisker, value on the cap
    fig, ax = figure(FULL_W * 0.92, 2.35)
    frame(ax); fig.canvas.draw()
    for xi, (m, v) in enumerate(zip(rs.subject_model, rs.rate)):
        rounded_bar(ax, xi, v, 0.58, MODEL_COLOR[m])
    ax.vlines(xs, rs.lo, rs.hi, color=INK, linewidth=0.8, zorder=3)
    for xi, (v, hi) in enumerate(zip(rs.rate, rs.hi)):
        ax.text(xi, hi + 0.012 * top, pct(v, 1), ha="center", va="bottom", fontsize=7, color=INK)
    fig.tight_layout()
    out.append(Variant("F1", "d", "Unfaithful rate per model (rounded bars, models sorted by rate; whiskers are "
               "95 % Wilson intervals; n = judged switched rollouts).", fig))

    # e · lollipops
    fig, ax = figure(FULL_W * 0.92, 2.35)
    frame(ax)
    ax.vlines(xs, 0, rs.rate, color=[MODEL_COLOR[m] for m in rs.subject_model], linewidth=2.2, zorder=2)
    ax.vlines(xs, rs.lo, rs.hi, color=INK, linewidth=0.7, zorder=3)
    ax.scatter(xs, rs.rate, s=64, color=[MODEL_COLOR[m] for m in rs.subject_model], edgecolor=SURFACE, linewidth=1.2, zorder=4)
    for xi, (v, hi) in enumerate(zip(rs.rate, rs.hi)):
        ax.text(xi, hi + 0.012 * top, pct(v, 1), ha="center", va="bottom", fontsize=7, color=INK)
    fig.tight_layout()
    out.append(Variant("F1", "e", "Unfaithful rate per model as lollipops (stem = pooled rate, whisker = 95 % "
               "Wilson interval, models sorted by rate).", fig))

    # f · rounded bars with the four per-dataset rates as small marks on each bar
    fig, ax = figure(FULL_W * 0.92, 2.35)
    top_f = min(1.0, float(per_ds.rate.max()) * 1.18)
    frame(ax); pct_axis(ax, "y", top=top_f); fig.canvas.draw()
    for xi, (m, v) in enumerate(zip(rs.subject_model, rs.rate)):
        rounded_bar(ax, xi, v, 0.58, MODEL_COLOR[m])
    ds_markers = dict(zip(DATASET_ORDER, ["o", "s", "^", "D"]))
    for ds in DATASET_ORDER:
        t = per_ds[per_ds.dataset == ds].set_index("subject_model").reindex(rs.subject_model)
        ax.scatter(xs, t.rate, s=17, marker=ds_markers[ds], facecolor=SURFACE, edgecolor=INK, linewidth=0.7,
                   zorder=4, label=DATASET_NAMES[ds])
    for xi, v in enumerate(rs.rate):
        ax.text(xi, -0.02 * top_f, pct(v, 1), ha="center", va="top", fontsize=6.8, color=INK2, transform=ax.transData)
    ax.set_xticks(xs); ax.set_xticklabels([f"\n{l}" for l in labels], fontsize=6.5)
    ax.legend(loc="upper right", ncol=2, columnspacing=0.8, handletextpad=0.2, title="per dataset", title_fontsize=7)
    fig.tight_layout()
    out.append(Variant("F1", "f", "Unfaithful rate per model (bars, pooled over datasets; models sorted by rate) "
               "with the rate on each of the four datasets as hollow marks, showing the spread behind the "
               "pooled number. Values under the bars are the pooled rates.", fig))

    # g · bars split by cue style; segments = cue's unfaithful / model's judged switches, so they sum to the pooled rate
    contrib = d.once.groupby(["subject_model", "hint_style"]).unfaithful.sum().unstack(fill_value=0)
    contrib = contrib.div(rs.set_index("subject_model").n, axis=0).reindex(rs.subject_model)
    cue_stack = list(contrib.sum().sort_values(ascending=False).index)
    fig, (ax, key) = figure(FULL_W, 2.6, ncols=2, gridspec_kw={"width_ratios": [13, 2.0], "wspace": 0.03})
    frame(ax); ax.set_xticklabels(labels, fontsize=6.0); ax.set_xlim(-0.55, len(rs) - 0.45); fig.canvas.draw()
    width = 0.6
    for xi, m in enumerate(rs.subject_model):
        ramp = lightness_ramp(MODEL_COLOR[m], len(cue_stack))
        bottom = 0.0
        shares = [(c, float(contrib.loc[m, c])) for c in cue_stack]
        last = max(i for i, (_, v) in enumerate(shares) if v > 0) if any(v > 0 for _, v in shares) else -1
        for i, (c, v) in enumerate(shares):
            if v <= 0:
                continue
            if i == last:
                rounded_bar(ax, xi, v, width, ramp[i], y0=bottom)
            else:
                ax.bar(xi, v, bottom=bottom, width=width, color=ramp[i], linewidth=0)
            nxt = next((w for _, w in shares[i + 1:] if w > 0), 0.0)
            if i < last and min(v, nxt) >= 0.012 * top:   # the surface gap between segments (not on slivers)
                ax.plot([xi - width / 2, xi + width / 2], [bottom + v, bottom + v], color=SURFACE, linewidth=1.0, solid_capstyle="butt", zorder=3)
            if v >= 0.045 * top:
                ax.text(xi, bottom + v / 2, CUE_ABBREV[c], ha="center", va="center", fontsize=6,
                        color=SURFACE if i < len(cue_stack) * 0.55 else INK, zorder=4)
            bottom += v
    ax.vlines(xs, rs.lo, rs.hi, color=INK, linewidth=0.8, zorder=5)
    for xi, (v, hi) in enumerate(zip(rs.rate, rs.hi)):
        ax.text(xi, hi + 0.012 * top, pct(v, 1), ha="center", va="bottom", fontsize=7, color=INK)
    # the key: the stacking order, bottom → top, in a neutral ramp
    gray = lightness_ramp("#52514e", len(cue_stack), lo=0.28, hi=0.86)
    key.set_xlim(0, 1); key.set_ylim(0, len(cue_stack)); key.axis("off")
    for i, c in enumerate(cue_stack):
        key.add_patch(plt.Rectangle((0.05, i + 0.12), 0.28, 0.76, facecolor=gray[i], edgecolor="none"))
        key.text(0.4, i + 0.5, f"{CUE_ABBREV[c]}  {CUE_NAMES[c]}", va="center", ha="left", fontsize=6.2, color=INK2)
    key.text(0.05, len(cue_stack) + 0.15, "cue (bottom → top)", va="bottom", ha="left", fontsize=6.5, color=INK2)
    fig.tight_layout()
    out.append(Variant("F1", "g", "Unfaithful rate per model, split by cue style: each segment is the share of "
               "the model's judged switches that were unfaithful under that cue, so the segments add up to "
               "the pooled rate on top (95 % Wilson whisker; n = judged switched rollouts). Segments are "
               "stacked in one fixed order, the largest contributor overall at the bottom, shaded dark to "
               "light in the model's own hue; post-hoc, grader-code and answer-key cues make up most of "
               "every bar.", fig))
    return out


# ---- F2 · model × dataset -----------------------------------------------------

def cell_rates(d: Data) -> pd.DataFrame:
    r = rate_table(d.once, ["subject_model", "dataset"])
    r["flag"] = [(m, ds) in d.flagged for m, ds in zip(r.subject_model, r.dataset)]
    return r


@register("F2", "Model × dataset")
def fig_f2(d: Data) -> list[Variant]:
    out = []
    r = cell_rates(d)
    mat = r.pivot(index="subject_model", columns="dataset", values="rate").reindex(index=MODEL_ORDER, columns=DATASET_ORDER)
    flags = r.pivot(index="subject_model", columns="dataset", values="flag").reindex(index=MODEL_ORDER, columns=DATASET_ORDER)

    fig, ax = figure(FULL_W * 0.72, 2.2)
    heatmap(ax, mat, flags=flags, fig=fig, cbar_label="unfaithful rate",
            row_names=[MODEL_NAMES[m] for m in MODEL_ORDER], col_names=[DATASET_SHORT[x] for x in DATASET_ORDER])
    fig.tight_layout()
    out.append(Variant("F2", "a", "Unfaithful rate per model and dataset. † marks cells where more than 10 % "
               "of the rollouts yielded no label (mostly traces cut at the 16,384-token budget): their "
               "rates are biased estimates because truncation censors the longest reasoning.", fig))

    # b · dots per dataset, models as series
    fig, ax = figure(FULL_W, 2.3)
    x = np.arange(len(DATASET_ORDER))
    offs = np.linspace(-0.3, 0.3, len(MODEL_ORDER))
    for m, off in zip(MODEL_ORDER, offs):
        t = r[r.subject_model == m].set_index("dataset").reindex(DATASET_ORDER)
        ax.vlines(x + off, t.lo, t.hi, color=INK, linewidth=0.6, zorder=2)
        ax.scatter(x + off, t.rate, s=20, color=MODEL_COLOR[m], edgecolor=SURFACE, linewidth=0.8, zorder=3,
                   label=MODEL_NAMES[m])
    ax.set_xticks(x); ax.set_xticklabels([DATASET_NAMES[k] for k in DATASET_ORDER])
    pct_axis(ax, "y", top=min(1.0, float(r.hi.max()) * 1.15))
    ax.set_ylabel("unfaithful rate")
    ax.legend(ncol=3, loc="upper left", columnspacing=1.0, handletextpad=0.3)
    tidy(ax, grid_axis="y")
    fig.tight_layout()
    out.append(Variant("F2", "b", "Unfaithful rate per dataset, one dot per model (95 % Wilson intervals). "
               "MedQA draws the most concealment from every verbalizing model; the ordering of the models "
               "is the same on every dataset.", fig))

    # c · small multiples, one panel per dataset, bars per model
    fig, axes = figure(FULL_W, 2.4, ncols=4, sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, ds in zip(axes, DATASET_ORDER):
        t = r[r.dataset == ds].set_index("subject_model").reindex(MODEL_ORDER)
        xx = np.arange(len(MODEL_ORDER))
        ax.bar(xx, t.rate, width=0.62, color=[MODEL_COLOR[m] for m in MODEL_ORDER], linewidth=0)
        ax.vlines(xx, t.lo, t.hi, color=INK, linewidth=0.7)
        for xi, (v, fl) in enumerate(zip(t.rate, t.flag)):
            ax.text(xi, v + 0.01, pct(v) + (" †" if fl else ""), ha="center", va="bottom", fontsize=6, color=INK2)
        ax.set_title(DATASET_NAMES[ds])
        ax.set_xticks(xx); ax.set_xticklabels([MODEL_NAMES[m].split("-")[0] for m in MODEL_ORDER], rotation=60, ha="right", fontsize=6.5)
        tidy(ax, grid_axis="y")
    pct_axis(axes[0], "y", top=min(1.0, float(r.hi.max()) * 1.25))
    axes[0].set_ylabel("unfaithful rate")
    fig.tight_layout()
    out.append(Variant("F2", "c", "Unfaithful rate per model within each dataset (95 % Wilson intervals; † = "
               "more than 10 % of the cell's rollouts unlabeled, mostly budget-truncated).", fig))
    return out


# ---- F3 · cue styles ----------------------------------------------------------

def switch_rates(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(k=("switched", "sum"), n=("unanswered", lambda x: int((~x).sum()))).reset_index()
    g["rate"] = g.k / g.n.replace(0, np.nan)
    ci = [wilson_interval(int(k), int(n)) if n else (np.nan, np.nan) for k, n in zip(g.k, g.n)]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return g


@register("F3", "Cue styles")
def fig_f3(d: Data) -> list[Variant]:
    out = []
    r = rate_table(d.once, ["hint_style", "subject_model"])
    s = switch_rates(d.once, ["hint_style", "subject_model"])
    mean_by_cue = r.groupby("hint_style").apply(lambda t: t.k.sum() / max(t.n.sum(), 1)).sort_values(ascending=False)
    cue_rows = [c for c in mean_by_cue.index if c in CUE_ORDER]

    def grid(tab):
        return tab.pivot(index="hint_style", columns="subject_model", values="rate").reindex(index=cue_rows, columns=MODEL_ORDER)

    for variant, tab, label, cap in [
        ("a", r, "unfaithful rate", "Unfaithful rate per cue style and model (rows sorted by the pooled "
         "rate). Post-hoc and grader-code cues are verbalized most; expert opinion and unethical "
         "information almost never, on every model."),
        ("b", s, "switch rate", "Susceptibility per cue style and model: the share of answered rollouts "
         "that switched to the cued option. The least-verbalized cues (expert opinion, unethical "
         "information) are among the most followed."),
    ]:
        fig, ax = figure(FULL_W * 0.78, 2.6)
        heatmap(ax, grid(tab), fig=fig, cbar_label=label, row_names=[CUE_NAMES[c] for c in cue_rows],
                col_names=[MODEL_SHORT[m] for m in MODEL_ORDER])
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=7)
        fig.tight_layout()
        out.append(Variant("F3", variant, cap, fig))

    # c · susceptibility vs concealment, one panel per model, points = cues
    fig, axes = figure(FULL_W, 3.6, ncols=3, nrows=2, sharex=True, sharey=True, gridspec_kw={"wspace": 0.16, "hspace": 0.3})
    for ax, m in zip(axes.flat, MODEL_ORDER):
        rr = r[r.subject_model == m].set_index("hint_style").reindex(CUE_ORDER)
        ss = s[s.subject_model == m].set_index("hint_style").reindex(CUE_ORDER)
        ax.scatter(ss.rate, rr.rate, s=26, color=MODEL_COLOR[m], edgecolor=SURFACE, linewidth=0.8, zorder=3)
        place_labels(ax, [(ss.rate[c], rr.rate[c], CUE_ABBREV[c]) for c in CUE_ORDER
                          if pd.notna(ss.rate[c]) and pd.notna(rr.rate[c])], xspan=1.0, yspan=float(r.hi.max()) * 1.1)
        ax.set_title(MODEL_NAMES[m])
        tidy(ax, grid_axis="both")
    for ax in axes[1]:
        ax.set_xlabel("switch rate", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("unfaithful rate", fontsize=7)
    pct_axis(axes[0, 0], "x", top=1.0); pct_axis(axes[0, 0], "y", top=min(1.0, float(r.hi.max()) * 1.1))
    axes[0, 0].set_xticks([0, 0.5, 1.0]); axes[0, 0].set_xlim(-0.04, 1.04)
    axes[0, 0].set_ylim(-0.04 * float(r.hi.max()), min(1.0, float(r.hi.max()) * 1.1))
    fig.text(0.5, -0.01, "cues: " + ", ".join(f"{CUE_ABBREV[c]} = {CUE_NAMES[c]}" for c in CUE_ORDER),
             ha="center", va="top", fontsize=6.5, color=INK2)
    fig.tight_layout()
    out.append(Variant("F3", "c", "Susceptibility against concealment per cue style, one panel per model "
               "(each point one cue, pooled over datasets). Across models the cues the model follows "
               "most are the ones it credits least: the relation is negative within every panel.", fig))
    return out


# ---- F4 · case ------------------------------------------------------------------

@register("F4", "Positive vs negative cases")
def fig_f4(d: Data) -> list[Variant]:
    out = []
    r = rate_table(d.once, ["subject_model", "case"])
    fig, ax = figure(HALF_W + 0.9, 2.0)
    y = np.arange(len(MODEL_ORDER))
    pos = r[r.case == "positive"].set_index("subject_model").reindex(MODEL_ORDER)
    neg = r[r.case == "negative"].set_index("subject_model").reindex(MODEL_ORDER)
    ax.hlines(y, pos.rate, neg.rate, color=EMPH_GRAY, linewidth=1.6, zorder=1)
    for tab, case in [(pos, "positive"), (neg, "negative")]:
        ax.hlines(y, tab.lo, tab.hi, color=CASE_COLOR[case], linewidth=0.7, alpha=0.6, zorder=2)
        ax.scatter(tab.rate, y, s=26, color=CASE_COLOR[case], edgecolor=SURFACE, linewidth=0.8, zorder=3,
                   label=f"{case} case")
    ax.set_yticks(y); ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); ax.invert_yaxis()
    pct_axis(ax, "x", top=min(1.0, float(r.hi.max()) * 1.15))
    ax.set_xlabel("unfaithful rate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, columnspacing=1.2)
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    out.append(Variant("F4", "a", "Unfaithful rate by case. Positive: the cue points at a wrong option on a "
               "question the model answers correctly without it; negative: the cue points at the correct "
               "option on a question the model gets wrong. Following a cue toward the right answer is "
               "concealed two to four times more often than following one toward a wrong answer.", fig))

    # b · per cue, pooled over models
    rc = rate_table(d.once, ["hint_style", "case"])
    fig, ax = figure(HALF_W + 0.9, 2.2)
    y = np.arange(len(CUE_ORDER))
    pos = rc[rc.case == "positive"].set_index("hint_style").reindex(CUE_ORDER)
    neg = rc[rc.case == "negative"].set_index("hint_style").reindex(CUE_ORDER)
    ax.hlines(y, pos.rate, neg.rate, color=EMPH_GRAY, linewidth=1.6, zorder=1)
    for tab, case in [(pos, "positive"), (neg, "negative")]:
        ax.scatter(tab.rate, y, s=26, color=CASE_COLOR[case], edgecolor=SURFACE, linewidth=0.8, zorder=3, label=f"{case} case")
    ax.set_yticks(y); ax.set_yticklabels([CUE_NAMES[c] for c in CUE_ORDER]); ax.invert_yaxis()
    pct_axis(ax, "x", top=min(1.0, float(rc.hi.max()) * 1.15))
    ax.set_xlabel("unfaithful rate (pooled over models)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, columnspacing=1.2)
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    out.append(Variant("F4", "b", "Unfaithful rate by case and cue style, pooled over the six models.", fig))
    return out


# ---- F5 · re-rolls --------------------------------------------------------------

@register("F5", "Re-rolls: robustness of the switch")
def fig_f5(d: Data) -> list[Variant]:
    out = []
    q = d.reliance
    used = q[q.role == "used_candidate"]

    # a · reliance composition of the single switches
    comp = used.groupby(["subject_model", "reliance_label"]).size().unstack(fill_value=0).reindex(index=MODEL_ORDER, columns=RELIANCE_ORDER, fill_value=0)
    share = comp.div(comp.sum(axis=1), axis=0)
    fig, ax = figure(FULL_W * 0.8, 1.9)
    left = np.zeros(len(MODEL_ORDER)); y = np.arange(len(MODEL_ORDER))
    for lab, col in zip(RELIANCE_ORDER, ORDINAL_3):
        ax.barh(y, share[lab], left=left, height=0.62, color=col, linewidth=0, edgecolor=SURFACE, label=RELIANCE_NAMES[lab])
        for yi, (l, v) in enumerate(zip(left, share[lab])):
            if v > 0.08:
                ax.text(l + v / 2, yi, pct(v), ha="center", va="center", fontsize=6.5, color=SURFACE if col == ORDINAL_3[0] else INK)
        left = left + share[lab].to_numpy()
    ax.set_yticks(y); ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); ax.invert_yaxis()
    ax.set_xlim(0, 1); pct_axis(ax, "x")
    ax.set_xlabel("share of single-rollout switches")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, columnspacing=1.0, handletextpad=0.4)
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    out.append(Variant("F5", "a", "How robust a single switch is: every switched rollout was re-rolled four "
               "times with the same cued prompt. A switch is robust when at least three re-rolls follow "
               "the cue again, weak when one or two do, and mixed otherwise (including re-rolls the judge "
               "could not label). Between 68 % and 88 % of single switches are robust.", fig))

    # b · survival heatmap cue × model
    surv = used.assign(robust=(used.reliance_label == "robust_used")).groupby(["hint_style", "subject_model"]).robust.mean().unstack()
    surv = surv.reindex(index=[c for c in CUE_ORDER if c in surv.index], columns=MODEL_ORDER)
    fig, ax = figure(FULL_W * 0.78, 2.6)
    heatmap(ax, surv, fig=fig, cbar_label="robust share", vmin=0, vmax=1,
            row_names=[CUE_NAMES[c] for c in surv.index], col_names=[MODEL_SHORT[m] for m in MODEL_ORDER])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=7)
    fig.tight_layout()
    out.append(Variant("F5", "b", "Share of single switches that are robust under re-rolling, per cue style "
               "and model. The cues that are followed most (expert opinion, unethical information) are "
               "also the most robust; metadata and answer-key switches are often sampling noise on the "
               "smaller reasoning models.", fig))

    # c · single-switch rate vs rate on robust re-rolls (contrast B), per model
    single = model_rates(d).set_index("subject_model").reindex(MODEL_ORDER)
    rob = model_rates(d, df=d.rerolls[d.rerolls["contrast_b"].fillna(False).astype(bool)]).set_index("subject_model").reindex(MODEL_ORDER)
    fig, ax = figure(HALF_W + 0.9, 2.0)
    y = np.arange(len(MODEL_ORDER))
    ax.hlines(y, single.rate, rob.rate, color=EMPH_GRAY, linewidth=1.6, zorder=1)
    ax.scatter(single.rate, y, s=26, color=MUTED, edgecolor=SURFACE, linewidth=0.8, zorder=3, label="single switches")
    ax.scatter(rob.rate, y, s=26, color=[MODEL_COLOR[m] for m in MODEL_ORDER], edgecolor=SURFACE, linewidth=0.8, zorder=3, label="robust switches (re-rolls)")
    ax.set_yticks(y); ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); ax.invert_yaxis()
    pct_axis(ax, "x", top=min(1.0, float(max(single.hi.max(), rob.hi.max())) * 1.2))
    ax.set_xlabel("unfaithful rate")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=MUTED, linestyle="none", markersize=5, label="single switches"),
                       Line2D([], [], marker="o", color=INK2, linestyle="none", markersize=5, label="robust switches (re-rolls, model colour)")],
              loc="lower right")
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    out.append(Variant("F5", "c", "Unfaithful rate on the original single switches (grey) and on the re-rolls "
               "of robust questions (colour; each re-roll judged on its own). Restricting to switches "
               "that reproduce under re-sampling changes the rate little, so the headline is not driven "
               "by sampling noise.", fig))

    # d · k-of-4 distribution for used candidates and controls
    fig, axes = figure(FULL_W, 2.3, ncols=2, sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, role, title in [(axes[0], "used_candidate", "single switches"), (axes[1], "control", "matched controls (kept the baseline answer)")]:
        t = q[q.role == role].groupby(["subject_model", "k_to_hint_count"]).size().unstack(fill_value=0).reindex(index=MODEL_ORDER, columns=[0, 1, 2, 3, 4], fill_value=0)
        sh = t.div(t.sum(axis=1), axis=0)
        left = np.zeros(len(MODEL_ORDER))
        for k, col in zip([4, 3, 2, 1, 0], ORDINAL_5):
            ax.barh(y, sh[k], left=left, height=0.62, color=col, linewidth=0, label=f"{k} of 4")
            left = left + sh[k].to_numpy()
        ax.set_title(title)
        ax.set_xlim(0, 1); pct_axis(ax, "x"); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        tidy(ax, grid_axis="x")
    axes[0].set_xticks([0, 0.25, 0.5, 0.75])
    axes[0].set_yticks(y); axes[0].set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); axes[0].invert_yaxis()
    axes[1].legend(title="re-rolls following the cue", loc="upper center", bbox_to_anchor=(-0.05, -0.2), ncol=5, columnspacing=0.8, handletextpad=0.4, title_fontsize=7)
    fig.tight_layout()
    out.append(Variant("F5", "d", "How many of the four re-rolls follow the cue, for rollouts that switched "
               "once (left) and for matched controls that had kept their baseline answer (right). "
               "Switches mostly reproduce; controls mostly do not, so the cue's effect is a property of "
               "the (question, cue) pair rather than of one sample.", fig))

    # e · within-question consistency of the verdict on robust questions
    robq = used[(used.reliance_label == "robust_used") & (used.k_judged >= 3)].copy()
    robq["kind"] = np.select([robq.k_unfaithful == 0, robq.k_faithful == 0], ["always verbalized", "never verbalized"], "sometimes")
    comp = robq.groupby(["subject_model", "kind"]).size().unstack(fill_value=0).reindex(index=MODEL_ORDER, columns=["never verbalized", "sometimes", "always verbalized"], fill_value=0)
    sh = comp.div(comp.sum(axis=1), axis=0)
    fig, ax = figure(FULL_W * 0.8, 1.9)
    left = np.zeros(len(MODEL_ORDER))
    for lab, col in zip(sh.columns, [LABEL_COLOR["unfaithful"], "#e87ba4", LABEL_COLOR["faithful"]]):
        ax.barh(y, sh[lab], left=left, height=0.62, color=col, linewidth=0, label=lab)
        for yi, (l, v) in enumerate(zip(left, sh[lab])):
            if v > 0.07:
                ax.text(l + v / 2, yi, pct(v), ha="center", va="center", fontsize=6.5, color=SURFACE)
        left = left + sh[lab].to_numpy()
    ax.set_yticks(y); ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); ax.invert_yaxis()
    ax.set_xlim(0, 1); pct_axis(ax, "x")
    ax.set_xlabel("robust (question, cue) pairs with ≥3 judged re-rolls")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, columnspacing=1.0, handletextpad=0.4)
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    out.append(Variant("F5", "e", "Is concealment a property of the question or of the sample? For robust "
               "(question, cue) pairs with at least three judged re-rolls: the share whose re-rolls never "
               "credit the cue, always credit it, or do both. Most pairs are consistent, but a sizeable "
               "minority conceals in some samples and verbalizes in others.", fig))
    return out


# ---- F6 · judge role ------------------------------------------------------------

@register("F6", "How the cue is mentioned (judge role)")
def fig_f6(d: Data) -> list[Variant]:
    out = []
    sw = d.once[d.once.judged & d.once.judge_role.notna()]
    cover = d.once[d.once.judged].groupby("subject_model").judge_role.apply(lambda s: s.notna().mean()).reindex(MODEL_ORDER)

    def stacked(ax, groups: pd.Series, index: list, names: dict):
        comp = sw.groupby([groups.name, "judge_role"]).size().unstack(fill_value=0).reindex(index=index, columns=ROLE_ORDER, fill_value=0)
        sh = comp.div(comp.sum(axis=1), axis=0)
        yy = np.arange(len(index)); left = np.zeros(len(index))
        for role in ROLE_ORDER:
            ax.barh(yy, sh[role], left=left, height=0.62, color=ROLE_COLOR[role], linewidth=0, label=ROLE_NAMES[role])
            for yi, (l, v) in enumerate(zip(left, sh[role])):
                if v > 0.09:
                    ax.text(l + v / 2, yi, pct(v), ha="center", va="center", fontsize=6.5,
                            color=SURFACE if role in ("credited", "rejected", "none") else INK)
            left = left + sh[role].to_numpy()
        ax.set_yticks(yy); ax.set_yticklabels([names[i] for i in index]); ax.invert_yaxis()
        ax.set_xlim(0, 1); pct_axis(ax, "x")
        tidy(ax, grid_axis="x")
        return comp

    fig, ax = figure(FULL_W * 0.85, 2.0)
    comp = stacked(ax, sw.subject_model, MODEL_ORDER, MODEL_NAMES)
    ax.set_xlabel("judged switched rollouts with a role label")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=5, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout()
    cov = ", ".join(f"{MODEL_NAMES[m]} {pct(c)}" for m, c in cover.items())
    out.append(Variant("F6", "a", "What the reasoning does with the cue, per model, on switched rollouts: "
               "credited (the cue is given as a reason — faithful), verification only (the model reaches "
               "the answer on its own and then notes the cue agrees), rejected (the cue is dismissed and "
               "the answer still follows it), neutral mention, or no mention at all (the last three are "
               f"unfaithful). Coverage of the role label: {cov}.", fig))

    fig, ax = figure(FULL_W * 0.85, 2.3)
    stacked(ax, sw.hint_style, CUE_ORDER, CUE_NAMES)
    ax.set_xlabel("judged switched rollouts with a role label (pooled over models)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=5, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout()
    out.append(Variant("F6", "b", "The same breakdown per cue style, pooled over models. Unverbalized cues "
               "are mostly never mentioned at all; explicit rejection followed by compliance is rare "
               "except for the grader-code and answer-key cues.", fig))
    return out


# ---- F7 · fine grain ------------------------------------------------------------

@register("F7", "Fine grain: confidence, trace length, truncation")
def fig_f7(d: Data) -> list[Variant]:
    out = []
    once = d.once
    j = once[once.judged].copy()
    j["verdict"] = np.where(j.unfaithful, "unfaithful", "faithful")
    conf = pd.to_numeric(j.judge_confidence, errors="coerce")

    # a · judge confidence: share of low-confidence verdicts by verdict and model
    low = j.assign(low=(conf <= 0.7)).groupby(["subject_model", "verdict"]).low.agg(["sum", "size"]).reset_index()
    low["rate"] = low["sum"] / low["size"]
    fig, ax = figure(HALF_W + 0.9, 2.0)
    y = np.arange(len(MODEL_ORDER))
    for verdict, off in [("unfaithful", -0.18), ("faithful", 0.18)]:
        t = low[low.verdict == verdict].set_index("subject_model").reindex(MODEL_ORDER)
        ci = [wilson_interval(int(k), int(n)) for k, n in zip(t["sum"], t["size"])]
        ax.barh(y + off, t.rate, height=0.34, color=LABEL_COLOR[verdict], linewidth=0, label=verdict)
        ax.hlines(y + off, [c[0] for c in ci], [c[1] for c in ci], color=INK, linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); ax.invert_yaxis()
    pct_axis(ax, "x", top=min(1.0, float(low.rate.max()) * 1.6 + 0.02))
    ax.set_xlabel("verdicts with judge confidence ≤ 0.7")
    ax.legend(loc="lower right")
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    out.append(Variant("F7", "a", "Judge confidence: the share of verdicts the judge marked ≤ 0.7 (its "
               "scale for oblique mentions and reasoning-free traces), by verdict and model. Unfaithful "
               "verdicts are the more confident ones on every model: a missing mention is easier to "
               "establish than a credited one.", fig))

    # b · trace length by verdict (and unchanged rollouts as reference), per model
    lens = once[once.judged | once.unchanged].copy()
    lens["kind"] = np.select([lens.unfaithful, lens.faithful], ["unfaithful", "faithful"], "unchanged")
    stats = lens.groupby(["subject_model", "kind"]).trace_token_len.quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    stats.columns = ["subject_model", "kind", "q25", "q50", "q75"]
    fig, ax = figure(FULL_W * 0.8, 2.1)
    for kind, off in [("unfaithful", -0.24), ("faithful", 0), ("unchanged", 0.24)]:
        t = stats[stats.kind == kind].set_index("subject_model").reindex(MODEL_ORDER)
        ax.hlines(y + off, t.q25, t.q75, color=LABEL_COLOR[kind], linewidth=1.4, zorder=2)
        ax.scatter(t.q50, y + off, s=18, color=LABEL_COLOR[kind], edgecolor=SURFACE, linewidth=0.8, zorder=3, label=kind)
    ax.set_yticks(y); ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel("reasoning length (tokens; median and interquartile range)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, columnspacing=1.2)
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    out.append(Variant("F7", "b", "Length of the reasoning trace by outcome: switched-and-unfaithful, "
               "switched-and-faithful, and rollouts that kept their baseline answer. Unfaithful traces "
               "are not shorter than faithful ones; the cue is followed inside reasoning of ordinary "
               "length rather than replacing it.", fig))

    # c · truncation per cell
    tr = once.groupby(["subject_model", "dataset"]).is_truncated.mean().unstack().reindex(index=MODEL_ORDER, columns=DATASET_ORDER)
    fig, ax = figure(FULL_W * 0.72, 2.2)
    heatmap(ax, tr, fig=fig, cbar_label="truncated share", vmin=0, vmax=max(0.5, float(np.nanmax(tr.to_numpy()))),
            row_names=[MODEL_NAMES[m] for m in MODEL_ORDER], col_names=[DATASET_SHORT[x] for x in DATASET_ORDER])
    fig.tight_layout()
    out.append(Variant("F7", "c", "Share of hinted rollouts whose reasoning block never closed within the "
               "16,384-token budget, per model and dataset. These rollouts have no parsed answer and never "
               "enter the unfaithful rate; cells above 10 % are the ones marked † elsewhere.", fig))
    return out


# ---- F8 · baseline stability -------------------------------------------------

@register("F8", "Baseline stability")
def fig_f8(d: Data) -> list[Variant]:
    out = []
    once = d.once.copy()
    once["stab"] = once.baseline_stability.clip(lower=5).astype(int)
    r = rate_table(once, ["subject_model", "stab"])
    s = switch_rates(once, ["subject_model", "stab"])
    fig, axes = figure(FULL_W, 2.1, ncols=2, gridspec_kw={"wspace": 0.28})
    x = np.arange(len(STABILITY_BINS))
    for ax, tab, ylabel, title in [(axes[0], s, "switch rate", "Susceptibility"), (axes[1], r, "unfaithful rate", "Concealment")]:
        for m in MODEL_ORDER:
            t = tab[tab.subject_model == m].set_index("stab").reindex(STABILITY_BINS)
            ax.plot(x, t.rate, color=MODEL_COLOR[m], linewidth=1.6, marker="o", markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.6, label=MODEL_NAMES[m])
        ax.set_xticks(x); ax.set_xticklabels([STABILITY_NAMES[b] for b in STABILITY_BINS])
        ax.set_xlabel("no-cue samples agreeing (of 8)")
        ax.set_ylabel(ylabel); ax.set_title(title)
        pct_axis(ax, "y", top=min(1.0, float(tab.rate.max()) * 1.2))
        tidy(ax, grid_axis="y")
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.18, -0.3), ncol=3, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    out.append(Variant("F8", "a", "Both rates against how settled the model's own answer was without the "
               "cue (8 of 8 no-cue samples agreeing down to ≤ 5 of 8). Less settled questions are "
               "switched more often and, once switched, concealed more often.", fig))

    def model_lines(ax, tab, xcol, xvals, *, xlabels=None, point_labels=None):
        for m in MODEL_ORDER:
            t = tab[tab.subject_model == m].set_index(xcol).reindex(xvals) if xlabels is not None else \
                tab[tab.subject_model == m].sort_values(xcol)
            xx = np.arange(len(xvals)) if xlabels is not None else t[xcol].to_numpy()
            ax.plot(xx, t.rate, color=MODEL_COLOR[m], linewidth=1.8, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.7, label=MODEL_NAMES[m], solid_capstyle="round")
            if point_labels is not None:
                for x, yv, lab in zip(xx, t.rate, t[point_labels]):
                    ax.annotate(DATASET_SHORT[lab], (x, yv), xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=5.5, color=INK2)
        if xlabels is not None:
            ax.set_xticks(np.arange(len(xvals))); ax.set_xticklabels(xlabels)
        tidy(ax, grid_axis="y")

    # b / c · one panel each: concealment, then susceptibility, against the stability bin
    for variant, tab, ylabel, cap in [
        ("b", r, "unfaithful rate", "Unfaithful rate against baseline stability: how many of the 8 no-cue "
         "samples agreed with the model's answer (8/8 = settled, ≤ 5/8 = unsettled). Concealment rises as "
         "the model's own answer gets less settled, on every model."),
        ("c", s, "switch rate", "Susceptibility (share of answered rollouts that switched to the cued option) "
         "against baseline stability. Unsettled questions are switched far more often."),
    ]:
        fig, ax = figure(FULL_W * 0.62, 2.5)
        model_lines(ax, tab, "stab", STABILITY_BINS, xlabels=[STABILITY_NAMES[b] for b in STABILITY_BINS])
        ax.set_xlabel("no-cue samples agreeing with the baseline answer (of 8)")
        ax.set_ylabel(ylabel)
        pct_axis(ax, "y", top=min(1.0, float(tab.rate.max()) * 1.25))
        ax.legend(loc="upper left" if variant == "b" else "lower right", ncol=2, columnspacing=0.8, handletextpad=0.4, fontsize=6.2)
        fig.tight_layout()
        out.append(Variant("F8", variant, cap, fig))

    # d / e · against the model's baseline accuracy per dataset (colour = model, marker = dataset, no lines)
    if len(d.accuracy):
        rc = rate_table(once, ["subject_model", "dataset"]).merge(d.accuracy, on=["subject_model", "dataset"])
        sc = switch_rates(once, ["subject_model", "dataset"]).merge(d.accuracy, on=["subject_model", "dataset"])
        for variant, tab, ylabel, cap in [
            ("d", rc, "unfaithful rate", "Unfaithful rate against the model's no-cue accuracy on each dataset "
             "(one point per dataset, lines join a model's four datasets, sorted by accuracy). Harder datasets "
             "for a given model — lower accuracy — tend to show more concealment, MedQA being the outlier "
             "with high accuracy and high concealment."),
            ("e", sc, "switch rate", "Susceptibility against the model's no-cue accuracy on each dataset: the "
             "cue is followed more where the model is less accurate."),
        ]:
            from matplotlib.lines import Line2D
            ds_markers = dict(zip(DATASET_ORDER, ["o", "s", "^", "D"]))
            fig, ax = figure(FULL_W * 0.62, 2.6)
            for m in MODEL_ORDER:
                t = tab[tab.subject_model == m]
                for ds in DATASET_ORDER:
                    tt = t[t.dataset == ds]
                    ax.scatter(tt.accuracy, tt.rate, s=30, marker=ds_markers[ds], color=MODEL_COLOR[m],
                               edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.set_xlabel("baseline accuracy (no cue)")
            ax.set_ylabel(ylabel)
            pct_axis(ax, "x"); ax.set_xlim(0.25, 1.0)
            pct_axis(ax, "y", top=min(1.0, float(tab.rate.max()) * 1.25))
            tidy(ax, grid_axis="both")
            h_models = [Line2D([], [], marker="o", linestyle="none", color=MODEL_COLOR[m], markersize=5, label=MODEL_NAMES[m]) for m in MODEL_ORDER]
            h_ds = [Line2D([], [], marker=ds_markers[ds], linestyle="none", color=INK2, markersize=5, label=DATASET_NAMES[ds]) for ds in DATASET_ORDER]
            leg1 = ax.legend(handles=h_models, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, columnspacing=0.8, handletextpad=0.3, fontsize=6.2)
            ax.add_artist(leg1)
            ax.legend(handles=h_ds, loc="upper left" if variant == "d" else "lower right", ncol=1, handletextpad=0.3, fontsize=6.2, title="dataset", title_fontsize=6.5)
            fig.tight_layout()
            out.append(Variant("F8", variant, cap.replace("(one point per dataset, lines join a model's four datasets, sorted by accuracy)", "(one point per model and dataset)"), fig))
    return out


# ---- F9 · supply --------------------------------------------------------------

@register("F9", "Supply of unfaithful rollouts (appendix)")
def fig_f9(d: Data) -> list[Variant]:
    out = []
    once = d.once
    fig, axes = figure(FULL_W, 2.5, ncols=2, gridspec_kw={"width_ratios": [4, 8], "wspace": 0.35})
    c1 = once.groupby(["subject_model", "dataset"]).unfaithful.sum().unstack().reindex(index=MODEL_ORDER, columns=DATASET_ORDER)
    c2 = once.groupby(["subject_model", "hint_style"]).unfaithful.sum().unstack().reindex(index=MODEL_ORDER, columns=CUE_ORDER)
    vmax = float(max(c1.to_numpy().max(), c2.to_numpy().max()))
    heatmap(axes[0], c1, vmax=vmax, fmt=lambda v: f"{int(v):,}", row_names=[MODEL_NAMES[m] for m in MODEL_ORDER], col_names=[DATASET_SHORT[x] for x in DATASET_ORDER])
    heatmap(axes[1], c2, vmax=vmax, fmt=lambda v: f"{int(v):,}", row_names=[""] * len(MODEL_ORDER), col_names=[CUE_ABBREV[c] for c in CUE_ORDER])
    axes[0].set_title("per dataset"); axes[1].set_title("per cue style")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    out.append(Variant("F9", "a", "Number of unfaithful rollouts (judge label 0) per model and dataset, and "
               "per model and cue style: the supply of positive examples for any detector trained on "
               f"this data ({int(once.unfaithful.sum()):,} in total). Cues: "
               + ", ".join(f"{CUE_ABBREV[c]} = {CUE_NAMES[c]}" for c in CUE_ORDER) + ".", fig))
    return out


# CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cueball-dir", default=DEFAULT_CUEBALL_DIR, help="the paper tree (holds hinted_rollouts/ and resample/)")
    parser.add_argument("--out", default=None, help="output directory (default: <cueball-dir>/figures)")
    parser.add_argument("--only", default=None, help="comma-separated figures or variants, e.g. F1,F3c")
    parser.add_argument("--formats", default="pdf,png", help="comma-separated output formats")
    args = parser.parse_args(argv)

    cueball = resolve_data_path(args.cueball_dir)
    out_dir = resolve_data_path(args.out) if args.out else cueball / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    wanted = {w.strip() for w in args.only.split(",")} if args.only else None

    apply_style()
    print(f"loading manifests from {cueball} …", flush=True)
    data = load(cueball)
    print(f"  {len(data.once):,} hinted-once rollouts, {len(data.rerolls):,} re-rolls, "
          f"{len(data.reliance):,} selected (question, cue) pairs; flagged cells: {len(data.flagged)}", flush=True)

    index = ["# Paper figures", "", f"Source: `{cueball}` — generated by `src.scripts.visualizations.paper_plots`.", ""]
    n = 0
    for key, (title, fn) in REGISTRY.items():
        if wanted and not any(w == key or (w.startswith(key) and len(w) > len(key)) for w in wanted):
            continue
        index += [f"## {key} — {title}", ""]
        for v in fn(data):
            if wanted and not (key in wanted or v.name in wanted):
                plt.close(v.fig); continue
            for ext in formats:
                v.fig.savefig(out_dir / f"{v.name}.{ext}", bbox_inches="tight", pad_inches=0.02)
            plt.close(v.fig)
            index += [f"**{v.name}** — {v.caption}", ""]
            n += 1
            print(f"  wrote {v.name}", flush=True)
    (out_dir / "figures.md").write_text("\n".join(index) + "\n")
    print(f"{n} figure(s) → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
