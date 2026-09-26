"""Build dose-response figures and question-cluster bootstrap tables.

Inputs and output directories are supplied to ``build(paths)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import common as rc
from . import survival as style

CATEGORIES = ["incoherent", "verification_only", "rejected"]
NAMES = {"incoherent": "Incoherent", "verification_only": "Verification only", "rejected": "Rejected"}
COLORS = {"incoherent": "#7B3294", "verification_only": "#0072B2", "rejected": "#D55E00"}
EXPECTED_MINUS1 = dict(zip(style.MODELS, [1448, 1245, 202, 707, 73]))


def build_tables(paths):
    """Tabulate non-truncated, judged, on-target traces of selected pairs.

    Original and fresh re-roll traces each carry one observation. The denominator
    includes every retained category except ``none``; incoherent verdicts override
    emitted roles. Dose is the target-hit count among the four fresh re-rolls.
    """
    paths.derived.mkdir(parents=True, exist_ok=True)
    pairs_path = paths.results / "pairs_master.parquet"
    rerolls_path = paths.results / "rerolls_long.parquet"
    pairs = pd.read_parquet(pairs_path)
    rerolls = pd.read_parquet(rerolls_path)
    assert not pairs.rollout_id.duplicated().any()
    selected = pairs[pairs.has_reliance].copy()
    assert (selected.k_hit == selected.qr_k_to_hint_count).all()
    assert selected.k_hit.isin(range(5)).all()

    original = selected[selected.to_hint].copy()
    original["lab"] = original.role_label
    original["category"] = original.role_orig
    original["is_original"] = True
    original["pair_id"] = original.rollout_id
    fresh = rerolls[rerolls.hit_any].merge(
        selected[["rollout_id", "rel_role", "k_hit", "qkey"]].rename(columns={"rollout_id": "pair_id"}),
        left_on="source_rollout_id", right_on="pair_id", validate="many_to_one", how="left", indicator=True,
    )
    assert (fresh._merge == "both").all()
    fresh["category"] = fresh.judge_role
    fresh["is_original"] = False
    fresh["truncated"] = fresh.truncated_b
    columns = ["rollout_id", "pair_id", "subject_model", "case", "qkey", "lab", "category", "is_original", "truncated", "rel_role", "k_hit"]
    raw = pd.concat([original[columns], fresh[columns]], ignore_index=True)
    assert not raw.rollout_id.duplicated().any()
    assert not (raw.is_original & (raw.rel_role == "control")).any()
    raw["k"] = raw.k_hit.astype(int)
    # Exactly the category/exclusion priority in production section3_claims:
    # -1 -> incoherent even if the raw role is missing or none; truncation wins.
    raw.loc[raw.lab.eq(-1).fillna(False), "category"] = "incoherent"
    raw["status"] = "kept"
    raw.loc[raw.lab.notna() & raw.category.isna(), "status"] = "missing_role"
    raw.loc[raw.lab.isna(), "status"] = "unjudged"
    raw.loc[raw.truncated.fillna(False), "status"] = "truncated"
    kept = raw[raw.status == "kept"].copy()
    assert kept.lab.isin([-1, 0, 1]).all()
    assert kept.category.isin(["credited", "neutral", "verification_only", "rejected", "none", "incoherent"]).all()
    incoherent = kept[kept.category == "incoherent"].groupby("subject_model").size().to_dict()
    assert incoherent == EXPECTED_MINUS1, (incoherent, EXPECTED_MINUS1)
    used = kept[(kept.rel_role == "used_candidate") & (kept.category != "none")].copy()

    # Use the same question frame and shared draws as the other figure tables.
    bs = rc.ClusterBootstrap(pairs.qkey)
    assert bs.Q == 3982
    tables, endpoints = [], []
    for grouping, keys in [("pooled", ["k"]), ("case", ["case", "k"]), ("model_case", ["subject_model", "case", "k"])]:
        for category in CATEGORIES:
            table = rc.ratio_table(bs, used, used.category.eq(category).astype(float), np.ones(len(used)), keys, "rate")
            if grouping == "pooled":
                boot = table.attrs["boot"]
                i0, i4 = boot.levels.index("0"), boot.levels.index("4")
                diff = boot.reps[:, i4] - boot.reps[:, i0]
                point = float(table.iloc[i4].rate - table.iloc[i0].rate)
                endpoints.append({"category": category, "contrast": "k4_minus_k0", **bs.summarize(diff, point)})
            table = rc.plain(table)
            table.insert(0, "grouping", grouping)
            table.insert(1, "category", category)
            tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(paths.derived / "dose_response_clusterCI.csv", index=False)
    pd.DataFrame(endpoints).to_csv(paths.derived / "dose_response_endpoint_differences.csv", index=False)
    raw.groupby(["subject_model", "status"]).size().rename("n").reset_index().to_csv(paths.derived / "dose_response_exclusions.csv", index=False)
    kept.groupby(["subject_model", "category"]).size().rename("n").reset_index().to_csv(paths.derived / "dose_response_all_selected_category_counts.csv", index=False)


def draw_series(ax, data, color, *, offset=0, marker="o", linewidth=.8, markersize=3):
    data = data.sort_values("k")
    xs, ys = data.k.to_numpy() + offset, data.rate.to_numpy() * 100
    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=.75)
    for x, y, (_, row) in zip(xs, ys, data.iterrows()):
        alpha = .30 if row.den < 20 else 1
        ax.plot([x, x], [100 * row.rate_ci_lo, 100 * row.rate_ci_hi], color=color, alpha=alpha, linewidth=.75)
        ax.plot(x, y, marker=marker, color=color, alpha=alpha, markersize=markersize, linestyle="none")


def draw_figures(paths):
    table = pd.read_csv(paths.derived / "dose_response_clusterCI.csv")
    pooled = table[table.grouping == "pooled"]
    fig, ax = plt.subplots(figsize=(2.695, 2.12))
    fig.subplots_adjust(left=.18, right=.98, bottom=.39, top=.96)
    for category, offset, marker in zip(CATEGORIES, [-.035, 0, .035], ["o", "s", "^"]):
        draw_series(ax, pooled[pooled.category == category], COLORS[category], offset=offset, marker=marker, markersize=2.7)
    ax.set(xlim=(-.15, 4.15), ylim=(0, 10), xticks=range(5), yticks=[0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Target re-rolls (of 4)", fontsize=7, labelpad=2)
    ax.set_ylabel("Retained rollouts (%)", fontsize=7, labelpad=3)
    ax.tick_params(labelsize=7)
    style.clean(ax, grid="y")
    fig.legend([Line2D([], [], color=COLORS[c], marker=m, markersize=2.7) for c, m in zip(CATEGORIES, ["o", "s", "^"])], [NAMES[c] for c in CATEGORIES], loc="lower center", bbox_to_anchor=(.55, .002), ncol=1, frameon=False, fontsize=7, labelspacing=.3, handlelength=1.5)
    style.finish(fig, "fig_dose_response")

    data = table[table.grouping == "model_case"]
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 6.0))
    fig.subplots_adjust(left=.12, right=.98, bottom=.16, top=.945, hspace=.57, wspace=.31)
    for i, category in enumerate(CATEGORIES):
        for j, case in enumerate(style.CASES):
            ax = axes[i, j]
            sub = data[(data.category == category) & (data.case == case)]
            for k, (model, color) in enumerate(zip(style.MODELS, style.MODEL_COLORS)):
                draw_series(ax, sub[sub.subject_model == model], color, offset=(k - 2) * .025, markersize=2.6)
            maximum = sub.rate_ci_hi.max() * 100
            ax.set(xlim=(-.18, 4.18), ylim=(0, min(100, max(1, maximum) * 1.10)), xticks=range(5))
            ax.set_title(f"{NAMES[category]} - {case}", fontsize=8.5)
            ax.set_xlabel("Target re-rolls (of 4)", fontsize=7.5, labelpad=2)
            ax.set_ylabel("Retained rollouts (%)", fontsize=7.5, labelpad=3)
            style.clean(ax, grid="y")
    fig.legend([Line2D([], [], color=c, marker="o", markersize=3) for c in style.MODEL_COLORS], style.MODEL_SHORT, loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False, fontsize=7.5, columnspacing=1, handlelength=1.5)
    style.finish(fig, "fig_dose_response_by_case")


def build(paths):
    """Recompute all dose-response intervals, then render both PDFs."""
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.derived.mkdir(parents=True, exist_ok=True)
    build_tables(paths)
    style.configure(paths)
    with plt.rc_context(style.STYLE):
        draw_figures(paths)
