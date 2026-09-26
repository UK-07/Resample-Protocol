#!/usr/bin/env python3
"""Render five rate figures and three complete LaTeX benchmark tables.

Input tables in ``paths.results`` contain the 10,000-replicate question-cluster
percentile intervals (seed 42, stratified by dataset). No intervals are inferred
from aggregate counts. SSP uses binary-judged switched originals; RSP uses
judged on-target re-rolls of persistent pairs and counts incoherent verdicts as
unfaithful. Their population, judge, and weighting units differ; the separate
matched-comparison analysis makes those differences explicit. All 10 pooled,
40 dataset, 80 cue, and 320 detailed rows are validated against exact totals.

Only ``build(paths)`` reads/writes files. This repository port preserves the
September 26 manuscript's numeric values, rounding, and vector appearance.
"""
from pathlib import Path
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA: Path
FIG: Path
TABLE: Path
MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
NAMES = ["Nemotron", "Qwen3-8B", "Qwen3.5-9B", "OLMo-3-7B", "Gemma-4-12B"]
FULL_NAMES = ["Nemotron-Nano-9B-v2", "Qwen3-8B", "Qwen3.5-9B", "OLMo-3-7B-Think", "Gemma-4-12B"]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
MODEL_NAME = dict(zip(MODELS, NAMES))
COLOR = dict(zip(MODELS, COLORS))
CUES = ["expert_opinion", "unethical_info", "tool_output", "consensus", "metadata", "answer_key_artifact", "grader_hacking", "post_hoc"]
CUE_NAMES = ["Expert opinion", "Unethical info", "Tool output", "Consensus", "Metadata", "Answer key", "Grader code", "Post-hoc"]
CUE_NAME = dict(zip(CUES, CUE_NAMES))
DATASETS = ["commonsense_qa", "medqa", "gpqa", "mmlu_pro"]
DATASET_NAME = dict(zip(DATASETS, ["CSQA", "MedQA", "GPQA-Ext.", "MMLU-Pro"]))
CASES = ["positive", "negative"]
STYLE = {"font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .6,
    "savefig.facecolor": "white"}


def read(relative):
    with (DATA / relative).open() as f:
        return list(csv.DictReader(f))


RATES: list[dict]
BY_CUE: list[dict]
BY_DATASET: list[dict]
CELLS: list[dict]
POOLED: list[dict]


def number(row, key):
    try:
        return float(row[key])
    except (ValueError, KeyError):
        return float("nan")


def one(rows, **keys):
    found = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(found) == 1, (keys, len(found))
    return found[0]


def validate():
    assert [len(POOLED), len(BY_CUE), len(BY_DATASET), len(CELLS)] == [10, 80, 40, 320]
    for rows in [POOLED, BY_CUE, BY_DATASET, CELLS]:
        assert np.nansum([number(r, "ssp_unf") for r in rows]) == 5656
        assert np.nansum([number(r, "ssp_labeled") for r in rows]) == 65772
        assert np.nansum([number(r, "rsp_unf_incl_minus1") for r in rows]) == 20000
        assert np.nansum([number(r, "rsp_judged") for r in rows]) == 219054
        assert np.nansum([number(r, "rsp_minus1") for r in rows]) == 1223
        for r in rows:
            for metric, num, den in [("ssp_rate", "ssp_unf", "ssp_labeled"), ("rsp_rate", "rsp_unf_incl_minus1", "rsp_judged")]:
                n, d = number(r, num), number(r, den)
                if math.isfinite(d) and d > 0:
                    assert math.isclose(number(r, metric), n / d, abs_tol=1e-12)
                if math.isfinite(number(r, metric)):
                    assert 0 <= number(r, metric + "_ci_lo") <= number(r, metric + "_ci_hi") <= 1
    # Repeated plotting exports must agree with the follow-up benchmark exports.
    for r in RATES:
        if r["grouping"] == "subject_model+hint_style+case":
            q = one(BY_CUE, subject_model=r["subject_model"], hint_style=r["hint_style"], case=r["case"])
        elif r["grouping"] == "subject_model+dataset+case":
            q = one(BY_DATASET, subject_model=r["subject_model"], dataset=r["dataset"], case=r["case"])
        else:
            continue
        for k in ["ssp_rate", "rsp_rate", "ssp_rate_ci_lo", "ssp_rate_ci_hi", "rsp_rate_ci_lo", "rsp_rate_ci_hi"]:
            assert math.isclose(number(r, k), number(q, k), abs_tol=1e-12)


def save_pdf(fig, filename):
    # The old renderer embedded the current time; omit it for reproducible bytes.
    fig.savefig(FIG / filename, metadata={"CreationDate": None, "ModDate": None})


def clean_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dddddd", lw=.45)
    ax.set_axisbelow(True)


def dumbbell(ax, r, y, color, offset=.08):
    s, p = number(r, "ssp_rate") * 100, number(r, "rsp_rate") * 100
    small = min(number(r, "ssp_labeled"), number(r, "rsp_judged")) < 20
    alpha = .35 if small else 1
    ax.plot([s, p], [y + offset, y - offset], color=color, lw=.8, alpha=alpha, zorder=2)
    for metric, yy, filled in [("ssp_rate", y + offset, False), ("rsp_rate", y - offset, True)]:
        x = number(r, metric) * 100
        lo, hi = number(r, metric + "_ci_lo") * 100, number(r, metric + "_ci_hi") * 100
        if math.isfinite(x):
            ax.hlines(yy, lo, hi, color=color, lw=.8, alpha=alpha, zorder=2)
            ax.plot(x, yy, "o", ms=3.4, mec=color, mew=.8,
                    mfc=color if filled else "white", alpha=alpha, zorder=3)


def protocol_legend(fig, **kwargs):
    handles = [Line2D([], [], marker="o", lw=0, color="#333333", mfc="white", ms=4, label="SSP"),
               Line2D([], [], marker="o", lw=0, color="#333333", ms=4, label="RSP")]
    fig.legend(handles=handles, ncol=2, frameon=False, handletextpad=.45, columnspacing=1, **kwargs)


def main_rates():
    # Compact main-text profile: native 3.3-inch width beside the judge panel.
    # Keep text at 7.5--8.5 pt; a shared x label avoids squeezing both panels.
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.275, right=.975, bottom=.225, top=.775, wspace=.19)
    for ax, case in zip(axes, CASES):
        for i, model in enumerate(MODELS):
            dumbbell(ax, one(POOLED, subject_model=model, case=case), 4-i, COLOR[model])
        ax.set_yticks(range(5), list(reversed(NAMES)))
        ax.set_ylim(-.55, 4.55)
        ax.set_xlim(-.7, 30)
        ax.set_xticks([0, 15, 30])
        ax.set_title(case.capitalize() + " case", pad=5)
        clean_axis(ax)
    fig.text(.625, .045, "Unfaithful rate (%)", ha="center", fontsize=8)
    protocol_legend(fig, loc="upper center", bbox_to_anchor=(.625, 1.005))
    save_pdf(fig, "fig_rates_dumbbell.pdf")
    plt.close(fig)


def dataset_rates():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.24, right=.985, bottom=.095, top=.90, wspace=.15)
    labels, positions = [], []
    for mi, model in enumerate(MODELS):
        for di, dataset in enumerate(DATASETS):
            y = (4-mi)*5 + 3-di
            positions.append(y)
            labels.append(DATASET_NAME[dataset])
            for ax, case in zip(axes, CASES):
                dumbbell(ax, one(BY_DATASET, subject_model=model, dataset=dataset, case=case), y, COLOR[model])
        for ax in axes:
            ax.text(0, (4-mi)*5+4.03, MODEL_NAME[model], fontsize=8, weight="bold", color=COLOR[model], ha="left", va="center")
    for ax, case in zip(axes, CASES):
        ax.set_yticks(positions, labels)
        ax.set_ylim(-.7, 24.4)
        ax.set_xlim(-1, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_title(case.capitalize() + " case", pad=6)
        ax.set_xlabel("Unfaithful rate (%)")
        clean_axis(ax)
    protocol_legend(fig, loc="upper center", bbox_to_anchor=(.64, .995))
    save_pdf(fig, "fig_rates_dumbbell_by_dataset.pdf")
    plt.close(fig)


def cue_rates():
    for case in CASES:
        fig, axes = plt.subplots(3, 2, figsize=(5.5, 6.5), sharex=True)
        fig.subplots_adjust(left=.215, right=.985, bottom=.075, top=.92, hspace=.48, wspace=.69)
        for mi, (ax, model) in enumerate(zip(axes.flat, MODELS)):
            for ci, cue in enumerate(CUES):
                dumbbell(ax, one(BY_CUE, subject_model=model, hint_style=cue, case=case), 7-ci, COLOR[model])
            ax.set_yticks(range(8), list(reversed(CUE_NAMES)))
            ax.set_ylim(-.55, 7.55)
            ax.set_xlim(-2, 80)
            ax.set_xticks([0, 25, 50, 75])
            ax.set_title(MODEL_NAME[model], color=COLOR[model], weight="bold", pad=5)
            ax.tick_params(axis="x", labelbottom=True)
            clean_axis(ax)
            ax.set_xlabel("Unfaithful rate (%)", labelpad=2)
        axes.flat[-1].axis("off")
        protocol_legend(fig, loc="upper center", bbox_to_anchor=(.6, .995))
        save_pdf(fig, "fig_rates_dumbbell_by_style_" + case + ".pdf")
        plt.close(fig)


def rank_average(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            ranks[i] = (start + 1 + end) / 2
        start = end
    return np.array(ranks)


def susceptibility():
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 4.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.38, top=.91, wspace=.12)
    for ax, case in zip(axes, CASES):
        data = [r for r in BY_CUE if r["case"] == case]
        for r in data:
            x, y = 100*number(r, "robust_susc"), 100*number(r, "rsp_rate")
            color, marker = COLOR[r["subject_model"]], markers[CUES.index(r["hint_style"])]
            ax.hlines(y, 100*number(r, "robust_susc_ci_lo"), 100*number(r, "robust_susc_ci_hi"), color=color, lw=.5, alpha=.6)
            ax.vlines(x, 100*number(r, "rsp_rate_ci_lo"), 100*number(r, "rsp_rate_ci_hi"), color=color, lw=.5, alpha=.6)
            ax.plot(x, y, marker, ms=4, color=color, mec="white", mew=.3)
        rho = np.corrcoef(rank_average([number(r, "robust_susc") for r in data]), rank_average([number(r, "rsp_rate") for r in data]))[0,1]
        ax.set_title(case.capitalize() + rf" case ($\rho={rho:.2f}$)")
        ax.set_xlim(0, 102)
        ax.set_ylim(-1, 82)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_xlabel("Robust susceptibility (%)")
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(color="#dddddd", lw=.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("RSP unfaithful rate (%)")
    model_handles = [Line2D([], [], marker="o", lw=0, color=c, label=n, ms=4) for n,c in zip(NAMES,COLORS)]
    cue_handles = [Line2D([], [], marker=m, lw=0, color="#333333", label=n, ms=4) for n,m in zip(CUE_NAMES,markers)]
    fig.legend(handles=model_handles, loc="lower center", bbox_to_anchor=(.52, .185), ncol=3, frameon=False, columnspacing=1.2, handletextpad=.4)
    fig.legend(handles=cue_handles, loc="lower center", bbox_to_anchor=(.52, .005), ncol=3, frameon=False, columnspacing=1.2, handletextpad=.4)
    save_pdf(fig, "fig_susceptibility_vs_unfaithfulness.pdf")
    plt.close(fig)


def tex_metric(row, metric, den=None):
    value, lo, hi = [number(row, metric + suffix) for suffix in ["", "_ci_lo", "_ci_hi"]]
    if not math.isfinite(value):
        return "---"
    dagger = r"$^\dagger$" if den and number(row, den) < 20 else ""
    return r"\shortstack{" + f"{100*value:.1f}" + dagger + r"\\(" + f"{100*lo:.1f}--{100*hi:.1f}" + ")}"


def tex_metric_inline(row, metric):
    values = [100 * number(row, metric + suffix) for suffix in ["", "_ci_lo", "_ci_hi"]]
    return f"{values[0]:.1f} ({values[1]:.1f}--{values[2]:.1f})"


def main_table():
    out = ["% Generated by scripts/build_revision_rates.py; intervals are question-cluster bootstrap.",
           r"\begingroup", r"\fontsize{8}{9.5}\selectfont", r"\setlength{\tabcolsep}{2pt}",
           r"\renewcommand{\arraystretch}{1.05}", r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
           r"Case & Model & SSP rate & RSP rate \\", r"\midrule"]
    for case in CASES:
        if case == "negative": out.append(r"\midrule")
        for i, model in enumerate(MODELS):
            r = one(POOLED, subject_model=model, case=case)
            out.append(" & ".join([case[:3] + "." if i == 0 else "", MODEL_NAME[model], tex_metric_inline(r, "ssp_rate"), tex_metric_inline(r, "rsp_rate")]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    (TABLE / "rates_main.tex").write_text("\n".join(out))


CAPTION = r"Robust susceptibility is the share of non-truncated, parsed cued originals whose pair reaches the target in $\geq$3 of 4 re-rolls. RSP unfaithfulness uses judged, to-target re-rolls of persistent pairs; incoherent verdicts ($-1$) count as unfaithful, and their exact counts appear in the $-1$ column. SSP uses sample~0 and the binary judge. Rates differ in population, judge and weighting unit. Values are percentages; parentheses contain question-cluster bootstrap 95\% intervals (10,000 replicates, seed 42, stratified by dataset). $^\dagger$: denominator $<20$; ---: no defined rate."


def benchmark_table(per_dataset):
    rows = CELLS if per_dataset else BY_CUE
    kind = "appendix" if per_dataset else "main"
    out = ["% Generated by scripts/build_revision_rates.py. Source: analysis/revision_analysis/results/plot_tables_followup/.",
           "% Preserve all 320 model/dataset/cue/case cells or all 80 model/cue/case cells; exact -1 counts."]
    for case in CASES:
        title = "Benchmark per dataset" if per_dataset else "Benchmark with datasets pooled"
        out += [r"\begingroup", r"\fontsize{8}{9}\selectfont", r"\setlength{\tabcolsep}{4pt}",
                r"\renewcommand{\arraystretch}{1.08}", r"\setlength{\LTleft}{\fill}", r"\setlength{\LTright}{\fill}",
                r"\begin{longtable}{@{}lccrcc@{}}", r"\caption{" + title + ", " + case + " case. " + CAPTION + "}" + r"\label{tab:benchmark_" + kind + "_" + case + r"}\\", r"\toprule",
                r"Cue & Robust susc. & RSP rate & $-1$ & SSP susc. & SSP rate \\", r"\midrule", r"\endfirsthead",
                r"\multicolumn{6}{l}{\textit{" + title + ", " + case + r" case (continued)}}\\", r"\toprule",
                r"Cue & Robust susc. & RSP rate & $-1$ & SSP susc. & SSP rate \\", r"\midrule", r"\endhead",
                r"\midrule", r"\multicolumn{6}{r}{\textit{Continued on next page}}\\", r"\endfoot", r"\bottomrule", r"\endlastfoot"]
        for model, full_name in zip(MODELS, FULL_NAMES):
            for dataset in DATASETS if per_dataset else [None]:
                name = full_name + (" --- " + DATASET_NAME[dataset] if dataset else "")
                out += [r"\addlinespace[3pt]", r"\multicolumn{6}{@{}l}{\textbf{" + name + r"}}\\*", r"\addlinespace[2pt]"]
                for cue in CUES:
                    keys = dict(subject_model=model, hint_style=cue, case=case)
                    if dataset: keys["dataset"] = dataset
                    r = one(rows, **keys)
                    cells = [CUE_NAME[cue], tex_metric(r,"robust_susc","n_clean"), tex_metric(r,"rsp_rate","rsp_judged"),
                             str(int(number(r,"rsp_minus1"))), tex_metric(r,"ssp_susc","n_eligible_clean"), tex_metric(r,"ssp_rate","ssp_labeled")]
                    # Keep a model/dataset block together; leave breathing room
                    # between successive point/interval pairs.
                    ending = r" \\*[1pt]" if cue != CUES[-1] else r" \\[1pt]"
                    out.append(" & ".join(cells) + ending)
        out += [r"\end{longtable}", r"\endgroup", r"\clearpage", ""]
    (TABLE / ("benchmark_" + kind + ".tex")).write_text("\n".join(out))


FIGURE_NAMES = (
    "fig_rates_dumbbell.pdf", "fig_rates_dumbbell_by_dataset.pdf",
    "fig_rates_dumbbell_by_style_positive.pdf", "fig_rates_dumbbell_by_style_negative.pdf",
    "fig_susceptibility_vs_unfaithfulness.pdf",
)
TABLE_NAMES = ("rates_main.tex", "benchmark_main.tex", "benchmark_appendix.tex")


def build(paths) -> list[Path]:
    """Validate prepared CSVs and write figures/tables to explicit output paths."""
    global DATA, FIG, TABLE, RATES, BY_CUE, BY_DATASET, CELLS, POOLED
    DATA, FIG, TABLE = Path(paths.results), Path(paths.figures), Path(paths.tables)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    RATES = read("plot_tables/fig_rates_dumbbell_ssp_vs_rsp_clusterCI.csv")
    BY_CUE = read("plot_tables_followup/benchmark_cells_model_cue_case_clusterCI.csv")
    BY_DATASET = read("plot_tables_followup/benchmark_cells_model_dataset_case_clusterCI.csv")
    CELLS = read("plot_tables_followup/benchmark_cells_model_dataset_cue_case_clusterCI.csv")
    POOLED = [r for r in RATES if r["grouping"] == "subject_model+case"]
    with plt.rc_context(STYLE):
        _build()
    return [FIG / name for name in FIGURE_NAMES] + [TABLE / name for name in TABLE_NAMES]


def _build():
    validate()
    main_rates()
    dataset_rates()
    cue_rates()
    susceptibility()
    main_table()
    benchmark_table(False)
    benchmark_table(True)
