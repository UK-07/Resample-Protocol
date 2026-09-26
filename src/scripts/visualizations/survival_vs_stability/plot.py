"""Survival vs stability — plot. Reads ONLY data/survival_by_model_bin.csv and data/survival_by_model.csv.

Grouped bars (x = stability bin 5/8..8/8, one bar per model), Wilson 95 % whiskers, n above every bar, bars with
n < 20 faded; a dashed horizontal marker per model at its overall (all-bin) survival, labelled in the right margin.

Run:
    python -m src.scripts.visualizations.survival_vs_stability.plot [--data D] [--out-dir D]
Figure: <out-dir>/survival_vs_stability.{png,pdf} (default <cueball>/plots/survival_vs_stability/).
"""
from matplotlib.lines import Line2D

from src.scripts.visualizations.common import section1_plot as SP
from src.scripts.visualizations.common.section1_plot import pd, plt

PLOT = "survival_vs_stability"
BINS = [5, 6, 7, 8]
BIN_NAMES = {5: "5/8", 6: "6/8", 7: "7/8", 8: "8/8"}
HATCH = {"qwen3.5-9b": "////"}   # teal vs Gemma green are close in the paper palette


def main(argv=None):
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    by_bin = pd.read_csv(a.data / "survival_by_model_bin.csv")
    overall = pd.read_csv(a.data / "survival_by_model.csv").set_index("subject_model")
    SP.apply_style()
    models = [m for m in SP.MODELS if m in set(by_bin.subject_model)]
    fig, ax = plt.subplots(figsize=(SP.FULL_W, 3.0))
    faded = SP.grouped_bars(ax, BINS, [(m, SP.MODEL_COLOR[m], HATCH.get(m)) for m in models],
                            SP.table_getter(by_bin, "stability_bin", "subject_model"), group_w=0.84,
                            n_backing=True)   # white-backed n labels sit above the dashed lines
    xmin, xmax = -0.5, len(BINS) - 0.5
    labels = []
    for m in models:
        r = float(overall.loc[m, "rate"])
        ax.hlines(r, xmin, xmax, colors=SP.MODEL_COLOR[m], linestyles=(0, (4, 2)), linewidth=0.9, zorder=1.5)
        labels.append([r, r, m])          # [text y, line y, model]
    labels.sort()
    for j in range(1, len(labels)):       # spread only the TEXT; the squares stay at their line's y
        labels[j][0] = max(labels[j][0], labels[j - 1][0] + 0.075)
    shift = max(0.0, labels[-1][0] - 1.1) if labels else 0.0
    for y_text, y_line, m in labels:
        o = overall.loc[m]
        ax.plot([xmax], [y_line], marker="s", ms=3, color=SP.MODEL_COLOR[m], clip_on=False, zorder=5)
        ax.annotate(f"{SP.MODEL_SHORT[m]} {100 * o.rate:.0f}% (n={int(o.denominator):,})", xy=(xmax, y_line),
                    xytext=(9, y_text - shift), textcoords=("offset points", "data"), va="center", ha="left",
                    fontsize=6.3, color=SP.INK2, annotation_clip=False)   # the label names its model
    ax.set_xticklabels([BIN_NAMES[b] for b in BINS])
    ax.set_xlabel("baseline stability: no-cue samples agreeing with the modal answer (of 8; binning only)")
    ax.set_ylabel("survival rate")
    SP.pct_axis(ax, "y", top=1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    SP.tidy(ax, grid_axis="y")
    ax.set_title("positive case; SSP-unfaithful = flip vs sample 0 ∧ binary judge 0; "
                 "survived = ≥3/4 re-rolls on target", fontsize=6.5, color=SP.INK2)
    handles = SP.legend_handles([(SP.MODEL_NAMES[m], SP.MODEL_COLOR[m], HATCH.get(m)) for m in models], faded,
                                [Line2D([], [], color=SP.MUTED, linestyle=(0, (4, 2)), linewidth=0.9,
                                        label="model overall")])
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, columnspacing=1.0,
              handletextpad=0.4)
    fig.tight_layout()
    SP.save(fig, a.out_dir / PLOT)


if __name__ == "__main__":
    main()
