"""Survival by dataset — plot. Reads ONLY data/survival_by_model_dataset_case.csv.

One panel per model (2 x 3 grid, legend in the sixth cell), x = dataset, bars = positive / negative case
(manifest case), Wilson 95 % whiskers, n above every bar, n < 20 faded.

Run:
    python -m src.scripts.visualizations.survival_by_dataset.plot [--data D] [--out-dir D]
Figure: <out-dir>/survival_by_dataset.{png,pdf} (default <cueball>/plots/survival_by_dataset/).
"""
from src.scripts.visualizations.common import section1_plot as SP
from src.scripts.visualizations.common.section1_plot import pd, plt

PLOT = "survival_by_dataset"
CASES = ["positive", "negative"]


def main(argv=None):
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    t = pd.read_csv(a.data / "survival_by_model_dataset_case.csv")
    SP.apply_style()
    models = [m for m in SP.MODELS if m in set(t.subject_model)]
    datasets = [d for d in SP.DATASET_ORDER if d in set(t.dataset)]
    fig, axes = plt.subplots(2, 3, figsize=(SP.FULL_W, 3.9), sharey=True)
    faded = False
    flat = axes.ravel()
    for ax, m in zip(flat, models):
        d = t[t.subject_model == m]
        faded |= SP.grouped_bars(ax, datasets, [(c, SP.CASE_COLOR[c], None) for c in CASES],
                                 SP.table_getter(d, "dataset", "case"), group_w=0.72)
        ax.set_xticklabels([SP.DATASET_SHORT[x] for x in datasets], fontsize=6.3)
        SP.pct_axis(ax, "y", top=1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        SP.tidy(ax, grid_axis="y")
        ax.set_title(SP.MODEL_NAMES[m], fontsize=7.5)
    for ax in flat[len(models):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("survival", fontsize=7)
    handles = SP.legend_handles([(f"{c} case", SP.CASE_COLOR[c], None) for c in CASES], faded)
    (flat[len(models)] if len(models) < len(flat) else fig).legend(handles=handles, loc="center")
    fig.suptitle("survival = P(robust_used | SSP-unfaithful), styles pooled", fontsize=6.5, color=SP.INK2,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    SP.save(fig, a.out_dir / PLOT)


if __name__ == "__main__":
    main()
