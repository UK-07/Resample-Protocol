"""Survival slice: positive vs negative case — plot. Reads ONLY data/survival_by_model_slice.csv.

x = model (paper order), one bar per slice level, Wilson 95 % whiskers, n above every bar, n < 20 faded.

Run:
    python -m src.scripts.visualizations.survival_slice_case.plot [--data D] [--out-dir D]
Figure: <out-dir>/survival_slice_case.{png,pdf} (default <cueball>/plots/survival_slice_case/).
"""
from src.scripts.visualizations.common import section1_plot as SP
from src.scripts.visualizations.common.section1_plot import pd, plt

PLOT = "survival_slice_case"
LEVELS = ["positive", "negative"]
COLORS = SP.CASE_COLOR


def main(argv=None):
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    t = pd.read_csv(a.data / "survival_by_model_slice.csv")
    note = "SSP-unfaithful rollouts, styles and datasets pooled"
    SP.apply_style()
    models = [m for m in SP.MODELS if m in set(t.subject_model)]
    fig, ax = plt.subplots(figsize=(SP.FULL_W, 2.5))
    faded = SP.grouped_bars(ax, models, [(l, COLORS[l], None) for l in LEVELS],
                            SP.table_getter(t, "subject_model", "slice"), group_w=0.6, n_rotation=0)
    ax.set_xticklabels([SP.MODEL_NAMES[m] for m in models], fontsize=6.8)
    ax.set_ylabel("survival  P(robust_used | SSP-unfaithful)", fontsize=7)
    SP.pct_axis(ax, "y", top=1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    SP.tidy(ax, grid_axis="y")
    ax.set_title(note, fontsize=6.5, color=SP.INK2)
    ax.legend(handles=SP.legend_handles([(l, COLORS[l], None) for l in LEVELS], faded), loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.tight_layout()
    SP.save(fig, a.out_dir / PLOT)


if __name__ == "__main__":
    main()
