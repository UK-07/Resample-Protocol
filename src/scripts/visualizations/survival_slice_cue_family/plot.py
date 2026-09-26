"""Survival slice: social vs artifact cue family — plot. Reads ONLY data/survival_by_model_slice.csv.

Two panels side by side (positive case | negative case, manifest case); in each, x = model (paper order), one bar
per slice level, Wilson 95 % whiskers, n above every bar, n < 20 faded. Styles within a level and datasets pooled.

Run:
    python -m src.scripts.visualizations.survival_slice_cue_family.plot [--data D] [--out-dir D]
Figure: <out-dir>/survival_slice_cue_family.{png,pdf} (default <cueball>/plots/survival_slice_cue_family/).
"""
from src.scripts.visualizations.common import section1_plot as SP
from src.scripts.visualizations.common.section1_plot import pd, plt

PLOT = "survival_slice_cue_family"
LEVELS = ["social", "artifact"]
COLORS = {"social": SP.SLOTS[6], "artifact": SP.SLOTS[3]}
CASES = ["positive", "negative"]


def main(argv=None):
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    t = pd.read_csv(a.data / "survival_by_model_slice.csv")
    SP.apply_style()
    models = [m for m in SP.MODELS if m in set(t.subject_model)]
    fig, axes = plt.subplots(1, 2, figsize=(SP.FULL_W, 2.6), sharey=True)
    faded = False
    for ax, case in zip(axes, CASES):
        d = t[t.case == case]
        faded |= SP.grouped_bars(ax, models, [(l, COLORS[l], None) for l in LEVELS],
                                 SP.table_getter(d, "subject_model", "slice"), group_w=0.72)
        ax.set_xticklabels([SP.MODEL_SHORT[m] for m in models], fontsize=6.2, rotation=30, ha="right")
        SP.pct_axis(ax, "y", top=1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        SP.tidy(ax, grid_axis="y")
        ax.set_title(f"{case} case", fontsize=7.5)
    axes[0].set_ylabel("survival  P(robust_used | SSP-unfaithful)", fontsize=6.8)
    fig.legend(handles=SP.legend_handles([(l, COLORS[l], None) for l in LEVELS], faded), loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3)
    fig.suptitle("SSP-unfaithful rollouts, datasets pooled", fontsize=6.5, color=SP.INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    SP.save(fig, a.out_dir / PLOT)


if __name__ == "__main__":
    main()
