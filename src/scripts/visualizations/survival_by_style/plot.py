"""Survival by cue style — plot. Reads ONLY data/survival_by_model_style_case.csv.

One panel per model (stacked), x = the 8 cue styles, bars = positive / negative case (manifest case), Wilson 95 %
whiskers, n above every bar, n < 20 faded.

Run:
    python -m src.scripts.visualizations.survival_by_style.plot [--data D] [--out-dir D]
Figure: <out-dir>/survival_by_style.{png,pdf} (default <cueball>/plots/survival_by_style/).
"""
from src.scripts.visualizations.common import section1_plot as SP
from src.scripts.visualizations.common.section1_plot import pd, plt

PLOT = "survival_by_style"
CASES = ["positive", "negative"]


def main(argv=None):
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    t = pd.read_csv(a.data / "survival_by_model_style_case.csv")
    SP.apply_style()
    models = [m for m in SP.MODELS if m in set(t.subject_model)]
    styles = [s for s in SP.CUE_ORDER if s in set(t.hint_style)]
    fig, axes = plt.subplots(len(models), 1, figsize=(SP.FULL_W, 1.35 * len(models) + 0.5), sharex=True,
                             squeeze=False)
    faded = False
    for ax, m in zip(axes[:, 0], models):
        d = t[t.subject_model == m]
        faded |= SP.grouped_bars(ax, styles, [(c, SP.CASE_COLOR[c], None) for c in CASES],
                                 SP.table_getter(d, "hint_style", "case"), group_w=0.74)
        SP.pct_axis(ax, "y", top=1.18)
        ax.set_yticks([0, 0.5, 1.0])
        SP.tidy(ax, grid_axis="y")
        ax.set_title(SP.MODEL_NAMES[m], fontsize=7.5)
        ax.set_ylabel("survival", fontsize=7)
    axes[-1, 0].set_xticklabels([SP.CUE_NAMES[s].replace(" ", "\n") for s in styles], rotation=0, fontsize=6.6)
    axes[-1, 0].set_xlabel("cue style")
    handles = SP.legend_handles([(f"{c} case", SP.CASE_COLOR[c], None) for c in CASES], faded)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3)
    fig.suptitle("survival = P(robust_used | SSP-unfaithful): share of single-sample unfaithful rollouts whose "
                 "question re-flips in ≥3/4 re-rolls", fontsize=6.5, color=SP.INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    SP.save(fig, a.out_dir / PLOT)


if __name__ == "__main__":
    main()
