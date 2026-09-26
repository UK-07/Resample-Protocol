"""Re-roll distribution — plot. Reads ONLY data/reroll_hist_by_model_case.csv.

One panel per model (1 x 5), x = number of the 4 re-rolls that land on the target (0–4), bars = positive /
negative case (manifest case); y = share of that model × case's SSP flips (any binary verdict), Wilson 95 %
whiskers, the count above every bar; a model × case whose denominator is < 20 is faded.

Run:
    python -m src.scripts.visualizations.reroll_distribution.plot [--data D] [--out-dir D]
Figure: <out-dir>/reroll_distribution.{png,pdf} (default <cueball>/plots/reroll_distribution/).
"""
from src.scripts.visualizations.common import section1_plot as SP
from src.scripts.visualizations.common.section1_plot import pd, plt

PLOT = "reroll_distribution"
CASES = ["positive", "negative"]
KS = [0, 1, 2, 3, 4]


def main(argv=None):
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    t = pd.read_csv(a.data / "reroll_hist_by_model_case.csv")
    SP.apply_style()
    models = [m for m in SP.MODELS if m in set(t.subject_model)]
    top = min(1.0, float(t.ci_hi.max()) * 1.2)
    fig, axes = plt.subplots(1, len(models), figsize=(SP.FULL_W, 2.2), sharey=True, squeeze=False)
    faded_any = False
    for ax, m in zip(axes[0], models):
        d = t[t.subject_model == m]
        idx = {(r.k_to_target, r.case): (r.share, r.ci_lo, r.ci_hi, r.count, r.denominator) for r in d.itertuples()}
        # fade by the model x case denominator (the histogram's n), print the per-bar count
        def get(k, c):
            v = idx.get((k, c))
            return None if v is None else v
        for i, c in enumerate(CASES):
            bw = 0.8 / len(CASES)
            for k in KS:
                v = get(k, c)
                if v is None or v[4] == 0:
                    continue
                share, lo, hi, cnt, den = v
                faded = den < SP.SMALL_N
                faded_any |= faded
                xi = k - 0.4 + bw * (i + 0.5)
                ax.bar(xi, share, width=bw * 0.88, color=SP.CASE_COLOR[c], alpha=SP.FADE_ALPHA if faded else 1,
                       linewidth=0, zorder=2)
                ax.vlines(xi, lo, hi, color=SP.INK, linewidth=0.5, zorder=3)
                ax.text(xi, hi + 0.012 * top, SP.fmt_n(cnt), rotation=90, ha="center", va="bottom",
                        fontsize=SP.N_FONT - 0.4, color=SP.MUTED if faded else SP.INK2)
        dens = {c: int(d[d.case == c].denominator.max()) if (d.case == c).any() else 0 for c in CASES}
        ax.set_title(f"{SP.MODEL_SHORT[m]}\n" + "\n".join(f"{c[:3]} n={dens[c]:,}" for c in CASES), fontsize=6.0)
        ax.set_xticks(KS)
        ax.set_xlim(-0.6, 4.6)
        SP.pct_axis(ax, "y", top=top * 1.12)
        SP.tidy(ax, grid_axis="y")
        ax.set_xlabel("re-rolls on target", fontsize=6.3)
    axes[0, 0].set_ylabel("share of SSP flips")
    handles = SP.legend_handles([(f"{c} case", SP.CASE_COLOR[c], None) for c in CASES], faded_any)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    SP.save(fig, a.out_dir / PLOT)


if __name__ == "__main__":
    main()
