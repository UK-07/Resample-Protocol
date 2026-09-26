"""Reliance-label composition — plot. Reads ONLY data/composition_{a_ssp_unfaithful,b_ssp_flips}_by_model_style.csv.

Same layout for every figure: one panel per model (2 x 3 grid, legend in the sixth cell), x = cue style, 100 % stacked
bars robust_used / weak_used / mixed (0/4 re-rolls on target). Positive case = the main figures, negative case =
companions with the suffix _negative (four figures in all). n (rows with a reliance label) above
each bar; bars with n < 20 hatched (colours kept at full strength).

Run:
    python -m src.scripts.visualizations.reliance_composition.plot [--data D] [--out-dir D]
Figures: <out-dir>/reliance_composition_ssp_{unfaithful,flips}[_negative].{png,pdf}
(default <cueball>/plots/reliance_composition/).
"""
from pathlib import Path

from matplotlib.patches import Patch

from src.scripts.visualizations.common import section1_plot as SP
from src.scripts.visualizations.common.section1_plot import np, pd, plt

PLOT = "reliance_composition"
LABELS = ["robust_used", "weak_used", "mixed"]
NAMES = {"robust_used": "robust_used (≥3/4 re-rolls on target)", "weak_used": "weak_used (1–2/4)",
         "mixed": "mixed (0/4)"}
COLORS = dict(zip(LABELS, SP.ORDINAL_3))
SMALL_HATCH = "////"   # n < 20: colours kept at full strength (a fade would make robust_used look like weak_used)
FIGS = {
    "a_ssp_unfaithful": ("reliance_composition_ssp_unfaithful",
                         "(a) SSP-unfaithful rollouts (flip vs sample 0 ∧ binary judge 0), {case} case"),
    "b_ssp_flips": ("reliance_composition_ssp_flips",
                    "(b) all SSP flips (any binary verdict), {case} case"),
}
CASES = ["positive", "negative"]   # positive = main figures; negative = companions (file suffix _negative)


def draw(t: pd.DataFrame, title: str, stem: Path) -> None:
    models = [m for m in SP.MODELS if m in set(t.subject_model)]
    styles = [s for s in SP.CUE_ORDER if s in set(t.hint_style)]
    fig, axes = plt.subplots(2, 3, figsize=(SP.FULL_W, 4.0), sharey=True)
    flat = axes.ravel()
    faded_any = False
    x = np.arange(len(styles))
    for ax, m in zip(flat, models):
        d = t[t.subject_model == m].set_index("hint_style").reindex(styles)
        for xi, s in zip(x, styles):
            r = d.loc[s]
            n = 0 if pd.isna(r.n_labeled) else int(r.n_labeled)
            if n == 0:
                ax.text(xi, 0.01, "n=0", rotation=90, ha="center", va="bottom", fontsize=SP.N_FONT, color=SP.MUTED)
                continue
            faded = n < SP.SMALL_N
            faded_any |= faded
            bottom = 0.0
            for lab in LABELS:
                h = float(r[f"share_{lab}"])
                ax.bar(xi, h, bottom=bottom, width=0.78, color=COLORS[lab], edgecolor=SP.SURFACE, linewidth=0.5, zorder=2)
                bottom += h
            if faded:   # a dark hatch over the whole (full-colour) bar marks n < 20
                ax.bar(xi, bottom, width=0.78, facecolor="none", hatch=SMALL_HATCH, edgecolor=SP.INK2,
                       linewidth=0, alpha=0.45, zorder=2.5)
            ax.text(xi, 1.015, SP.fmt_n(n), rotation=90, ha="center", va="bottom", fontsize=SP.N_FONT,
                    color=SP.MUTED if faded else SP.INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([SP.CUE_ABBREV[s] for s in styles], fontsize=6.3)
        ax.set_xlim(-0.6, len(styles) - 0.4)
        SP.pct_axis(ax, "y", top=1.16)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        SP.tidy(ax, grid_axis="y")
        ax.set_title(SP.MODEL_NAMES[m], fontsize=7.5)
    for ax in flat[len(models):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of rollouts", fontsize=7)
    handles = [Patch(facecolor=COLORS[l], edgecolor="none", label=NAMES[l]) for l in LABELS]
    if faded_any:
        handles.append(Patch(facecolor=SP.SURFACE, hatch=SMALL_HATCH, edgecolor=SP.INK2, linewidth=0, alpha=0.45,
                             label=f"n < {SP.SMALL_N} (hatched)"))
    items = [f"{SP.CUE_ABBREV[s]} {SP.CUE_NAMES[s]}" for s in styles]
    abbrev = "\n".join("   ".join(items[i:i + 2]) for i in range(0, len(items), 2))
    leg_ax = flat[len(models)] if len(models) < len(flat) else None
    if leg_ax is not None:
        leg_ax.legend(handles=handles, loc="upper left", fontsize=6.3)
        leg_ax.text(0.02, 0.02, abbrev,
                    transform=leg_ax.transAxes, fontsize=5.3, color=SP.INK2, va="bottom")
    fig.suptitle(title, fontsize=6.8, color=SP.INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    SP.save(fig, stem)


def main(argv=None):
    a = SP.plot_args(PLOT, argv, doc=__doc__)
    SP.apply_style()
    for pool, (stem, title) in FIGS.items():
        t = pd.read_csv(a.data / f"composition_{pool}_by_model_style.csv")
        for case in CASES:
            suffix = "" if case == "positive" else f"_{case}"
            draw(t[t.case == case], title.format(case=case), a.out_dir / f"{stem}{suffix}")


if __name__ == "__main__":
    main()
