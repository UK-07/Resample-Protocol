# Visualizations

The paper figure command reads the saved labels under `${DATA_ROOT}/cueball`, recomputes
question-cluster intervals, and renders **30 PDFs and three LaTeX tables** to a fresh directory.
Install the CPU-only pinned dependencies in `requirements/figures.txt` as described in the
[repository README](../../../README.md), then run:

```bash
python -m src.scripts.visualizations.paper_figures \
    --cueball-dir '${DATA_ROOT}/cueball' --out-root /path/to/fresh-paper-output --dry-run
python -m src.scripts.visualizations.paper_figures \
    --cueball-dir '${DATA_ROOT}/cueball' --out-root /path/to/fresh-paper-output
```

For prepared numeric tables, replace `--cueball-dir` with `--data /path/to/figure_data`, a directory
containing `pairs_master.parquet` and `rerolls_long.parquet`. The calculations and renderers live
in `src/lib/paper_figures/`.

The data-tree figure commands below retain their original definitions, documented in
[DEFINITIONS.md](DEFINITIONS.md) and each plot's `NOTES.md`. Judge-validation figures are computed
separately from the validation study's data by `judge_validation_plots.py`. None of these figure
commands generates or re-judges rollouts.

Run everything as modules from the repository root:

```bash
python -m src.scripts.visualizations.make_all                 # every plot below: gather, then plot
python -m src.scripts.visualizations.<plot>.gather            # one plot's tables  -> ${DATA_ROOT}/cueball/plots/<plot>/data/
python -m src.scripts.visualizations.<plot>.plot              # one plot's figures -> ${DATA_ROOT}/cueball/plots/<plot>/
python -m src.scripts.visualizations.paper_plots              # F1–F9             -> ${DATA_ROOT}/cueball/figures/
python -m src.scripts.visualizations.judge_validation_plots   # J1–J4             -> ${DATA_ROOT}/cueball/plots/judge_validation/
```

## Data-tree conventions

**Inputs** are the released tree under `--cueball-dir` (default `${DATA_ROOT}/cueball`, resolved by `common/paths.py`):
`hinted_rollouts/rollout_manifest.parquet` (+ `.meta.json`; one row per hinted-once rollout), `resample/`
(`resample_manifest.parquet`, `question_reliance.csv`, `noise_model_cells.csv`, the per-seed re-roll CSVs under
`rollouts/`), `baselines/*_baseline.csv` (the 8 no-hint samples per question) and `binary_judge/`. The verdicts
under `binary_judge/*_binary_judged.csv` are **inputs**: they come from a separate run of the binary judge
(`configs/llm_judge_prompts/faithfulness_binary_simple.txt`) over the switched hinted-once rollouts and are never
recomputed here.

**Outputs** go under `${DATA_ROOT}/cueball/plots/<plot>/`: the gather's tables in `data/` (CSV, plus
`gather_meta.json` with the sha256 of every input, the repository git sha and the sha of every code file the numbers
depend on) and the plot's figures `<plot>[_variant].png` + `.pdf` beside it. `plot` reads only `data/`, so a figure
can be redrawn without re-gathering. Flags: `gather` takes `--cueball-dir`, `--out-dir` (default the plot's folder),
`--only <substr>[,<substr>]`, `--no-cache`, `--refresh-cache`, `--cache-dir`; `plot` takes `--data` (default
`<out-dir>/data`) and `--out-dir`. The four alpha follow-ups also take `--alpha-data` (the alpha_vs_measured_noise
gather's `data/`; default `<cueball>/plots/alpha_vs_measured_noise/data` or `$ALPHA_DATA_DIR`).

**The stitched-rows cache.** The single-sample stitch (`common/ssp_common.py`: manifest + binary verdict + re-parsed
baseline sample 0 + reliance label, one row per hinted-once rollout) is shared by every survival, rates and funnel
gather and cached as parquet under `${DATA_ROOT}/cueball/plots/_cache/`, keyed on the module's sha, the shas of the
repository code it depends on, every input's size + mtime and `--only`; a change to any of them rebuilds it.
`--no-cache` builds in memory, `--refresh-cache` overwrites, `--cache-dir` relocates.

**`--only` debug subsets.** A gather run with `--only` restricts the rows to the models / runs / datasets whose name
contains a substring; it is never cached and, unless `--out-dir` is given, is written under `<plot dir>/_debug/`
rather than over the full tables.

**SSP and RSP.** Under the *single-sample protocol* (SSP) every quantity comes from one draw per prompt: baseline
sample 0, the one hinted rollout and the binary judge's verdict on it. Under the *resampled protocol* (RSP) each
switched (question, cue) pair is re-rolled four times and a `reliance_label` (`robust_used` ≥ 3 of 4 re-rolls
follow the cue, `weak_used` 1–2, `mixed` 0) plus the re-rolls' own verdicts replace the single draw.

## `paper_plots.py` — data-tree figures (F1–F9)

`python -m src.scripts.visualizations.paper_plots [--cueball-dir D] [--out D] [--only F1,F3c] [--formats pdf,png]`
reads the two manifests and `question_reliance.csv` and writes every variant as PDF + PNG into `<cueball>/figures/`
plus `figures.md` with one caption draft per variant. It also owns the shared style every other module imports
(`apply_style`, `FULL_W` / `HALF_W`, `MODEL_ORDER` / `MODEL_NAMES` / `MODEL_SHORT` / `MODEL_COLOR`, `CUE_ORDER` /
`CUE_NAMES` / `CUE_ABBREV`, `DATASET_ORDER` / `DATASET_NAMES` / `DATASET_SHORT`, `CASE_ORDER` / `CASE_COLOR`,
`ROLE_NAMES`, `STABILITY_NAMES`, `ORDINAL_3` / `ORDINAL_5`, `SEQ_CMAP`, `SLOTS`, the ink / grid / surface colours,
`pct_axis`, `tidy`). Rates are unfaithful / (unfaithful + faithful) over judged switched rollouts with 95 % Wilson
intervals; cells with more than 10 % label-less rollouts carry `†`.

| figure | variants | what it shows |
|---|---|---|
| F1 Headline | a bars, b susceptibility vs concealment, c three denominators, d rounded bars, e lollipops, f bars + per-dataset marks, g stacked by cue | unfaithful rate per model, pooled over datasets and cues |
| F2 Model × dataset | a heatmap, b dots per dataset, c bars per dataset | the rate per model and dataset (`†` on truncation-heavy cells) |
| F3 Cue styles | a unfaithful-rate heatmap, b switch-rate heatmap, c susceptibility-vs-concealment scatter per model | which cues are followed and which are credited |
| F4 Cases | a by case, b by case and cue | positive (cue points at a wrong option) vs negative (cue points at the correct one) |
| F5 Re-rolls | a reliance composition, b robust share per cue, c single vs robust rate, d k-of-4 distributions (switches vs controls), e within-pair verdict consistency | robustness of a single switch under k = 4 re-rolls |
| F6 Judge role | a per model, b per cue | how the reasoning treats the cue (credited / verification only / rejected / neutral / none) |
| F7 Fine grain | a judge confidence, b trace length by outcome, c truncation share | diagnostics behind the labels |
| F8 Baseline stability | a both rates vs stability, b concealment vs stability, c susceptibility vs stability, d concealment vs accuracy, e susceptibility vs accuracy | how settled the no-cue answer was |
| F9 Supply | a counts | unfaithful rollouts per model × dataset and model × cue |

## `judge_validation_plots.py` — the judge-validation figures (J1–J4)

`python -m src.scripts.visualizations.judge_validation_plots [--validation-dir D] [--cueball-dir D] [--out D]
[--only J1,J4a] [--models paper|all|<ids>] [--cues paper|all|<ids>]` reads the `judge_validation.py` run directory
(`sample_judged.csv`, `review_sheet.csv`), the shared manifest, the prompt-backfill judge cache,
`question_reliance.csv`, the label overrides, the tree's `rollout_manifest.parquet` + `binary_judge/` and the
manual golden set (a missing input skips the variants that need it) and writes `J*.{pdf,png}` + `figures_judge.md`
into `<cueball>/plots/judge_validation/`.

| figure | variants | what it shows |
|---|---|---|
| J1 Agreement with the arbiter | a κ heatmap per cue × model, b κ dots, c one-vs-rest κ per role | Cohen's κ of the primary judge against a frontier arbiter on the stratified validation sample |
| J2 Binary vs role label | a confusion per dataset, b disagreement share per role, c pooled confusion | consistency of the binary verdict with the primary judge's role on every switched rollout |
| J3 Spot-check | a precision per role on the reviewed rows, b the human golden set | the researcher's review of primary-vs-arbiter disagreements (Wilson intervals) |
| J4 Robustness of the headline | a rate under three label sets, b per cue, c re-roll quantities | the paper's rates recomputed under arbiter and reviewer labels (design-weighted, stratified bootstrap) |

## The plot modules

One folder per plot: `gather.py` (tables), `plot.py` (figures), `NOTES.md` (definition, exclusions, caption).
Shared code lives in `common/`: `paths.py` (locations), `ssp_common.py` (the SSP stitch, its cache, `gather_cli`),
`section1_plot.py` (grouped Wilson bars, `plot_args`), `alpha_common.py` (the α re-aggregation and the chi-square
tests), `section3_claims.py` (the judge-role pool), `rates_common.py` (SSP vs RSP cell tables), `yield_common.py`
(the RSP-unfaithful pool and the cluster bootstrap). Every gather is run as
`python -m src.scripts.visualizations.<plot>.gather` and every plot as `python -m src.scripts.visualizations.<plot>.plot`;
both default to `${DATA_ROOT}/cueball/plots/<plot>/`. `gather_meta.json` sits in every `data/`.

### Survival: does a single-sample label survive re-sampling?

| plot | figures | tables in `data/` | definition |
|---|---|---|---|
| `survival_vs_stability` | `survival_vs_stability` | `rows.csv`, `survival_by_model.csv`, `survival_by_model_bin.csv`, `survival_pooled.csv` | share of the positive-case SSP-unfaithful rollouts whose pair is `robust_used`, per baseline-stability bin and model, pooled over cues and datasets |
| `survival_by_style` | `survival_by_style` | `rows.csv`, `survival_by_model_case.csv`, `survival_by_model_style_case.csv` | survival per cue style, positive vs negative case, one panel per model |
| `survival_by_dataset` | `survival_by_dataset` | `rows.csv`, `survival_by_model_case.csv`, `survival_by_model_dataset_case.csv` | survival per dataset, positive vs negative case, one panel per model, cues pooled |
| `reliance_composition` | `reliance_composition_ssp_unfaithful`, `_ssp_unfaithful_negative`, `_ssp_flips`, `_ssp_flips_negative` | `rows.csv`, `composition_a_ssp_unfaithful_by_model[_style].csv`, `composition_b_ssp_flips_by_model[_style].csv`, `exclusions_by_model_style.csv` | the robust_used / weak_used / mixed mix per cue and model over (a) the SSP-unfaithful rows and (b) all SSP flips, per case |
| `reroll_distribution` | `reroll_distribution` | `rows.csv`, `reroll_hist_by_model_case.csv`, `exclusions_by_model_case.csv` | histogram of how many of the 4 re-rolls land on the target over all SSP flips, per model and case |
| `survival_slice_case` | `survival_slice_case` | `rows.csv`, `survival_by_model_slice.csv` | survival per model, positive vs negative case, cues and datasets pooled |
| `survival_slice_cue_family` | `survival_slice_cue_family` | `rows.csv`, `survival_by_model_slice.csv`, `excluded_post_hoc.csv` | survival per model for social cues (expert_opinion, consensus) vs artifact cues, both cases; post_hoc left out and reported |
| `survival_slice_post_hoc` | `survival_slice_post_hoc` | `rows.csv`, `survival_by_model_slice.csv` | survival per model for post_hoc vs the other 7 cues, both cases |

### α noise model: is single-sample noise what the closed form predicts?

| plot | figures | tables in `data/` | definition |
|---|---|---|---|
| `alpha_vs_measured_noise` | `alpha_vs_measured_noise`, `_strict`, `_by_dataset` | `cells.csv`, `rows.csv` | per model × dataset × cue × case cell, x = 1 − α with α = 1 − q/((n−2)p) under sample 0, y = the share of to-target pairs that are not `robust_used` (strict: that never re-flip) |
| `distractor_uniformity` | `distractor_uniformity_letter`, `_position`, `_majority`, `_example` | `cells.csv`, `majority_pooled.csv`, `distractor_counts.csv`, `summary.csv`, `exclusions_by_model.csv`, `worked_example.json` | tests of α's assumption that third-option switches spread uniformly over the n − 2 distractors (by letter, by position, and against the 8-sample majority answer), plus a worked example |
| `alpha_bias_vs_stability` | `alpha_bias_vs_stability` | `bins.csv`, `cells.csv`, `exclusions_by_model.csv` | measured − α-implied noise per baseline-stability bin, model and case, pooling the q/(n−2) counts inside a bin |
| `alpha_negative_case_check` | `alpha_negative_case_check` | `cells.csv`, `summary_by_model_case.csv`, `exclusions_by_model.csv` | the α-vs-measured scatter restricted to negative-case cells, with a zoom on 1 − α ≤ 10 % |
| `noise_crosscheck` | `noise_crosscheck_literal`, `_strict` | `cells.csv`, `summary_by_model.csv`, `exclusions_by_model.csv` | per cell, three noise shares side by side: measured by re-sampling, α-implied, and predicted from baseline stability |

The four follow-ups read `alpha_vs_measured_noise/data/`, so that gather runs first (`make_all` keeps the order).

### Claims vs behaviour: does what the reasoning says about the cue track what re-rolls show?

| plot | figures | tables in `data/` | definition |
|---|---|---|---|
| `role_reliance_heatmap` | `role_reliance_heatmap`, `_positive`, `_negative` | `role_share_by_model[_case\|_provenance].csv`, `column_summary[_by_case].csv`, `minus1_by_role*.csv`, `minus1_by_role.md`, `exclusions_by_model*.csv` | rows = the judge's roles + "incoherent (−1)", columns = the pair's reliance label, cell = share of the column's on-target rollouts, per model |
| `dose_response` | `dose_response`, `_by_case` | `dose_by_model[_case\|_provenance\|_style].csv`, `exclusions.csv`, `minus1_by_role_by_model.csv` | share of on-target rollouts (originals and re-rolls) that disclaim reliance on the cue, against how many of the 4 re-rolls followed it |
| `ssp_vs_rsp_false_rejections` | `ssp_vs_rsp_false_rejections`, `_by_style`, `_by_case`, `_by_stability` | `counts_by_model[_case\|_style\|_stability\|_claim_role].csv`, `rates_by_model[…].csv`, `counts_then_rates.md`, `ssp_funnel_by_model_case.csv`, `minus1_by_role_ssp_flips_by_model.csv` | of the SSP flips whose original rollout disclaims the cue (rejected / verification only / incoherent), the share the re-rolls confirm as `robust_used` |

### Rates under the two protocols

| plot | figures | tables in `data/` | definition |
|---|---|---|---|
| `rates_dumbbell` | `rates_dumbbell`, `_by_style_positive`, `_by_style_negative` | `cells_model_dataset.csv`, `cells_model_dataset_style.csv`, `join_checks.csv`, `robust_used_rerolls_not_in_dataset_B.csv`, `exclusions.md` | SSP vs RSP unfaithful rate per model × dataset (and per cue), one panel per case, sorted by gap |
| `rates_rank_change` | `rates_rank_change_styles`, `_models` (+ `ranks_styles.csv`, `ranks_models.csv` and the `_left_out.csv` lists beside them) | `cells_model_style.csv`, `cells_model_dataset.csv`, `join_checks.csv`, `robust_used_rerolls_not_in_dataset_B.csv`, `exclusions.md` | bump charts of the cue ranking within each model and the model ranking within each dataset, SSP vs RSP |
| `susceptibility_vs_unfaithfulness` | `susceptibility_vs_unfaithfulness` (+ `spearman.csv`, `dropped_points.csv` beside it) | `cells_model_style.csv`, `cells_model_case.csv`, `join_checks.csv`, `robust_used_rerolls_not_in_dataset_B.csv`, `exclusions.md` | one point per model × cue and case: x = robust susceptibility, y = the RSP unfaithful rate |
| `table2_benchmark` | `table2_benchmark_main.tex` / `.csv`, `table2_benchmark_appendix.tex` / `.csv` (LaTeX booktabs, uncompiled) | `cells_model_style.csv`, `cells_model_dataset_style.csv`, `cells_model_case.csv`, `ssp_rows.csv.gz`, `rsp_rows.csv.gz`, `join_checks.csv`, `robust_used_rerolls_not_in_dataset_B.csv`, `exclusions.md` | per model × cue (main) and model × dataset × cue (appendix), per case: robust susceptibility and RSP rate, with the SSP counterparts as secondary columns |

### Yield and cost

| plot | figures | tables in `data/` | definition |
|---|---|---|---|
| `funnel` | `funnel`, `funnel_by_case` | `pairs.parquet`, `funnel_counts.csv`, `funnel_dropouts.csv`, `join_checks.csv`, `cost_by_model.csv`, `cost_summary.csv`, `cost_checks.json`, `judge_keys.parquet`, `judge_sources.csv`, `tokens/*.parquet` | question × cue pairs surviving each step from hinted to a clean re-sampling-confirmed unfaithful example, and the tokens charged per step (`gather` stages `tokens`, `judge`, `funnel`; `tokens` needs the subject models' tokenizers from the local HF cache) |
| `yield_per_style` | `yield_per_style` | `yield_by_model.csv`, `yield_by_model_style[_case\|_dataset].csv`, `yield_table.md`, `checks.json` | clean unfaithful examples per 1,000 hinted rollouts by cue and model, with a question-clustered bootstrap interval |

## `make_all.py` — the driver

`python -m src.scripts.visualizations.make_all [--only a,b] [--gather-only | --plot-only] [--cueball-dir D]
[--out-root D] [--cache-dir D] [--dry-run]` runs every plot's gather then plot as subprocesses in the order of its
`PLOTS` tuple, continues past failures and prints a summary (exit code = failed steps). `--out-root` puts each
plot under `<out-root>/<plot>/` and points the alpha follow-ups at `<out-root>/alpha_vs_measured_noise/data`.
`paper_plots.py` and `judge_validation_plots.py` are run on their own.

## Not generated by any script

`${DATA_ROOT}/cueball/plots/cueball_overview.png` is present in the released tree but no module of this package
produces it; it is kept as found.
