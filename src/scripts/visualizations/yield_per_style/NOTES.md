# Yield per cue style

Clean unfaithful examples per 1,000 hinted rollouts, by cue style and model: which cues give the most confirmed
unfaithful training examples per unit of hinting.

## Run

```bash
python -m src.scripts.visualizations.yield_per_style.gather   # tables -> <cueball>/plots/yield_per_style/data/
python -m src.scripts.visualizations.yield_per_style.plot     # figure -> <cueball>/plots/yield_per_style/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--only <substr>` (debug subset of the
rows whose run / model / style contains it; written under `<plot dir>/_debug` unless `--out-dir` is given),
`--reps` (bootstrap reps, default 2000), and `--no-cache` / `--refresh-cache` / `--cache-dir` (accepted for
uniformity; this gather reads the manifests directly and uses no cache). `plot` flags: `--data` (default
`<out-dir>/data`), `--out-dir`. The pool and the bootstrap live in `common/yield_common.py`; its sha256, the repo
git sha, the shas of the repo code it depends on and every input's sha256 are recorded in `data/gather_meta.json`.

## Files

`data/yield_by_model_style.csv` (plotted), `data/yield_by_model.csv`, `data/yield_by_model_style_case.csv`,
`data/yield_by_model_style_dataset.csv`, `data/yield_table.md` (the plotted pivot), `data/checks.json`,
`data/gather_meta.json`. Figure: `yield_per_style.{png,pdf}`.

## Inputs (the paper tree only)

- `hinted_rollouts/rollout_manifest.parquet`: the denominator (one row per question x style pair, both cases).
- `resample/resample_manifest.parquet`: the re-rolls, their `reliance_label`, `to_hint`, `judge_label_final`,
  `exclude_reason`.
- `resample/question_reliance.csv`: `reliance_label` per (question, hint), used as a consistency check only.

## Definitions

- **Numerator (clean unfaithful examples)** = `yield_common.rsp_unfaithful_pool` rows with label 0 **or −1**:
  - the `src.lib.selection.select_rows("dataset_B", …)` rows of `resample_manifest.parquet`: re-rolls
    (`resample_k4`) of `robust_used` pairs that went to the target themselves, with their own
    `judge_label_final` ∈ {0, 1} and `exclude_reason` null. Only label 0 is counted;
  - plus the re-rolls that meet the same conditions but have label −1 and `exclude_reason == incoherent`. They
    count as unfaithful and are reported apart (`clean_unfaithful_label_-1`, `yield_per_1000_label0_only`).

  An earlier exclusion reason (e.g. truncated) still excludes a re-roll. Re-rolls are never noise flips; this is
  asserted, so the −1 admission hides nothing.
- **Denominator** = the original hinted rollouts of `rollout_manifest.parquet`, per model x style: one per
  question x style pair, both manifest cases, all 4 datasets.
- **Yield** = 1000 x numerator / denominator. It may exceed 1,000, because each pair has up to 4 re-rolls. The
  figure pools the cases; the per-case values are in `yield_by_model_style_case.csv`.
- **CI:** question-clustered percentile bootstrap. Questions (`question_id`) are resampled with replacement within
  the cell, and the statistic is the ratio of sums (`--reps` reps, seed 0). Wilson does not apply, since the
  quantity is a count per rollout.
- **Fading:** bars with a denominator below 20 are faded; none are expected.
- **Filters:** those of the resample scripts (no smoke runs, the 8 styles) and the 5 models. Qwen3.6-27B is excluded.
- **Nemotron re-rolls** were generated at a 24,576-token budget (others at 16,384). They are kept as generated.
- **Secondary columns** (not plotted): `rsp_unfaithful_rate` = (0 + −1) / (0 + 1 + −1); `pairs_with_any`;
  `n_questions`; `rerolls`.

## Exclusions

`data/checks.json` records the row counts of every step: hinted pairs, re-rolls, dataset_B rows per label, the
admitted −1 rows, the pool rows whose original is missing from the rollout manifest (must be 0; the gather
aborts otherwise), the pool rows whose reliance label disagrees with `question_reliance.csv` (must be 0), and the
to-target re-rolls of robust_used pairs left out of the pool, by (label, exclude_reason): unjudged re-rolls and
the ones an earlier reason (truncated) excludes.

## Plot

Grouped bars, x = the 8 cue styles (paper order), one bar per model (paper colours, Qwen3.5-9B hatched),
95 % bootstrap whiskers, bars with a denominator < 20 faded and annotated with their n.

## Caption

*Where confirmed unfaithful examples come from.* Clean unfaithful examples per 1,000 hinted rollouts, by cue style
and model. An example is a re-roll that meets all of the following: it comes from a question x cue pair whose cue
is robustly used (≥ 3 of 4 re-rolls follow it); it follows the cue itself; the judge finds the cue's influence
unverbalised, or calls the trace incoherent (its conclusion contradicts the hinted answer); it passes every other
exclusion. The denominator counts the single hinted rollout of every pair (both cases pooled, four datasets). Each
pair can contribute up to four examples, so values may exceed 1,000. Whiskers are 95 % bootstrap intervals with
questions resampled. Nemotron's re-rolls used a 24,576-token budget (others 16,384). Qwen3.6-27B is not shown.
