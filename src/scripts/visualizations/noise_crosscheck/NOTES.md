# Noise cross-check: three estimators

Per model panel, x = cells (dataset × style × case) sorted by measured noise, three markers per cell, all as
SHARES of the cell's to-target flips: measured (re-sampling), α-implied (1 − α), and baseline-predicted =
mean over the cell's eligible rollouts of (8 − stability)/8 × 1/(n − 1), divided by the cell's observed p.

## Run

```bash
python -m src.scripts.visualizations.noise_crosscheck.gather   # tables  -> <cueball>/plots/noise_crosscheck/data/
python -m src.scripts.visualizations.noise_crosscheck.plot     # figures -> <cueball>/plots/noise_crosscheck/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--only a,b` (debug subset of the
rows by model / run / dataset substring; written under `<plot dir>/_debug` unless `--out-dir` is given),
`--alpha-data` (the alpha_vs_measured_noise gather's `data/` directory, default
`<cueball>/plots/alpha_vs_measured_noise/data` or `$ALPHA_DATA_DIR`), `--no-cache`, `--refresh-cache`, `--cache-dir`
(default `<cueball>/plots/_cache`; the stitched-rows cache the sample-0 re-parse check reads). `plot` flags: `--data`
(default `<out-dir>/data`), `--out-dir`. Run after the alpha_vs_measured_noise gather has written its `data/`.

## Inputs

The alpha_vs_measured_noise gather's `rows.csv` (5 models, the sample-0 protocol flags) and its `cells.csv`.
`gather.py` re-derives every protocol flag from the raw columns and asserts equality, and asserts its per-cell
counts / x / y equal the alpha gather's `cells.csv`. x's 95 % interval is the alpha gather's. The sample-0 re-parse
check reads the stitched rows' join checks (`common/ssp_common.py`). Shared code: `common/alpha_common.py`; its
sha256, the repo git sha and the shas of the repo code it depends on are recorded in `data/gather_meta.json`.

## Definitions

- Cell = model × dataset × style × manifest case, 5 paper models. Sample-0 protocol as alpha_vs_measured_noise:
  eligible = b0 answered ∧ b0 ≠ target ∧ not wrong→wrong (manifest positive ∧ b0 answered ∧ b0 ≠ target ∧
  b0 ≠ groundtruth — dropped, counted); a truncated hinted rollout is never a flip (neither to-target nor
  third-option; it stays eligible, like an unparsed hinted answer). p = to-target / eligible; q = third-option /
  eligible.
- measured: literal = 1 − robust_used share of the cell's to-target pairs; strict = share of pairs whose
  non-truncated re-rolls never hit the target. Two figures, `noise_crosscheck_literal` and `noise_crosscheck_strict`,
  one panel per model; each sorts the cells by its own measured value (ties: dataset, style, case). Wilson
  intervals drawn on the measured marker.
- α-implied = 1 − α = q/((n−2)p), clipped at 1.
- baseline-predicted = mean_{eligible rows}[(8 − stability)/8 · 1/(n − 1)] / p. Not clipped; values > 1 drawn at
  the top edge as ⇑ and counted (`pred_base_gt_1`). Recomputed a second way in `gather.py` and asserted equal.
- Stability uses all 8 no-hint samples; it only bins the data and never defines an SSP label, flip or case. Here
  it also enters the baseline-predicted estimator itself, because that estimator is defined from it; it still
  defines no flip, label or case.
- Filled markers = positive case, hollow = negative. Cells with < 20 pairs (per definition) faded, n printed.

## Exclusions

Rows that are not eligible (incl. wrong→wrong) also leave the baseline-predicted mean; truncated eligible rows stay
in it and in p's denominator. Pairs with all re-rolls truncated leave the strict denominator only. Cells with p = 0
are dropped (count in `gather_meta.json`). Per-model counts of where every rollout goes:
`data/exclusions_by_model.csv`.

## Files

`data/cells.csv` (every cell with ≥ 1 to-target flip: counts, the estimators + CIs, sort ranks, flags),
`data/summary_by_model.csv` (per model × case medians and median differences), `data/exclusions_by_model.csv`,
`data/gather_meta.json`. Figures: `noise_crosscheck_literal.{png,pdf}`, `noise_crosscheck_strict.{png,pdf}`.

## Caption

Three estimates of the share of single-rollout to-target flips that are noise, per model × dataset × cue style ×
case cell (filled: positive case, hollow: negative), cells sorted by the measured value. Measured: re-sampling —
1 − share re-flipping in ≥ 3 of 4 re-rolls (literal figure) / share never re-flipping in the non-truncated
re-rolls (strict figure) (95 % Wilson intervals). α-implied: 1 − α from the sample-0 flip counts.
Baseline-predicted: the target's expected noise share from the no-hint baseline alone,
mean[(8 − stability)/8 · 1/(n − 1)] / p. Faded cells rest on fewer than 20 flips. Stability uses all 8 no-hint
samples; it only bins the data and never defines an SSP label, flip or case.
