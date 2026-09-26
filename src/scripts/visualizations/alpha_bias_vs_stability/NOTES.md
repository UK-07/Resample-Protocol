# α bias vs baseline stability

y = measured − α-implied noise, x = baseline-stability bin, faceted by manifest case (rows) and measured-noise
definition (columns), one line per model with question-cluster bootstrap intervals. Computed per model × case × bin,
pooling datasets and styles inside the bin — the implied-noise **counts** (q/(n−2) per cell) are pooled, never α
values. Tests whether α's error concentrates in unstable questions.

## Run

```bash
python -m src.scripts.visualizations.alpha_vs_measured_noise.gather        # first: writes rows.csv / cells.csv
python -m src.scripts.visualizations.alpha_bias_vs_stability.gather        # tables -> <cueball>/plots/alpha_bias_vs_stability/data/
python -m src.scripts.visualizations.alpha_bias_vs_stability.plot          # figure -> <cueball>/plots/alpha_bias_vs_stability/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--alpha-data` (the alpha_vs_measured_noise
gather's `data/`; default `<cueball>/plots/alpha_vs_measured_noise/data`, or `$ALPHA_DATA_DIR`), `--only <substr>`
(debug subset of the rows by model / run / dataset substring; written under `<plot dir>/_debug` unless `--out-dir`
is given — the per-cell check against the alpha gather then needs an `--alpha-data` built with the same subset),
`--no-cache`, `--refresh-cache`, `--cache-dir` (default `<cueball>/plots/_cache`; the stitched-rows cache the
sample-0 re-parse check reads). `plot` flags: `--data` (default `<out-dir>/data`), `--out-dir`.

## Inputs

`<alpha data>/rows.csv` (5 models, one row per hinted-once rollout with the sample-0 protocol flags) and its
`cells.csv`. `gather` re-derives every protocol flag from the raw columns and asserts equality, asserts its
per-cell counts / x / y equal the alpha gather's `cells.csv`, and asserts the stored sample-0 letter equals the
repo parser's re-parse for every source (from the stitched-rows cache of `common/ssp_common.py`). Shared code:
`common/alpha_common.py` (its sha256, the repo git sha and the shas of the repo code it depends on are recorded
in `data/gather_meta.json` next to every input's sha256).

## Definitions

- 5 paper models, 8 styles, case = manifest `case`; b0 = stored sample-0 letter (= its re-parse, checked); h =
  `model_answer`. Sample-0 protocol as in alpha_vs_measured_noise: eligible = b0 answered ∧ b0 ≠ target ∧ not
  wrong→wrong (manifest positive ∧ b0 answered ∧ b0 ≠ target ∧ b0 ≠ groundtruth — dropped, counted); a truncated
  hinted rollout is never a flip (neither to-target nor third-option; it stays eligible, like an unparsed hinted
  answer). To-target = eligible ∧ not truncated ∧ h = t; third-option = eligible ∧ not truncated ∧ h answered ∧
  h ∉ {t, b0}.
- Bin = manifest `baseline_stability` (5, 6, 7, 8; the tree has no question below 5/8 — checked; 5 is labelled
  ≤5/8).
- **Stability uses all 8 no-hint samples; it only bins the data and never defines an SSP label, flip or case.**
- Per model × case × bin, over all its datasets and styles:
  - α-implied noise = min(1, Σ_rows [third-option]/(n_row − 2) / Σ_rows [to-target]).
  - literal measured = 1 − n_robust_used / n_pairs_labeled over the to-target pairs (robust_used = ≥ 3 of 4
    re-rolls on target; stored labels).
  - strict measured = n_pairs never re-flipping / n_pairs with ≥ 1 non-truncated re-roll.
  - y = measured − implied, for each definition (two columns of panels).
- 95 % CI: question-cluster bootstrap (`N_BOOT` = 2,000 draws, `BOOT_SEED`) inside each (model, case, bin): a
  question (model-specific) has one stability and one case (checked), so it is resampled with all its styles. The
  point estimate is recomputed from the per-question sums and asserted equal to the table. Wilson CIs of the two
  measured shares are also in `bins.csv`.
- Fade rule: points with < 20 pairs (per definition) faded, n printed.

## Exclusions

Under the sample-0 protocol a hinted-once rollout is not eligible when b0 is unanswered, b0 is the target, or the
row is wrong→wrong (dropped). Of the eligible rows, those without a parsed hinted answer and the truncated ones
are never flips; the to-target flips are all re-sampled and labelled (pairs whose re-rolls are all truncated
leave the strict denominator only). Per-model counts: `data/exclusions_by_model.csv`.

## Outputs

`alpha_bias_vs_stability.{png,pdf}` (2 × 2: rows = case, columns = literal | strict). `data/bins.csv` (model × case
× bin: all counts, the three estimators, both differences + CIs), `data/cells.csv` (model × dataset × style × case
× bin counts, audit trail), `data/exclusions_by_model.csv`, `data/gather_meta.json`.

## Caption

Where does α go wrong? Difference between the noise share re-sampling measures and the share α implies, per
baseline-stability bin (number of the 8 no-hint samples agreeing with the modal answer), pooled over datasets and
cue styles by summing q/(n−2) counts; rows: positive / negative case; left: measured = 1 − robust_used share,
right: measured = share of flips that never recur in the non-truncated re-rolls. Bars: 95 % question-cluster
bootstrap intervals; faded points rest on fewer than 20 flips. Stability uses all 8 no-hint samples; it only bins
the data and never defines an SSP label, flip or case.
