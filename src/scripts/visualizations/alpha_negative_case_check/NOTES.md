# Alpha negative-case check

The alpha-implied vs measured-noise scatter restricted to negative-case cells, points by model. In the negative
case the cue targets the correct answer, so noise should favour the target and α (uniform noise over the n − 1
non-baseline options) should underestimate it most; the figure tests that.

## Run

```bash
python -m src.scripts.visualizations.alpha_negative_case_check.gather   # tables -> <cueball>/plots/alpha_negative_case_check/data/
python -m src.scripts.visualizations.alpha_negative_case_check.plot     # figure -> <cueball>/plots/alpha_negative_case_check/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--alpha-data` (the
alpha_vs_measured_noise gather's `data/` directory, default `<cueball>/plots/alpha_vs_measured_noise/data/`; the env
var `ALPHA_DATA_DIR` is the same override), `--only <substr>[,<substr>]` (debug subset of the rows whose model,
run or dataset contains a substring; written under `<plot dir>/_debug` unless `--out-dir` is given; the cell
check then needs an `--alpha-data` written with the same subset), `--no-cache`, `--refresh-cache`, `--cache-dir`
(default `<cueball>/plots/_cache`; only the sample-0 re-parse check reads the stitched-rows cache). `plot` flags:
`--data` (default `<out-dir>/data`), `--out-dir`. Run `gather` after the alpha_vs_measured_noise gather has
written its `data/`.

## Files

`data/cells.csv` (plotted: the negative cells with every count, x + CI, both y + CIs, y − x, faded flags),
`data/summary_by_model_case.csv` (both cases, for the contrast: cells with pairs, median x, median y, median
y − x, cells with y > x / y < x / CI disjoint above, per y definition), `data/exclusions_by_model.csv` (where every
rollout goes under the sample-0 protocol, per model), `data/gather_meta.json` (inputs with sha256, the shared
code's sha, the repo git sha, constants, self-test, checks). Figure: `alpha_negative_case_check.{png,pdf}`.

## Inputs

`alpha_vs_measured_noise/data/rows.csv` (the 5 paper models' hinted-once rollouts with the sample-0 protocol
flags and the re-sampling fields joined) and its `cells.csv`. `gather` re-derives every protocol flag from the raw
columns and asserts equality, and asserts its per-cell counts / x / y equal the alpha gather's `cells.csv`; x's
95 % interval is the alpha gather's (Jeffreys draws, taken from its `cells.csv`). The shared code is
`common/alpha_common.py` (sha256 in `data/gather_meta.json`); `common/ssp_common.py` is used only for the
sample-0 stored-vs-re-parsed check.

## Definitions

Exactly the alpha_vs_measured_noise gather's, restricted to manifest case = negative. Sample-0 protocol:
eligible = b0 answered ∧ b0 ≠ target ∧ not wrong→wrong (manifest positive ∧ b0 answered ∧ b0 ≠ target ∧
b0 ≠ groundtruth — dropped, counted; it only exists in the positive case, so it removes nothing here); a
truncated hinted rollout is never a flip (neither to-target nor third-option; it stays eligible, like an unparsed
hinted answer).

- cell = model × dataset × style (negative case); p = to-target / eligible, q = third-option / eligible;
- x = 1 − α = q/((n−2)p), clipped at 1 (α < 0 drawn hollow at x = 1);
- y_literal = 1 − robust_used share of the cell's to-target pairs; y_strict = share of pairs whose non-truncated
  re-rolls never hit the target (pairs with ≥ 1 non-truncated re-roll). Top / bottom row. Wilson CIs.
- A cell without a to-target flip has no x or y and is dropped from the figure (`has_pairs`; counted in
  `gather_meta.json`). Stability is not used.

## Plot

Layout 2 × 2: rows = literal / strict y, columns = all cells | zoom on 1 − α ≤ 10 % (as the alpha plot; the
dotted box on the left panel marks the zoom). Colour = model (paper palette), marker = dataset, light 95 %
intervals on both axes, dashed y = x. Cells with fewer than 20 pairs are faded with n printed (kept). The left
panels share one x range (the full x range of the negative cells).

## Caption

Negative case only (the cue points at the correct answer). Each point is one model × dataset × cue style cell; x:
the noise share implied by α = 1 − q/((n−2)p) under the sample-0 baseline; y: the noise share re-sampling finds —
top row, 1 − the share of to-target flips that re-flip in ≥ 3 of 4 re-rolls; bottom row, the share that never
re-flip in the non-truncated re-rolls. Left column: all cells; right column: zoom on 1 − α ≤ 10 %. Dashed: y = x.
Faded points rest on fewer than 20 flips. Contrary to the expectation that α underestimates noise most when the
cue points at the truth, the gap is smaller than in the positive case.
