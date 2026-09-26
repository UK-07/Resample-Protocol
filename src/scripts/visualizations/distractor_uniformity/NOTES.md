# Distractor uniformity

α = 1 − q/((n−2)p) assumes that off-target noise spreads **uniformly over the n − 2 distractors** (the options other
than the target and the baseline answer). Three tests of that assumption over the third-option switches of the
single-sample protocol, plus a worked example.

## Run (after `alpha_vs_measured_noise.gather` has written its `data/`)

```bash
python -m src.scripts.visualizations.distractor_uniformity.gather   # tables -> <cueball>/plots/distractor_uniformity/data/
python -m src.scripts.visualizations.distractor_uniformity.plot     # figures -> <cueball>/plots/distractor_uniformity/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--alpha-data` (the
alpha_vs_measured_noise gather's `data/`; default `<cueball>/plots/alpha_vs_measured_noise/data`, or
`$ALPHA_DATA_DIR`), `--only <a,b>` (debug subset by model / run / dataset substring; needs an alpha data dir built
with the same subset; never cached, written under `<plot dir>/_debug` unless `--out-dir` is given), `--no-cache`,
`--refresh-cache`, `--cache-dir` (default `<cueball>/plots/_cache`; the stitched-rows cache backs only the sample-0
re-parse check). `plot` flags: `--data` (default `<out-dir>/data`), `--out-dir`.

## Inputs

Only the alpha_vs_measured_noise gather's `rows.csv` (one row per hinted-once rollout of the 5 models, with the
protocol flags) and its `cells.csv`. Shared code: `common/alpha_common.py`; its sha256, the repo git sha, the shas
of the repo code it depends on and the inputs' sha256 go into `data/gather_meta.json`. `gather` checks the data
against the alpha gather:
- it re-derives every protocol flag from the raw columns and asserts equality;
- it asserts that its per-cell counts equal the alpha gather's `cells.csv`;
- it asserts the stored sample-0 letter equals the repo parser's re-parse on every source (`common/ssp_common`).

## Definitions

- **Scope.** The 5 paper models and the 8 kept styles. Case = manifest `case`.
- **b0** = sample 0 of the 8 no-hint samples. **h** = the hinted-once `model_answer`. **t** = the target.
- **Eligible** = b0 answered ∧ b0 ≠ t ∧ not wrong→wrong.
  - wrong→wrong = manifest positive ∧ b0 answered ∧ b0 ≠ t ∧ b0 ≠ groundtruth. These rows are dropped and counted.
- **Third-option switch** = eligible ∧ hinted rollout not truncated ∧ h answered ∧ h ≠ t ∧ h ≠ b0. This is the q
  numerator of α.
- **Distractors** = the run's n letters minus {t, b0}, so n − 2 of them. h is one of them.
- **Null hypothesis:** every switch picks uniformly among its own n − 2 distractors.
- **Letter test** (figure 1). One test per cell (model × dataset × style × case, both cases).
  - Categories are the option letters. Rows have different distractor sets, so it is a Wald test:
    W = (O − E)ᵀ Σ⁺ (O − E), with E = Σ_r π_r, Σ = Σ_r (diag π_r − π_r π_rᵀ), df = rank Σ.
  - It reduces exactly to Pearson with df n − 3 when every row shares one set (unit-checked).
  - Qualifies with ≥ 5·(n − 2) third-option switches (`qualifies`).
- **Position test** (figure 2). Same cells and qualifying rule.
  - Categories are the rank of h among the row's distractors in letter order. Pearson, df n − 3.
- **Majority-answer test** (figure 3), **pooled over models**.
  - **Pooled cell** = dataset × style × case.
  - **Population** = the switches whose 8-sample majority (modal) no-hint answer is one of the row's distractors,
    i.e. sample 0 was a minority answer.
  - The modal answer only names a category. It never defines a flip, a label or a case.
  - Under the null, P(h = modal) = 1/(n − 2) for every row of a dataset, so rows from different models pool into one
    binomial.
  - The test is Pearson on {modal, other distractors}, df 1. The exact upper binomial tail is stored too.
  - **Negative cells only.** Once wrong→wrong rows are dropped, the positive population is empty by construction
    (`gather` asserts it). In a positive cell the modal answer is the correct one. It is a distractor only when b0
    is a wrong minority answer, and such a row is wrong→wrong.
  - **Qualifies** with ≥ 5·(n − 2) switches in the pooled population. This guarantees an expected count ≥ 5
    (asserted).
  - Pooled cells with a smaller, non-empty population are **not tested** and are counted.
  - Per-model population and hit counts are kept as `pop__<model>` / `hits__<model>` columns.
- **Rejection:** p < 0.05 per qualifying cell. Benjamini–Hochberg is applied per test, over that test's qualifying
  cells.
- **Chi-square survival function** (self-implemented in `common/alpha_common.py`): the regularised upper incomplete
  gamma, computed by a series or a Lentz continued fraction. Its unit checks run on every gather and are recorded
  in `gather_meta.json` (`selftest`): textbook critical values, the 95 % quantiles for df 1 / 5 / 9, closed forms
  for df 1–5 and 8, Wald = Pearson for a shared support, Pearson with explicit expected counts, the binomial tail,
  BH, and the Monte Carlo size under H0 — overall and in the regimes where cells actually qualify
  (`selftest.mc.regimes`: n = 4 with 10 / 20 switches, n = 5 with 15 / 25). **The tests are conservative near the
  qualifying threshold** (≈ 2 % at n = 4 with 10 switches), so the letter / position result has limited power:
  most qualifying cells sit just above the threshold.
- **Worked example:** the **qualifying** pooled cell with the most switches in the majority population; ties are
  broken by dataset, then style. The MMLU-Pro pooled cell with the most population switches is kept in
  `worked_example.json` as `context_mmlu_pro_cell`; it is context only and is not drawn.
- **Fade rule:** points with n < 20 are drawn faded with n printed. n is the population size for the majority test
  and the third-option count for letter / position.
- No literal / strict y pair: there is no re-sampling quantity.

## Exclusions (full grid, 5 models)

- 141,336 hinted-once rollouts. The Qwen3.6-27B rows are dropped upstream, in the alpha gather.
- Not eligible: 2,656 with b0 unanswered; 2,965 with b0 = target; **3,091 wrong→wrong (dropped)**.
- Of the 132,624 eligible rollouts: 10,042 have no parsed hinted answer; **8,202 are truncated and never switches**
  (5 of them would have been to-target flips, 0 third-option); 65,833 went to the target; **2,054 are third-option
  switches**. They spread over 205 of the 320 per-model cells; 155 of those cells have fewer than 5·(n−2) and are
  not tested by letter / position.
- **Majority population:** 226 switches, all negative-case. It fills 27 of the 32 pooled negative cells; 20 of those
  are below the rule and not tested (114 switches).

## Files

- `distractor_uniformity_letter.{png,pdf}`: per-model strips of the letter-test p-values (log y), qualifying cells
  only. Colour = case, marker = dataset.
- `distractor_uniformity_position.{png,pdf}`: the same for the position test.
- `distractor_uniformity_majority.{png,pdf}`: per-dataset strips of the pooled majority-test p-values, qualifying
  negative cells only, with style abbreviations. The number of not-tested cells is printed per dataset, and a
  footnote says the positive case is untestable.
- `distractor_uniformity_example.{png,pdf}`: the worked example. Observed vs uniform-expected switches, majority
  answer vs other distractors, with the Pearson p (and its BH-adjusted value) and the exact binomial p.
- All strips: dashed line at p = 0.05, and "rejected / qualifying" per strip.
- `data/`:
  - `cells.csv`: per-model cells — counts, `qualifies`, the letter / position tests with BH and reject flags, and the
    cell's majority population and hit counts;
  - `majority_pooled.csv`: pooled cells — population, hits, expected, both p-values, qualifies, not_tested, BH, reject
    flags, and per-model columns;
  - `distractor_counts.csv`: long table, per-model cell × letter / position category, observed vs expected;
  - `summary.csv`: per model × test, and per dataset for the majority test;
  - `exclusions_by_model.csv`, `worked_example.json`, `gather_meta.json`.

## Caption

Does off-target noise spread uniformly over the distractors, as α assumes? We take the third-option switches under
the single-sample protocol: hinted rollouts that answer neither the cue's target nor the sample-0 no-hint answer.
(a, b) One point per model × dataset × cue style × case cell with at least 5·(n−2) such switches. y is the p-value
of a test that the switches are uniform over the n−2 remaining options, (a) by option letter and (b) by rank among
the remaining options. Uniformity is rarely rejected (2 and 1 of 50 cells at p < 0.05, none after Benjamini–Hochberg).
(c) Negative-case dataset × style cells, pooled over models, restricted to switches whose majority no-hint answer
(over 8 samples) is one of the remaining options, i.e. the baseline draw was not the model's preferred answer. y is
the p-value that such switches land on the majority answer at the uniform rate 1/(n−2). All 7 qualifying cells
reject: switches return to the model's preferred answer (209 of 226 switches overall, against 88 expected). Six of
the seven rest on 10–19 switches and are drawn faded; only MedQA / metadata (25 switches) is at full strength. The
MMLU-Pro cells depart even more strongly (exact binomial p down to 4e-13) but hold at most 18 switches, below that
dataset's 40-switch rule, and are not tested. Near the rule the tests are conservative (simulated size 2–5 %), so
(a, b) have limited power to detect small departures. α's uniformity assumption therefore holds in general but
fails exactly when the baseline draw was not the modal answer, which sample 0 alone cannot detect. (d) Worked
example: MedQA / metadata / negative, models pooled. 24 of 25 such switches return to the majority answer, where
uniformity predicts 12.5. The positive case cannot be tested: there the majority answer is the correct one, and a
sample 0 that is a wrong minority answer is a wrong→wrong row, which the protocol drops. Dashed line: p = 0.05;
faded points rest on fewer than 20 switches.
