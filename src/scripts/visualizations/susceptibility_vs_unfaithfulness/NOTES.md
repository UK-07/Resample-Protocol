# Susceptibility vs unfaithfulness

Do susceptibility and unfaithfulness dissociate? One point per model × cue style (datasets pooled), one panel
per manifest case: x = robust susceptibility, y = the resampled-protocol (RSP) unfaithful rate.

## Run

```bash
python -m src.scripts.visualizations.susceptibility_vs_unfaithfulness.gather   # tables -> <cueball>/plots/susceptibility_vs_unfaithfulness/data/
python -m src.scripts.visualizations.susceptibility_vs_unfaithfulness.plot     # figure -> <cueball>/plots/susceptibility_vs_unfaithfulness/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--only <substr>` (debug subset of the
binary-judged runs; never cached, written under `<plot dir>/_debug` unless `--out-dir` is given), `--no-cache`,
`--refresh-cache`, `--cache-dir` (default `<cueball>/plots/_cache`). `plot` flags: `--data` (default
`<out-dir>/data`), `--out-dir`. The logic lives in `common/rates_common.py` (which imports `common/ssp_common.py`
for the per-run stitching and the SSP flags); their sha256, the repo git sha, the shas of the repo code they
depend on and every input's sha256 are recorded in `data/gather_meta.json`.

## Files

| file | role |
| --- | --- |
| `susceptibility_vs_unfaithfulness.{png,pdf}` | one point per model × cue style (datasets pooled), one panel per manifest case |
| `data/cells_model_style.csv` | the points (every count / rate / CI; keys `subject_model`, `hint_style`, `case`) |
| `data/cells_model_case.csv` | per model × case totals |
| `spearman.csv` | written by `plot` next to the figure: Spearman ρ per case (all points; points with both n ≥ 20) |
| `dropped_points.csv` | written by `plot` next to the figure: cells not drawn because `n_clean == 0` or `rsp_den == 0` |
| `data/exclusions.md`, `join_checks.csv`, `robust_used_rerolls_not_in_dataset_B.csv`, `gather_meta.json` | audit |

## Drawing

- x = robust susceptibility; y = RSP unfaithful rate. Both axes run 0–100 %.
- Colour = model; marker = cue style.
- Qwen3.5-9B markers get a dark edge, because its green is close to Gemma-4-12B's.
- Wilson whiskers are shown on both axes.
- A point with `n_clean < 20` or `rsp_den < 20` (`rsp_den` = dataset_B rows + admitted −1 rows) is faded and
  labelled "n=x/y".
- A point with `n_clean = 0` or `rsp_den = 0` is not drawn; it is listed in `dropped_points.csv`.
- The panel title gives Spearman ρ, both over all points and over n ≥ 20 points only.
- The SSP quantities are not drawn here; they are in the CSV. The two axes use the same protocol (RSP), and the
  y axis uses the v2 judge.

## Inputs (the paper tree only)

- `hinted_rollouts/rollout_manifest.parquet`: one row per hinted-once rollout. Supplies manifest `case`,
  target, groundtruth, `model_answer` (the hinted answer), `truncated`, `to_hint`, `exclude_reason`.
- `binary_judge/<stem>_binary_judged.csv`: the SSP label (`judge_label`: 0 unfaithful, 1 faithful, NaN judge
  error).
- `baselines/<run>_baseline.csv` (named by the collect recipe sidecar): `sample_answers[0]` = b0, re-parsed from
  `sample_rollouts[0]` as a check.
- `resample/question_reliance.csv`: `reliance_label` of each (question, hint) that was re-rolled.
- `resample/resample_manifest.parquet`: the k = 4 re-rolls, with their own v2 verdicts (`judge_label_final`).
  **dataset_B** is taken from it with `src.lib.selection.select_rows("dataset_B", …)` and is not
  re-implemented here.

## Filters

- Models: the five with a binary-judge run (Nemotron-Nano-9B, Qwen3-8B, Qwen3.5-9B, Olmo-3-7B-Think,
  Gemma-4-12B). Qwen3.6-27B is excluded everywhere, including from dataset_B.
- No smoke runs. Only the 8 styles are kept (`src.lib.resample.filter_manifest`).
- Grouping is always by the manifest `case`, for both SSP and RSP. A dataset_B re-roll inherits the case of its
  question × hint. gather checks that every dataset_B `source_rollout_id` is an SSP row of the same
  model / dataset / style / case.

## Definitions (exact)

Per hinted-once rollout, with b0 = sample 0 of the 8 no-hint samples, h = its parsed hinted answer and t = the
cue target:

- **SSP status**. The statuses are mutually exclusive and are tested in this order:
  - `b0_unanswered`: not a flip; counted.
  - `b0_is_target`: cannot flip; counted.
  - `wrong_to_wrong`: a manifest-positive row whose b0 is answered and is neither the groundtruth nor t.
    DROPPED and counted.
  - `eligible`: everything else.

  In the negative case t is the groundtruth, so `wrong_to_wrong` cannot occur there.
- **SSP flip** = eligible ∧ h == t ∧ the hinted rollout is **not truncated** (a truncated rollout is never a
  flip; the eligible truncated rows whose parsed answer is the target are counted as
  `n_truncated_to_target_excluded`). Since b0 ≠ t on eligible rows, h ≠ b0 holds automatically. Every SSP flip is
  therefore a clean row (asserted).
- **SSP rate** = n(flip ∧ binary label 0) / n(flip ∧ binary label ∈ {0, 1}). A binary-judge error (NaN) is not a
  label; it is counted as `n_ssp_binary_missing`.
- **clean** = not `truncated` and `model_answer` parsed. On the manifest this is identical to
  `exclude_reason ∉ {truncated, unanswered, parse_fail}`, which gather checks (`n_clean_vs_exclude_reason_mismatch`).
- **SSP susceptibility** = n(SSP flip) / n(clean ∧ eligible). This is the table's secondary column.
- **robust susceptibility** (the x axis) = n(clean hinted rollouts whose (question, hint) has
  `reliance_label == robust_used`) / n(clean hinted rollouts in the cell), where clean removes truncated,
  unanswered AND parse_fail rows. The cell's hinted rollouts are all its manifest rows, positive or negative as
  grouped. The reliance labels are the stored ones, with no re-censoring.
- **RSP rate** (the y axis) = (label 0 + label −1) / (label 0 + label 1 + label −1), over the **RSP set**: every
  k = 4 re-roll (`provenance == resample_k4`) of a robust_used question that switched to the target (`to_hint`),
  with its OWN v2 verdict (`judge_label_final`) and no other exclusion. **v2-incoherent (−1) counts as
  unfaithful** and is reported separately as `n_rsp_incoherent` in every CSV, table and caption. How the set is
  built:
  - `selection.select_rows("dataset_B", …)` gives the 0/1 rows (`in_dataset_B`, `n_dataset_B`).
  - The −1 rows are added by re-using `selection.mask("dataset_B", …)` unchanged on a copy of exactly the rows with
    `judge_label_final == −1` and `exclude_reason == "incoherent"`. In that copy the verdict is set to 0 and the
    exclusion to null, for the mask evaluation only; the stored −1 is kept.
  - Because `exclude_reason` is the manifest's first applicable reason, `incoherent` means no earlier reason
    (truncated, unanswered, parse_fail) applies. Re-rolls never carry the later `noise_flip`, so the −1 rows meet
    every other dataset_B condition.

  The unit is the re-roll rollout, so one question contributes up to 4 rows. `rsp_rate_excl_incoherent` (the
  plain dataset_B share of 0) is kept in the CSVs for reference only.
- **Nemotron re-rolls** were generated with a 24,576-token budget (every other model and every original rollout
  used 16,384). They are used as generated, with no censoring at 16,384. The number of RSP rows above 16,384 CoT
  tokens is reported as `n_dataset_B_over_budget`.
- **CIs**: Wilson 95 % (`src.lib.resample.wilson_interval`), for the RSP rate too. **Caveat:** the RSP interval
  treats the up to 4 re-rolls of one question as independent, which ignores the per-question clustering, so it
  is somewhat too narrow. `n_rsp_questions` is reported alongside.
- **n < 20**: a value whose own denominator is below 20 is faded in the figure. The denominators are `ssp_den`,
  `rsp_den` (= `n_rsp_rows` = dataset_B rows + admitted −1 rows), `n_clean` and `n_eligible_clean`.
- **Spearman ρ**: rank correlation of `robust_susc` against `rsp_rate` over the drawn points of a panel (all
  points, and points with both denominators ≥ 20), written to `spearman.csv`.

**The two rates differ in judge as well as protocol.** The SSP rate uses the **binary** judge on the one hinted
rollout. The RSP rate uses the **v2 role-based** verbalisation judge (`judge_label_final`) on the re-rolls. An
SSP–RSP difference therefore mixes the protocol change (sample-0 vs resampled reliance) with the judge change.
This figure cannot separate the two.

Per cell, `data/*.csv` also report:
- `n_binary_labeled_rows`: every row of the cell with a binary label in {0, 1}. The binary judge labelled the
  manifest `to_hint` rows, a superset of the SSP flips.
- `ssp_den`: SSP flips with a binary label.
- `n_dataset_B`: dataset_B rows (v2 0/1).
- `n_rsp_incoherent`: admitted −1 rows. `n_rsp_rows` = `n_dataset_B` + `n_rsp_incoherent`.
- `n_to_hint_modal`, `n_robust_used`, `n_truncated_to_target_excluded`, `n_dataset_B_over_budget`.

`baseline_stability` is not used by this figure.

## Exclusions

`data/exclusions.md` holds the per model × case exclusion counts (one row per model × case plus the total) and
the re-rolls of robust_used questions that are not in the RSP set, by reason; `data/gather_meta.json` holds the
same under `exclusions`, next to the join-check totals, the b0 stored-vs-re-parsed mismatch count and the
clean-vs-`exclude_reason` mismatch count (gather aborts when a join check fails).

## Caption

**Robust susceptibility and unfaithfulness dissociate.** Each point is one model (colour) × cue style (marker),
with datasets pooled; left panel positive case, right panel negative case.
- x axis: *robust susceptibility*, the share of hinted rollouts (truncated, unanswered and parse-fail rollouts
  removed) whose question follows the cue in at least 3 of 4 re-rolls.
- y axis: the unfaithful rate among those to-target re-rolls under the v2 role-based judge (RSP).

Whiskers are Wilson 95 % intervals, and faded points have n < 20. ρ is Spearman's rank correlation. Incoherent
v2 verdicts (−1) count as unfaithful; their number is reported separately in the data. RSP intervals ignore the
clustering of up to 4 re-rolls per question. Nemotron-Nano-9B's re-rolls were generated with a 24,576-token
budget (16,384 elsewhere). Qwen3.6-27B is excluded for consistency with the SSP figures.
