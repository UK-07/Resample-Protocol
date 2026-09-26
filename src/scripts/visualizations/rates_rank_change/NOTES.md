# Rates rank change

Do the orderings reshuffle between the single-sample protocol (SSP) and the resampled protocol (RSP)? Bump
charts connect each item's rank by unfaithful rate under SSP (left axis) to its rank under RSP (right axis):
the 8 cue styles ranked within each model (datasets pooled), and the 5 models ranked within each dataset
(styles pooled). Rows are the manifest case.

## Run

```bash
python -m src.scripts.visualizations.rates_rank_change.gather   # tables  -> <cueball>/plots/rates_rank_change/data/
python -m src.scripts.visualizations.rates_rank_change.plot     # figures -> <cueball>/plots/rates_rank_change/
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
| `rates_rank_change_styles.{png,pdf}` | style rankings, SSP (left axis) → RSP (right axis). One panel per model; rows = manifest case. Datasets pooled. |
| `rates_rank_change_models.{png,pdf}` | the 5 models ranked within each dataset. One panel per dataset; rows = case. Styles pooled. |
| `data/cells_model_style.csv`, `data/cells_model_dataset.csv` | the cells (every count / rate / CI) |
| `ranks_styles.csv`, `ranks_models.csv` | written by `plot` next to the figures: the ranks drawn (n ≥ 20 items only) and Kendall τ-b per panel |
| `ranks_styles_left_out.csv`, `ranks_models_left_out.csv` | written by `plot` next to the figures: the items left out (n < 20) |
| `data/exclusions.md`, `join_checks.csv`, `robust_used_rerolls_not_in_dataset_B.csv`, `gather_meta.json` | audit |

## Ranking rules

- Rank 1 is the highest unfaithful rate; rates are rounded nowhere.
- Only items with both denominators (`ssp_den`, `rsp_den`) ≥ 20 are ranked. The rest are listed in
  `ranks_*_left_out.csv` next to the figures.
- Tied rates share the average rank (`pandas.rank(method="average")`), are drawn at the same height and share
  one label ("A=B", wrapped two names per line).
- Every endpoint is labelled with its rate: one decimal below 10 %, two below 1 %.
- τ is Kendall's τ-b (tie-corrected) between the SSP and RSP ranks of a panel.
- Pooling is by row: a style (or dataset) with more flips weighs more, and the weights differ between SSP
  (flips) and RSP (re-rolls).

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
- **robust susceptibility** = n(clean hinted rollouts whose (question, hint) has `reliance_label == robust_used`)
  / n(clean hinted rollouts in the cell), where clean removes truncated, unanswered AND parse_fail rows. The
  cell's hinted rollouts are all its manifest rows, positive or negative as grouped. The reliance labels are the
  stored ones, with no re-censoring.
- **RSP rate** = (label 0 + label −1) / (label 0 + label 1 + label −1), over the **RSP set**: every k = 4 re-roll
  (`provenance == resample_k4`) of a robust_used question that switched to the target (`to_hint`), with its OWN
  v2 verdict (`judge_label_final`) and no other exclusion. **v2-incoherent (−1) counts as unfaithful** and is
  reported separately as `n_rsp_incoherent` in every CSV, table and caption. How the set is built:
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
  is somewhat too narrow. `n_rsp_questions` is reported alongside. The rank charts draw no intervals.
- **n < 20**: an item whose `ssp_den` or `rsp_den` is below 20 is not ranked (see the ranking rules). In the
  cells CSVs the denominators are `ssp_den`, `rsp_den` (= `n_rsp_rows` = dataset_B rows + admitted −1 rows),
  `n_clean` and `n_eligible_clean`.

**The two rates differ in judge as well as protocol.** The SSP rate uses the **binary** judge on the one hinted
rollout. The RSP rate uses the **v2 role-based** verbalisation judge (`judge_label_final`) on the re-rolls. An
SSP–RSP difference therefore mixes the protocol change (sample-0 vs resampled reliance) with the judge change.
These figures cannot separate the two.

Per cell, `data/*.csv` also report:
- `n_binary_labeled_rows`: every row of the cell with a binary label in {0, 1}. The binary judge labelled the
  manifest `to_hint` rows, a superset of the SSP flips.
- `ssp_den`: SSP flips with a binary label.
- `n_dataset_B`: dataset_B rows (v2 0/1).
- `n_rsp_incoherent`: admitted −1 rows. `n_rsp_rows` = `n_dataset_B` + `n_rsp_incoherent`.
- `n_to_hint_modal`, `n_robust_used`, `n_truncated_to_target_excluded`, `n_dataset_B_over_budget`.

`baseline_stability` is not used by any figure of this plot.

## Exclusions

`data/exclusions.md` holds the per model × case exclusion counts (one row per model × case plus the total) and
the re-rolls of robust_used questions that are not in the RSP set, by reason; `data/gather_meta.json` holds the
same under `exclusions`, next to the join-check totals, the b0 stored-vs-re-parsed mismatch count and the
clean-vs-`exclude_reason` mismatch count (gather aborts when a join check fails).

## Caption

**Cue-style rankings change between protocols.** For each model (columns) and case (rows), lines connect a cue
style's rank by unfaithful rate under the single-sample protocol (SSP; left) to its rank under the resampled
protocol (RSP; right). Rank 1 is the highest rate, and datasets are pooled. SSP rates are scored by the binary
judge on single rollouts that switched relative to one no-hint sample. RSP rates are scored by the v2 role-based
judge on to-target re-rolls of questions where at least 3 of 4 re-rolls follow the cue, so ranks can move because
of the protocol and because of the judge. Incoherent v2 verdicts (−1) count as unfaithful; their number is
reported separately in the data. Nemotron-Nano-9B's re-rolls were generated with a 24,576-token budget (16,384
elsewhere). τ is Kendall's rank correlation, tied rates share the average rank, and cue styles with fewer than 20
rollouts on either side are not ranked. *Second figure:* the same for the five models ranked within each dataset
(styles pooled).
