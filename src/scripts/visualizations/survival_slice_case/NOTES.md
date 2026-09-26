# Survival slice: positive vs negative case

Survival per model, positive vs negative case (manifest case), cue styles and datasets pooled.

## Run

```bash
python -m src.scripts.visualizations.survival_slice_case.gather   # tables -> <cueball>/plots/survival_slice_case/data/
python -m src.scripts.visualizations.survival_slice_case.plot     # figure -> <cueball>/plots/survival_slice_case/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--only <substr>` (debug subset of the
binary-judged runs; never cached, written under `<plot dir>/_debug` unless `--out-dir` is given), `--no-cache`,
`--refresh-cache`, `--cache-dir` (default `<cueball>/plots/_cache`). `plot` flags: `--data` (default
`<out-dir>/data`), `--out-dir`. The stitching lives in `common/ssp_common.py`; its sha256, the repo git sha,
the shas of the repo code it depends on and every input's sha256 are recorded in `data/gather_meta.json`.

## Files

`data/survival_by_model_slice.csv` (plotted; `slice` = case), `data/rows.csv` (all rows, both cases),
`data/gather_meta.json`. Figure: `survival_slice_case.{png,pdf}`.

## Inputs (the paper tree only; nothing regenerated, re-parsed beyond sample 0, or re-judged)

- `hinted_rollouts/rollout_manifest.parquet`: keys, manifest `case`, `target_option`, `groundtruth`,
  `model_answer` (the hinted answer), `baseline_stability`, `truncated`, `exclude_reason`, `to_hint` (check only).
- `binary_judge/<stem>_binary_judged.csv`: the binary judge's `judge_label` — 0 = unfaithful, 1 = faithful,
  NaN = judge error (not a label).
- `baselines/<run>_baseline.csv` + `.meta.json` (found via the collect recipe sidecar `baseline_csv`; asserted
  inside the paper tree): `sample_rollouts[0]` is **re-parsed** with `src.lib.parsing.parse_answer_from_response`
  (run letters, delimiters and `thinking` from the sidecar, row `choices`). The stored `sample_answers[0]` is kept
  as a check (`b0_stored_vs_reparsed_mismatch_questions` in the join checks).
- `resample/question_reliance.csv`: `role`, `reliance_label`, `k_n`, `k_to_hint_count` per (question, hint),
  joined on `rollout_id`. Stored labels, no re-censoring.

Joins (all keyed, never positional; uniqueness asserted, a multiplying join aborts): manifest ↔ binary CSV via
`source_csv` = `<stem>_judged.csv` ↔ `<stem>_binary_judged.csv` + (`original_index`, `hint_style` = `hint_name`);
manifest ↔ baseline on `original_index` within the run's own baseline; manifest ↔ reliance on `rollout_id`
(and re-checked column by column on model, run, original_index, hint_style, case). The stitched rows are cached
under `<cueball>/plots/_cache/ssp_rows__<key>.parquet`, key = sha256 of (`ssp_common.py`, the repo code deps,
every input's size + mtime, `--only`).

Every join/consistency count is in `gather_meta.json` → `ssp_rows.join_checks` (per run) and `ssp_rows.flag_checks`;
gather **aborts** if any of these is non-zero: missing rows on either side, binary `final_answer` ≠ manifest
`model_answer`, target / groundtruth / case / modal answer / stability disagreements, reliance key mismatches,
baselines without 8 samples, stability ≤ 4 or null; and it aborts if an SSP flip is not a manifest `to_hint` row, is
re-sampled as a control, has a label outside {robust_used, weak_used, mixed}, has `k_n` ≠ 4, or has a label that
disagrees with its `k_to_hint_count` (≥3 robust, 1–2 weak, 0 mixed).

Filters: the five binary-judged models (Nemotron-Nano-9B, Qwen3-8B, Qwen3.5-9B, Olmo-3-7B-Think, Gemma-4-12B;
Qwen3.6-27B has no binary-judge run and is excluded); smoke runs and the dropped styles (`authority`, `few_shot`,
`visual_pattern`, `pushback`) removed as in the resample scripts; the 8 styles kept. The manifest `exclude_reason` is
**not** applied (the SSP rules below are the whole selection).

## Definitions (implemented in `common/ssp_common.py`)

Per hinted-once rollout, b0 = sample 0 of the 8 no-hint samples (re-parsed), h = `model_answer`, t = target:

- **SSP status** (mutually exclusive, first match wins):
  1. `b0_unanswered` — b0 has no parsed answer → not a flip (counted).
  2. `b0_is_target` — b0 == t → cannot flip (counted). In the positive case this also covers b0 wrong-and-equal-to-t.
  3. `wrong_to_wrong` — manifest-positive row with b0 answered and ≠ groundtruth (and ≠ t) → **dropped** (counted).
  4. `eligible` — everything else. (Positive: b0 correct. Negative: b0 answered and ≠ t = groundtruth, i.e. wrong.)
- **SSP flip** = eligible ∧ h == t ∧ the hinted rollout is **not truncated** (h ≠ b0 follows). A truncated hinted
  rollout is never an SSP flip. Eligible truncated rollouts whose parsed answer is the target are excluded and
  counted in `n_truncated_to_target_excluded` (row flag `ssp_flip_truncated`, which despite its name marks an
  *excluded* row); all eligible truncated rollouts are counted in `n_eligible_truncated`.
- **SSP label** = the binary judge's `judge_label` on that rollout. **SSP-unfaithful** = SSP flip ∧ label 0.
  Flips with a NaN label (judge error) are neither unfaithful nor faithful (`n_ssp_flip_binary_missing`).
- **Case** = the manifest `case` (positive: cue points at a wrong option; negative: cue points at the groundtruth).
  The 8-sample majority never defines a flip, label or case.
- **Reliance label** (resampled protocol, question × hint, from the k=4 re-rolls; stored): `robust_used` ≥ 3/4
  re-rolls on target, `weak_used` 1–2/4, `mixed` 0/4 (for a used candidate `mixed` means exactly 0 of 4, checked).
- **Survival** = P(robust_used | SSP-unfaithful): numerator = SSP-unfaithful rows whose question is `robust_used`;
  denominator = SSP-unfaithful rows with a reliance label (rows without one are counted in
  `n_ssp_unfaithful_no_reliance` and left out). The v2 verdict (`judge_label_final`) and `judge_role` play **no** role.
- CIs: Wilson 95 % (`src.lib.resample.wilson_interval`). Bars with denominator < 20 are drawn faded and kept; every
  bar carries its n.

## Table columns

`n_rows` (manifest rows in the group) = `n_b0_unanswered` + `n_b0_is_target` + `n_wrong_to_wrong_dropped` +
`n_eligible`; `n_eligible_truncated` (⊆ eligible), `n_truncated_to_target_excluded` (⊆ eligible truncated);
`n_ssp_flip` (⊆ eligible, disjoint from the truncated rows); `n_ssp_flip_binary_missing`,
`n_ssp_flip_binary_faithful`, `n_ssp_unfaithful` (flip = missing + faithful + unfaithful);
`n_ssp_unfaithful_no_reliance`; `denominator`; `numerator` (robust_used); `n_weak_used`, `n_mixed_0of4`
(numerator + weak + mixed = denominator); `rate`, `ci_lo`, `ci_hi`; `faded_n_lt_20`. The `slice` column repeats
`case` (the slice variable of this plot).

`data/rows.csv` is the row-level audit file: one row per manifest row of both cases with every key, b0
(stored and re-parsed), h, the SSP status/flip/label flags and the joined reliance fields (`rel_role`,
`reliance_label`, `k_n`, `k_to_hint_count`); every table number can be recomputed from it by a groupby. Note that
control rows (never SSP flips) also carry a reliance label (`robust_ignored`/`mixed`) — `rel_role` tells them apart.

## Plot

One panel, x = model (paper order), two bars per model (positive / negative case in the paper's case colours),
Wilson whiskers, n above every bar, n < 20 faded.

## Caption

Survival of single-sample unfaithful labels by case. Per model, the share of single-sample unfaithful rollouts
whose question re-flips to the cued option in at least 3 of 4 re-rolls, for cues pointing at a wrong option
(positive) or at the correct option (negative); cue styles and datasets pooled. Whiskers: Wilson 95 % intervals;
faded bars: n < 20; n above each bar.
