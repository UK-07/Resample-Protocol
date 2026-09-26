# SSP vs RSP false rejections

Of the rollouts the single-sample protocol (SSP) would call *false rejections* — the CoT rejects the cue, uses it
only to verify, or reasons to a different option than the one it gives, yet the single draw switched to the cue's
target — what share does the resampled protocol (RSP) confirm as genuine reliance (`robust_used`)? The
complement is what SSP overstates.

## Run

```bash
python -m src.scripts.visualizations.ssp_vs_rsp_false_rejections.gather   # tables -> <cueball>/plots/ssp_vs_rsp_false_rejections/data/
python -m src.scripts.visualizations.ssp_vs_rsp_false_rejections.plot     # figures -> <cueball>/plots/ssp_vs_rsp_false_rejections/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--only <substr>` (debug subset of the
binary-judged runs; never cached, written under `<plot dir>/_debug` unless `--out-dir` is given), `--no-cache`,
`--refresh-cache`, `--cache-dir` (default `<cueball>/plots/_cache`), `--minus1 own_category|exclude` (`exclude` is
the superseded sensitivity rule — −1 originals excluded — and belongs in a separate `--out-dir`). `plot` flags:
`--data` (default `<out-dir>/data`), `--out-dir`. The SSP stitching lives in `common/ssp_common.py`, the role /
category logic in `common/section3_claims.py`; their sha256, the repo git sha, the shas of the repo code they
depend on and the inputs' sha256 are recorded in `data/gather_meta.json`.

## Files

| file | role |
| --- | --- |
| `gather.py` | stitches the SSP rows (`common/ssp_common.py`) with the v2 roles of the original rollouts; writes `data/` |
| `plot.py` | reads only `data/`; writes `ssp_vs_rsp_false_rejections.{png,pdf}` (one bar per model), `_by_style` (style × model heatmap, the primary slice), `_by_case`, `_by_stability` |
| `data/counts_by_<slice>.csv` | COUNTS for slices `model`, `model_style`, `model_case`, `model_stability`, `model_claim_role` (category rejected / verification_only / incoherent): truncated to-target rows (not flips), SSP flips, exclusions (no role / unjudged), categories of the kept flips (incl. `n_cat_incoherent`), false rejections, of which robust_used / weak_used / mixed (0/4), −1 flips that carried `rejected` / `verification_only` |
| `data/rates_by_<slice>.csv` | the same rows + `rate` = `n_fr_robust_used / n_fr_with_reliance`, Wilson `ci_lo`/`ci_hi`, `faded_n_lt_20`, and the reference `ref_all_flips_rate` (robust_used share over ALL kept SSP flips; NaN on the claim-role slice) |
| `data/counts_then_rates.md` | every counts table first, then every rates table, then the −1 × role table for SSP flips (markdown; also printed to stdout in that order) |
| `data/minus1_by_role_ssp_flips_by_model.csv` | SSP flips with v2 −1, by the role the judge also emitted |
| `data/ssp_funnel_by_model_case.csv` | the sample-0 funnel from ssp_common (b0 unanswered / b0 = target / wrong→wrong dropped / eligible / flips) |
| `data/gather_meta.json` | provenance (code shas, git sha, inputs), the ssp_common row-cache meta, checks, `minus1_rule` |

## Definitions

- **SSP rows / flips** come unchanged from `common/ssp_common.py` (`load_rows`), the single-sample rules:
  b0 = sample 0 of the 8 no-hint samples, re-parsed with the repo parser; hinted answer = manifest
  `model_answer`; per row, in order: b0 unanswered → not a flip; b0 == target → cannot flip; manifest-positive row
  with b0 answered and ≠ groundtruth (wrong→wrong) → dropped; otherwise eligible. **SSP flip** = eligible ∧
  hinted answer == target ∧ the hinted rollout is not truncated (truncated to-target rows are counted,
  `n_truncated_to_target_excluded`). Grouping by manifest `case`. The 8-sample majority never defines a flip.
  Every SSP flip is a manifest `to_hint` row re-sampled as a used_candidate (checked; `checks` in the meta).
- **SSP false rejection** = SSP flip whose ORIGINAL (hinted-once) rollout makes a non-reliance claim: v2
  `judge_role` ∈ {rejected, verification_only}, **or v2 verdict −1** (category incoherent, whatever its role) —
  from `hinted_rollouts/rollout_manifest.parquet`, joined on `rollout_id`, one-to-one.
- **Excluded** (counted in every counts table and in the figure footers): SSP flips whose original rollout has no
  v2 verdict, or a 0/1 verdict without a role.
- **Confirmed** = the pair's `reliance_label` (`resample/question_reliance.csv`) == `robust_used` (≥ 3/4
  re-rolls on target). **Rate** = confirmed / false rejections with a reliance label (every flip has one:
  k_n = 4). Wilson 95 %.
- **Reference tick** (grey, main figure only): robust_used share over all kept SSP flips, any role — whether
  false rejections are more or less robust than an average flip.
- **Slices**: style × model (primary; heatmap, cell = share and k/n), manifest case, stability bin, and (table
  only) claim category rejected / verification_only / incoherent. Bars/cells with n < 20 faded; n printed.
- **Stability axis** (`_by_stability` and the `model_stability` tables): the manifest's `baseline_stability`,
  the number of the 8 no-hint samples that agree with the modal answer; it only bins the data and never defines an
  SSP label, flip or case. The tree has no rows below 5/8; the lowest bin is labelled ≤5/8.
- `exclude_reason` is not applied.

## Population caveat

Missing v2 roles (originals judged before the role-emitting prompt) remove a large share of some models' SSP
flips; their bars describe the covered subset, which need not be representative. The figure footers print the
coverage per model (`n_kept / n_ssp_flips` in `data/counts_by_model.csv`). `rejected` alone is rare per model, so
the per-style cells are mostly verification_only + −1 and many are n < 20; the counts table decides whether a
claim is per style or pooled.

## Caption

How many single-sample "false rejections" are real? A false rejection under the single-sample protocol (SSP) is a
rollout that switched to the cue's target relative to one no-hint sample while its reasoning rejects the cue or
uses it only to verify an answer, or reasons to a different option than the one it gives (v2 judge role / verdict
incoherent). Bars show the share of these rollouts whose (question, cue) pair is confirmed as robustly cue-driven
by re-sampling (≥ 3 of 4 re-rolls of the same prompt follow the cue); the grey tick is the same share over all
SSP flips. Whiskers: Wilson 95 %; labels k/n; faded bars n < 20. Flips whose original was judged without a role
are excluded (footer). [Stability panel: stability uses all 8 no-hint samples; it only bins the data and never
defines an SSP label, flip or case.]
