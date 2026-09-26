# Alpha-implied noise vs measured noise

α = 1 − q/((n−2)p), clipped into [0, 1]. p = P(switch to the hint), q = P(switch to a non-hint option other than
the baseline), both under the **single-sample protocol** baseline. One point per cell (model × dataset × cue style
× case). x = 1 − α, y = 1 − (share of to-target pairs labeled `robust_used`). Diagonal y = x. Colour = case,
marker shape = model, marker area ∝ √(re-rolled to-target pairs).

## Run

```bash
python -m src.scripts.visualizations.alpha_vs_measured_noise.gather   # tables -> <cueball>/plots/alpha_vs_measured_noise/data/
python -m src.scripts.visualizations.alpha_vs_measured_noise.plot     # figures -> <cueball>/plots/alpha_vs_measured_noise/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--only <substr>[,<substr>]` (debug subset
of the runs whose model or run name contains a substring; written under `<plot dir>/_debug` unless `--out-dir` is
given), and `--no-cache` / `--refresh-cache` / `--cache-dir` (accepted for a uniform CLI; this gather reads its
inputs directly and keeps no cache). The gather is heavy: it reads every run's baseline CSV (`usecols`). `plot`
flags: `--data` (default `<out-dir>/data`), `--out-dir`. The repo git sha, the shas of the code the numbers depend
on (`src/lib/resample.py`, `gather.py`) and every input's sha256 are recorded in `data/gather_meta.json`.

## Files

`data/cells.csv` (one row per cell, plotted), `data/rows.csv` (one row per hinted-once rollout: every cell number
can be recomputed from it by a groupby), `data/gather_meta.json` (inputs, constants, the checks and their
results). Figures: `alpha_vs_measured_noise.{png,pdf}` (left: all cells on equal linear axes, so y = x is the true
diagonal; right: zoom on 1 − α ≤ 10 % with the full y range, since most cells have x ≈ 0),
`alpha_vs_measured_noise_strict.{png,pdf}` (companion, same layout, y = `y_strict`) and
`alpha_vs_measured_noise_by_dataset.{png,pdf}` (the main figure's points split by dataset, with 95 % intervals on
both axes).

## Inputs (the paper tree only; nothing regenerated, re-parsed or re-judged)

| file | used for |
|---|---|
| `hinted_rollouts/rollout_manifest.parquet` (+ `.meta.json`) | one row per hinted-once rollout: keys, case, target, `model_answer` (the hinted answer), `truncated`, `to_hint`/`changed` (modal comparison only); the meta names each run's baseline CSV (asserted inside the paper tree) |
| `baselines/<run>_baseline.csv` | `sample_answers` (JSON list of 8 parsed letters, `""` = unanswered) → sample 0; `choices` → n; `baseline_answer` (cross-check only) |
| `resample/question_reliance.csv` | per re-rolled (question, hint): `role`, `reliance_label`, `k_n`, `k_to_hint_count`, `k_truncated` |
| `resample/resample_manifest.parquet` | the re-rolls: recount of `k_to_hint_count` (check) and the censored variant |
| `resample/noise_model_cells.csv` | cross-check of the modal-baseline x against the repo's noise model |

## Filters

- The five paper models; Qwen3.6-27B is dropped (counted as `manifest_rows_excluded_models`).
- Smoke runs and the dropped styles (`authority`, `few_shot`, `visual_pattern`, `pushback`) are removed with
  `src.lib.resample.filter_manifest`, as the resample scripts do.
- No `exclude_reason` filter. In particular `noise_flip` rows must stay: they are exactly the noise-shaped flips
  the figure is about. `unanswered` / `parse_fail` rows have no parsed hinted answer, so they count as neither a
  switch to the hint nor to another option; `incoherent` is a judge verdict and irrelevant here.

## Definitions (exact)

Per hinted-once rollout (row), with b0 = sample 0 of the no-hint baseline, h = `model_answer`, t = `target_option`:

- `b0_answered` = b0 parsed (non-empty string; a JSON null is unanswered).
- `wrong_to_wrong` = manifest positive ∧ b0 answered ∧ b0 ≠ t ∧ b0 ≠ groundtruth — dropped from the protocol,
  counted as `n_wrong_to_wrong`.
- `eligible` = b0 answered ∧ b0 ≠ t ∧ not wrong→wrong. A rollout whose sample 0 already *is* the target cannot
  "switch to the hint", and its n − 1 non-baseline options contain no target, so it does not fit the α model; it
  is dropped from both p and q (`n_b0_is_target`). Rollouts whose sample 0 is unanswered are dropped too
  (`n_b0_unanswered`).
- A **truncated hinted rollout is never a flip**, neither to the target nor to another option. It stays eligible,
  like a hinted rollout without a parsed answer (the eligible count only enters p and q, and cancels from α).
  Counted as `n_truncated_eligible`, `n_truncated_to_hint_excluded`, `n_truncated_to_other_excluded`.
- `p_to_hint` = eligible ∧ not truncated ∧ h = t (h ≠ b0 follows).
- `q_to_other` = eligible ∧ not truncated ∧ h parsed ∧ h ≠ t ∧ h ≠ b0.
- Cell: p = n_to_hint / n_eligible, q = n_to_other / n_eligible. The denominator cancels in α
  (1 − α = n_to_other / ((n − 2) · n_to_hint)), so whether unanswered hinted rollouts sit in it is immaterial.
- n = the dataset's option count, verified against every baseline's `choices` width (CommonsenseQA 5, MedQA 4,
  GPQA-Extended 4, MMLU-Pro 10 — each run is constant-width; `src.lib.resample.N_OPTIONS_BY_DATASET`).
- `alpha_raw` = 1 − q/((n−2)p); α ≤ 1 always holds (q, p ≥ 0), so the only clipping that can happen is α < 0
  (off-target flips per option exceed to-target flips), set to α = 0 → x = 1, drawn as a **hollow** marker and
  flagged `alpha_clipped`. p = 0 → α undefined; such a cell has no to-target pair either and is not plotted.
- x = `x_alpha_noise` = 1 − α. 95 % interval `x_lo`/`x_hi`: `N_BOOT` draws from independent Jeffreys posteriors
  p ~ Beta(n_to_hint + ½, n_eligible − n_to_hint + ½), q ~ Beta(n_to_other + ½, n_eligible − n_to_other + ½),
  x = min(1, q/((n−2)p)) per draw, 2.5/97.5 percentiles (a plug-in bootstrap gives zero-width intervals whenever
  n_to_other = 0).
- `n_to_other_is_modal`: the q rows whose hinted answer is the modal baseline answer, i.e. a minority sample 0
  "reverting" to the modal answer. They count as switches to another option under the single-sample protocol
  and inflate q (and x) relative to the modal-baseline convention.
- **To-target pairs** = the protocol's `p_to_hint` rows. Each is one (question, hint) whose single rollout switched
  to the target. Joined to `question_reliance.csv` on `rollout_id` (= `<model>:<run>:<hint>:<original_index>`;
  the join is also checked column by column on model, run, original_index, hint_style).
- y = `y_measured_noise` = 1 − n_robust_used / n_pairs_labeled, over to-target pairs with a non-null
  `reliance_label` (`robust_used` = ≥ 3 of the 4 re-rolls to the target). 95 % Wilson interval `y_lo`/`y_hi`.
  Truncated re-rolls count as not-to-target (they have no answer); `n_pairs_with_truncated_reroll` counts the
  affected pairs per cell. `weak_used` (1–2 of 4) and `mixed` (0 of 4) both count as noise; the "0 of 4" share
  alone is `zero_of_4_share`.
- **Strict companion** `y_strict` = n_zero_strict / n_pairs_strict: among labeled pairs with at least one
  non-truncated re-roll (`k_truncated < k_n`), the share whose re-rolls never reach the target
  (`k_to_hint_count == 0`). Wilson interval `y_strict_lo`/`y_strict_hi`.
- `y_measured_noise_censored`: y recounted with re-rolls generated under a budget above `CENSOR_TOKENS` and
  longer than it treated as truncated (`n_robust_used_censored`; the plotted y uses the stored labels).
- Modal-baseline counterparts of x and y (`x_modal`, `y_modal`, `n_*_modal`, the manifest's own `to_hint` /
  `changed` definition) are kept for comparison; `x_modal` is checked against `noise_model_cells.csv`.
- Case = the manifest's `case`, i.e. the cue direction (checked: negative ⇔ target = groundtruth). It is not
  re-derived from sample 0.
- **Fade rule:** every cell with a y is drawn; a cell with n_pairs_labeled < `MIN_PAIRS` = 20 (`n_pairs_strict` for
  the strict figure) is drawn faded with its n printed (`faded` / `faded_strict`).

## Checks gather.py enforces (fails loudly otherwise; results in `gather_meta.json`)

- No duplicate (model, run, original_index, hint_style) in the manifest; no duplicate `original_index` per baseline.
- Every run's baseline is inside the paper tree, has constant choice width = the registered n and exactly 8
  samples per question; every manifest row joins its baseline row.
- Manifest `baseline_modal_answer` = baseline CSV `baseline_answer` on every row (the right baseline file).
- case ⇔ target/groundtruth; protocol to-target ⊆ manifest `to_hint`; every protocol pair found in
  `question_reliance.csv` has role `used_candidate`; the re-roll recount of `k_to_hint_count` from
  `resample_manifest.parquet` equals the stored one on every pair.
- Reported, not enforced: cells with α clipped, cells with p = 0, protocol pairs without a reliance row or label,
  the modal-x vs noise-model max difference.

## Caption (main figure)

Each point is one model × dataset × cue style × case cell. x: the noise share implied by α = 1 − q/((n−2)p)
under a single-sample baseline (sample 0 of the no-hint samples; positive rows whose sample 0 is wrong are
dropped, truncated hinted rollouts never count as switches); y: the share of that cell's single-rollout
switches to the cue's target that do not re-flip to it in ≥ 3 of 4 re-rolls. The y measure counts partially
reliant `weak_used` pairs (1–2 of 4) as noise — the strict companion counts only pairs that never re-flip; it
also counts truncated re-rolls as not following the cue, which raises y for models whose re-rolls are often
truncated. Conversely, x is inflated where sample 0 is a minority answer and the hinted rollout returns to the
modal answer, which the single-sample protocol counts as an off-target switch. Colour: case; marker: model;
marker area ∝ √(re-rolled to-target pairs); faded: fewer than 20 pairs (n printed). Dashed line: y = x.
