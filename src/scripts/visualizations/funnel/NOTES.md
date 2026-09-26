# Yield funnel and token cost

How many question × style pairs survive each step from "hinted" to a clean, re-sampling-confirmed unfaithful
example, per model, and what that costs in **tokens** (never GPU-hours).

## Run

```bash
python -m src.scripts.visualizations.funnel.gather          # all three stages -> <cueball>/plots/funnel/data/
python -m src.scripts.visualizations.funnel.gather funnel   # just the SSP stitch + stage counts (+ cost tables when
                                                            # tokens/ and judge_keys.parquet already exist)
python -m src.scripts.visualizations.funnel.plot            # figures -> <cueball>/plots/funnel/
```

`gather` stages: `tokens` (heavy: tokenizes every stored generation with the subject model's own tokenizer, loaded
offline from the HF cache; resumable, existing `tokens/*.parquet` are kept unless `--force`; `--workers`, default 3),
`judge` (reads the judge caches), `funnel` (SSP stitch, stage counts, cost tables); `all` (the default) runs the three.
Flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir` (default `<cueball>/plots/funnel`; tables land in
`<out-dir>/data`), `--only <substr>` (debug subset of the binary-judged runs; never cached, written under
`<plot dir>/_debug` unless `--out-dir` is given), `--limit-rows N` (tokens: first N rows per file), `--no-cache`,
`--refresh-cache`, `--cache-dir` (default `<cueball>/plots/_cache`, the stitched SSP rows shared with the survival
figures). `plot` flags: `--data` (default `<out-dir>/data`), `--out-dir`.

## Files

| file | role |
| --- | --- |
| `data/pairs.parquet` | one row per (question, style) pair = hinted-once rollout, with every flag and stage column (audit) |
| `data/funnel_counts.csv` | per model × case group (`all` / `positive` / `negative`): the counts of each stage, the `ssp_status` exclusion counts, `n_truncated_to_target_excluded`, the v2-label split of the clean pairs |
| `data/funnel_dropouts.csv` | why pairs leave the chain at each step, with counts |
| `data/join_checks.csv` | `ssp_common`'s per-run join checks (all 0 except the size columns), plus `reliance_mismatch_vs_resample_manifest` |
| `data/tokens/*.parquet`, `token_stats.jsonl` | row-level token counts per CSV row, plus per-file timing |
| `data/judge_keys.parquet`, `judge_sources.csv` | per judged key: tokens of the served call and of all calls; per judge pass: coverage and the low-completion flag |
| `data/cost_by_model.csv` | per scope (`charged` = what the figure shows; `as_run` = everything generated or judged for these runs) × model × case group × component: output tokens, `input_tokens_reconstructed` (table only), `judge_prompt_tokens_served`, units, `in_figure`, notes |
| `data/cost_summary.csv`, `cost_checks.json` | per-model SSP vs RSP token totals, tokens per clean pair, and the cost join checks |
| `data/gather_meta.json` | the repo git sha, this script's sha256 and the shas of the shared code the numbers depend on (`common/yield_common.py`, `common/ssp_common.py`, `src/lib/{selection,resample,parsing,baseline}.py`), every input's sha256 (manifests, reliance) or size + mtime (generation CSVs), and `ssp_common_build` (the stitch's input fingerprint and flag checks) |

Figures: `funnel.{png,pdf}` (main, cases pooled), `funnel_by_case.{png,pdf}` (companion, per manifest case).

## Inputs (the paper tree only; nothing regenerated, re-parsed beyond sample 0, or re-judged)

Five models (Nemotron-Nano-9B, Qwen3-8B, Qwen3.5-9B, Olmo-3-7B-Think, Gemma-4-12B) × 4 datasets = 20 runs;
Qwen3.6-27B has no binary-judge run and is absent. Smoke runs are skipped.
- **SSP rows** come from `common/ssp_common.py` (`load_rows`), shared with the survival figures so every figure's SSP
  flips are identical: run discovery from `binary_judge/*_binary_judged.csv`, the manifest ↔ binary CSV ↔ baseline ↔
  `question_reliance.csv` joins on keys, the sample-0 re-parse with the repo parser, and all join checks.
- Pairs come from `hinted_rollouts/rollout_manifest.parquet`, filtered to the 5 models, no smoke runs and the 8 styles;
  `judge_label_final` (the v2 label) and `baseline_hint_votes` are joined from it on `rollout_id`.
- `reliance_label` comes from `resample/question_reliance.csv` (`role == used_candidate`, matched on `rollout_id`)
  and is checked against the resample manifest's own `reliance_label` on the original rows.
- `noise_flip_min_votes` is read from `hinted_rollouts/rollout_manifest.meta.json`.

## Definitions (unit = one question × style pair = one hinted-once rollout)

| stage | definition |
| --- | --- |
| hinted pairs | every manifest row, both manifest cases |
| SSP flip (sample 0) | `ssp_status == eligible` ∧ hinted answer (manifest `model_answer`) == target ∧ the rollout is not truncated. This implies hinted answer ≠ b0. b0 = sample 0 of the 8 baseline samples, re-parsed. |
| robust_used | SSP flip ∧ the pair's `reliance_label == robust_used` (≥ 3 of 4 re-rolls on target; stored labels, no re-censoring) |
| binary verdict 0 | robust_used ∧ the **original** rollout's binary-judge label == 0 (NaN is not a label) |
| clean | binary verdict 0 ∧ the original rollout's `exclude_reason` is null **or** `incoherent` (a v2 −1 is not dropped). Every other reason, `noise_flip` included, still drops the pair. The −1 pairs are reported apart (`n_clean_v2_label_-1`). |

`ssp_status` (`ssp_common`) takes the first of these that applies, and the counts are reported per model and case:
1. `b0_unanswered`: not a flip.
2. `b0_is_target`: cannot flip. This also covers positive rows whose b0 is the (wrong) target.
3. `wrong_to_wrong`: a manifest-positive row whose answered b0 is wrong; dropped.
4. `eligible`.

The 8-sample majority never defines a flip. The manifest `case` is used only for grouping (the `positive` /
`negative` panels).

**Truncation.** A truncated hinted rollout is never an SSP flip. Such rollouts are counted as
`n_truncated_to_target_excluded`.

**The −1 admission and the noise-flip rule.** In `EXCLUDE_REASONS`, `incoherent` ranks *before* `noise_flip`, so an
original with v2 −1 that also meets the noise-flip rule carries `exclude_reason == incoherent`. For such rows the
clean stage re-tests the noise-flip rule itself: `to_hint` ∧ `baseline_hint_votes ≥ noise_flip_min_votes` (the manifest
meta's value). Those pairs are dropped and counted as `n_binary0_incoherent_but_noise_flip_dropped`, and in
`funnel_dropouts.csv` as "noise_flip (recorded as incoherent)". Reasons that rank before `incoherent` (truncated,
unanswered, …) are never hidden this way.

**Stability** (`baseline_stability`) plays no role in this figure.

## Cost definitions

The figure charges:
- generation **output** tokens of the baseline samples of the funnel's questions (questions with at least one hinted
  pair; sample 0 and samples 1–7 shown apart), of every hinted rollout, and of the 4 re-rolls of **SSP-flip pairs
  only**;
- **served** judge calls (prompt + completion, one call per label used):
  - the binary judge on the SSP-flip originals;
  - the v2 judge on the to-target re-rolls of the SSP-flip pairs (the RSP labels).

The v2 judge on the originals is not used by the chain (a −1 does not drop a pair). It appears in the tables only
(`in_figure == False`); its `charged` row is restricted to SSP-flip originals, like the binary judge.

The `as_run` scope in `cost_by_model.csv` keeps the full spend: every re-roll (including controls and weak/mixed
pairs), every baseline question, and every judge pass. Nemotron's re-rolls were generated at a **24,576-token budget**
(everything else at 16,384) and are kept as generated, with no censoring.

**Generation tokens** come from the subject model's own tokenizer, loaded offline from the HF cache (`HF_HOME`).
Every text is tokenized with `add_special_tokens=False`. Nothing is estimated (on a sample of hinted rows,
`tokenizer(reasoning)` reproduces the manifest's `trace_token_len` exactly).

- **Output tokens** = tokens of the stored generation. The sources are:
  - baselines: each of the 8 `sample_rollouts` entries;
  - hinted CSVs and re-roll CSVs (`resample/rollouts/*_rs{43..46}.csv`): the `rollout` column.

  Two small undercounts are not included: the end-of-sequence token that vLLM strips, and Nemotron's / Olmo's
  pre-injected `<think>`, which counts as prompt.
- **Input tokens are reconstructed and appear in the tables only (never in the figure).** Each prompt is rendered
  with the tokenizer's chat template and the *current* system prompt for the run's thinking mode
  (`get_thinking_config`, `activation_store.build_messages`). The generation prompt is included; multi-turn hints
  include their extra turns. Runs from before a system-prompt change may have used slightly different text. A
  baseline question's prompt is counted **once per sample** (8 requests; no prefix-cache credit). Template failures
  are counted in `cost_checks.json`.

**Judge tokens** are the `prompt_tokens` / `completion_tokens` (and `cost_usd`) that the API reported, stored on every
record of every judge cache this chain uses. Every successful record carries non-zero usage; records with 0 tokens are
errored calls whose true cost is unknown (this affects only the "all calls" figure, never the served one).

The caches are:
- binary judge on the switched originals: the binary sidecar's `cache_dir` (`<cueball>/judge_cache_binary/<stem>/`);
- v2 judge on the switched originals: the judged sidecar's `cache_dir` where a `_judged.meta.json` exists; runs with
  only a `_judged_summary.json` follow the repo convention `${DATA_ROOT}/judge_cache/<stem>`, with the prompt hash
  from the summary;
- v2 judge on the to-target re-rolls: `${DATA_ROOT}/judge_cache/<stem>_rs<seed>/`.

The judge tokens are reported in two ways:
- **served** (plotted): one call per labelled row. It is the latest non-error record of the key **whose label equals
  the label actually used** (binary CSV `judge_label`; manifest / resample-manifest `judge_label`).
  - `judge_sources.csv` counts the keys whose *latest* record carried a different label (`served_label_mismatch`,
    where duplicate records disagree).
  - It also counts keys with no record carrying the used label at all (`served_no_record_with_used_label`, which then
    fall back to the latest record).
- **Low completion counts (flagged, not imputed).** `completion_tokens_flag` marks caches whose median served
  completion is below 1/4 of the median for the same model and judge pass: the provider usage there appears to
  exclude the judge's reasoning tokens (about 10× lower). Those models' "judge: v2, their re-rolls" bars are therefore
  **understated**. Nothing is imputed.
- **all calls** (in `cost_by_model.csv` notes and `judge_sources.csv`): every record on the paper tree's rows,
  including retries, errored calls and superseded re-judges.

Records on keys outside the paper tree's rows are excluded and counted (`records_outside_cueball_rows`); for example the
other questions of a larger MMLU-Pro cache. Other judges' files in the same directory are ignored and listed
(`other_cache_files_ignored`; an arbiter file from the judge validation is not the pipeline).

**Components** (panel b):

| component | what is charged |
| --- | --- |
| baseline sample 0 | the SSP's one no-hint sample per funnel question |
| baseline samples 1–7 | RSP only: stability and the modal answer |
| hinted rollouts | one per pair (both protocols) |
| k=4 re-rolls | re-rolls of SSP-flip pairs → robust_used |
| judge: binary, SSP flips | the SSP label (stage "binary verdict 0") |
| judge: v2, their re-rolls | the RSP labels of those pairs' to-target re-rolls |

The bar end shows **charged tokens per clean pair**.

## Exclusions, with counts (`funnel_counts.csv`, `funnel_dropouts.csv`, `cost_summary.csv`)

**Counts** (cases pooled):

| model | hinted pairs | b0 unans. / b0 = target / wrong→wrong | truncated → target excl. | SSP flips | robust_used | binary 0 | clean (−1) | noise flip hidden as incoherent | charged tokens per clean pair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nemotron-9B | 28,816 | 408 / 808 / 776 | 0 | 11,165 | 8,350 | 1,151 | 584 (94) | 12 | 0.90 M |
| Qwen3-8B | 29,416 | 456 / 542 / 466 | 0 | 13,910 | 11,872 | 1,023 | 615 (95) | 10 | 1.35 M |
| Qwen3.5-9B | 28,712 | 344 / 350 / 498 | 0 | 17,708 | 15,497 | 74 | 32 (5) | 0 | 54.1 M |
| Olmo-3-7B | 25,728 | 1,064 / 804 / 828 | 4 | 9,434 | 6,417 | 917 | 409 (24) | 4 | 1.72 M |
| Gemma-4-12B | 28,664 | 384 / 461 / 523 | 1 | 13,616 | 11,201 | 645 | 355 (0) | 0 | 3.55 M |

**Per case** (clean, with −1 in brackets):

| model | positive | negative |
| --- | --- | --- |
| Nemotron-9B | 401 (83) | 183 (11) |
| Qwen3-8B | 358 (75) | 257 (20) |
| Qwen3.5-9B | 21 (5) | 11 (0) |
| Olmo-3-7B | 218 (21) | 191 (3) |
| Gemma-4-12B | 215 (0) | 140 (0) |

**Checks:**
- **Cost coverage:** all join checks are 0. Every hinted, re-roll and baseline row has token counts (0 blank rollouts,
  0 template failures, every baseline row with 8 samples). 2,533 of the 20,200 baseline questions have no hinted pair
  and are not charged.
- **Judge caches:** 0 labelled rows lack a cache record. There are 13,589 records outside the paper tree's rows
  (excluded), and 28 served-label mismatches (Gemma MedQA rs46, each resolved to the record carrying the used label).
- **Low-completion caches** (`completion_tokens_flag`; median served completion ≈ 220–295 tokens against
  ≈ 2,300–2,900 for the same model's other re-roll caches): Qwen3.5-9B MMLU-Pro rs43; Olmo-3-7B MMLU-Pro rs43;
  Gemma-4-12B GPQA-Ext. rs43, rs44 and rs45, and MMLU-Pro rs43, rs45 and rs46. The v2 re-roll judge bars of these
  three models are **understated**. Nothing is imputed.

## Caption

*Yield and cost of confirmed unfaithful examples.*
- **(a)** Question × cue-style pairs remaining at each step, per model (log scale; both cases pooled; per-case
  companion in `funnel_by_case`):
  - every hinted pair;
  - single-sample flips: the hinted answer is the cue's target and differs from one no-cue sample, and that sample
    was answered, not the target, and (positive case) correct; truncated rollouts are never flips;
  - pairs whose cue is robustly used (≥ 3 of 4 re-rolls follow it);
  - pairs whose original rollout the binary judge labels unfaithful;
  - pairs that pass every exclusion, including noise flips. Rollouts the second judge calls incoherent are kept and
    counted as unfaithful.
- **(b)** Tokens charged per model:
  - generation output tokens of the baseline samples of these questions, of the hinted rollouts, and of the 4 re-rolls
    of single-sample flips;
  - provider-reported tokens of one judge call per label used.

  The number at the bar end is tokens per clean pair.
- The v2 re-roll judge bars of Qwen3.5-9B, Olmo-3-7B and Gemma-4-12B are understated. For some of their re-roll judge
  caches (Qwen3.5-9B and Olmo MMLU-Pro rs43; Gemma GPQA rs43–45 and MMLU-Pro rs43/45/46), the provider-reported
  completion tokens appear to exclude reasoning, about 10× lower.
- End labels: the legend gives each model's clean count and, in brackets, how many of those the second judge called
  incoherent (−1).
- The single-sample protocol needs only baseline sample 0, the hinted rollouts and the binary judge.
- Nemotron's re-rolls used a 24,576-token budget (others 16,384).

## Decisions

- The main figure pools the cases; a per-case companion is drawn.
- Cost = served judge calls + generation output tokens. Reconstructed inputs appear in the tables only.
- Only the re-rolls of SSP-flip pairs and the baseline samples of funnel questions are charged.
- A v2 −1 is kept at the clean stage and reported apart.
- Nemotron's 24,576-token re-rolls are kept as generated.
- Truncated rollouts are never flips.
