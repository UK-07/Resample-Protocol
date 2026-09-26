# `configs/cueball/` — the paper grid (6 models × 4 datasets)

One `run_pipeline_<model>_<dataset>.yaml` per cell, all under one recipe, all writing into the
paper's own data tree `${DATA_ROOT}/cueball/` (layout in `DATA.md`). Models: Nemotron-Nano-9B-v2,
Qwen3-8B, Qwen3.5-9B, Qwen3.6-27B, Olmo-3-7B-Think, Gemma-4-12B-it. Datasets: CommonsenseQA
(validation, 1,221 questions, A–E), MedQA USMLE (test, 1,273, A–D), GPQA-Extended (546, A–D),
MMLU-Pro (test, 10-option questions, the shared 1,000-question set, A–J).

| stage | recipe |
| --- | --- |
| baseline | 8 samples at T 0.7, accept the top answer at ≥ 5 votes, 16,384 new tokens (24,576 ctx), `top_p 1.0` / `top_k -1` (untruncated sampling, pinned in every config; the repo default is 0.95 / 20) |
| hinted rollouts | `cases: both`, the 8 cue styles, T 0.7, `top_p 0.95` / `top_k 20`, seed 42, 16,384 new tokens |
| re-sampling | k = 4 (seeds 43–46) of every switched rollout + a matched control per cell, T 0.7, 0.95 / 20, 16,384 new tokens |
| judge | `z-ai/glm-5.3-flash`, the built-in verbalization prompt (`configs/llm_judge_prompts/faithfulness_default.txt`), `judge_max_tokens` 16,384 |
| MMLU-Pro | one shared set: `require_n_options: 10` + `max_samples: 1000` + seed 42 (a permutation prefix; `${DATA_ROOT}/cueball/mmlu_pro_1000_questions.csv`) |

## Running a cell

```bash
uv run python -m src.scripts.run_pipeline --config configs/cueball/run_pipeline_q38b_csqa.yaml --dry-run
uv run python -m src.scripts.run_pipeline --config configs/cueball/run_pipeline_q38b_csqa.yaml
```

The config runs `baseline → rollouts → judge` for the cell and then the tree-level stages
(`manifest`, `resample_select`, `rerolls`, `judge_rerolls`, `relabel`, `summary`, `figures`) over
`${DATA_ROOT}/cueball`; every stage resumes, so the same command continues an interrupted run
(`configs/pipeline/README.md` has the stage table and resume rules). `--stages baseline,rollouts`
on a GPU box and `--stages judge` where the OpenRouter key is splits the work. The tree stages act
on the whole tree and can be rerun after every cell. `run_pipeline_gemma412b_smoke.yaml` is the
8-question smoke test to run once on a new machine or model.

Eight cells were generated before the grid was assembled and reused as they are (Nemotron ×4,
Qwen3.5-9B CommonsenseQA / MedQA / GPQA-Extended, Qwen3-8B MedQA): their configs point at the
copied files, whose stems keep the original run tags (`0909`, `0910`, `0911`, `0828`, none) instead
of `cueball`. Stage 1 resumes past their baselines; stage 2 finds every pair generated and loads no
model; stage 3 is served from the judge cache.

Other files: `resample_hinted_rollouts.yaml` (the re-roll recipe the `rerolls` stage generates its
flat config from), `batch_judge_resample.yaml` (judges every `_rs<seed>.csv` of the tree by hand),
`judge_rollouts_roles.yaml` (re-judges a cell into a fresh cache so every verdict carries
`judge_role`).

## Outputs the paper reads

`hinted_rollouts/rollout_manifest.parquet` (one row per hinted rollout: switch flags, judge label
and role, baseline stability, truncation, exclusion reason), `resample/resample_manifest.parquet`
and `question_reliance.csv` (the re-rolls and the per-(question, cue) reliance label),
`resample/noise_model*.csv`, `resample/survival_*.csv`,
`hinted_rollouts/unfaithfulness_metrics_summary.md`, `figures/` and `plots/` (see
`src/scripts/visualizations/README.md`). Group by the manifest's `dataset` column, not by `run`.

## Known deviations from the one recipe

- Nemotron's re-rolls were generated with a 24,576-token budget (the rest at 16,384); 0.03–1.9 %
  of them run past 16,384 CoT tokens and are treated as truncated at analysis time.
- The reused baselines were sampled without a per-request seed; new ones seed vLLM's sampler with 42
  (same distribution).
- 16,384 tokens truncates the long thinkers: Qwen3.6-27B on GPQA-Extended (43 %) and MMLU-Pro
  (29 %), Olmo-3-7B-Think on GPQA-Extended (12 %). Qwen3.6-27B's truncations are mostly cue-induced
  repetition loops, not long deliberation, and the rate is style-dependent: report "did not
  terminate" as its own outcome next to switched / kept, never as missing data.
- The hosted judge's verdicts drifted over time (see the README's reproducibility section): the
  judged CSVs and the judge caches are part of the released data.
