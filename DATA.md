# The released data tree

Every stage reads and writes one directory layout rooted at `${DATA_ROOT}` (from `.env`). The
paper's data is a self-contained copy of that layout under `${DATA_ROOT}/cueball/`, which the grid
configs select with `data_dir: "${DATA_ROOT}/cueball"`.

## Layout

| path | contents | written by |
| --- | --- | --- |
| `baselines/<stem>_baseline.csv` + `.meta.json` | one row per question: prompt, choices, ground truth, the 8 no-hint samples and their majority (`baseline_status`); the sidecar records the recipe and is the resume identity | stage 1 |
| `hinted_rollouts/<stem>_baseline_hinted_rollouts.csv` + `.meta.json` | one row per (question, cue, case): hinted prompt, rollout, extracted reasoning, parsed answer; the sidecar is the generation recipe | stage 2 |
| `hinted_rollouts/<…>_judged.csv`, `_judged_summary.json`, `_judged.meta.json` | the same rows plus `judge_*` columns for the rollouts that switched to the cue | stage 3 |
| `hinted_rollouts/rollout_manifest.parquet` + `.meta.json` | one row per hinted rollout: keys, switch flags, judge label and role, baseline stability, truncation, exclusion reason, split | stage 4 |
| `hinted_rollouts/judge_label_overrides.csv` | the researcher-adjudicated label overrides the manifest applies (`judge_label_final`) | judge validation |
| `hinted_rollouts/unfaithfulness_metrics_summary.md` | per-cell counts and rates; `verify_manifest` recomputes it from the manifest | stage 9 |
| `resample/resample_selection.csv` + `.meta.json`, `resample_selection_report.csv` | every switched rollout (`used_candidate`) and its matched control per cell | stage 5 |
| `resample/items/<stem>_resample_items.csv` + `.meta.json` | the selected rows' prompt columns, what the re-roll stage consumes | stage 5 |
| `resample/rollouts/<stem>_rs<seed>.csv` (+ `_judged.csv`, sidecars, summaries) | the k = 4 re-rolls, one file per sampling seed | stages 6–7 |
| `resample/resample_manifest.parquet`, `question_reliance.csv`, `noise_model*.csv`, `survival_*.csv`, `*.md` | re-roll rows with reliance labels, per-(question, cue) counts, the noise model and survival tables | stage 8 |
| `judge_cache/<stem>/<judge>_binary_<prompt hash>.jsonl` | content-addressed verdicts; a rerun of a judge stage is served from here | stages 3, 7 |
| `pipeline/<stem>_pipeline.json`, `pipeline/generated_configs/` | run records and the flat per-stage configs the entry point wrote | entry point |
| `figures/` | the paper figures F1–F9 (+ `figures.md`) | stage 10 |
| `plots/<plot>/{data/, *.png, *.pdf}`, `plots/judge_validation/` | the additional figures of `src/scripts/visualizations/` (tables + figures) and J1–J4 | visualizations |
| `binary_judge/<stem>_binary_judged.csv`, `judge_cache_binary/` | a second, single-question judging of every switched rollout (binary prompt), an input of the single-sample-protocol plots | separate judge run |
| `mmlu_pro_1000_questions.csv` + `.meta.json` | the shared 1,000-question MMLU-Pro set (seeded permutation prefix of the 10-option test split) | `make_cueball_subsample` |
| `sources.json` | provenance of the cells copied in from earlier runs (source path, size, mtime per file) | `assemble_cueball_tree` |
| `probe_datasets/`, `probe_activations/`, `trained_probes/` | the opt-in downstream part: probe datasets per label configuration, activation stores, training runs | stages 12–14 |

Cell stems are `<model short name>_<dataset tag>_<run tag>` (for example
`qwen3-8b_commonsense_qa-validation_cueball`); cells generated before the grid keep their original
run tags (`0909`, `0910`, `0911`, `0828`, or none). Group results by the manifest's `dataset` and
`subject_model` columns, never by the run tag.

## The paper grid

Six models (Nemotron-Nano-9B-v2, Qwen3-8B, Qwen3.5-9B, Qwen3.6-27B, Olmo-3-7B-Think,
Gemma-4-12B-it) × four datasets (CommonsenseQA validation, MedQA test, GPQA-Extended, the MMLU-Pro
1,000-question set), one recipe: 8 baseline samples at T 0.7 accepted at ≥ 5 votes, untruncated
baseline sampling, 16,384-token budgets, the 8 cue styles at T 0.7 / top-p 0.95 / top-k 20 / seed
42, judge `z-ai/glm-5.3-flash` with the default verbalization prompt, k = 4 re-rolls with seeds
43–46. `configs/cueball/README.md` has the per-cell configs and the known deviations (Nemotron's
re-rolls used a 24,576-token budget; the long thinkers truncate at 16,384 tokens, reported per cell).

## Provenance and what a release must contain

- Every generated file has a sidecar with its recipe; the manifest's sidecar lists its sources with
  sizes and mtimes; the judge cache records are content-addressed (`input_sha`).
- The judged CSVs and the judge caches are **data**, not derivable: the hosted judge's verdicts on
  identical inputs changed over time (see the README's reproducibility section). Ship them.
- `hinted_rollouts/judge_label_overrides.csv` must be present: the released manifest applied 16
  researcher overrides from it, and the builder refuses to rebuild without the file once a manifest
  recorded applied overrides.
- The judge-validation study (`configs/judge_validation.yaml`) lives in its own directory
  `${DATA_ROOT}/judge_validation/<run>/` with the arbiter's cache under `${DATA_ROOT}/judge_cache/`
  and the manual golden set under `${DATA_ROOT}/manual_golden_dataset_judge/`; ship them with the
  tree if the J-figures are to be regenerated.

## What to leave out when packaging

Working-state directories a run leaves behind, none of which any stage reads: `_backup_*/`, `tmp/`,
`pods/`, `*.log`, `plots/.pod_logs/`, `plots/_lib/` and `plots/<plot>/v*/{gather,plot}.py`
(that code now lives in `src/scripts/visualizations/`; keep the `data/` tables and figures),
`plots/*/scratch/`, superseded plot versions.

## Assembling a tree from earlier runs

`src/scripts/make_cueball_subsample.py` derives the shared MMLU-Pro question set and restricts one
run's artifacts to it; `src/scripts/assemble_cueball_tree.py` copies full-split runs generated before
the grid into the tree (sidecars re-pointed, selection rows merged, `sources.json` written). Both
read `${DATA_ROOT}` and write `${DATA_ROOT}/cueball`; the run lists at the top of each script name
the cells the paper reused.
