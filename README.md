# CueBall — generating and validating chain-of-thought unfaithfulness data

Code for *CueBall: a statistical framework for generating unfaithfulness evaluation data and
validating the generation and labeling process*. Given a model and a multiple-choice dataset,
the pipeline

1. samples a **baseline** answer distribution per question (repeated sampling, majority vote);
2. injects a **cue** (eight styles: social, artifact and multi-turn) that points at another option
   and generates one hinted reasoning rollout per (question, cue);
3. has an **LLM judge** label every rollout that switched to the cued option: does the reasoning
   verbalize the cue as a reason for the answer (faithful) or not (unfaithful);
4. builds the **per-rollout manifest** and **re-samples** every switched rollout (k = 4 with
   matched controls) to separate cue reliance from sampling noise;
5. reports the statistics (noise model, reliance labels, unfaithfulness rates with confidence
   intervals, cross-model figures) and validates the judge against a second judge family.

A downstream, optional part trains white-box probes (linear, attention, TF-IDF) on residual-stream
activations of the re-rolled rollouts.

## Setup

```bash
uv sync                      # Python ≥ 3.10; installs vLLM, transformers, torch, spaCy, …
cp .env.example .env         # fill in the keys below
```

`.env` (git-ignored, loaded on every `import src.lib`; a variable already set in the shell wins):

| key | used by |
| --- | --- |
| `OPENROUTER_API_KEY` | the LLM judge stages (OpenRouter, OpenAI-compatible API) |
| `HF_TOKEN` | gated datasets (GPQA) and the optional Hugging Face activation store |
| `DATA_ROOT` | root of the data tree every `${DATA_ROOT}/...` config path resolves against (unset = `<repo>/data`) |

Generation stages need a CUDA GPU (vLLM); activation collection and probe training need a GPU and
`transformers`; every other stage is GPU-free. Run every script as a module from the repo root:
`uv run python -m src.scripts.<name> --config <yaml>` and/or flags (`--help` lists them).

## Running the pipeline

One config = one **model × dataset cell**. `src/scripts/run_pipeline.py` runs the stages below in
order, each as a subprocess on a generated flat config (or explicit flags), and resumes whatever is
already done. `configs/pipeline/run_pipeline.yaml` documents every section;
[configs/pipeline/README.md](configs/pipeline/README.md) has the details (paths, resume rules).

```bash
uv run python -m src.scripts.run_pipeline --config configs/pipeline/run_pipeline.yaml --dry-run  # print every command
uv run python -m src.scripts.run_pipeline --config configs/pipeline/run_pipeline.yaml            # run
uv run python -m src.scripts.run_pipeline --config ... --stages baseline,rollouts                 # GPU box only
uv run python -m src.scripts.run_pipeline --config ... --stages judge                             # judge key only
```

| # | stage | script | scope | needs |
| --- | --- | --- | --- | --- |
| 1 | `baseline` | `compute_baseline` | cell | GPU |
| 2 | `rollouts` | `collect_hinted_rollouts` | cell | GPU |
| 3 | `judge` | `judge_rollouts` | cell | judge key |
| 4 | `manifest` | `build_rollout_manifest` | tree | |
| 5 | `resample_select` | `select_resample_set` | tree | |
| 6 | `rerolls` | `resample_hinted_rollouts` | cell | GPU |
| 7 | `judge_rerolls` | `batch_judge_rollouts` | cell | judge key |
| 8 | `relabel` | `resample_noise_model`, `resample_relabel` | tree | |
| 9 | `summary` | `unfaithfulness_metrics_summary`, `verify_manifest` | tree | |
| 10 | `figures` | `paper_plots` | tree | |
| 11 | `splits` | `assign_splits` | tree | opt-in |
| 12 | `probe_dataset` | `build_probe_dataset` | model | opt-in |
| 13 | `activations` | `collect_probe_activations` | model | opt-in, GPU |
| 14 | `train_probe` | `train_probe` | model | opt-in, GPU (TF-IDF: CPU) |

"Cell" stages read and write the cell's own files; "tree" stages run over the whole data tree
(`data_dir`, default `${DATA_ROOT}`) and are safe to rerun after every cell. Stages 1–10 are the
default; 11–14 run only when named in the config's `stages:` list or `--stages`.

The config's top level holds what every stage shares (`extends: ../<model>_base.yaml` brings
`model_name`, `seed`, `thinking`, `inference`), then one section per stage script carrying that
script's own keys. Per-model bases live in `configs/*_base.yaml` (the six paper models: Qwen3-8B, Qwen3.5-9B,
Qwen3.6-27B, Nemotron-Nano-9B-v2, Olmo-3-7B-Think, Gemma-4-12B-it; `src/lib/model_utils.py`
registers a few more); datasets are
`mmlu`, `mmlu_pro`, `gpqa`, `medqa`, `aqua`, `commonsense_qa` (`src/lib/dataset.py`); cue styles are
the `HINTS` registry in `src/lib/hints.py`; judge prompts are in `configs/llm_judge_prompts/`.

## Reproducing the paper

The paper grid (Nemotron-Nano-9B-v2, Qwen3.5-9B, Qwen3-8B, Qwen3.6-27B, Olmo-3-7B-Think,
Gemma-4-12B-it × CommonsenseQA, MedQA, GPQA-Extended, MMLU-Pro-1000) is `configs/cueball/`, one
`run_pipeline_<model>_<dataset>.yaml` per cell under one recipe (8 baseline samples at T 0.7
accepted at ≥ 5 votes, 16,384-token budgets, the 8 cue styles, judge `z-ai/glm-5.3-flash`, k = 4
re-rolls with seeds 43–46). Its data tree is `${DATA_ROOT}/cueball/`; see
[configs/cueball/README.md](configs/cueball/README.md) for the recipe, the tree layout and the notes on
truncation. With the released tree in place (it must include `cueball/hinted_rollouts/judge_label_overrides.csv`,
the researcher-adjudicated label overrides the manifest applies), the tree-level stages rebuild the
manifest, the re-sampling labels, the summary tables and the figures from the released rollouts.
They act on the whole tree, so one cell config is enough; add `splits` (and the opt-in probe stages)
for the downstream part:

```bash
export DATA_ROOT=/path/to/data          # holds cueball/
uv run python -m src.scripts.run_pipeline --config configs/cueball/run_pipeline_q38b_csqa.yaml \
    --stages manifest,resample_select,relabel,summary,figures,splits
```

Rerunning a cell config with its default stages resumes past everything that exists (baseline,
rollouts, judged verdicts, re-rolls) and regenerates only what is missing; the generation stages
need a GPU and the judge stages an OpenRouter key.

The paper's numbers come from `cueball/hinted_rollouts/rollout_manifest.parquet` (one row per hinted
rollout: switch flags, judge label and role, baseline stability, truncation),
`cueball/resample/resample_manifest.parquet` + `question_reliance.csv` (the re-rolls and the
per-(question, cue) reliance label), `cueball/resample/noise_model*.csv`, `survival_*.csv`,
`cueball/hinted_rollouts/unfaithfulness_metrics_summary.md` and `cueball/figures/`. Group by the
manifest's `dataset` column, not by `run`. `src/scripts/judge_validation.py`
(`configs/judge_validation.yaml`) is the judge-reliability study: a stratified sample re-judged by
an arbiter from another model family, agreement / κ per stratum and cue style, the researcher's
review sheet and the label overrides applied to the manifest. `src/scripts/visualizations/judge_validation_plots.py`
draws the paper's judge-validation figures from that run, the binary-judge verdicts under
`cueball/binary_judge/` and the manual golden set (κ per cue style and role against the arbiter, the
binary-vs-role confusion, the researcher's spot-check precision per role, and the headline rates
recomputed under arbiter and reviewer labels) into `cueball/plots/judge_validation/J*.{pdf,png}` + `figures_judge.md`. The whole figure catalogue — F1–F9, J1–J4 and the per-plot `gather` / `plot` modules under `src/scripts/visualizations/<plot>/` with their definitions — is indexed in [src/scripts/visualizations/README.md](src/scripts/visualizations/README.md).

## Reproducibility of the released tree

Checked against the released tree with this code (read-only; everything regenerated into a scratch copy):

| artifact | result |
| --- | --- |
| question sets and prompts of all 24 baselines, the 1,000-question MMLU-Pro draw | identical on every row and column |
| every hinted prompt of every cell (170,224 rows), from the baseline CSV + recipe sidecar | byte-identical |
| judge inputs (row scope + content hash): reruns on 3 cells and 2 re-roll files | 100 % served from cache, output CSVs byte-identical |
| manifest, re-sample selection, reliance labels, noise model, survival tables, summary, 31 figures | byte-identical |
| the nine probe datasets | byte-identical, same fingerprints |

Regenerating one cell end to end (Gemma-4-12B-it × CommonsenseQA, the first 100 questions of the
seeded draw, vLLM + the same judge) reproduces the generated stages up to sampling variance:
prompts identical, majority answers equal on 93 of 100 questions, switch-to-cue rate 0.781 vs
0.777, reliance labels equal on 94 % of the pairs selected in both runs, mean re-rolls on target
3.77 vs 3.72. The unfaithful rates came out higher (7.9 % vs 4.5 % hinted-once, 9.3 % vs 5.8 % on
re-rolls) because the hosted judge changed: re-judging the *released* rollouts of the same
questions fresh (identical inputs, temperature 0, same prompt) gives 9.6 % unfaithful against the
stored 4.5 %, with 93 % label agreement. The judged CSVs and judge caches are therefore part of
the data: a re-judge does not reproduce the paper's labels, and the judge date belongs in the
method section.

## Repository layout

```
src/scripts/                one CLI per stage (run_pipeline.py chains them); each takes --config and/or flags
src/scripts/visualizations/ the figure catalogue (paper_plots.py, judge_validation_plots.py, one folder per plot)
src/lib/                    the shared library — the single source of truth every script imports from
configs/                    per-model bases, the paper grid (cueball/), the reference pipeline config, judge prompts, probe specs
tests/                      stdlib unittest, one <module>_test.py per source file, GPU/network mocked
DATA.md                     the released data tree: layout, provenance, what to ship and what to leave out
AGENTS.md, CLAUDE.md, .claude/skills/cueball/   guidance for agents using or working on the repo
```

Key library modules: `hints.py` (the cue styles and their judge-facing excerpts), `parsing.py`
(answer / reasoning extraction, per-model delimiters), `model_utils.py` (model registry, vLLM and
HF loaders, thinking-mode resolution), `hinted_rollouts.py` (the hinted-rollout row contract and
work items), `llm_judge_verb.py` (the binary LLM judge and its content-addressed cache),
`rollout_manifest.py` (the per-rollout manifest schema and exclusion rules), `resample.py`
(noise model, re-sample selection, reliance labels), `selection.py` (named row predicates and
label configurations — the only place rollouts are selected), `splits.py`, `probe_datasets.py`,
`activation_store.py`, `probe_training.py`, `probes.py`, `report.py`, `probe_metrics.py`.

Conventions: every artifact keeps `original_index` + dataset/split identity as its join key; answer
letters are per run (A–D, A–E or A–J), never a global constant; reasoning delimiters are a
per-model property; all data paths go through `resolve_data_path` (`${DATA_ROOT}/...`).

## Tests

```bash
uv run python -m unittest discover -s tests -t . -p '*_test.py' -b
```
