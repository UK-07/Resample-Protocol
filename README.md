# One Draw Is Not Enough: Resampling-Based Labels for Chain-of-Thought Unfaithfulness

This repository contains **CueBall**, the pipeline used in the paper. Given a model and a
multiple-choice dataset, it

1. samples a **baseline** answer distribution per question (eight draws, accepting an answer
   with at least five votes in the paper configuration);
2. injects a **cue** (eight styles: social, artifact and multi-turn) that points at another option
   and generates one hinted reasoning rollout per (question, cue);
3. has an **LLM judge** check coherence and assign the cue's role in each switched rollout:
   credited, verification_only, rejected, neutral or none;
4. builds the **per-rollout manifest** and draws **four fresh re-rolls** of every switch and
   matched controls to measure target-answer recurrence;
5. reports persistence, judge-assigned unfaithfulness rates and aggregate noise estimates,
   and audits judge agreement with another model family and human adjudication.

The machine-readable persistence labels retain their existing names: `robust_used` means at
least three of four fresh re-rolls answer the target; `weak_used` means one or two do; for switched
pairs with four observed outcomes, `mixed` means none does. Failing the persistence threshold
is not itself evidence of sampling noise, and persistence does not establish that the cue caused
an individual answer or that a trace's claim of non-reliance is false.

The pipeline and existing configuration/data paths retain the name `cueball`. Optional downstream
work trains linear or attention probes on activations and TF-IDF baselines on text.

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

Generation stages need a CUDA GPU (vLLM); activation collection and activation-based probe training
need a GPU and `transformers`. TF-IDF training and the remaining stages are GPU-free. Run every script as a module from the repo root:
`uv run python -m src.scripts.<name> --config <yaml>` and/or flags (`--help` lists them).
Figure modules live under `src.scripts.visualizations.<name>`.

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
script's own keys. Per-model bases live in `configs/*_base.yaml`. The five models in the paper are
Qwen3-8B, Qwen3.5-9B, Nemotron-Nano-9B-v2, Olmo-3-7B-Think and Gemma-4-12B-it; the broader release
also includes Qwen3.6-27B, and `src/lib/model_utils.py` registers additional models. Datasets are
`mmlu`, `mmlu_pro`, `gpqa`, `medqa`, `aqua`, `commonsense_qa` (`src/lib/dataset.py`); cue styles are
the `HINTS` registry in `src/lib/hints.py`; judge prompts are in `configs/llm_judge_prompts/`.

## Reproducing the paper

For the **September 26 revised figures**, use the repository's revision workflow. It recomputes
plotting tables and question-cluster intervals from the numeric bundle, then writes **33 vector
PDFs and three LaTeX tables** to a fresh directory. No separate paper checkout, GPU or judge key
is needed. See [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) for the input checksum, lightweight
pinned environment, output map and numerical validation.

```bash
python -m src.scripts.visualizations.make_all \
    --paper-revision-bundle /path/to/revision_analysis_bundle_2026-09-26.tar.gz \
    --out-root /path/to/fresh-paper-output --dry-run
python -m src.scripts.visualizations.make_all \
    --paper-revision-bundle /path/to/revision_analysis_bundle_2026-09-26.tar.gz \
    --out-root /path/to/fresh-paper-output
```

The following pipeline commands reproduce the underlying released tree and retained legacy
figures. They do not replace the revision workflow above for the new cluster-interval figures.

The paper evaluates the five models above on CommonsenseQA, MedQA, GPQA-Extended and
MMLU-Pro-1000. `configs/cueball/` contains one `run_pipeline_<model>_<dataset>.yaml` per cell,
including four additional Qwen3.6-27B cells in the broader six-model release. Tree-level stages
process the full available tree; generic figure outputs can therefore include that additional
model. Use the five-model selection of the paper's analysis modules for manuscript comparisons.

The recipe uses 8 baseline samples at T 0.7 accepted at ≥ 5 votes, 8 cue styles, judge
`z-ai/glm-5.3-flash`, and 4 fresh re-rolls with seeds 43–46. Baselines use top-p 1.0 without top-k
truncation; cued rollouts and re-rolls use top-p 0.95 / top-k 20. The configured generation budget
is 16,384 tokens; the released Nemotron re-rolls were generated with 24,576. See
[configs/cueball/README.md](configs/cueball/README.md) for the configuration recipe and
[src/scripts/visualizations/DEFINITIONS.md](src/scripts/visualizations/DEFINITIONS.md) for the
analysis populations and exclusion rules.

Keep the released data read-only and reproduce into a **writable scratch copy**. Its layout must
retain `cueball/` and the shared inputs used by the selected stages; see [DATA.md](DATA.md).
Include `cueball/hinted_rollouts/judge_label_overrides.csv`, the researcher-adjudicated overrides
applied by the manifest. One cell config is enough to run the tree-level stages:

```bash
export DATA_ROOT=/path/to/scratch-data  # contains a writable copy of the released tree
uv run python -m src.scripts.run_pipeline --config configs/cueball/run_pipeline_q38b_csqa.yaml \
    --stages manifest,resample_select,relabel,summary,figures,splits --dry-run
uv run python -m src.scripts.run_pipeline --config configs/cueball/run_pipeline_q38b_csqa.yaml \
    --stages manifest,resample_select,relabel,summary,figures,splits
```

Generation and judging stages resume using their completion checks and caches. Derived stages
such as `manifest`, `relabel`, `summary` and `figures` run again when selected and can overwrite
outputs in the scratch tree. Generation needs a GPU; judging needs an OpenRouter key. The
`splits` stage assigns question-level train/validation/test membership, shared by cues and re-rolls.

The paper's numbers come from `cueball/hinted_rollouts/rollout_manifest.parquet` (one row per hinted
rollout: switch flags, judge label and role, baseline stability, truncation),
`cueball/resample/resample_manifest.parquet` + `question_reliance.csv` (the re-rolls and the
per-(question, cue) persistence label, stored as `reliance_label`), `cueball/resample/noise_model*.csv`, `survival_*.csv`,
`cueball/hinted_rollouts/unfaithfulness_metrics_summary.md` and `cueball/figures/`. Group by the
manifest's `dataset` column, not by `run`. `src/scripts/judge_validation.py`
(`configs/judge_validation.yaml`) is the judge-reliability study: a stratified sample re-judged by
an arbiter from another model family, agreement / κ per stratum and cue style, the researcher's
review sheet and the label overrides applied to the manifest. `src/scripts/visualizations/judge_validation_plots.py`
draws the paper's judge-validation figures from that run, the binary-judge verdicts under
`cueball/binary_judge/` and the manual golden set (κ per cue style and role against the arbiter, the
binary-vs-role confusion, the researcher's spot-check precision per role, and the headline rates
recomputed under arbiter and reviewer labels) into `cueball/plots/judge_validation/J*.{pdf,png}` + `figures_judge.md`.

Judge-validation plots default to the five paper models and **all eight cues**, including
`grader_hacking`: `--cues paper` and `--cues all` both select all eight. A comma-separated cue
list restricts the J1a/J1b per-cue and pooled summaries, J4b cue rows, and J2's rollout population;
J1c, J3 and J4a/J4c continue to use all sampled cues. For example:

```bash
uv run python -m src.scripts.visualizations.judge_validation_plots \
    --models paper --cues all
```

The whole figure catalogue — F1–F9, J1–J4 and the per-plot `gather` / `plot` modules under `src/scripts/visualizations/<plot>/` with their definitions — is indexed in [src/scripts/visualizations/README.md](src/scripts/visualizations/README.md).

## Interpretation and current analysis scope

The paper's about-40% non-persistence result concerns **positive-case SSP-unfaithful originals at
temperature 0.7 on baseline-screened questions**. It is not an error rate for temperature-zero
studies. The alpha-implied noise share is compared with non-recurrence and non-persistence as
different quantities; it does not identify individual causal reliance.

SSP uses baseline sample 0 and a binary judge on original flips; RSP uses the role judge on
on-target re-rolls from persistent pairs. The September 26 manuscript revision separates judge,
persistence-filter, trace, population and weighting changes with matched IDs and paired
question-cluster intervals (10,000 replicates, seed 42, questions resampled within dataset).
On 65,752 shared originals, binary and role unfaithfulness are 8.590% and 9.804%. Holding the
judge and pair weights fixed, replacing persistent-pair originals with fresh re-rolls changes
the rate from 8.406% to 8.542%: +0.136 percentage points [−0.049, 0.320]. Full-pool RSP rates
are 9.130% with re-roll weighting and 9.352% with equal-pair weighting.

The revised manuscript's survival, alpha, role, dose-response, rate and yield figures and
benchmark tables use question-cluster intervals. The separate judge audit retains its stated
Wilson spot-check and stratified-bootstrap intervals. The revised figure builders and
question-cluster calculations are maintained here under `src/lib/paper_revision/`, with the thin
`src.scripts.visualizations.paper_revision` CLI and the `make_all --paper-revision-bundle`
entry point; the three judge plots redraw the supplied audit summary. The original data-tree
plotting modes remain available for historical outputs and
retained legacy figures.

Remaining limitations include the baseline/cued decoding mismatch and missing structured
roles on 18,972 originals. Production persistence still treats truncated or unparseable
re-rolls as misses; the manuscript now reports effective-k, complete-four sensitivity and
full-population missing-answer bounds. Positive SSP-unfaithful survival lies between 59.70%
and 65.63% under those bounds. Complete-case estimates describe selected subsets rather than
correcting missingness. `post_hoc` is reported separately as instructed justification:
excluding it lowers positive survival to 54.8% and pooled RSP unfaithfulness to 5.67%.
The primary judge audit covers one model and dataset; its inter-LLM agreement does not measure
human-label accuracy or independently validate incoherence or the separate SSP binary judge.
No new model generations or judge runs were performed for this reanalysis.

The five-model release contains 20,000 unfaithful or incoherent re-rolls (1,223 incoherent)
from 9,997 source pairs and 2,610 canonical questions. The 1,995 strict clean pairs used in the
cost funnel form a different subset. Train/validation/test contain 13,886/2,962/3,152 traces,
6,941/1,501/1,555 source pairs and 1,810/397/403 questions; no question or source pair crosses
splits. These counts were checked by independently aggregating the archived label records.

## Reproducibility of the released tree

The initial release documented these checks on the **six-model tree** (24 cells), with all
regeneration performed in a scratch copy. They are historical reproduction results, not a new
validation of the revised five-model manuscript. Figure-byte equality predates the subsequent
judge-plot correction:

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
configs/                    per-model bases, the release grid (cueball/), the reference pipeline config, judge prompts, probe specs
tests/                      stdlib unittest, one <module>_test.py per source file, GPU/network mocked
DATA.md                     the released data tree: layout, provenance, what to ship and what to leave out
AGENTS.md, CLAUDE.md, .claude/skills/cueball/   guidance for agents using or working on the repo
```

Key library modules: `hints.py` (the cue styles and their judge-facing excerpts), `parsing.py`
(answer / reasoning extraction, per-model delimiters), `model_utils.py` (model registry, vLLM and
HF loaders, thinking-mode resolution), `hinted_rollouts.py` (the hinted-rollout row contract and
work items), `llm_judge_verb.py` (the verbalization/role judge and its content-addressed cache),
`rollout_manifest.py` (the per-rollout manifest schema and exclusion rules), `resample.py`
(noise model, re-sample selection, persistence labels under existing reliance-label keys), `selection.py` (named row predicates and
label configurations — the only place rollouts are selected), `splits.py`, `probe_datasets.py`,
`activation_store.py`, `probe_training.py`, `probes.py`, `report.py`, `probe_metrics.py`.

Conventions: every artifact keeps `original_index` + dataset/split identity as its join key; answer
letters are per run (A–D, A–E or A–J), never a global constant; reasoning delimiters are a
per-model property; all data paths go through `resolve_data_path` (`${DATA_ROOT}/...`).

## Tests

```bash
uv run python -m unittest discover -s tests -t . -p '*_test.py' -b
```
