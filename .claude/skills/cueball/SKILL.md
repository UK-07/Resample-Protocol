---
name: cueball
description: Use the CueBall pipeline in this repository — generate chain-of-thought unfaithfulness data for a model × dataset cell (baseline, cued rollouts, LLM judge, re-sampling), reproduce the paper's tables and figures, add a model or a cue style, or train the downstream probes. Use when asked to run, resume, extend or reproduce anything in this repo.
---

# CueBall pipeline

Everything runs from the repo root through one entry point on one config per **cell**
(model × dataset). Read `README.md` for the overview, `configs/pipeline/README.md` for the config
contract, `DATA.md` for the data tree, `src/scripts/visualizations/README.md` for the figures.

## Before anything

1. `uv sync`; copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` (judge stages), `HF_TOKEN`
   (gated datasets and models), `DATA_ROOT` (an absolute path; the paper tree is `${DATA_ROOT}/cueball`).
2. Run scripts as modules: `uv run python -m src.scripts.<name> …` (or `.venv/bin/python -m …` when
   several processes share the venv). Never `cd` into `src/`.
3. `--dry-run` on the entry point prints every command a run would execute and writes nothing. Use
   it first, every time.
4. Never write into a released data tree you are asked to analyse: point `data_dir` (or the
   script's output flags) at a scratch copy. Never hardcode answer letters, reasoning delimiters or
   `/absolute/data/paths`; use the registry, the per-model config and `${DATA_ROOT}`.

## Recipes

**Run one cell end to end**
```bash
cp configs/cueball/run_pipeline_q38b_csqa.yaml configs/cueball/run_pipeline_<model>_<dataset>.yaml
# edit: extends (the model base), compute_baseline.dataset, the explicit output paths / run_name, data_dir
uv run python -m src.scripts.run_pipeline --config configs/cueball/run_pipeline_<model>_<dataset>.yaml --dry-run
uv run python -m src.scripts.run_pipeline --config configs/cueball/run_pipeline_<model>_<dataset>.yaml
```
Stages in order: `baseline, rollouts, judge, manifest, resample_select, rerolls, judge_rerolls,
relabel, summary, figures` (default), then opt-in `splits, probe_dataset, activations, train_probe`.
`--stages a,b` restricts; the GPU stages are 1, 2, 6 (and 13–14); the judge stages 3 and 7 need the
OpenRouter key. Every stage resumes: rerun the same command after an interruption. Tree stages
(4, 5, 8, 9, 10) run over the whole `data_dir` and are safe to rerun after every cell.

**Split GPU and judge work across machines**: `--stages baseline,rollouts` on the GPU box, then
`--stages judge` where the key is.

**Reproduce the paper's tables and figures from the released tree** (no GPU, no key):
```bash
uv run python -m src.scripts.run_pipeline --config configs/cueball/run_pipeline_q38b_csqa.yaml \
    --stages manifest,resample_select,relabel,summary,figures,splits
uv run python -m src.scripts.visualizations.make_all            # the additional plots, or --only <plot>
```
The tree must hold `hinted_rollouts/judge_label_overrides.csv`; the judged CSVs and caches are data
(the hosted judge's labels are not reproducible by re-judging).

**Judge only what is missing**: judge stages are served from `judge_cache/<stem>/`; a rerun buys
only rows with no record (`retry_errors` controls whether cached errors are retried).

**Add a model**: register it in `src/lib/model_utils.py` (`MODEL_CONFIGS`: short name, thinking
mode, reasoning delimiters, layer stack, hidden size, loader kwargs), add `configs/<model>_base.yaml`
(`model_name`, `seed`, `thinking`, `inference`, `layers`), run the smoke config pattern
(`configs/cueball/run_pipeline_gemma412b_smoke.yaml`: `max_samples: 8`) before a full cell, and
check the parser finds the close delimiter in the stored samples (a wrong delimiter silently empties
every CoT).

**Add a cue style**: add a `format_fn(HintInput) -> HintResult` entry to the `HINTS` registry in
`src/lib/hints.py` with a judge-facing `description` and a `hint_text` excerpt, make
`extract_hint_text` recover the excerpt from the stored prompt, and extend `tests/lib/hints_test.py`
(the round-trip test asserts excerpt recovery). Name it in the cell config's `hints:` list.

**Add a dataset**: a loader class in `src/lib/dataset.py` registered in `DATASET_REGISTRY`,
returning the common `(df, options)` schema with a stable `original_index`; option letters come
from the loader, never from a constant.

**Downstream probes**: `splits` → `probe_dataset` (one spec per label configuration, predicates in
`src/lib/selection.py`) → `activations` (GPU, `collect_probe_activations`) → `train_probe`
(`configs/probes/<model>_{tfidf,linear,attention}.yaml`); list them in the config's `stages:`.

## Verify

- `uv run python -m unittest discover -s tests -t . -p '*_test.py' -b` (GPU and network mocked).
- `verify_manifest` (part of the `summary` stage) recomputes the summary from the manifest and
  exits non-zero on any disagreement.
- Compare a regenerated artifact with the released one cell by cell (`rollout_id`-aligned) rather
  than trusting a row count.

## Where things are

`src/lib/` is the single source of truth (hints, parsing, model registry, judge, manifest,
re-sampling, selection, probes); `src/scripts/` one CLI per stage; `configs/cueball/` the paper
grid; `configs/llm_judge_prompts/faithfulness_default.txt` the judge prompt (its hash is part of
the cache identity — changing it orphans every cached verdict).
