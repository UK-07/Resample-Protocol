# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Research code released with the paper *CueBall*: a pipeline that generates chain-of-thought
unfaithfulness data (baseline answers → cued rollouts → LLM-judged verbalization → per-rollout
manifest → k = 4 re-sampling with matched controls → statistics and figures) and, downstream, trains
white-box probes on the re-rolled rollouts' activations. `README.md` is the reproduction guide;
`configs/pipeline/README.md` documents the single entry point; `configs/cueball/README.md` the paper grid.

## Environment

- Dependencies: `uv sync` (`pyproject.toml` + `uv.lock`). Run scripts as modules from the repo root:
  `uv run python -m src.scripts.<name> --config <yaml>` and/or flags (`--help` lists them).
- Secrets and `DATA_ROOT` come from the git-ignored repo-root `.env` (`.env.example` lists the keys),
  loaded once by `src/lib/env.py` on `import src.lib`; a variable already set in the shell wins.
  Never put a key in a config or in code.
- GPU (vLLM / transformers) is needed by `compute_baseline`, `collect_hinted_rollouts`,
  `resample_hinted_rollouts`, `collect_probe_activations` and the activation probes of `train_probe`;
  the judge stages need `OPENROUTER_API_KEY`; everything else is GPU-free.
- Tests: `uv run python -m unittest discover -s tests -t . -p '*_test.py' -b` — stdlib `unittest`, one
  `tests/<lib|scripts>/<module>_test.py` per source file, GPU and network mocked. No linter or
  formatter is configured. When several processes share the venv, call `.venv/bin/python` directly
  instead of `uv run` (which may re-sync it).

## Layout

- `src/scripts/` — one CLI per stage. `run_pipeline.py` is the **single entry point**: one config =
  one (model, dataset) cell with one section per stage script, stages `baseline, rollouts, judge,
  manifest, resample_select, rerolls, judge_rerolls, relabel, summary, figures` by default and
  `splits, probe_dataset, activations, train_probe` opt-in; every stage runs as a subprocess on a
  generated flat config or explicit flags (`--dry-run` prints them) and resumes.
  `src/scripts/visualizations/` holds the paper's figure modules (`paper_plots.py`,
  `judge_validation_plots.py`, one `<plot>/{gather,plot}.py` folder per figure, `make_all.py`); its
  [README.md](src/scripts/visualizations/README.md) indexes them and `DEFINITIONS.md` fixes the plot definitions.
- `src/lib/` — the shared library; change behaviour here, never by copying logic into a script.
  `hints.py` (cue registry), `parsing.py` (answers, reasoning delimiters per model), `model_utils.py`
  (model registry, loaders, thinking resolution), `hinted_rollouts.py` (row contract, work items),
  `llm_judge_verb.py` (judge + content-addressed cache), `rollout_manifest.py` (manifest schema,
  exclusion reasons), `resample.py`, `selection.py` (the only place rollouts are selected by
  predicate), `splits.py`, `probe_datasets.py`, `activation_store.py`, `probe_training.py`,
  `probes.py`, `report.py`, `probe_metrics.py`, `judge_validation.py`, `string_judge.py`.
- `configs/` — per-model bases (`<model>_base.yaml`: `model_name`, `seed`, `thinking`, `inference`,
  `layers`; a config inherits one with `extends:`, one level deep), the reference pipeline config
  `configs/pipeline/run_pipeline.yaml`, the paper grid under `configs/cueball/` (one config per cell),
  judge prompts under `configs/llm_judge_prompts/`, probe dataset specs and probe-training configs.
- `DATA.md` describes the released data tree; `AGENTS.md` and `.claude/skills/cueball/SKILL.md` are the
  agent-facing guides (task recipes); this file is for working on the code.

## Conventions that must not be broken

- **Provenance**: every artifact carries `original_index` and the dataset/split identity; downstream
  joins are keyed on them, never on row position.
- **Answer letters are per run** (A–D, A–E, A–J): resolve them with `resolve_option_letters` /
  the loader's `possible_options`; never hardcode `OPTION_LETTERS` in a script.
- **Reasoning delimiters are per model** (`MODEL_CONFIGS[...]["delimiters"]`): thread them into every
  parsing call; a wrong delimiter silently empties every CoT. Gemma 4's delimiters are special tokens
  and must be kept at generation time (`sampling_kwargs_for`).
- **Thinking mode** is resolved once by `get_thinking_config`; the activation collector and the
  re-sampler refuse thinking-off runs.
- **Paths** go through `resolve_data_path` (`${DATA_ROOT}/...` or absolute); `DATA_ROOT` defaults
  to `<repo>/data`.
- **Row selection** for datasets happens in `src/lib/selection.py` (named predicates); `case` is
  never a training filter outside the sanctioned `cases=` argument and the report's case views.
- **On-disk compatibility**: the released data tree contains files written by earlier versions
  (sidecars without newer keys, judge-cache records without `input_sha`). Keep the code paths that
  read them.
- The pod's usable RAM is the container cgroup limit, not what `free` shows; size in-memory buffers
  (activation parts, big frames) against it.
- `transformers` is capped `<5.15`: vLLM cannot load Gemma 4 under 5.15's `per_layer_config`.
