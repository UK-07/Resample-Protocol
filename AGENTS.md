# Working with this repository as an agent

CueBall generates and validates chain-of-thought unfaithfulness data: baseline answers → cued
rollouts → LLM-judged verbalization → per-rollout manifest → k = 4 re-sampling with matched
controls → statistics and figures, plus optional white-box probes. Start with `README.md`
(overview and reproduction), `configs/pipeline/README.md` (the config contract),
`DATA.md` (the data tree), `src/scripts/visualizations/README.md` (the figures) and `CLAUDE.md`
(conventions for changing the code). The same recipes are packaged as a Claude Code skill in
`.claude/skills/cueball/SKILL.md`.

## Ground rules

- `uv sync`, then a repo-root `.env` (see `.env.example`): `OPENROUTER_API_KEY`, `HF_TOKEN`,
  `DATA_ROOT`. Secrets never go into a config or the code.
- Run scripts as modules from the repo root: `uv run python -m src.scripts.<name> …`.
- One config = one model × dataset cell; `src/scripts/run_pipeline.py` runs its stages in order
  and resumes. `--dry-run` prints every command and writes nothing: use it before every real run.
- A released data tree is read-only: regenerate into a scratch copy and compare `rollout_id`-aligned.
- Answer letters are per run (A–D, A–E, A–J), reasoning delimiters per model, data paths through
  `${DATA_ROOT}`; never hardcode any of them.
- Behaviour lives in `src/lib/`; scripts are thin CLIs. Keep code paths that read older on-disk
  formats. Row selection for datasets goes through `src/lib/selection.py` only.

## Tasks

| task | how |
| --- | --- |
| run / resume a cell | copy a `configs/cueball/run_pipeline_*.yaml`, set the model base, dataset, output paths and `data_dir`, dry-run, run; `--stages` restricts (GPU: `baseline`, `rollouts`, `rerolls`; judge key: `judge`, `judge_rerolls`) |
| reproduce the paper's numbers and figures | `--stages manifest,resample_select,relabel,summary,figures,splits` on any cell config over the released tree; `python -m src.scripts.visualizations.make_all` for the additional plots |
| add a model | `MODEL_CONFIGS` in `src/lib/model_utils.py` + `configs/<model>_base.yaml`; smoke-test with `max_samples: 8` |
| add a cue style | `HINTS` registry in `src/lib/hints.py` (+ `extract_hint_text`, tests) |
| add a dataset | loader in `src/lib/dataset.py`, registered in `DATASET_REGISTRY` |
| judge with another model or prompt | `judge_rollouts.judge_model` / `judge_prompt_file` in the cell config (a new prompt hash means a new cache) |
| train probes | opt-in stages `splits, probe_dataset, activations, train_probe` |
| check yourself | `uv run python -m unittest discover -s tests -t . -p '*_test.py' -b`; the `summary` stage's `verify_manifest` fails on any disagreement |

## Layout

`src/lib/` (library) · `src/scripts/` (one CLI per stage, `run_pipeline.py` chains them) ·
`src/scripts/visualizations/` (figures) · `configs/` (model bases, the paper grid `cueball/`, the
reference config `pipeline/run_pipeline.yaml`, judge prompts, probe specs) · `tests/` (one test
module per source file).
