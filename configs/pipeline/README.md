# `run_pipeline` — the single entry point

`src/scripts/run_pipeline.py` runs the whole pipeline for **one model × one dataset** (a *cell*)
from one config, then the tree-level stages over the data tree the cell writes into.
`configs/pipeline/run_pipeline.yaml` documents every section with the paper's defaults.

```bash
python -m src.scripts.run_pipeline --config configs/pipeline/run_pipeline.yaml --dry-run   # print every command
python -m src.scripts.run_pipeline --config configs/pipeline/run_pipeline.yaml             # run
python -m src.scripts.run_pipeline --config ... --stages baseline,rollouts                  # GPU box
python -m src.scripts.run_pipeline --config ... --stages judge                              # judge key only
```

| # | stage | script(s) | scope | needs |
| --- | --- | --- | --- | --- |
| 1 | `baseline` | `compute_baseline` | cell | GPU |
| 2 | `rollouts` | `collect_hinted_rollouts` | cell | GPU |
| 3 | `judge` | `judge_rollouts` | cell | `OPENROUTER_API_KEY` |
| 4 | `manifest` | `build_rollout_manifest` | tree | |
| 5 | `resample_select` | `select_resample_set --extend` | tree | |
| 6 | `rerolls` | `resample_hinted_rollouts --model <model> --only <cell>` | cell | GPU |
| 7 | `judge_rerolls` | `batch_judge_rollouts` | cell | `OPENROUTER_API_KEY` |
| 8 | `relabel` | `resample_noise_model`, `resample_relabel` | tree | |
| 9 | `summary` | `unfaithfulness_metrics_summary`, `verify_manifest` | tree | |
| 10 | `figures` | `paper_plots` | tree | |
| 11 | `splits` | `assign_splits` | tree | opt-in |
| 12 | `probe_dataset` | `build_probe_dataset` (one spec per predicate) | model | opt-in |
| 13 | `activations` | `collect_probe_activations` | model | opt-in, GPU |
| 14 | `train_probe` | `train_probe` (one run per entry) | model | opt-in, GPU (tfidf: CPU) |

Stages 1–10 are the default; 11–14 (the white-box downstream task) run only when named in the
config's `stages:` list or `--stages`. The order is always the one above.

## Config layout

- **Top level**: `extends` (the per-model base: `model_name`, `seed`, `thinking`, `inference`),
  `run_name`, `data_dir` (default `${DATA_ROOT}`; the paper tree is `${DATA_ROOT}/cueball`),
  `pipeline_dir` (default `<data_dir>/pipeline`), `stages` (null = the default ten).
- **One section per stage script**, named after the script, carrying the script's own config
  keys (`compute_baseline`, `collect_hinted_rollouts`, `judge_rollouts`,
  `resample_hinted_rollouts`, `batch_judge_rollouts`, `collect_probe_activations`) or its CLI
  flags as snake_case keys (`build_rollout_manifest`, `select_resample_set`,
  `resample_noise_model`, `resample_relabel`, `unfaithfulness_metrics_summary`,
  `verify_manifest`, `paper_plots`, `assign_splits`). `build_probe_dataset` carries a
  `predicates` list plus the spec keys; `train_probe` a `runs` list of trainer configs.
  Every section is optional; unknown keys are rejected.
- **Paths**: the cell stem is `<model short>_<dataset tag>[_<run_name>]`. Defaults land under
  `data_dir`: `baselines/<stem>_baseline.csv`, `hinted_rollouts/<stem>_baseline_hinted_rollouts.csv`
  (+ `_judged.csv`, `_judged_summary.json`), `judge_cache/<rollouts stem>/` (stage 3's judge
  cache, overridable with `judge_rollouts.cache_dir`; stage 7's `batch_judge_rollouts` caches
  under the shared `${DATA_ROOT}/judge_cache/<re-roll CSV stem>/` by the script's own rule),
  `hinted_rollouts/rollout_manifest.parquet`, `resample/{items,rollouts}`,
  `resample/resample_manifest.parquet`, `hinted_rollouts/unfaithfulness_metrics_summary.md`,
  `figures/`, `probe_datasets/<model short>_<predicate>.parquet`,
  `probe_activations/<model short>/`, `trained_probes/probes/`. Explicit path keys win, with
  two constraints checked at plan time: stages 4–7 find the cell by the judged CSV's stem
  under the manifest's directory, so `judge_rollouts.output_csv` must stay the default
  whenever a stage after `judge` is requested; and `paper_plots` reads
  `<cueball_dir>/hinted_rollouts/rollout_manifest.parquet` and `<cueball_dir>/resample`, so
  `paper_plots.cueball_dir` must name the tree holding the run's manifest and re-sample dir
  (shown in `--dry-run`, refused at run time).
- **Cue styles**: `collect_hinted_rollouts.hints` defaults to the paper's eight
  (`PAPER_HINT_STYLES`: `unethical_info`, `metadata`, `grader_hacking`, `expert_opinion`,
  `tool_output`, `answer_key_artifact`, `post_hoc`, `consensus`) — the registry styles the
  re-sampling keeps.
- **Re-rolls**: `resample_hinted_rollouts` inherits stage 2's generation budget and, when its
  `top_p` / `top_k` are unset, stage 2's sampling truncation. Its `model_bases` defaults to the
  config's `extends` base; when no base pins the model (e.g. `--model`), a flat base of the
  shared keys is generated at `<pipeline_dir>/generated_configs/<stem>_reroll_base.yaml` when
  the stage runs (`--dry-run` only prints the path).
- **Secrets** (`OPENROUTER_API_KEY`, `HF_TOKEN`) and `DATA_ROOT` come from the `.env` file
  (see `.env.example`) or the environment, never from a config.

Each stage runs as a subprocess on a generated flat config
(`<pipeline_dir>/generated_configs/`) or an explicit argv — `--dry-run` prints every command
without writing anything — so any stage can be re-run by hand. The run record
`<pipeline_dir>/<stem>_baseline_pipeline.json` is rewritten after every stage.

## Resume

Rerun the same command after an interruption: the baseline is skipped when its sidecar is
complete and matches the config (`--force-baseline`), the hinted generation when the run record
marks the recipe finished (`--force-generate`), judge verdicts are cached, re-rolls resume
row-wise per seed, the re-sample selection and the split assignment use `--extend`, an
existing probe dataset is skipped (`force: true` rebuilds it), `manifest` / `relabel` /
`summary` / `figures` always rerun (cheap and deterministic). A failed stage stops the run
(exit 1). The judge slug is preflighted before any GPU work whenever a judge stage is requested.
