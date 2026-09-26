#!/usr/bin/env python3
"""Run the pipeline for one model × one dataset from a single config.

Fixed stage order (a stage is a subprocess on a generated flat config or an argv,
so every one can be re-run by hand and vLLM memory is released between loads):

  cell (this model × dataset)         tree (the data tree the cell writes into)
   1 baseline        compute_baseline
   2 rollouts        collect_hinted_rollouts
   3 judge           judge_rollouts
                                        4 manifest         build_rollout_manifest
                                        5 resample_select  select_resample_set
   6 rerolls         resample_hinted_rollouts
   7 judge_rerolls   batch_judge_rollouts
                                        8 relabel          resample_noise_model + resample_relabel
                                        9 summary          unfaithfulness_metrics_summary + verify_manifest
                                       10 figures          paper_plots
  optional white-box stages (only when named in `stages:` / --stages):
                                       11 splits           assign_splits
  12 probe_dataset   build_probe_dataset (per predicate)
  13 activations     collect_probe_activations
  14 train_probe     train_probe (per run)

Config: the shared model keys at the top level (``model_name``, ``seed``,
``thinking``, ``inference`` — usually via ``extends:``), the pipeline keys
(``run_name``, ``data_dir``, ``pipeline_dir``, ``stages``) and one optional
section per stage script, named after the script and carrying its own config
keys or CLI flags as snake_case keys. Default paths derive from ``data_dir`` and
the cell stem ``<model short>_<dataset tag>[_<run_name>]``; explicit path keys
win. A run record (``<pipeline_dir>/<stem>_baseline_pipeline.json``) is
rewritten after every stage; a failed stage stops the run (exit 1) and the same
command resumes it.

Usage:
    python -m src.scripts.run_pipeline --config configs/pipeline/run_pipeline.yaml [--dry-run]
    python -m src.scripts.run_pipeline --config ... --stages baseline,rollouts,judge
"""

from __future__ import annotations

import argparse
import dataclasses
import glob as globlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src.lib.config import (
    LEGACY_TOP_K,
    LEGACY_TOP_P,
    _deep_merge,
    get_inference_config,
    get_sampling_config,
    load_config,
    utc_now,
)
from src.lib.hinted_rollouts import CASE_MODES, validate_cases
from src.lib.hints import HINTS
from src.lib.paths import REPO_ROOT, resolve_data_path
from src.scripts.judge_rollouts import JudgeRolloutsConfig

STAGES = (
    "baseline", "rollouts", "judge", "manifest", "resample_select", "rerolls", "judge_rerolls",
    "relabel", "summary", "figures", "splits", "probe_dataset", "activations", "train_probe",
)
DEFAULT_STAGES = STAGES[:10]
OPTIONAL_STAGES = STAGES[10:]

# Stage → the config section(s) it reads, named after the script(s) it runs.
STAGE_SECTIONS = {
    "baseline": ("compute_baseline",),
    "rollouts": ("collect_hinted_rollouts",),
    "judge": ("judge_rollouts",),
    "manifest": ("build_rollout_manifest",),
    "resample_select": ("select_resample_set",),
    "rerolls": ("resample_hinted_rollouts",),
    "judge_rerolls": ("batch_judge_rollouts",),
    "relabel": ("resample_noise_model", "resample_relabel"),
    "summary": ("unfaithfulness_metrics_summary", "verify_manifest"),
    "figures": ("paper_plots",),
    "splits": ("assign_splits",),
    "probe_dataset": ("build_probe_dataset",),
    "activations": ("collect_probe_activations",),
    "train_probe": ("train_probe",),
}
STAGE_BLOCKS = {stage: sections[0] for stage, sections in STAGE_SECTIONS.items()}
STAGE_MODULES = {stage: f"src.scripts.{block}" for stage, block in STAGE_BLOCKS.items()}
STAGE_MODULES["figures"] = "src.scripts.visualizations.paper_plots"
SECTIONS = tuple(s for sections in STAGE_SECTIONS.values() for s in sections)

DEFAULT_DATA_DIR = "${DATA_ROOT}"
DEFAULT_PIPELINE_JUDGE_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_SAMPLING_TEMPERATURE = 0.7   # compute_baseline's fallback when sampling without a temperature
DEFAULT_SAMPLE_SEEDS = (43, 44, 45, 46)
DEFAULT_REROLL_TEMPERATURE = 0.7
DEFAULT_REROLL_CHUNK_SIZE = 512
DEFAULT_PREDICATE = "used_vs_ignored"
# The paper's cue styles (the re-sampling drops authority / few_shot / visual_pattern / pushback).
PAPER_HINT_STYLES = ("unethical_info", "metadata", "grader_hacking", "expert_opinion", "tool_output",
                     "answer_key_artifact", "post_hoc", "consensus")

PIPELINE_KEYS = {"run_name", "pipeline_dir", "data_dir", "stages", *SECTIONS}

JUDGE_CONFIG_KEYS = {f.name for f in dataclasses.fields(JudgeRolloutsConfig)}
SHARED_KEYS = {"model_name", "seed", "thinking", "inference", "top_p", "top_k", "layers"}
BASELINE_CONFIG_KEYS = SHARED_KEYS | {
    "dataset", "max_samples", "consistency_samples", "temperature", "vote_threshold",
    "random_answers_order", "batch_size", "output_csv",
}
ROLLOUTS_CONFIG_KEYS = SHARED_KEYS | {
    "baseline_csv", "output_csv", "cases", "hints", "hint_n_examples", "sample_questions_csv",
    "exclusion_list", "temperature", "chunk_size",
}

# Config keys of the config-style stage scripts (each script's KNOWN_CONFIG_KEYS; drift-tested).
RESAMPLE_CONFIG_KEYS = {
    "items_dir", "output_dir", "model_bases", "sample_seeds", "temperature", "top_p", "top_k",
    "inference", "chunk_size",
}
BATCH_JUDGE_CONFIG_KEYS = {
    "rollouts", "judge_model", "judge_prompt_file", "judge_workers", "retry_errors", "judge_max_tokens",
}
COLLECT_CONFIG_KEYS = {
    "model_name", "seed", "thinking", "inference", "layers", "seq_layers", "layer_stack_path",
    "dataset", "store_folder", "storage", "max_file_gb", "upload_workers", "staging", "staging_dir",
    "hf_model_kwargs",
}
COLLECT_FLAG_KEYS = ("resume", "force", "only", "limit")
COLLECT_SECTION_KEYS = COLLECT_CONFIG_KEYS | {"predicate", *COLLECT_FLAG_KEYS}
PROBE_DATASET_SPEC_KEYS = ("manifest", "runs", "hint_styles", "cases", "balance", "seed", "output_dir")
PROBE_DATASET_SECTION_KEYS = ("predicates", *PROBE_DATASET_SPEC_KEYS, "force", "chunk_rows")
TRAIN_BLOCKS = ("dataset", "model", "probe", "training", "search", "dataloader", "output")
TRAIN_RUN_KEYS = ("extends", *TRAIN_BLOCKS, "device")

# Flag-style sections: the script's CLI flags as snake_case keys, in argv order.
FLAG_KEYS = {
    "build_rollout_manifest": ("dir", "output", "only", "chunk_rows", "no_token_len", "include_smoke",
                               "noise_flip_min_votes", "label_overrides", "no_label_overrides"),
    "select_resample_set": ("manifest", "output_dir", "only", "seed", "k", "sample_seeds",
                            "controls_per_used", "rollouts_dir", "no_items", "extend", "chunk_rows"),
    "resample_noise_model": ("manifest", "output_dir", "only", "skip_width_check"),
    "resample_relabel": ("resample_dir", "manifest", "no_token_len", "chunk_rows"),
    "unfaithfulness_metrics_summary": ("dir", "output", "chunk_rows", "include_smoke"),
    "verify_manifest": ("manifest", "summary"),
    "paper_plots": ("cueball_dir", "out", "only", "formats"),
    "assign_splits": ("manifest", "resample_manifest", "seed", "fractions", "force", "extend", "dry_run"),
}
BOOL_FLAGS = {"no_token_len", "include_smoke", "no_label_overrides", "no_items", "extend",
              "skip_width_check", "force", "dry_run"}
PATH_FLAGS = {"dir", "output", "label_overrides", "manifest", "output_dir", "rollouts_dir", "resample_dir",
              "summary", "cueball_dir", "out", "resample_manifest"}

# Baseline sidecar fields the skip check compares (`thinking` joins them when the config pins it).
BASELINE_IDENTITY_FIELDS = (
    "model_name", "dataset", "consistency_samples", "temperature",
    "sampling_enabled", "vote_threshold", "max_samples", "seed", "max_tokens",
    "random_answers_order", "top_p", "top_k",
)
SAMPLING_ONLY_IDENTITY_FIELDS = ("top_p", "top_k")   # greedy decoding ignores both


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def parse_dataset_param(text: str) -> tuple[str, object]:
    """``key=value`` → ``(key, value)`` with the value YAML-parsed."""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError(f"--dataset-param expects key=value, got {text!r}")
    return key, yaml.safe_load(raw) if raw.strip() else None


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """``cfg`` with the CLI overrides applied (input untouched). ``--dataset`` replaces the
    dataset block including its params; ``--dataset-param`` entries overlay single keys."""
    out = dict(cfg)
    baseline = dict(out.get("compute_baseline") or {})
    if args.model:
        out["model_name"] = args.model
    if args.dataset:
        baseline["dataset"] = {"name": args.dataset, "params": {}}
    if args.dataset_param:
        block = dict(baseline.get("dataset") or {})
        params = dict(block.get("params") or {})
        for item in args.dataset_param:
            key, value = parse_dataset_param(item)
            params[key] = value
        block["params"] = params
        baseline["dataset"] = block
    out["compute_baseline"] = baseline
    if args.run_name:
        out["run_name"] = args.run_name
    if getattr(args, "cases", None):
        out["collect_hinted_rollouts"] = {
            **(out.get("collect_hinted_rollouts") or {}), "cases": args.cases,
        }
    if getattr(args, "judge_model", None):
        out["judge_rollouts"] = {
            **(out.get("judge_rollouts") or {}), "judge_model": args.judge_model,
        }
    return out


# Split each loader falls back to when `dataset.params.split` is unset (keep in step with
# compute_baseline.build_loader_kwargs).
DEFAULT_SPLITS = {"commonsense_qa": "validation", "aqua": "train"}
FALLBACK_SPLIT = "test"


def derive_dataset_tag(dataset_cfg: dict) -> str:
    """Filesystem tag of the dataset selection (name + split/subset only), e.g. ``mmlu-test``,
    ``gpqa-diamond``, ``commonsense_qa-validation``."""
    name = str(dataset_cfg["name"]).strip().lower()
    params = dict(dataset_cfg.get("params") or {})
    if name == "gpqa":
        subset = str(params.get("config", "gpqa_diamond"))
        suffix = subset[len("gpqa_"):] if subset.startswith("gpqa_") else subset
        return f"gpqa-{suffix}"
    split = str(params.get("split", DEFAULT_SPLITS.get(name, FALLBACK_SPLIT)))
    return f"{name}-{split}"


def shared_config(cfg: dict) -> dict:
    """The top-level keys every stage inherits (everything but the pipeline's own keys)."""
    return {k: v for k, v in cfg.items() if k not in PIPELINE_KEYS}


def check_keys(block: dict, known, name: str) -> dict:
    """``block`` as a dict; an unknown key raises listing the known ones."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"`{name}:` must be a mapping, got {type(block).__name__}")
    unknown = sorted(set(block) - set(known))
    if unknown:
        raise ValueError(
            f"Unknown key(s) in `{name}:`: {', '.join(unknown)}. Known keys: {', '.join(sorted(known))}"
        )
    return dict(block)


def build_stage_config(cfg: dict, stage: str) -> dict:
    """The flat config of a cell stage: its section laid over the shared keys (nested dicts such
    as ``inference`` merge). The judge stage gets only its own keys."""
    block = dict(cfg.get(STAGE_BLOCKS[stage]) or {})
    if stage == "judge":
        return check_keys(block, JUDGE_CONFIG_KEYS, "judge_rollouts")
    merged = _deep_merge(shared_config(cfg), block)
    if stage == "baseline":
        merged.pop("layers", None)   # activation-collector-only
    return merged


def overlay(defaults: dict, section: dict) -> dict:
    """``defaults`` with the section's non-null keys laid over (an explicit null keeps the default)."""
    return {**defaults, **{k: v for k, v in section.items() if v is not None}}


def pick(section: dict, key: str, default):
    value = section.get(key)
    return default if value is None else value


def flag_argv(section_name: str, values: dict) -> list[str]:
    """The argv of a flag-style section: ``--key value`` per set key (booleans as bare flags,
    lists comma-joined, paths resolved, ``None`` / ``False`` omitted). An unknown key raises."""
    known = FLAG_KEYS[section_name]
    check_keys(values, known, section_name)
    argv: list[str] = []
    for key in known:
        value = values.get(key)
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if key in BOOL_FLAGS:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag, ",".join(str(v) for v in value)]
        elif key in PATH_FLAGS:
            argv += [flag, str(resolve_data_path(value))]
        else:
            argv += [flag, str(value)]
    return argv


def parse_seeds(value) -> list[int]:
    """A seed list from a YAML list or a comma-separated string."""
    if isinstance(value, str):
        return [int(s) for s in value.split(",") if s.strip()]
    return [int(s) for s in value]


def resolve_baseline_sampling(baseline_cfg: dict) -> dict:
    """The effective sampling regime, mirroring ``compute_baseline`` (greedy unless
    ``consistency_samples > 1``; majority threshold by default)."""
    k = baseline_cfg.get("consistency_samples")
    n = int(k) if k else 1
    enabled = n > 1
    temperature = baseline_cfg.get("temperature")
    if enabled:
        eff_temp = float(temperature if temperature is not None else DEFAULT_SAMPLING_TEMPERATURE)
    else:
        eff_temp = 0.0
    threshold = baseline_cfg.get("vote_threshold")
    if enabled:
        eff_threshold = int(threshold) if threshold is not None else n // 2 + 1
    else:
        eff_threshold = None
    return {
        "consistency_samples": n,
        "temperature": eff_temp,
        "sampling_enabled": enabled,
        "vote_threshold": eff_threshold,
    }


def baseline_identity(gen_cfg: dict) -> dict:
    """The sidecar fields a finished baseline must match for stage 1 to be skipped."""
    dataset = gen_cfg["dataset"]
    identity = {
        "model_name": str(gen_cfg["model_name"]),
        "dataset": {
            "name": str(dataset["name"]).strip().lower(),
            "params": dict(dataset.get("params") or {}),
        },
        **resolve_baseline_sampling(gen_cfg),
        "max_samples": gen_cfg.get("max_samples"),
        "seed": gen_cfg.get("seed", 42),
        "max_tokens": get_inference_config(gen_cfg).max_tokens,
        "random_answers_order": bool(gen_cfg.get("random_answers_order", False)),
        "top_p": get_sampling_config(gen_cfg).top_p,
        "top_k": get_sampling_config(gen_cfg).top_k,
    }
    # Unset `thinking` means the model's registry default: compare it only when pinned.
    if isinstance(gen_cfg.get("thinking"), bool):
        identity["thinking"] = gen_cfg["thinking"]
    return identity


def load_json(path: Path) -> dict:
    """A JSON file's mapping ({} when absent/unreadable)."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_sidecar(csv_path: Path) -> dict:
    """A CSV's ``.meta.json`` sidecar ({} when absent/unreadable)."""
    return load_json(csv_path.with_suffix(".meta.json"))


def baseline_is_complete(baseline_csv: Path, identity: dict) -> tuple[bool, str]:
    """``(skip, reason)`` — skip stage 1 only when the sidecar says ``complete`` and every
    identity field matches; on a mismatch the stage runs and ``compute_baseline`` decides."""
    if not baseline_csv.exists():
        return False, "baseline CSV not found"
    sidecar = load_sidecar(baseline_csv)
    if not sidecar:
        return False, "baseline sidecar missing/unreadable"
    if not sidecar.get("complete"):
        return False, "baseline sidecar says the run is incomplete"
    sidecar.setdefault("random_answers_order", False)
    sidecar.setdefault("thinking", True)
    sidecar.setdefault("top_p", LEGACY_TOP_P)   # pre-knob sidecars sampled at vLLM's defaults
    sidecar.setdefault("top_k", LEGACY_TOP_K)
    greedy = not (sidecar.get("sampling_enabled") or identity.get("sampling_enabled"))
    mismatches = []
    for field in (*BASELINE_IDENTITY_FIELDS, "thinking"):
        if field == "thinking" and "thinking" not in identity:
            continue
        if greedy and field in SAMPLING_ONLY_IDENTITY_FIELDS:
            continue
        have, want = sidecar.get(field), identity.get(field)
        if field == "model_name":
            have, want = str(have).lower(), str(want).lower()
        if have != want:
            mismatches.append(f"{field}: existing={have!r} current={want!r}")
    if mismatches:
        return False, "baseline sidecar differs from this run (" + "; ".join(mismatches) + ")"
    return True, f"baseline complete ({sidecar.get('n_rows')} rows, accuracy {sidecar.get('accuracy')})"


def generation_key(gen_cfg: dict) -> str:
    """Identity of one finished hinted-rollouts generation: everything that changes the rollout set."""
    return "|".join([
        Path(str(gen_cfg["output_csv"])).name,
        f"cases={gen_cfg.get('cases', 'positive_cases')}",
        f"hints={','.join(sorted(str(h) for h in gen_cfg['hints']))}",
        f"temp={gen_cfg.get('temperature', 0.0)}",
        f"seed={gen_cfg.get('seed', 42)}",
        f"max_tokens={get_inference_config(gen_cfg).max_tokens}",
    ])


def rollouts_sidecar(gen_cfg: dict) -> dict:
    """The generation-recipe sidecar beside the rollouts CSV (provenance for the judge summary
    and the manifest)."""
    inference = get_inference_config(gen_cfg)
    sampling = get_sampling_config(gen_cfg)
    return {
        "model_name": gen_cfg["model_name"],
        "baseline_csv": str(gen_cfg["baseline_csv"]),
        "cases": gen_cfg.get("cases", "positive_cases"),
        "hints": list(gen_cfg["hints"]),
        "temperature": gen_cfg.get("temperature", 0.0),
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "seed": gen_cfg.get("seed", 42),
        "thinking": gen_cfg.get("thinking"),
        "max_tokens": inference.max_tokens,
        "max_model_len": inference.max_model_len,
        "updated_utc": utc_now(),
    }


def resolve_stages(text: str | None, configured=None) -> list[str]:
    """The stages to run, in pipeline order: ``--stages`` (comma list), else the config's
    ``stages:`` list, else the default ten."""
    if text:
        chosen = [s.strip() for s in text.split(",") if s.strip()]
    elif configured is not None:
        if not isinstance(configured, (list, tuple)):
            raise ValueError(f"`stages:` must be a list of stage names or null, got {configured!r}")
        chosen = [str(s).strip() for s in configured]
    else:
        return list(DEFAULT_STAGES)
    unknown = [s for s in chosen if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}. Stages: {', '.join(STAGES)}")
    if not chosen:
        raise ValueError("No stage requested.")
    return [s for s in STAGES if s in chosen]


def config_base_path(config_path: Path) -> Path | None:
    """The absolute path of the base a config ``extends`` (None without one)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    ref = raw.get("extends")
    return (config_path.parent / ref).resolve() if ref else None


def manifest_has_split(manifest: Path) -> bool:
    """Whether the manifest already carries a split assignment (sidecar block or column)."""
    if load_json(manifest.with_suffix(".meta.json")).get("split"):
        return True
    if not manifest.exists():
        return False
    import pandas as pd

    try:
        return bool(pd.read_parquet(manifest, columns=["split"])["split"].notna().any())
    except (OSError, ValueError, KeyError):
        return False


@dataclasses.dataclass(frozen=True)
class StageCommand:
    """One script invocation of a stage: ``python -m <module> [--config <config_path>] <argv>``,
    with the generated ``config`` written to ``config_path`` before the run."""

    module: str
    argv: tuple[str, ...] = ()
    config: dict | None = None
    config_path: Path | None = None
    note: str | None = None
    files: tuple[tuple[Path, dict], ...] = ()   # other generated configs written before the run

    def cmd(self) -> list[str]:
        out = [sys.executable, "-m", self.module]
        if self.config_path is not None:
            out += ["--config", str(self.config_path)]
        return out + list(self.argv)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class Plan:
    """Every path and flat stage config resolved up front, before any model load."""

    def __init__(self, cfg: dict, config_path: str | os.PathLike | None = None, *, stages=None):
        if not cfg.get("model_name"):
            raise ValueError("`model_name` is required (shared key, base via `extends`, or --model).")
        baseline_block = cfg.get(STAGE_BLOCKS["baseline"]) or {}
        if not (baseline_block.get("dataset") or {}).get("name"):
            raise ValueError("`compute_baseline.dataset.name` is required (or --dataset).")
        for name in SECTIONS:
            if cfg.get(name) is not None and not isinstance(cfg[name], dict):
                raise ValueError(f"`{name}:` must be a mapping, got {type(cfg[name]).__name__}")

        self.cfg = cfg
        self.config_path = Path(config_path).resolve() if config_path else None
        self.config_dir = self.config_path.parent if self.config_path else None
        self.model_name: str = str(cfg["model_name"])
        self._model_short: str | None = None
        self.dataset_cfg: dict = baseline_block["dataset"]
        self.dataset_tag = derive_dataset_tag(self.dataset_cfg)
        self.run_name = str(cfg.get("run_name") or "").strip() or None
        self.data_dir = resolve_data_path(cfg.get("data_dir") or DEFAULT_DATA_DIR)
        self.pipeline_dir = resolve_data_path(cfg.get("pipeline_dir") or str(self.data_dir / "pipeline"))
        self.generated_dir = self.pipeline_dir / "generated_configs"

        # Stage 1 — compute_baseline.
        self.baseline_cfg = build_stage_config(cfg, "baseline")
        if self.baseline_cfg.get("output_csv"):
            self.baseline_csv = resolve_data_path(self.baseline_cfg["output_csv"])
        else:
            self.baseline_csv = self.data_dir / "baselines" / f"{self.default_stem()}_baseline.csv"
        self.baseline_cfg["output_csv"] = str(self.baseline_csv)
        self.baseline_identity = baseline_identity(self.baseline_cfg)
        self.stem = self.baseline_csv.stem
        self.record_path = self.pipeline_dir / f"{self.stem}_pipeline.json"

        # Stage 2 — collect_hinted_rollouts.
        self.rollouts_cfg = build_stage_config(cfg, "rollouts")
        if self.rollouts_cfg.get("baseline_csv"):
            self.rollouts_input = resolve_data_path(self.rollouts_cfg["baseline_csv"])
        else:
            self.rollouts_input = self.baseline_csv
        self.rollouts_cfg["baseline_csv"] = str(self.rollouts_input)
        if self.rollouts_cfg.get("output_csv"):
            self.rollouts_csv = resolve_data_path(self.rollouts_cfg["output_csv"])
        else:
            self.rollouts_csv = (
                self.data_dir / "hinted_rollouts" / f"{self.rollouts_input.stem}_hinted_rollouts.csv"
            )
        self.rollouts_cfg["output_csv"] = str(self.rollouts_csv)
        self.cases = validate_cases(self.rollouts_cfg.get("cases", "positive_cases"))
        self.rollouts_cfg["cases"] = self.cases
        self.hints = [str(h) for h in (self.rollouts_cfg.get("hints") or PAPER_HINT_STYLES)]
        unknown_hints = [h for h in self.hints if h not in HINTS]
        if unknown_hints:
            raise ValueError(f"Unknown hint style(s) {unknown_hints}. Available: {list(HINTS)}")
        self.rollouts_cfg["hints"] = list(self.hints)
        self.generation_key = generation_key(self.rollouts_cfg)

        # Stage 3 — judge_rollouts.
        self.judge_cfg = build_stage_config(cfg, "judge")
        if self.judge_cfg.get("input_csv"):
            self.judge_input = resolve_data_path(self.judge_cfg["input_csv"])
        else:
            self.judge_input = self.rollouts_csv
        self.judge_cfg["input_csv"] = str(self.judge_input)
        if self.judge_cfg.get("output_csv"):
            self.judged_csv = resolve_data_path(self.judge_cfg["output_csv"])
        else:
            self.judged_csv = self.judge_input.with_name(f"{self.judge_input.stem}_judged.csv")
        self.judge_cfg["output_csv"] = str(self.judged_csv)
        self.judge_model = str(self.judge_cfg.get("judge_model") or DEFAULT_PIPELINE_JUDGE_MODEL)
        self.judge_cfg["judge_model"] = self.judge_model
        self.judge_prompt_file = self.judge_cfg.get("judge_prompt_file") or None
        if self.judge_cfg.get("cache_dir"):
            self.judge_cache_dir = resolve_data_path(self.judge_cfg["cache_dir"])
        else:
            self.judge_cache_dir = self.data_dir / "judge_cache" / self.judge_input.stem
        self.judge_cfg["cache_dir"] = str(self.judge_cache_dir)
        self.summary_json = self.judged_csv.with_name(f"{self.judged_csv.stem}_summary.json")
        # Stages 4-7 name the items / re-roll files after the judged CSV's stem.
        self.cell_stem = self.judged_csv.stem.removesuffix("_judged")

        # Tree paths (explicit section keys win; later stages follow the earlier ones).
        manifest_section = self.section("build_rollout_manifest")
        self.hinted_dir = self.resolve(manifest_section.get("dir"), self.data_dir / "hinted_rollouts")
        self.manifest = self.resolve(manifest_section.get("output"), self.hinted_dir / "rollout_manifest.parquet")
        self.label_overrides = self.resolve(manifest_section.get("label_overrides"),
                                            self.hinted_dir / "judge_label_overrides.csv")
        select_section = self.section("select_resample_set")
        self.resample_dir = self.resolve(select_section.get("output_dir"), self.data_dir / "resample")
        self.sample_seeds = parse_seeds(select_section.get("sample_seeds") or DEFAULT_SAMPLE_SEEDS)
        reroll_section = self.section("resample_hinted_rollouts")
        self.items_dir = self.resolve(reroll_section.get("items_dir"), self.resample_dir / "items")
        self.rerolls_dir = self.resolve(reroll_section.get("output_dir"), self.resample_dir / "rollouts")
        self.reroll_seeds = parse_seeds(reroll_section.get("sample_seeds") or self.sample_seeds)
        self.resample_manifest = self.resample_dir / "resample_manifest.parquet"
        summary_section = self.section("unfaithfulness_metrics_summary")
        self.summary_dir = self.resolve(summary_section.get("dir"), self.hinted_dir)
        self.summary_md = self.resolve(summary_section.get("output"),
                                       self.summary_dir / "unfaithfulness_metrics_summary.md")
        plots_section = self.section("paper_plots")
        self.figures_root = self.resolve(plots_section.get("cueball_dir"), self.data_dir)
        self.figures_dir = self.resolve(plots_section.get("out"), self.figures_root / "figures")
        dataset_section = self.section("build_probe_dataset")
        self.probe_datasets_dir = self.resolve(dataset_section.get("output_dir"), self.data_dir / "probe_datasets")
        self.trained_probes_dir = self.data_dir / "trained_probes" / "probes"
        self.validate_sections()
        self.reroll_judge_model = str(self.section("batch_judge_rollouts").get("judge_model") or self.judge_model)
        if stages is not None and set(stages) - {"baseline", "rollouts", "judge"}:
            self.check_judged_csv()

    # --- helpers -----------------------------------------------------------

    def check_judged_csv(self) -> None:
        """The stages after judge find the cell by the judged CSV's stem under the manifest's
        directory: an explicit ``judge_rollouts.output_csv`` elsewhere desynchronises them."""
        expected = self.hinted_dir / f"{self.judge_input.stem}_judged.csv"
        if self.judged_csv != expected:
            raise ValueError(
                f"`judge_rollouts.output_csv` must be {expected} when a stage after judge is requested "
                f"(the manifest reads `build_rollout_manifest.dir` and the re-sample items / re-roll "
                f"files are named after the judged CSV stem), got {self.judged_csv}."
            )

    def validate_sections(self) -> None:
        """Every section with a known-key set is checked up front, requested or not."""
        for name, known in FLAG_KEYS.items():
            check_keys(self.section(name), known, name)
        check_keys(self.section("compute_baseline"), BASELINE_CONFIG_KEYS, "compute_baseline")
        check_keys(self.section("collect_hinted_rollouts"), ROLLOUTS_CONFIG_KEYS, "collect_hinted_rollouts")
        check_keys(self.section("resample_hinted_rollouts"), RESAMPLE_CONFIG_KEYS, "resample_hinted_rollouts")
        check_keys(self.section("batch_judge_rollouts"), BATCH_JUDGE_CONFIG_KEYS, "batch_judge_rollouts")
        check_keys(self.section("build_probe_dataset"), PROBE_DATASET_SECTION_KEYS, "build_probe_dataset")
        check_keys(self.section("collect_probe_activations"), COLLECT_SECTION_KEYS, "collect_probe_activations")
        check_keys(self.section("train_probe"), ("runs",), "train_probe")

    @staticmethod
    def resolve(explicit, default: Path) -> Path:
        return resolve_data_path(explicit) if explicit else default

    def section(self, name: str) -> dict:
        return dict(self.cfg.get(name) or {})

    @property
    def model_short(self) -> str:
        """The model's registry short name (a heavy import, resolved lazily)."""
        if self._model_short is None:
            from src.lib.model_utils import get_model_short_name

            self._model_short = get_model_short_name(self.model_name)
        return self._model_short

    def default_stem(self) -> str:
        """``<model short name>_<dataset tag>[_<run_name>]``."""
        stem = f"{self.model_short}_{self.dataset_tag}"
        return f"{stem}_{self.run_name}" if self.run_name else stem

    def stage_config(self, stage: str) -> dict:
        return {"baseline": self.baseline_cfg, "rollouts": self.rollouts_cfg,
                "judge": self.judge_cfg}[stage]

    def generated_path(self, name: str) -> Path:
        return self.generated_dir / f"{name}.yaml"

    def generated_config_path(self, stage: str) -> Path:
        return self.generated_path(f"{self.stem}_{STAGE_BLOCKS[stage]}")

    def rollouts_complete(self, record: dict) -> bool:
        """Whether the run record says this exact generation recipe finished."""
        return self.rollouts_csv.exists() and self.generation_key in (
            record.get("generations_complete") or {}
        )

    def probe_dataset_parquet(self, predicate: str) -> Path:
        return self.probe_datasets_dir / f"{self.model_short}_{predicate}.parquet"

    def reroll_glob(self) -> str:
        return str(self.rerolls_dir / f"{self.cell_stem}_rs*.csv")

    def items_csv(self) -> Path:
        return self.items_dir / f"{self.cell_stem}_resample_items.csv"

    # --- generated configs of the config-style stages ----------------------

    def reroll_base(self) -> tuple[Path, dict | None]:
        """The base config the re-rolls run this model under: the config's ``extends`` base (or
        the config itself) when it pins ``model_name``, else a generated flat base of the shared
        keys — ``(path, config)``, the config None for an existing file."""
        if self.config_path is not None:
            base = config_base_path(self.config_path) or self.config_path
            if load_config(str(base)).get("model_name") == self.model_name:
                return base, None
        return self.generated_path(f"{self.stem}_reroll_base"), shared_config(self.cfg)

    def reroll_config(self) -> dict:
        """The flat ``resample_hinted_rollouts`` config of this cell (the generation budget and
        sampling truncation default to the hinted-rollouts stage's, ``model_bases`` to the
        config's base)."""
        section = check_keys(self.section("resample_hinted_rollouts"), RESAMPLE_CONFIG_KEYS,
                             "resample_hinted_rollouts")
        bases = section.get("model_bases")
        if bases is None:
            bases = {self.model_name: self.reroll_base()[0]}
        elif self.config_dir is not None:
            bases = {m: (self.config_dir / str(p)).resolve() for m, p in bases.items()}
        stage2 = get_sampling_config(self.rollouts_cfg)
        sampling = get_sampling_config({"top_p": pick(section, "top_p", stage2.top_p),
                                        "top_k": pick(section, "top_k", stage2.top_k)})
        return {
            "items_dir": str(self.items_dir),
            "output_dir": str(self.rerolls_dir),
            "model_bases": {m: str(p) for m, p in bases.items()},
            "sample_seeds": list(self.reroll_seeds),
            "temperature": float(pick(section, "temperature", DEFAULT_REROLL_TEMPERATURE)),
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "inference": _deep_merge(dict(self.rollouts_cfg.get("inference") or {}),
                                     dict(section.get("inference") or {})),
            "chunk_size": int(pick(section, "chunk_size", DEFAULT_REROLL_CHUNK_SIZE)),
        }

    def reroll_judge_config(self) -> dict:
        """The flat ``batch_judge_rollouts`` config over this cell's re-roll CSVs (the judge
        defaults to the judge stage's)."""
        section = check_keys(self.section("batch_judge_rollouts"), BATCH_JUDGE_CONFIG_KEYS,
                             "batch_judge_rollouts")
        out = {
            "rollouts": [str(r) for r in section["rollouts"]] if section.get("rollouts") else [self.reroll_glob()],
            "judge_model": self.reroll_judge_model,
            "judge_prompt_file": section.get("judge_prompt_file") or self.judge_prompt_file,
            "judge_workers": int(section.get("judge_workers") or self.judge_cfg.get("workers")
                                 or JudgeRolloutsConfig.workers),
        }
        for key in ("retry_errors", "judge_max_tokens"):
            value = section.get(key)
            value = self.judge_cfg.get(key) if value is None else value
            if value is not None:
                out[key] = value
        return out

    def probe_dataset_specs(self) -> list[tuple[str, dict, list[str]]]:
        """One ``(predicate, spec, argv)`` per entry of ``build_probe_dataset.predicates``."""
        section = check_keys(self.section("build_probe_dataset"), PROBE_DATASET_SECTION_KEYS, "build_probe_dataset")
        predicates = section.get("predicates") or [DEFAULT_PREDICATE]
        if isinstance(predicates, str):
            predicates = [predicates]
        argv = ["--force"] if section.get("force") else []
        if section.get("chunk_rows") is not None:
            argv += ["--chunk-rows", str(section["chunk_rows"])]
        specs = []
        for predicate in predicates:
            spec = {
                "name": f"{self.model_short}_{predicate}",
                "manifest": str(self.resolve(section.get("manifest"), self.resample_manifest)),
                "predicate": str(predicate),
                "subject_models": [self.model_short],
                "output_dir": str(self.probe_datasets_dir),
            }
            for key in ("runs", "hint_styles", "cases", "balance", "seed"):
                if section.get(key) is not None:
                    spec[key] = section[key]
            specs.append((str(predicate), spec, list(argv)))
        return specs

    def default_storage(self) -> dict:
        return {"backend": "local", "local_dir": str(self.data_dir / "probe_activations" / self.model_short)}

    def activations_config(self) -> tuple[dict, list[str]]:
        """The flat ``collect_probe_activations`` config (the section over the shared keys) and
        its CLI flags."""
        section = check_keys(self.section("collect_probe_activations"), COLLECT_SECTION_KEYS,
                             "collect_probe_activations")
        flags = {k: section.pop(k, None) for k in COLLECT_FLAG_KEYS}
        predicate = section.pop("predicate", None) or DEFAULT_PREDICATE
        shared = {k: v for k, v in shared_config(self.cfg).items() if k in COLLECT_CONFIG_KEYS}
        cfg = _deep_merge(shared, {k: v for k, v in section.items() if v is not None})
        cfg["dataset"] = str(self.resolve(cfg.get("dataset"), self.probe_dataset_parquet(predicate)))
        cfg["storage"] = pick(cfg, "storage", self.default_storage())
        argv = [f"--{k}" for k in ("resume", "force") if flags[k]]
        only = flags["only"]
        for item in ([only] if isinstance(only, str) else (only or [])):
            argv += ["--only", str(item)]
        if flags["limit"] is not None:
            argv += ["--limit", str(flags["limit"])]
        return cfg, argv

    def train_runs(self) -> list[tuple[str, dict, list[str]]]:
        """One ``(run name, flat train config, argv)`` per entry of ``train_probe.runs``; an
        entry's ``extends`` resolves relative to the pipeline config, ``dataset.predicate``
        names the probe dataset in place of ``dataset.path``."""
        section = check_keys(self.section("train_probe"), ("runs",), "train_probe")
        runs = section.get("runs") or []
        if not isinstance(runs, list):
            raise ValueError("`train_probe.runs` must be a list of train configs.")
        collect_section = self.section("collect_probe_activations")
        collect_predicate = collect_section.get("predicate") or DEFAULT_PREDICATE
        out = []
        for index, entry in enumerate(runs):
            run = check_keys(entry, TRAIN_RUN_KEYS, f"train_probe.runs[{index}]")
            device = run.pop("device", None)
            ref = run.pop("extends", None)
            if ref:
                if self.config_dir is None:
                    raise ValueError(f"train_probe.runs[{index}].extends needs the pipeline config's path.")
                run = _deep_merge(load_config(str(self.config_dir / str(ref))), run)
            dataset = dict(run.get("dataset") or {})
            predicate = dataset.pop("predicate", None) or collect_predicate
            dataset["path"] = str(self.resolve(dataset.get("path"), self.probe_dataset_parquet(predicate)))
            run["dataset"] = dataset
            output = dict(run.get("output") or {})
            output["dir"] = str(self.resolve(output.get("dir"), self.trained_probes_dir))
            run["output"] = output
            if isinstance(run.get("model"), dict):
                model = dict(run["model"])
                model.setdefault("subject_model", self.model_short)
                if not model.get("activations"):
                    model["activations"] = {
                        **(collect_section.get("storage") or self.default_storage()),
                        "folder": collect_section.get("store_folder") or f"{self.model_short}_{collect_predicate}",
                    }
                run["model"] = model
            name = str(output.get("run_name") or f"run{index + 1}")
            out.append((name, run, ["--device", str(device)] if device else []))
        return out

    # --- commands ----------------------------------------------------------

    def commands(self, stage: str) -> list[StageCommand]:
        """The script invocations of a stage, in order (nothing is written here)."""
        module = STAGE_MODULES[stage]
        if stage in ("baseline", "rollouts", "judge"):
            return [StageCommand(module, config=self.stage_config(stage),
                                 config_path=self.generated_config_path(stage))]
        if stage == "manifest":
            values = overlay({"dir": str(self.hinted_dir), "output": str(self.manifest),
                              "label_overrides": str(self.label_overrides)}, self.section("build_rollout_manifest"))
            return [StageCommand(module, tuple(flag_argv("build_rollout_manifest", values)))]
        if stage == "resample_select":
            section = self.section("select_resample_set")
            values = overlay({"manifest": str(self.manifest), "output_dir": str(self.resample_dir),
                              "rollouts_dir": str(self.hinted_dir), "extend": True,
                              "sample_seeds": list(self.sample_seeds), "k": len(self.sample_seeds)}, section)
            return [StageCommand(module, tuple(flag_argv("select_resample_set", values)))]
        if stage == "rerolls":
            files, note = (), None
            if self.section("resample_hinted_rollouts").get("model_bases") is None:
                base, generated = self.reroll_base()
                if generated is not None:
                    files = ((base, generated),)
                    note = f"generated base {base} (no base config pins {self.model_name!r})"
            return [StageCommand(module, ("--model", self.model_name, "--only", self.cell_stem),
                                 config=self.reroll_config(),
                                 config_path=self.generated_path(f"{self.stem}_resample_hinted_rollouts"),
                                 note=note, files=files)]
        if stage == "judge_rerolls":
            return [StageCommand(module, config=self.reroll_judge_config(),
                                 config_path=self.generated_path(f"{self.stem}_batch_judge_rollouts"))]
        if stage == "relabel":
            noise = overlay({"manifest": str(self.manifest), "output_dir": str(self.resample_dir)},
                            self.section("resample_noise_model"))
            relabel = overlay({"resample_dir": str(self.resample_dir), "manifest": str(self.manifest)},
                              self.section("resample_relabel"))
            return [StageCommand("src.scripts.resample_noise_model", tuple(flag_argv("resample_noise_model", noise))),
                    StageCommand("src.scripts.resample_relabel", tuple(flag_argv("resample_relabel", relabel)))]
        if stage == "summary":
            summary = overlay({"dir": str(self.summary_dir), "output": str(self.summary_md)},
                              self.section("unfaithfulness_metrics_summary"))
            verify = overlay({"manifest": str(self.manifest), "summary": str(self.summary_md)},
                             self.section("verify_manifest"))
            return [StageCommand("src.scripts.unfaithfulness_metrics_summary",
                                 tuple(flag_argv("unfaithfulness_metrics_summary", summary))),
                    StageCommand("src.scripts.verify_manifest", tuple(flag_argv("verify_manifest", verify)))]
        if stage == "figures":
            values = overlay({"cueball_dir": str(self.figures_root), "out": str(self.figures_dir)}, self.section("paper_plots"))
            return [StageCommand(module, tuple(flag_argv("paper_plots", values)))]
        if stage == "splits":
            section = self.section("assign_splits")
            values = overlay({"manifest": str(self.manifest), "resample_manifest": str(self.resample_manifest)}, section)
            if section.get("extend") is None and not values.get("force"):
                values["extend"] = manifest_has_split(self.resolve(section.get("manifest"), self.manifest))
            note = "--extend: the manifest already carries a split" if values.get("extend") else None
            return [StageCommand(module, tuple(flag_argv("assign_splits", values)), note=note)]
        if stage == "probe_dataset":
            out = []
            for predicate, spec, argv in self.probe_dataset_specs():
                parquet = self.probe_dataset_parquet(predicate)
                note = None
                if parquet.exists() and "--force" not in argv:
                    note = f"skip: {parquet} exists (force: true rebuilds it)"
                out.append(StageCommand(module, tuple(argv), config=spec,
                                        config_path=self.generated_path(
                                            f"{self.model_short}_build_probe_dataset_{predicate}"),
                                        note=note))
            return out
        if stage == "activations":
            cfg, argv = self.activations_config()
            return [StageCommand(module, tuple(argv), config=cfg,
                                 config_path=self.generated_path(f"{self.model_short}_collect_probe_activations"))]
        if stage == "train_probe":
            return [StageCommand(module, tuple(argv), config=run,
                                 config_path=self.generated_path(f"{self.model_short}_train_probe_{name}"))
                    for name, run, argv in self.train_runs()]
        raise ValueError(f"Unknown stage {stage!r}")

    def missing_input(self, stage: str) -> str | None:
        """Why a stage cannot run yet (an input the earlier stages write is absent), else None."""
        checks = {
            "manifest": (self.hinted_dir, "judged rollouts directory"),
            "resample_select": (self.manifest, "rollout manifest"),
            "rerolls": (self.items_csv(), "re-sample items file (run resample_select)"),
            "relabel": (self.resample_dir / "resample_selection.csv", "re-sample selection"),
            "summary": (self.manifest, "rollout manifest"),
            "figures": (self.resample_manifest, "re-sample manifest (run relabel)"),
            "splits": (self.manifest, "rollout manifest"),
            "probe_dataset": (self.resample_manifest, "re-sample manifest (run relabel)"),
        }
        if stage == "figures" and (self.manifest != self.figures_root / "hinted_rollouts" / "rollout_manifest.parquet"
                                   or self.resample_dir != self.figures_root / "resample"):
            return (f"paper_plots reads <cueball_dir>/hinted_rollouts/rollout_manifest.parquet and <cueball_dir>/resample "
                    f"(cueball_dir = {self.figures_root}), but this run's manifest is {self.manifest} and its "
                    f"re-sample dir {self.resample_dir}: set `paper_plots.cueball_dir` to the tree holding them")
        if stage in checks:
            path, what = checks[stage]
            return None if path.exists() else f"{what} not found at {path}"
        if stage == "judge_rerolls":
            pattern = self.reroll_judge_config()["rollouts"]
            checkable = [p for p in pattern if "/" in str(p) or "$" in str(p)]   # bare names: the script resolves them
            hits = [p for pat in checkable for p in globlib.glob(str(resolve_data_path(pat)))
                    if p.endswith(".csv") and not p.endswith("_judged.csv")]
            return None if hits or not checkable else f"no re-roll CSV matches {pattern} (run rerolls)"
        if stage in ("activations", "train_probe"):
            for command in self.commands(stage):
                path = Path(command.config["dataset"] if stage == "activations" else command.config["dataset"]["path"])
                if not path.exists():
                    return f"probe dataset not found at {path} (run probe_dataset)"
        return None

    def outputs(self, stage: str) -> dict:
        """The artifacts a stage's run record entry names."""
        if stage == "probe_dataset":
            return {"probe_datasets": [str(self.probe_dataset_parquet(p)) for p, _, _ in self.probe_dataset_specs()]}
        if stage == "activations":
            return {"store": self.activations_config()[0].get("storage")}
        return {
            "manifest": {"manifest": str(self.manifest)},
            "resample_select": {"resample_dir": str(self.resample_dir)},
            "rerolls": {"rerolls": self.reroll_glob()},
            "judge_rerolls": {"rerolls": self.reroll_glob()},
            "relabel": {"resample_manifest": str(self.resample_manifest)},
            "summary": {"summary_md": str(self.summary_md)},
            "figures": {"figures_dir": str(self.figures_dir)},
            "splits": {"manifest": str(self.manifest)},
            "train_probe": {"output_dir": str(self.trained_probes_dir)},
        }.get(stage, {})

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset": self.dataset_cfg,
            "dataset_tag": self.dataset_tag,
            "run_name": self.run_name,
            "data_dir": str(self.data_dir),
            "baseline_csv": str(self.baseline_csv),
            "baseline_identity": self.baseline_identity,
            "rollouts_csv": str(self.rollouts_csv),
            "cases": self.cases,
            "hints": self.hints,
            "rollouts_temperature": float(self.rollouts_cfg.get("temperature", 0.0)),
            "generation_key": self.generation_key,
            "judged_csv": str(self.judged_csv),
            "summary_json": str(self.summary_json),
            "judge_model": self.judge_model,
            "judge_prompt_file": self.judge_prompt_file,
            "manifest": str(self.manifest),
            "resample_dir": str(self.resample_dir),
            "sample_seeds": list(self.sample_seeds),
            "reroll_judge_model": self.reroll_judge_model,
            "figures_dir": str(self.figures_dir),
        }

    def print_plan(self, stages: list[str], record: dict) -> None:
        skip_baseline, why = baseline_is_complete(self.baseline_csv, self.baseline_identity)
        print(f"Model:    {self.model_name}")
        print(f"Dataset:  {self.dataset_cfg['name']}  params={self.dataset_cfg.get('params') or {}}")
        print(f"Data dir: {self.data_dir}")
        print(f"Stages:   {', '.join(stages)}")
        if self.rollouts_csv.parent != self.hinted_dir and set(stages) - {"baseline", "rollouts", "judge"}:
            print(f"  ! the cell's rollouts ({self.rollouts_csv.parent}) are outside the tree stages' "
                  f"directory ({self.hinted_dir}); set `data_dir` to the tree the cell writes into")
        for stage in stages:
            n = STAGES.index(stage) + 1
            print(f"  {n:2d}. {STAGE_BLOCKS[stage]:<30s} {self.stage_target(stage)}")
            if stage == "baseline":
                print(f"        {'skip: ' if skip_baseline else 'run: '}{why}")
            elif stage == "rollouts":
                print(f"        cases={self.cases} hints={self.hints} "
                      f"temperature={self.rollouts_cfg.get('temperature', 0.0)}")
                if self.rollouts_input != self.baseline_csv:
                    print(f"        input: {self.rollouts_input} (explicit baseline_csv)")
                print("        " + ("skip: generation complete per run record"
                                    if self.rollouts_complete(record) else "run: generation pending"))
            elif stage == "judge":
                if self.judge_input != self.rollouts_csv:
                    print(f"        input: {self.judge_input} (explicit input_csv)")
                print(f"        judge={self.judge_model} prompt={self.judge_prompt_file or 'built-in default'}"
                      f"{'  (judged CSV exists — rewritten; cached verdicts reused)' if self.judged_csv.exists() else ''}")
            elif stage == "rerolls":
                print(f"        seeds={self.reroll_seeds} items={self.items_csv()}")
            elif stage == "judge_rerolls":
                print(f"        judge={self.reroll_judge_model}")
            missing = self.missing_input(stage) if stage not in ("baseline", "rollouts", "judge") else None
            if missing:
                print(f"        ! {missing}")
            commands = self.commands(stage)
            if not commands:
                print("        skip: nothing configured")
            for command in commands:
                if command.note:
                    print(f"        {command.note}")
                print(f"        $ {' '.join(command.cmd())}")
        print(f"Generated configs: {self.generated_dir}")
        print(f"Run record:        {self.record_path}")

    def stage_target(self, stage: str) -> str:
        targets = {
            "baseline": self.baseline_csv, "rollouts": self.rollouts_csv, "judge": self.judged_csv,
            "manifest": self.manifest, "resample_select": self.resample_dir / "resample_selection.csv",
            "rerolls": self.reroll_glob(), "judge_rerolls": self.rerolls_dir,
            "relabel": self.resample_manifest, "summary": self.summary_md, "figures": self.figures_dir,
            "splits": self.manifest, "probe_dataset": self.probe_datasets_dir,
            "activations": self.data_dir / "probe_activations", "train_probe": self.trained_probes_dir,
        }
        return f"→ {targets[stage]}"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_stage_script(stage: str, config_path: Path | None = None, *, module: str | None = None,
                     argv=()) -> int:
    """Run a stage script as a subprocess from the repo root; returns the exit code."""
    cmd = [sys.executable, "-m", module or STAGE_MODULES[stage]]
    if config_path is not None:
        cmd += ["--config", str(config_path)]
    cmd += [str(a) for a in argv]
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, env=stage_env()).returncode


def stage_env() -> dict:
    """The subprocess environment: the interpreter's own ``bin/`` first on ``PATH``, so the
    venv's tools (vLLM shells out to ``ninja``) are found without activating it."""
    env = dict(os.environ)
    bin_dir = str(Path(sys.executable).parent)
    path = env.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        env["PATH"] = bin_dir + (os.pathsep + path if path else "")
    return env


def write_generated_config(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def write_stage_config(plan: Plan, stage: str) -> Path:
    return write_generated_config(plan.generated_config_path(stage), plan.stage_config(stage))


def stage_baseline(plan: Plan, record: dict, *, force: bool) -> dict:
    skip, why = baseline_is_complete(plan.baseline_csv, plan.baseline_identity)
    if skip and not force:
        print(f"  skipping: {why} (--force-baseline to rerun)")
        return {"status": "skipped (already complete)", "baseline_csv": str(plan.baseline_csv)}
    print(f"  {why}" if not skip else "  --force-baseline: rerunning (resumes row-wise)")
    returncode = run_stage_script("baseline", write_stage_config(plan, "baseline"))
    if returncode != 0:
        return {"status": f"failed (compute_baseline exit {returncode})"}
    if not plan.baseline_csv.exists():
        return {"status": "failed: compute_baseline reported success but wrote no CSV"}
    sidecar = load_sidecar(plan.baseline_csv)
    return {
        "status": "ok",
        "baseline_csv": str(plan.baseline_csv),
        "n_rows": sidecar.get("n_rows"),
        "accuracy": sidecar.get("accuracy"),
    }


def stage_rollouts(plan: Plan, record: dict, *, force: bool) -> dict:
    if not plan.rollouts_input.exists():
        return {"status": f"failed: baseline CSV not found at {plan.rollouts_input}"}
    if plan.rollouts_complete(record) and not force:
        done = record["generations_complete"][plan.generation_key]
        print(f"  skipping: generation complete per run record ({done}) (--force-generate to redo)")
        return {"status": "skipped (already complete)", "rollouts_csv": str(plan.rollouts_csv)}
    plan.rollouts_csv.parent.mkdir(parents=True, exist_ok=True)
    returncode = run_stage_script("rollouts", write_stage_config(plan, "rollouts"))
    if returncode != 0:
        return {"status": f"failed (collect_hinted_rollouts exit {returncode})"}
    if not plan.rollouts_csv.exists():
        return {"status": "failed: collect_hinted_rollouts reported success but wrote no CSV"}
    with open(plan.rollouts_csv.with_suffix(".meta.json"), "w") as f:
        json.dump(rollouts_sidecar(plan.rollouts_cfg), f, indent=2)
    record.setdefault("generations_complete", {})[plan.generation_key] = utc_now()
    return {"status": "ok", "rollouts_csv": str(plan.rollouts_csv)}


def stage_judge(plan: Plan, record: dict) -> dict:
    if not plan.judge_input.exists():
        return {"status": f"failed: rollouts CSV not found at {plan.judge_input}"}
    returncode = run_stage_script("judge", write_stage_config(plan, "judge"))
    if returncode != 0:
        return {"status": f"failed (judge_rollouts exit {returncode})"}
    if not plan.judged_csv.exists():
        return {"status": "failed: judge_rollouts reported success but wrote no CSV"}
    summary = write_judge_summary(plan)
    n_judged = sum(int(r["n_judged"]) for r in summary["results"])
    n_errors = sum(int(r["n_judge_errors"]) for r in summary["results"])
    status = "ok" if n_errors == 0 else f"ok ({n_errors} judge errors — rerun retries them)"
    return {
        "status": status,
        "judged_csv": str(plan.judged_csv),
        "summary_json": str(plan.summary_json),
        "n_judged": n_judged,
    }


def stage_commands(plan: Plan, stage: str) -> dict:
    """Run a stage's script invocations in order (generated configs written first); the first
    non-zero exit ends the stage. Commands carrying a skip note are not run."""
    missing = plan.missing_input(stage)
    if missing:
        return {"status": f"failed: {missing}"}
    commands = plan.commands(stage)
    if not commands:
        return {"status": "skipped (nothing configured)"}
    to_run = [c for c in commands if not (c.note and c.note.startswith("skip"))]
    for command in commands:
        if command.note:
            print(f"  {command.note}")
    if not to_run:
        return {"status": "skipped (already complete)", **plan.outputs(stage)}
    for command in to_run:
        for path, cfg in command.files:
            write_generated_config(path, cfg)
        if command.config is not None:
            write_generated_config(command.config_path, command.config)
        returncode = run_stage_script(stage, command.config_path, module=command.module, argv=command.argv)
        if returncode != 0:
            return {"status": f"failed ({command.module.rsplit('.', 1)[-1]} exit {returncode})"}
    return {"status": "ok", **plan.outputs(stage)}


def write_judge_summary(plan: Plan) -> dict:
    """``<judged stem>_summary.json`` in the batch judge's format, computed from the judged CSV."""
    import pandas as pd

    from src.lib.llm_judge_verb import load_prompt_template, prompt_template_hash
    from src.scripts.batch_judge_rollouts import aggregate_judged

    labeled = pd.read_csv(plan.judged_csv)
    prompt_template = (
        load_prompt_template(plan.judge_prompt_file) if plan.judge_prompt_file else None
    )
    summary = {
        "rollouts_csv": str(plan.judge_input),
        "judged_csv": str(plan.judged_csv),
        "provenance": load_sidecar(plan.judge_input),
        "judge_model": plan.judge_model,
        "judge_prompt_hash": prompt_template_hash(prompt_template),
        "created_utc": utc_now(),
        "results": aggregate_judged(labeled),
    }
    with open(plan.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON → {plan.summary_json}")
    return summary


def preflight_judge(plan: Plan, stages: list[str]) -> None:
    """Fail on a bad judge slug / missing key before any GPU work."""
    from src.lib.llm_judge_verb import check_judge_model

    slugs = []
    if "judge" in stages:
        slugs.append(plan.judge_model)
    if "judge_rerolls" in stages and plan.reroll_judge_model not in slugs:
        slugs.append(plan.reroll_judge_model)
    for slug in slugs:
        if check_judge_model(slug):
            print(f"Judge preflight: {slug} OK")
        else:
            print(f"Judge preflight: {slug} unreachable right now (transient) — continuing")


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------

def load_record(path: Path) -> dict:
    return load_json(path)


def write_record(plan: Plan, record: dict) -> None:
    plan.record_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = plan.record_path.with_suffix(".json.tmp")
    record["updated_utc"] = utc_now()
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    tmp.replace(plan.record_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pipeline for one model × dataset from a single config"
    )
    parser.add_argument("--config", required=True, help="Path to the pipeline YAML config.")
    parser.add_argument("--model", default=None, help="HF model id (overrides the shared `model_name`).")
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset name (mmlu | mmlu_pro | gpqa | medqa | aqua | commonsense_qa); replaces "
             "`compute_baseline.dataset`, params start empty — add them with --dataset-param.",
    )
    parser.add_argument(
        "--dataset-param", action="append", default=None, metavar="KEY=VALUE",
        help="Set one `compute_baseline.dataset.params` key (repeatable; YAML-parsed value).",
    )
    parser.add_argument("--run-name", default=None, help="Extra tag in the default output stem (overrides `run_name`).")
    parser.add_argument(
        "--stages", default=None,
        help=f"Comma-separated subset of {','.join(STAGES)} (default: the config's `stages:`, "
             f"else {','.join(DEFAULT_STAGES)}; the rest are opt-in).",
    )
    parser.add_argument("--cases", default=None, choices=CASE_MODES,
                        help="Which baseline rows to hint (overrides `collect_hinted_rollouts.cases`).")
    parser.add_argument("--judge-model", default=None,
                        help="OpenRouter slug of the judge (overrides `judge_rollouts.judge_model`).")
    parser.add_argument("--force-baseline", action="store_true",
                        help="Run compute_baseline even when the sidecar says the baseline is complete.")
    parser.add_argument("--force-generate", action="store_true",
                        help="Rerun hinted-rollout generation even when the run record marks it complete.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan (every command, nothing executed or written), then exit.")
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args)
    stages = resolve_stages(args.stages, cfg.get("stages"))
    plan = Plan(cfg, config_path=args.config, stages=stages)
    record = load_record(plan.record_path)
    plan.print_plan(stages, record)
    if args.dry_run:
        print("\nDry run — nothing executed.")
        return

    preflight_judge(plan, stages)

    record.update({
        "config": str(Path(args.config).resolve()),
        "plan": plan.describe(),
        "stages_requested": stages,
        "started_utc": utc_now(),
        "stages": {},
    })
    write_record(plan, record)

    runners = {
        "baseline": lambda: stage_baseline(plan, record, force=args.force_baseline),
        "rollouts": lambda: stage_rollouts(plan, record, force=args.force_generate),
        "judge": lambda: stage_judge(plan, record),
        **{stage: (lambda s=stage: stage_commands(plan, s)) for stage in STAGES[3:]},
    }
    failed: str | None = None
    for i, stage in enumerate(stages, start=1):
        print(f"\n=== [{i}/{len(stages)}] {STAGE_BLOCKS[stage]} ===", flush=True)
        started = utc_now()
        try:
            result = runners[stage]()
        except Exception as e:  # noqa: BLE001 — recorded, then re-raised
            record["stages"][stage] = {
                "status": f"error: {type(e).__name__}: {e}",
                "started_utc": started, "finished_utc": utc_now(),
            }
            write_record(plan, record)
            raise
        record["stages"][stage] = {**result, "started_utc": started, "finished_utc": utc_now()}
        write_record(plan, record)
        print(f"  {STAGE_BLOCKS[stage]}: {result['status']}")
        if not str(result["status"]).startswith(("ok", "skipped")):
            failed = stage
            break

    print("\n=== Pipeline summary ===")
    for stage in stages:
        entry = record["stages"].get(stage)
        print(f"  {STAGE_BLOCKS[stage]}: {entry['status'] if entry else 'not run'}")
    print(f"  run record → {plan.record_path}")
    if failed:
        print(f"\nStopped at '{STAGE_BLOCKS[failed]}'; rerun the same command to resume.")
        sys.exit(1)
    if "judge" in stages:
        print(f"  judged CSV → {plan.judged_csv}\n  summary    → {plan.summary_json}")


if __name__ == "__main__":
    main()
