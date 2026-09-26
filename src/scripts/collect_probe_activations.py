#!/usr/bin/env python3
"""Collect residual-stream activations of a probe dataset into an activation store.

Every row of the dataset parquet is replayed teacher-forced (prompt + CoT + close
delimiter + post-CoT text up to the answer commit); per layer the CoT sequence
(``seq_layer_<L>_part<NN>``, for ``seq_layers``) and the four probe points
(``points_layer_<L>_part<NN>``, ``POINTS`` order) are written as bf16 safetensors
under ``<root>/<store folder>/`` next to ``manifest.csv`` / ``manifest.meta.json``.
The manifest is committed after every part, so ``--resume`` continues an
interrupted run against the same dataset fingerprint and ``--force`` rewrites the
folder. Rows without a usable CoT / close delimiter / answer commit, or longer than
the model's context, are listed under the meta's ``skipped`` and never stored.

Usage:
    python -m src.scripts.collect_probe_activations --config configs/collect_probe_activations_nemotron.yaml
    python -m src.scripts.collect_probe_activations --config ... --dry-run
    python -m src.scripts.collect_probe_activations --config ... --limit 50      # smoke run
    python -m src.scripts.collect_probe_activations --config ... --only post_hoc --only gpqa
    python -m src.scripts.collect_probe_activations --config ... --resume       # continue an interrupted run
    python -m src.scripts.collect_probe_activations --config ... --verify-only  # re-check a finished store
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.lib.activation_store import (
    CGROUP_MEMORY_MAX,
    DEFAULT_MAX_FILE_GB,
    DEFAULT_UPLOAD_WORKERS,
    GB,
    HF_HARD_FILE_LIMIT_GB,
    HF_RECOMMENDED_FILE_GB,
    POINTS,
    STAGING_MODES,
    STORE_DTYPE_NAME,
    STORE_FILES,
    STORE_MANIFEST_COLUMNS,
    ActivationCapturer,
    GroupResult,
    PartPlan,
    PartWriter,
    PointStore,
    SequenceStore,
    StoreBackend,
    StoreWriter,
    _tokenize,
    assert_layer_depth,
    build_messages,
    cgroup_memory_limit_bytes,
    is_store_file,
    load_store_state,
    make_backend,
    mamba_chunk_sizes,
    part_index,
    part_template,
    plan_parts,
    read_store_manifest,
    render_prompt_text,
    reset_read_caches,
    resolve_hidden_size,
    resolve_layer_stack,
    set_mamba_chunk_size,
)
from src.lib.config import load_config
from src.lib.model_utils import (
    get_model_config,
    get_model_short_name,
    get_text_config,
    get_thinking_config,
    load_model_hf,
)
from src.lib.parsing import _find_real_end_think_char, answer_commit_match
from src.lib.paths import resolve_data_path
from src.lib.probe_datasets import dataset_sidecars, read_dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STAGING_SUBDIR = "tmp/collect_probe_activations"
SPAN = "chain_of_thought"
DEFAULT_VERIFY_SAMPLE = 20

# Every key the config may carry; anything else is refused at parse time.
KNOWN_CONFIG_KEYS = {
    "model_name", "seed", "thinking", "inference", "layers", "seq_layers", "layer_stack_path",
    "dataset", "store_folder", "storage", "max_file_gb", "upload_workers", "staging", "staging_dir",
    "hf_model_kwargs",
}

# Skip reasons (the meta's ``skipped`` values start with one of these).
SKIP_EMPTY_COT = "empty_cot"
SKIP_NO_CLOSE = "no_close_delimiter"
SKIP_NO_COMMIT = "no_answer_commit"
SKIP_COT_NOT_IN_ROLLOUT = "cot_not_in_rollout"
SKIP_TOO_LONG = "too_long"
SKIP_CUDA_OOM = "cuda_oom"   # runtime: OOM even at the chunk-size floor

# CUDA OOM recovery halves every Mamba mixer's ``chunk_size`` down to this floor.
MIN_MAMBA_CHUNK_SIZE = 8

log = logging.getLogger("collect_probe_activations")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ProbeCollectConfig:
    model_name: str
    dataset: Path
    storage: dict
    layers: list[int]
    seed: int
    max_file_gb: float
    thinking: object = None
    layer_stack_path: Optional[str] = None
    upload_workers: int = DEFAULT_UPLOAD_WORKERS
    staging: str = "memory"
    staging_dir: Optional[Path] = None
    store_folder: Optional[str] = None   # None = the dataset's name
    hf_model_kwargs: dict = field(default_factory=dict)   # extra from_pretrained kwargs
    seq_layers: Optional[list[int]] = None   # layers with seq files (sorted, ⊆ layers); None = layers

    def __post_init__(self) -> None:
        if self.seq_layers is None:
            self.seq_layers = list(self.layers)


def parse_seq_layers(value, layers: list[int]) -> list[int]:
    """The config's ``seq_layers``: ``None`` = every layer of ``layers``; else a
    non-empty list of distinct ints, every one in ``layers``; returned sorted."""
    if value is None:
        return sorted(layers)
    if not isinstance(value, list) or not value:
        raise ValueError(f"`seq_layers` must be a non-empty list of layers from `layers` or null (= every layer), "
                         f"got {value!r}.")
    if any((not isinstance(l, int)) or isinstance(l, bool) for l in value):
        raise ValueError(f"`seq_layers` must be integers, got {value}.")
    if len(set(value)) != len(value):
        raise ValueError(f"`seq_layers` has duplicates: {value}.")
    outside = sorted(set(value) - set(layers))
    if outside:
        raise ValueError(f"`seq_layers` {outside} are not in `layers` {sorted(layers)}: every seq layer needs its "
                         "points file too, so `seq_layers` must be a subset of `layers`.")
    return sorted(value)


def parse_config(cfg: dict) -> ProbeCollectConfig:
    """Validate a loaded YAML config and resolve its data paths."""
    unknown = sorted(set(cfg) - KNOWN_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown config keys {unknown}. This script reads one probe dataset parquet "
            "(`dataset:`) into a `storage:` store — see configs/collect_probe_activations_nemotron.yaml."
        )
    model_name = cfg.get("model_name")
    if not model_name or not isinstance(model_name, str):
        raise ValueError("Config needs a `model_name` (usually via `extends:`).")
    dataset = cfg.get("dataset")
    if not dataset or not isinstance(dataset, str):
        raise ValueError("`dataset` must be the path of a probe dataset parquet (build_probe_dataset.py output).")
    storage = cfg.get("storage")
    if not isinstance(storage, dict):
        raise ValueError("`storage` must be a mapping {backend: hf|local, hf_repo_id, hf_private, local_dir}.")
    make_backend(dict(storage), api=object())   # validates the block without touching the Hub

    layers = cfg.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("`layers` must be a non-empty list of decoder block indices.")
    if any((not isinstance(l, int)) or isinstance(l, bool) or l < 0 for l in layers):
        raise ValueError(f"`layers` must be non-negative integers, got {layers}.")
    if len(set(layers)) != len(layers):
        raise ValueError(f"`layers` has duplicates: {layers}.")
    seq_layers = parse_seq_layers(cfg.get("seq_layers"), layers)

    max_file_gb = cfg.get("max_file_gb", DEFAULT_MAX_FILE_GB)
    if isinstance(max_file_gb, bool) or not isinstance(max_file_gb, (int, float)) or max_file_gb <= 0:
        raise ValueError(f"`max_file_gb` must be a positive number, got {max_file_gb!r}.")
    if max_file_gb > HF_HARD_FILE_LIMIT_GB:
        raise ValueError(
            f"`max_file_gb`={max_file_gb} exceeds the Hub's hard per-file limit of "
            f"{HF_HARD_FILE_LIMIT_GB:g} GB (docs.huggingface.co/hub/storage-limits)."
        )
    if max_file_gb > HF_RECOMMENDED_FILE_GB:
        print(f"[warn] max_file_gb={max_file_gb} is above the Hub's recommended "
              f"{HF_RECOMMENDED_FILE_GB:g} GB per file; uploads of that size resume poorly.")

    seed = cfg.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"`seed` must be an integer, got {seed!r}.")
    workers = cfg.get("upload_workers", DEFAULT_UPLOAD_WORKERS)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError(f"`upload_workers` must be a positive integer, got {workers!r}.")
    staging = cfg.get("staging", "memory")
    if staging not in STAGING_MODES:
        raise ValueError(f"`staging` must be one of {STAGING_MODES}, got {staging!r}.")
    staging_dir = None
    if staging == "disk":
        staging_dir = resolve_data_path(cfg.get("staging_dir") or f"${{DATA_ROOT}}/{DEFAULT_STAGING_SUBDIR}")
    layer_stack_path = cfg.get("layer_stack_path")
    if layer_stack_path is not None and not isinstance(layer_stack_path, str):
        raise ValueError(f"`layer_stack_path` must be a dotted attribute path or null, got {layer_stack_path!r}.")
    store_folder = cfg.get("store_folder")
    if store_folder is not None:
        name = store_folder.strip().strip("/") if isinstance(store_folder, str) else ""
        if not name or "/" in name or name in (".", ".."):
            raise ValueError(
                f"`store_folder` must be a non-empty folder name without '/' (not '.' or '..') or null (= the dataset's name), "
                f"got {store_folder!r}."
            )
        store_folder = name
    hf_model_kwargs = cfg.get("hf_model_kwargs") or {}
    if not isinstance(hf_model_kwargs, dict) or any(not isinstance(k, str) for k in hf_model_kwargs):
        raise ValueError(
            f"`hf_model_kwargs` must be a mapping of from_pretrained keyword arguments (e.g. {{chunk_size: 64}}), "
            f"got {hf_model_kwargs!r}."
        )

    return ProbeCollectConfig(
        model_name=model_name, dataset=resolve_data_path(dataset), storage=dict(storage),
        layers=sorted(layers), seed=seed, max_file_gb=float(max_file_gb), thinking=cfg.get("thinking"),
        layer_stack_path=layer_stack_path, upload_workers=workers, staging=staging, staging_dir=staging_dir,
        store_folder=store_folder, hf_model_kwargs=dict(hf_model_kwargs), seq_layers=seq_layers,
    )


# ---------------------------------------------------------------------------
# Items: the teacher-forced sequence of one rollout and its probe points
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One rollout's forwarded sequence: ``input_ids`` = prompt (+ head) + CoT +
    close delimiter + tail, ``cot_span`` the CoT's ``[start, end)``, ``point_idx``
    the (``pre_cot``, ``last_cot``, ``pre_answer``) positions."""
    rollout_id: str
    input_ids: np.ndarray
    cot_span: tuple[int, int]
    point_idx: tuple[int, int, int]
    commit_form: str
    n_seq_tokens: int
    n_cot_tokens: int
    subject_model: str = ""
    hint_style: str = ""
    case: str = ""
    split: str = ""

    @property
    def n_tokens(self) -> int:
        """What ``plan_parts`` sizes a seq file by: the CoT tokens."""
        return self.n_cot_tokens


@dataclass(frozen=True)
class Skip:
    rollout_id: str
    reason: str


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def build_item(
    row, *, tokenizer, system_prompt: str, enable_thinking, delimiters, max_positions: Optional[int] = None,
) -> Union[Item, Skip]:
    """The :class:`Item` of one dataset row, or a :class:`Skip` with the reason.

    ``input_ids`` = tok(rendered prompt + head) + tok(reasoning) + tok(close) +
    tok(rollout[close_end:commit_start]); ``head`` is the rollout's prefix before
    the CoT when the model wrote its own open delimiter. Each piece is tokenized
    separately, so the spans are exact by construction. Skips: an empty CoT, no
    real close delimiter, no commit after it, a CoT not found before the close, or
    a sequence longer than ``max_positions`` (``None`` = unchecked).
    """
    rollout_id = str(row["rollout_id"])
    rollout = _text(row.get("rollout"))
    cot = _text(row.get("reasoning"))
    if not cot.strip():
        return Skip(rollout_id, SKIP_EMPTY_COT)
    close_end = _find_real_end_think_char(rollout, delimiters=delimiters)
    if close_end is None:
        return Skip(rollout_id, SKIP_NO_CLOSE)
    commit = answer_commit_match(rollout, delimiters=delimiters)
    if commit is None:
        return Skip(rollout_id, SKIP_NO_COMMIT)
    commit_start, commit_form = commit
    cot_at = rollout.find(cot)
    if cot_at < 0 or cot_at + len(cot) > close_end - len(delimiters.close):
        return Skip(rollout_id, SKIP_COT_NOT_IN_ROLLOUT)
    head = rollout[:cot_at] if cot_at > 0 and delimiters.open in rollout[:cot_at] else ""

    prompt_text = render_prompt_text(tokenizer, build_messages(_text(row.get("hinted_prompt")), system_prompt),
                                     enable_thinking)
    prompt_ids = _tokenize(tokenizer, prompt_text + head)
    cot_ids = _tokenize(tokenizer, cot)
    close_ids = _tokenize(tokenizer, delimiters.close)
    tail_ids = _tokenize(tokenizer, rollout[close_end:commit_start])
    if not cot_ids:
        return Skip(rollout_id, SKIP_EMPTY_COT)
    if not prompt_ids:
        return Skip(rollout_id, "empty_prompt")
    n_prompt, n_cot = len(prompt_ids), len(cot_ids)
    ids = prompt_ids + cot_ids + close_ids + tail_ids
    n_seq = len(ids)
    if max_positions is not None and n_seq > int(max_positions):
        return Skip(rollout_id, f"{SKIP_TOO_LONG} ({n_seq} tokens > max_positions {int(max_positions)})")
    return Item(
        rollout_id=rollout_id,
        input_ids=np.asarray(ids, dtype=np.int32),
        cot_span=(n_prompt, n_prompt + n_cot),
        point_idx=(n_prompt - 1, n_prompt + n_cot - 1, n_seq - 1),
        commit_form=commit_form,
        n_seq_tokens=n_seq,
        n_cot_tokens=n_cot,
        subject_model=_text(row.get("subject_model")),
        hint_style=_text(row.get("hint_style")),
        case=_text(row.get("case")),
        split=_text(row.get("split")),
    )


def build_items(rows: pd.DataFrame, *, tokenizer, system_prompt: str, enable_thinking, delimiters,
                max_positions: Optional[int] = None) -> tuple[list[Item], list[Skip]]:
    items: list[Item] = []
    skips: list[Skip] = []
    for row in rows.to_dict("records"):
        out = build_item(row, tokenizer=tokenizer, system_prompt=system_prompt, enable_thinking=enable_thinking,
                         delimiters=delimiters, max_positions=max_positions)
        (items if isinstance(out, Item) else skips).append(out)   # type: ignore[arg-type]
    return items, skips


def resolve_max_positions(model_name: str) -> Optional[int]:
    """The model's ``max_position_embeddings`` from its HF config (text config of a
    multimodal wrapper), without loading weights; ``None`` (with a warning) when
    the config cannot be read or has no such field."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 — the check is advisory
        print(f"[warn] could not read {model_name}'s config for max_position_embeddings ({e}); "
              "sequence lengths are not checked.")
        return None
    value = getattr(get_text_config(config), "max_position_embeddings", None)
    return int(value) if value else None


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def select_rows(df: pd.DataFrame, *, short_name: str, only: Optional[list[str]] = None,
                limit: Optional[int] = None, exclude_ids=()) -> pd.DataFrame:
    """The dataset rows this run collects, in ``rollout_id`` order.

    Every row's ``subject_model`` must equal ``short_name`` (one store per
    model); ``only`` keeps rows whose ``hint_style`` or ``run`` contains any of
    the substrings; ``exclude_ids`` (the store's rows under ``--resume``) are
    dropped; ``limit`` keeps the first N of what remains.
    """
    models = sorted(set(df["subject_model"].astype(str)))
    if models != [short_name]:
        raise ValueError(
            f"The dataset holds rows of subject model(s) {models}, but this run collects "
            f"{short_name!r} (one store per model): build a dataset restricted to it."
        )
    rows = df.sort_values("rollout_id", kind="stable")
    if only:
        styles = rows["hint_style"].astype(str)
        runs = rows["run"].astype(str)
        mask = np.zeros(len(rows), dtype=bool)
        for sub in only:
            mask |= styles.str.contains(sub, regex=False).to_numpy() | runs.str.contains(sub, regex=False).to_numpy()
        rows = rows[mask]
    if exclude_ids:
        rows = rows[~rows["rollout_id"].astype(str).isin(set(map(str, exclude_ids)))]
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"--limit must be a positive integer, got {limit!r}")
        rows = rows.head(limit)
    return rows.reset_index(drop=True)


def existing_part_files(manifest: pd.DataFrame, layers: list[int], seq_layers: Optional[list[int]] = None) -> set[str]:
    """Every part file the manifest's rows name: the points file of every layer
    in ``layers`` and the seq file of every layer in ``seq_layers`` (``None`` =
    ``layers``)."""
    if manifest is None or manifest.empty or "seq_file" not in manifest.columns:
        return set()
    kind_layers = {"seq": list(layers) if seq_layers is None else list(seq_layers), "points": list(layers)}
    files: set[str] = set()
    for column, kind in (("seq_file", "seq"), ("points_file", "points")):
        for n in {part_index(v) for v in manifest[column].tolist()}:
            files.update(STORE_FILES[kind].format(layer=l, part=n) for l in kind_layers[kind])
    return files


def next_part_index(manifest: pd.DataFrame) -> int:
    if manifest is None or manifest.empty or "seq_file" not in manifest.columns:
        return 0
    return max(part_index(v) for v in manifest["seq_file"].tolist()) + 1


# ---------------------------------------------------------------------------
# Store identity and resume
# ---------------------------------------------------------------------------

# The fields an existing store must share with this run before --resume extends it
# (dataset_spec.fingerprint is compared separately; a meta without seq_layers reads as every layer).
IDENTITY_KEYS = ("model_name", "hidden_size", "layers", "seq_layers", "dtype", "thinking_mode", "reasoning_delimiters")


def store_identity(cc: ProbeCollectConfig, *, hidden: int, layer_stack_path: Optional[str], thinking,
                   ds_meta: dict, collected_utc: str) -> dict:
    return {
        "model_name": cc.model_name,
        "short_name": get_model_short_name(cc.model_name),
        "hidden_size": int(hidden),
        "dtype": STORE_DTYPE_NAME,
        "layers": sorted(cc.layers),
        "seq_layers": sorted(cc.seq_layers),
        "layer_stack_path": layer_stack_path,
        "points": list(POINTS),
        "span": SPAN,
        "dataset_spec": dataset_spec(ds_meta),
        "thinking_mode": thinking.mode,
        "reasoning_delimiters": [thinking.delimiters.open, thinking.delimiters.close],
        "max_file_gb": cc.max_file_gb,
        "hf_model_kwargs": dict(cc.hf_model_kwargs),   # informational, not resume identity
        "collected_utc": collected_utc,
    }


def dataset_spec(ds_meta: dict, n_rows: Optional[int] = None) -> dict:
    """The store meta's ``dataset_spec`` block (``n_rows``: the sidecar's count, else ``n_rows``)."""
    rows = ds_meta.get("n_rows")
    return {"name": ds_meta.get("name"), "built_utc": ds_meta.get("built_utc"),
            "fingerprint": ds_meta.get("fingerprint"),
            "n_rows": int(rows) if rows is not None else (None if n_rows is None else int(n_rows))}


def resume_mismatches(existing_meta: dict, identity: dict) -> list[str]:
    """The identity fields on which an existing store differs from this run
    (``dataset_spec.fingerprint`` included); empty for a matching store."""
    existing = dict(existing_meta)
    if existing.get("seq_layers") is None and existing.get("layers") is not None:
        existing["seq_layers"] = existing["layers"]   # no key = seq files for every layer
    out = [k for k in IDENTITY_KEYS
           if existing.get(k) is not None and existing.get(k) != identity.get(k)]
    stored = (existing_meta.get("dataset_spec") or {}).get("fingerprint")
    if stored != identity["dataset_spec"]["fingerprint"]:
        out.append("dataset_spec.fingerprint")
    return out


# ---------------------------------------------------------------------------
# Memory plan
# ---------------------------------------------------------------------------


def memory_plan(cc: ProbeCollectConfig, parts: list[PartPlan], *, n_layers: int,
                n_seq_layers: Optional[int] = None) -> dict:
    """The peak footprint of writing one part: ``n_seq_layers`` seq (``None`` = ``n_layers``)
    + ``n_layers`` points buffers open at once, plus (memory staging) the uploads in
    flight and the one hand-off copy ``PartWriter.finish`` makes."""
    n_seq = n_layers if n_seq_layers is None else int(n_seq_layers)
    max_seq = max((p.seq_bytes for p in parts), default=0)
    max_pts = max((p.points_bytes for p in parts), default=0)
    open_bytes = n_seq * max_seq + n_layers * max_pts
    if cc.staging == "memory":
        peak = open_bytes + (cc.upload_workers + 1) * max_seq
        return {"staging": "memory", "peak_ram_bytes": peak, "peak_disk_bytes": 0}
    return {"staging": "disk", "peak_ram_bytes": 0, "peak_disk_bytes": open_bytes + cc.upload_workers * max_seq}


def check_memory_plan(plan: dict, limit: Optional[int], *, dry_run: bool = False) -> None:
    """Refuse a memory-staged plan whose peak RAM exceeds the container's cap; a dry run only warns."""
    if plan["staging"] != "memory":
        return
    if limit is None:
        print("  [warn] container memory limit unreadable — the memory-staging plan is not checked")
        return
    if plan["peak_ram_bytes"] > limit:
        message = (
            f"staging: memory would peak at ≈ {plan['peak_ram_bytes'] / GB:.1f} GB, above the container's "
            f"{limit / GB:.1f} GB limit ({CGROUP_MEMORY_MAX}). Lower max_file_gb / upload_workers or use staging: disk."
        )
        if dry_run:
            print(f"  [warn] {message} (a real run on this machine would be refused)")
            return
        raise ValueError(message)


# ---------------------------------------------------------------------------
# Plan printout
# ---------------------------------------------------------------------------


def _fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / GB:.2f}"


def print_plan(items: list[Item], skips: list[Skip], parts: list[PartPlan], *, cc: ProbeCollectConfig,
               hidden: int, folder: str, n_resumed: int, rows: pd.DataFrame, spec: Optional[dict] = None) -> None:
    n_layers, n_seq = len(cc.layers), len(cc.seq_layers)
    files_per_part = n_seq + n_layers
    print(f"\nPlan — store folder {folder!r} on {cc.storage.get('backend')}, layers {cc.layers} (points), "
          f"seq_layers={cc.seq_layers}" + (" (= layers)" if cc.seq_layers == cc.layers else " (seq files for these only)")
          + f", hidden {hidden}, max_file_gb={cc.max_file_gb:g}, staging={cc.staging}, hf_model_kwargs={cc.hf_model_kwargs}")
    if spec:
        print(f"  collected from dataset {spec.get('name')!r} ({spec.get('n_rows')} rows, built {spec.get('built_utc')}, "
              f"fingerprint {str(spec.get('fingerprint'))[:12]}…)")
    by_style: dict[str, dict] = {}
    for it in items:
        by_style.setdefault(it.hint_style, {"items": 0, "tokens": 0, "skipped": 0})
        by_style[it.hint_style]["items"] += 1
        by_style[it.hint_style]["tokens"] += it.n_cot_tokens
    style_of = dict(zip(rows["rollout_id"].astype(str), rows["hint_style"].astype(str)))
    for sk in skips:
        by_style.setdefault(style_of.get(sk.rollout_id, "?"), {"items": 0, "tokens": 0, "skipped": 0})["skipped"] += 1
    print(f"  {'hint_style':<24}{'items':>8}{'skipped':>9}{'cot tokens':>12}{'seq GB/layer':>14}")
    for style in sorted(by_style):
        s = by_style[style]
        print(f"  {style:<24}{s['items']:>8}{s['skipped']:>9}{s['tokens']:>12}"
              f"{_fmt_gb(s['tokens'] * hidden * 2):>14}")
    print(f"\n  {'part':<8}{'items':>8}{'cot tokens':>12}{'seq GB/layer':>14}{'points GB/layer':>17}{'files':>7}")
    for p in parts:
        print(f"  {p.index:<8}{len(p.items):>8}{p.n_tokens:>12}{_fmt_gb(p.seq_bytes):>14}"
              f"{_fmt_gb(p.points_bytes):>17}{files_per_part:>7}")
    total_tokens = sum(p.n_tokens for p in parts)
    total_bytes = sum(p.total_bytes(n_layers, n_seq) for p in parts)
    print(f"  {'TOTAL':<8}{len(items):>8}{total_tokens:>12}{_fmt_gb(sum(p.seq_bytes for p in parts)):>14}"
          f"{_fmt_gb(sum(p.points_bytes for p in parts)):>17}{files_per_part * len(parts):>7}"
          f"   all files: {_fmt_gb(total_bytes)} GB ({n_seq} seq layer(s) + {n_layers} points layer(s))")
    if n_resumed:
        print(f"  (resume: {n_resumed} rollouts already in the store, not planned)")
    if skips:
        print(f"\n  {len(skips)} row(s) skipped:")
        for sk in skips:
            print(f"    - {sk.rollout_id}: {sk.reason}")
    else:
        print("\n  0 rows skipped")


# ---------------------------------------------------------------------------
# Manifest rows
# ---------------------------------------------------------------------------


def manifest_row(item: Item, part: PartPlan, *, collected_utc: str) -> dict:
    pre_cot, last_cot, pre_answer = item.point_idx
    return {
        "rollout_id": item.rollout_id,
        "subject_model": item.subject_model,
        "hint_style": item.hint_style,
        "case": item.case,
        "split": item.split,
        "n_cot_tokens": int(item.n_cot_tokens),
        "n_seq_tokens": int(item.n_seq_tokens),
        "pre_cot_idx": int(pre_cot),
        "last_cot_idx": int(last_cot),
        "pre_answer_idx": int(pre_answer),
        "commit_form": item.commit_form,
        "seq_file": part_template("seq", part.index),
        "points_file": part_template("points", part.index),
        "collected_utc": collected_utc,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def dataset_folder(dataset_path, store_folder: Optional[str] = None) -> str:
    """``store_folder`` when set, else the dataset's ``name`` from its ``.meta.json`` sidecar."""
    if store_folder:
        return str(store_folder)
    meta_file, _ = dataset_sidecars(resolve_data_path(dataset_path))
    if not meta_file.exists():
        raise FileNotFoundError(f"{meta_file} is missing — not a dataset written by build_probe_dataset.py")
    name = json.loads(meta_file.read_text()).get("name")
    if not name:
        raise ValueError(f"{meta_file} carries no dataset name")
    return str(name)


def verify_store(backend: StoreBackend, folder: str, dataset_path, *, sample: int = DEFAULT_VERIFY_SAMPLE,
                 seed: int = 0, tokenizer=None, thinking=None) -> dict:
    """Re-read ``sample`` random stored rollouts and check them against the dataset.

    Per rollout the manifest's token counts / point indices / ``commit_form`` must
    equal a fresh :func:`build_item`, every points layer's ``last_cot`` must be
    ``[hidden]``, every seq layer's tensor ``[n_cot_tokens, hidden]`` with
    ``points[last_cot] == seq[-1]``; the store's fingerprint must be the parquet's.
    Returns the report and raises ``ValueError`` listing every mismatch.
    """
    reset_read_caches()   # never serve a stale mmap/header for a file this process just rewrote
    manifest, meta = read_store_manifest(backend, folder)
    df, ds_meta = read_dataset(dataset_path)
    mismatches: list[str] = []
    stored_fp = (meta.get("dataset_spec") or {}).get("fingerprint")
    if stored_fp != ds_meta.get("fingerprint"):
        mismatches.append(f"dataset fingerprint: store {stored_fp!r} != parquet {ds_meta.get('fingerprint')!r}")
    model_name = meta["model_name"]
    if thinking is None:
        thinking = get_thinking_config(model_name, meta.get("thinking") or "on")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ids = sorted(manifest["rollout_id"].astype(str))
    rng = np.random.default_rng(int(seed))
    n = min(int(sample), len(ids))
    picked = [ids[i] for i in sorted(rng.choice(len(ids), size=n, replace=False))] if n else []
    rows = df.set_index(df["rollout_id"].astype(str))
    man = manifest.set_index(manifest["rollout_id"].astype(str))
    seq_store = SequenceStore(backend, folder, manifest, meta)
    point_store = PointStore(backend, folder, manifest, meta)
    hidden = int(meta["hidden_size"])
    checked: list[str] = []
    for rid in picked:
        if rid not in rows.index:
            mismatches.append(f"{rid}: in the store but not in the dataset")
            continue
        m = man.loc[rid]
        item = build_item(rows.loc[rid].to_dict(), tokenizer=tokenizer, system_prompt=thinking.system_prompt,
                          enable_thinking=thinking.enable_thinking, delimiters=thinking.delimiters)
        if isinstance(item, Skip):
            mismatches.append(f"{rid}: stored but rebuilds as a skip ({item.reason})")
            continue
        expected = {"n_cot_tokens": item.n_cot_tokens, "n_seq_tokens": item.n_seq_tokens,
                    "pre_cot_idx": item.point_idx[0], "last_cot_idx": item.point_idx[1],
                    "pre_answer_idx": item.point_idx[2], "commit_form": item.commit_form}
        for col, want in expected.items():
            got = m[col]
            if (str(got) if col == "commit_form" else int(got)) != want:
                mismatches.append(f"{rid}: manifest {col}={got!r} but re-tokenisation gives {want!r}")
        seq = seq_store.get(rid)
        last = point_store.get(rid, "last_cot")
        bad_points = set()
        for layer in point_store.layers:
            if tuple(last[layer].shape) != (hidden,):
                mismatches.append(f"{rid} layer {layer}: last_cot point shape {tuple(last[layer].shape)}")
                bad_points.add(layer)
        for layer in seq_store.layers:
            t = seq[layer]
            if tuple(t.shape) != (item.n_cot_tokens, hidden):
                mismatches.append(f"{rid} layer {layer}: seq shape {tuple(t.shape)} != ({item.n_cot_tokens}, {hidden})")
                continue
            if layer in bad_points:
                continue
            if not torch.equal(last[layer], t[-1]):
                mismatches.append(f"{rid} layer {layer}: points[last_cot] != seq[-1]")
        checked.append(rid)
    report = {"folder": folder, "backend": backend.describe(), "n_manifest": int(len(manifest)),
              "n_checked": len(checked), "rollout_ids": checked, "layers": list(seq_store.layers),
              "points_layers": list(point_store.layers), "mismatches": mismatches, "ok": not mismatches}
    if mismatches:
        raise ValueError(f"verify_store found {len(mismatches)} mismatch(es) in {folder!r}:\n  - "
                         + "\n  - ".join(mismatches))
    print(f"verify_store: {len(checked)} of {len(manifest)} rollouts re-checked — seq layers {report['layers']}, "
          f"points layers {report['points_layers']} — OK")
    return report


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _cuda_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def capture_item(capturer: ActivationCapturer, it: Item, *, model) -> Union[tuple, Skip]:
    """:meth:`ActivationCapturer.capture_points` for one item with CUDA-OOM recovery:
    on ``torch.OutOfMemoryError`` the item is retried after halving every Mamba
    mixer's ``chunk_size`` down to :data:`MIN_MAMBA_CHUNK_SIZE` (restored after the
    item); an item that still fails, or a model without such a mixer, is returned
    as a :class:`Skip` (``cuda_oom (n_seq_tokens=N)``). Retries and skips log at WARNING.
    """
    original = mamba_chunk_sizes(model)
    chunk = min(original.values()) if original else None
    changed = False
    try:
        while True:
            try:
                return capturer.capture_points(it.input_ids, cot_span=it.cot_span, point_idx=it.point_idx)
            except torch.OutOfMemoryError as exc:
                _cuda_empty_cache()
                if chunk is None or chunk <= MIN_MAMBA_CHUNK_SIZE:
                    log.warning("%s: CUDA OOM (n_seq_tokens=%d) at chunk_size %s — skipped: %s",
                                it.rollout_id, it.n_seq_tokens, chunk, str(exc).splitlines()[0][:200])
                    return Skip(it.rollout_id, f"{SKIP_CUDA_OOM} (n_seq_tokens={it.n_seq_tokens})")
                chunk = max(MIN_MAMBA_CHUNK_SIZE, chunk // 2)
                n = set_mamba_chunk_size(model, chunk)
                changed = True
                log.warning("%s: CUDA OOM (n_seq_tokens=%d) — retrying with Mamba chunk_size %d on %d module(s)",
                            it.rollout_id, it.n_seq_tokens, chunk, n)
    finally:
        if changed:
            modules = dict(model.named_modules())
            for name, value in original.items():
                modules[name].chunk_size = value
            _cuda_empty_cache()


def run(
    cc: ProbeCollectConfig,
    *,
    dry_run: bool = False,
    only: Optional[list[str]] = None,
    resume: bool = False,
    force: bool = False,
    limit: Optional[int] = None,
    tokenizer=None,
    model_loader: Optional[Callable] = None,
    api=None,
    max_positions: Optional[int] = None,
    verify_sample: int = DEFAULT_VERIFY_SAMPLE,
) -> dict:
    thinking = get_thinking_config(cc.model_name, cc.thinking)
    if not thinking.enabled:
        raise ValueError(
            f"collect_probe_activations requires thinking, but it is off for {cc.model_name}: "
            "the stored span is the chain of thought between the reasoning delimiters."
        )
    if resume and force:
        raise ValueError("--resume and --force are exclusive")
    n_layers, n_seq = len(cc.layers), len(cc.seq_layers)
    files_per_part = n_seq + n_layers
    hidden = get_model_config(cc.model_name)["hidden_size"]
    if not hidden:
        raise ValueError(
            f"{cc.model_name} has no `hidden_size` in MODEL_CONFIGS; register it so files "
            "can be planned before the model loads."
        )
    hidden = int(hidden)
    short_name = get_model_short_name(cc.model_name)
    collected_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- Backend first: the token/namespace check fails before any data is read.
    backend = make_backend(cc.storage, api=api)
    backend.preflight(dry_run=dry_run)

    # --- Dataset ------------------------------------------------------------
    print(f"Reading dataset {cc.dataset} …")
    df, ds_meta = read_dataset(cc.dataset)
    ds_meta = {**ds_meta, "n_rows": int(len(df))} if ds_meta.get("n_rows") is None else ds_meta
    spec_name = str(ds_meta["name"])
    folder = cc.store_folder or spec_name
    print(f"  {len(df)} rows, spec {spec_name!r}, built {ds_meta.get('built_utc')}, "
          f"fingerprint {str(ds_meta.get('fingerprint'))[:12]}…")
    print(f"  store folder {folder!r}" + (" (store_folder)" if cc.store_folder else " (= the dataset's name)"))
    identity = store_identity(
        cc, hidden=hidden, layer_stack_path=cc.layer_stack_path or get_model_config(cc.model_name)["layer_stack_path"],
        thinking=thinking, ds_meta=ds_meta, collected_utc=collected_utc,
    )

    # --- Existing store ------------------------------------------------------
    files, existing_manifest, existing_meta = load_store_state(backend, folder, columns=STORE_MANIFEST_COLUMNS)
    existing_manifest = existing_manifest.reindex(columns=STORE_MANIFEST_COLUMNS)
    has_store = bool(existing_meta) or not existing_manifest.empty
    stored_ids: set[str] = set()
    if has_store:
        print(f"Store {backend.describe()}/{folder}: {len(existing_manifest)} manifest rows, "
              f"{sum(1 for f in files if is_store_file(f))} part files")
        if force:
            print("  [force] the folder is rewritten: existing rows dropped, stale parts deleted after the first part")
            existing_manifest, existing_meta = pd.DataFrame(columns=STORE_MANIFEST_COLUMNS), {}
        elif resume:
            mismatched = resume_mismatches(existing_meta, identity)
            if mismatched:
                raise ValueError(
                    f"--resume refused: the store differs on {mismatched} "
                    f"(store dataset_spec={existing_meta.get('dataset_spec')}, layers={existing_meta.get('layers')}, "
                    f"seq_layers={existing_meta.get('seq_layers', 'absent = every layer')}; "
                    f"this run {identity['dataset_spec']}, layers={identity['layers']}, seq_layers={identity['seq_layers']}). "
                    "Rebuild the store with --force or point storage at a new root."
                )
            stored_ids = set(existing_manifest["rollout_id"].astype(str))
        else:
            raise ValueError(
                f"The store already holds {len(existing_manifest)} rollouts under {folder!r} on "
                f"{backend.describe()}. Pass --resume to add the missing rollouts (same dataset fingerprint) "
                "or --force to rewrite the folder."
            )

    # --- Rows and items ---------------------------------------------------------
    rows = select_rows(df, short_name=short_name, only=only, limit=limit, exclude_ids=stored_ids)
    n_resumed = len(stored_ids & set(df["rollout_id"].astype(str)))
    del df
    print(f"  {len(rows)} rows selected" + (f" (--only {only})" if only else "")
          + (f" (--limit {limit})" if limit else "") + (f"; {n_resumed} already stored" if resume else ""))
    if rows.empty:
        print("Nothing to collect.")
        return {"dry_run": dry_run, "folder": folder, "n_rows": 0, "n_items": 0, "n_skipped": 0,
                "skipped": {}, "n_resumed": n_resumed, "parts": [], "failed_parts": [], "uploaded_bytes": 0}

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cc.model_name, trust_remote_code=True)
    if max_positions is None:
        max_positions = resolve_max_positions(cc.model_name)
    print(f"Building {len(rows)} items (max_positions={max_positions}) …")
    items, skips = build_items(
        rows, tokenizer=tokenizer, system_prompt=thinking.system_prompt,
        enable_thinking=thinking.enable_thinking, delimiters=thinking.delimiters, max_positions=max_positions,
    )
    skipped = {**(existing_meta.get("skipped") or {}), **{sk.rollout_id: sk.reason for sk in skips}}

    # --- Parts ------------------------------------------------------------------
    parts = plan_parts(items, hidden=hidden, n_layers=n_layers, max_bytes=int(cc.max_file_gb * GB))
    offset = next_part_index(existing_manifest)
    for p in parts:
        p.index += offset
    print_plan(items, skips, parts, cc=cc, hidden=hidden, folder=folder, n_resumed=n_resumed, rows=rows,
               spec=identity["dataset_spec"])
    plan_files = {f for p in parts for f in p.files(cc.layers, seq_layers=cc.seq_layers)}
    keep_files = plan_files | existing_part_files(existing_manifest, cc.layers, cc.seq_layers)

    mem = memory_plan(cc, parts, n_layers=n_layers, n_seq_layers=n_seq)
    writer = StoreWriter(backend, folder, manifest=existing_manifest, meta={**existing_meta, **identity},
                         upload_workers=cc.upload_workers, staging=cc.staging, staging_dir=cc.staging_dir,
                         max_file_gb=cc.max_file_gb)
    print("  " + writer.staging_note(files_per_part=n_seq))   # only the seq files are full-size
    if mem["staging"] == "memory":
        print(f"  this plan: peak RAM ≈ {mem['peak_ram_bytes'] / GB:.1f} GB "
              f"({n_seq} seq + {n_layers} points buffers open per part, {cc.upload_workers} uploads in flight, one hand-off copy)")
    else:
        print(f"  this plan: peak transient disk ≈ {mem['peak_disk_bytes'] / GB:.1f} GB under {cc.staging_dir}")
    check_memory_plan(mem, cgroup_memory_limit_bytes(), dry_run=dry_run)

    summary = {
        "dry_run": dry_run, "folder": folder, "n_rows": int(len(rows)), "n_items": len(items),
        "n_skipped": len(skips), "skipped": {sk.rollout_id: sk.reason for sk in skips}, "n_resumed": n_resumed,
        "parts": [{"index": p.index, "n_items": len(p.items), "n_tokens": p.n_tokens,
                   "seq_bytes": p.seq_bytes, "points_bytes": p.points_bytes} for p in parts],
        "identity": identity, "memory_plan": mem,
    }
    if dry_run or not items:
        writer.close()
        print("\nDry run — nothing written." if dry_run else "\nNo item to collect (every row skipped).")
        return {**summary, "failed_parts": [], "uploaded_bytes": 0}

    # --- Model ----------------------------------------------------------------
    print(f"\nLoading {cc.model_name} …")
    model, _ = (model_loader or load_model_hf)(cc.model_name, model_kwargs=cc.hf_model_kwargs)
    model.eval()
    chunk_sizes = mamba_chunk_sizes(model)
    if chunk_sizes:
        print(f"  Mamba chunk_size {sorted(set(chunk_sizes.values()))} on {len(chunk_sizes)} module(s); "
              f"a CUDA OOM halves it down to {MIN_MAMBA_CHUNK_SIZE} before the rollout is skipped")
    layers, layer_path = resolve_layer_stack(model, path=identity["layer_stack_path"])
    assert_layer_depth(model, cc.model_name, layers, layer_path)
    out_of_range = [i for i in cc.layers if i >= len(layers)]
    if out_of_range:
        raise IndexError(f"Configured layers {out_of_range} are out of range for {cc.model_name}, "
                         f"which has {len(layers)} layers. Rescale the `layers` list.")
    expected_hidden = resolve_hidden_size(model, cc.model_name)
    if expected_hidden is not None and int(expected_hidden) != hidden:
        raise RuntimeError(f"{cc.model_name}: registry hidden_size {hidden} != model's {expected_hidden}")
    identity["layer_stack_path"] = layer_path
    writer.meta.update(identity)
    print(f"Layer stack: {layer_path} ({len(layers)} layers), hidden={expected_hidden}; capturing {cc.layers} "
          f"(seq files for {cc.seq_layers}). "
          f"Thinking mode={thinking.mode}, delimiters={thinking.delimiters.open!r}/{thinking.delimiters.close!r}")
    capturer = ActivationCapturer(model, layers, cc.layers, expected_hidden=expected_hidden)

    # --- Collect ---------------------------------------------------------------
    from tqdm import tqdm

    leftovers = writer.prepare_staging()
    if leftovers:
        print(f"  [warn] {len(leftovers)} leftover staged file(s) in {cc.staging_dir} "
              f"({sum(f.stat().st_size for f in leftovers) / GB:.1f} GB), e.g. {leftovers[0].name} — "
              "from an earlier interrupted run; delete them if no other collection is running.")
    kept_existing = existing_part_files(existing_manifest, cc.layers, cc.seq_layers)
    ok_files: set[str] = set(kept_existing)
    # A file that failed earlier but has since been written is no longer a failure.
    failed_files: list[str] = list(existing_meta.get("failed_files") or [])
    writer.meta.update({"skipped": skipped, "n_skipped": len(skipped), "files": sorted(ok_files),
                        "failed_files": sorted(set(failed_files) - ok_files), "n_rollouts": int(len(existing_manifest))})

    def update_meta(meta: dict, result: GroupResult) -> None:
        ok_files.update(result.ok_files)
        failed_files.extend(result.failed_files)
        meta["files"] = sorted(ok_files)
        meta["failed_files"] = sorted(set(failed_files) - ok_files)
        meta["n_rollouts"] = int(len(writer.manifest))
        meta["skipped"] = dict(skipped)
        meta["n_skipped"] = len(skipped)
        meta["collected_utc"] = collected_utc

    n_done = 0
    oom_skips: list[Skip] = []
    open_writers: dict[tuple[str, int], PartWriter] = {}
    try:
        for part in parts:
            key = f"part{part.index:02d}"
            base_meta = {"model_name": cc.model_name, "dataset_spec": spec_name, "store_folder": folder,
                         "layers": json.dumps(cc.layers), "seq_layers": json.dumps(cc.seq_layers),
                         "hidden_size": str(hidden), "dtype": STORE_DTYPE_NAME, "part": key,
                         "n_items": str(len(part.items)), "created_utc": collected_utc}
            for layer in cc.seq_layers:
                open_writers[("seq", layer)] = writer.new_part_writer(
                    part.seq_file(layer), part.entries("seq", hidden=hidden),
                    {**base_meta, "layer": str(layer), "kind": "seq", "span": SPAN})
            for layer in cc.layers:
                open_writers[("points", layer)] = writer.new_part_writer(
                    part.points_file(layer), part.entries("points", hidden=hidden),
                    {**base_meta, "layer": str(layer), "kind": "points", "points": json.dumps(list(POINTS))})
            rows_out: list[dict] = []
            for it in tqdm(part.items, desc=key, unit="rollout", leave=False):
                captured = capture_item(capturer, it, model=model)
                if isinstance(captured, Skip):
                    # The part's header is fixed by the plan: the slot stays zero-filled.
                    for pw in open_writers.values():
                        pw.skip(it.rollout_id)
                    oom_skips.append(captured)
                    skipped[captured.rollout_id] = captured.reason
                    continue
                seq, pts = captured
                for layer in cc.seq_layers:
                    open_writers[("seq", layer)].add(it.rollout_id, seq[layer])
                for layer in cc.layers:
                    open_writers[("points", layer)].add(it.rollout_id, pts[layer])
                del seq, pts
                rows_out.append(manifest_row(it, part, collected_utc=collected_utc))
                n_done += 1
            for (kind, layer), pw in list(open_writers.items()):
                filename = part.seq_file(layer) if kind == "seq" else part.points_file(layer)
                payload = pw.finish()
                del open_writers[(kind, layer)]
                writer.submit_file(key, filename, payload)
                del payload
            writer.finalize_group(key, rows=rows_out, keep=keep_files, stale_filter=is_store_file,
                                  meta_update=update_meta)
            print(f"[{key}] {len(part.items)} rollouts captured → {files_per_part} files")
    finally:
        for pw in open_writers.values():
            pw.abort()
        print("Waiting for pending uploads …")
        writer.close()

    print(f"\nDone. rollouts={n_done} uploaded={writer.uploaded_bytes / GB:.2f} GB "
          f"failed_parts={len(writer.failed_parts)} skipped={len(skips) + len(oom_skips)} "
          f"(cuda_oom={len(oom_skips)})")
    if oom_skips:
        print(f"\n!! {len(oom_skips)} rollout(s) skipped for CUDA OOM at the chunk-size floor "
              f"(listed in the meta's `skipped`, never in the manifest):")
        for sk in oom_skips:
            print(f"  - {sk.rollout_id}: {sk.reason}")
    if writer.failed_parts:
        print("\n!! Some files failed to upload (their part's rows are NOT in the manifest); rerun with --resume:")
        for f, err in writer.failed_parts:
            print(f"  - {f}: {err}")
    summary.update({
        "n_done": n_done, "uploaded_bytes": writer.uploaded_bytes, "failed_parts": writer.failed_parts,
        "n_skipped": len(skips) + len(oom_skips), "n_cuda_oom": len(oom_skips),
        "skipped": {**summary["skipped"], **{sk.rollout_id: sk.reason for sk in oom_skips}},
    })

    print("\nVerifying the store …")
    summary["verify"] = verify_store(backend, folder, cc.dataset, sample=verify_sample, seed=cc.seed,
                                     tokenizer=tokenizer, thinking=thinking)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read the dataset, tokenize and print the plan (parts, sizes, skipped rows); "
                             "no model load, nothing written.")
    parser.add_argument("--only", action="append", default=None, metavar="SUBSTR",
                        help="Only rows whose hint_style or run contains SUBSTR (repeatable).")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only the first N selected rows in rollout_id order (smoke runs).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rollouts the store already holds (same dataset fingerprint only).")
    parser.add_argument("--force", action="store_true",
                        help="Rewrite the store folder: drop its manifest rows and delete its stale parts.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run verify_store on the existing store and exit.")
    parser.add_argument("--sample", type=int, default=DEFAULT_VERIFY_SAMPLE,
                        help=f"Rollouts verify_store re-checks (default {DEFAULT_VERIFY_SAMPLE}).")
    args = parser.parse_args(argv)

    cc = parse_config(load_config(args.config))
    if args.verify_only:
        backend = make_backend(cc.storage)
        backend.preflight(dry_run=True)
        folder = dataset_folder(cc.dataset, cc.store_folder)
        thinking = get_thinking_config(cc.model_name, cc.thinking)
        report = verify_store(backend, folder, cc.dataset, sample=args.sample, seed=cc.seed, thinking=thinking)
        print(json.dumps({k: v for k, v in report.items() if k != "rollout_ids"}, indent=2))
        return 0
    summary = run(cc, dry_run=args.dry_run, only=args.only, resume=args.resume, force=args.force,
                  limit=args.limit, verify_sample=args.sample)
    return 1 if summary.get("failed_parts") else 0


if __name__ == "__main__":
    raise SystemExit(main())
