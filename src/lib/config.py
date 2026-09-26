"""Shared config loading helpers: YAML loading with single-level ``extends:``
inheritance, plus readers for the shared ``inference:`` and sampling knobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


def utc_now() -> str:
    """ISO-8601 UTC timestamp at second precision (sidecars, run records)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` with ``override`` applied. Nested dicts merge recursively;
    scalars and lists replace wholesale."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> dict:
    """Load a YAML config, deep-merging it over the ``extends:`` base (resolved
    relative to the config's directory) when one is declared. Inheritance is one
    level only: a base that itself declares ``extends`` raises ``ValueError``."""
    path = Path(path)
    cfg = _load_yaml(path)
    base_ref = cfg.pop("extends", None)
    if base_ref is None:
        return cfg

    base_path = path.parent / base_ref
    if not base_path.exists():
        raise FileNotFoundError(
            f"{path} extends '{base_ref}', but {base_path} does not exist."
        )
    base = _load_yaml(base_path)
    if "extends" in base:
        raise ValueError(
            f"{base_path} itself declares 'extends'; inheritance is one level only."
        )
    return _deep_merge(base, cfg)


# Truncation shared by every sampling stage. A baseline sidecar without the
# fields was sampled at the LEGACY_* values (vLLM's defaults: no truncation).
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
LEGACY_TOP_P = 1.0
LEGACY_TOP_K = -1


@dataclass(frozen=True)
class SamplingConfig:
    """Truncation knobs of a sampling stage."""

    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K


def get_sampling_config(cfg: dict) -> SamplingConfig:
    """``top_p`` / ``top_k`` from a config's top-level keys, else the shared
    defaults (``null`` counts as unset). Raises ``ValueError`` on a ``top_p``
    outside ``(0, 1]`` or a ``top_k`` that is neither -1 (off) nor positive."""
    top_p = cfg.get("top_p")
    top_k = cfg.get("top_k")
    if isinstance(top_p, bool) or isinstance(top_k, bool):
        raise ValueError(f"top_p / top_k must be numbers, not booleans (got {top_p!r} / {top_k!r}).")
    top_p = DEFAULT_TOP_P if top_p is None else float(top_p)
    if top_k is not None and float(top_k) != int(float(top_k)):
        raise ValueError(f"top_k must be an integer (got {top_k!r}).")
    top_k = DEFAULT_TOP_K if top_k is None else int(float(top_k))
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0.0, 1.0] (got {top_p}).")
    if top_k < -1 or top_k == 0:
        raise ValueError(f"top_k must be -1 (off) or a positive int (got {top_k}).")
    return SamplingConfig(top_p=top_p, top_k=top_k)


@dataclass(frozen=True)
class InferenceConfig:
    """Shared inference knobs. ``max_tokens`` is the generation budget,
    ``max_model_len`` the vLLM context window (unused on the HF path), and
    ``max_num_seqs`` caps vLLM's concurrent sequences (``None`` = vLLM's
    default; hybrid Mamba models need it at or below their cache block count)."""

    max_model_len: int = 16384
    max_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_num_seqs: int | None = None


def get_inference_config(
    cfg: dict, *, defaults: InferenceConfig = InferenceConfig()
) -> InferenceConfig:
    """Read the inference knobs from the ``inference:`` block, else from a
    legacy top-level key (``max_new_tokens`` aliases ``max_tokens``), else
    ``defaults``. ``null`` counts as unset on either side. Setting a knob both
    in the block and at the top level raises ``ValueError`` (the block would
    silently shadow the flat key)."""
    inf = dict(cfg.get("inference") or {})

    def pick(key: str, *aliases: str, default):
        inf_val = inf.get(key)
        top_keys = [k for k in (key, *aliases) if cfg.get(k) is not None]
        if inf_val is not None and top_keys:
            raise ValueError(
                f"Config sets both `inference.{key}` and top-level `{top_keys[0]}`. "
                f"Remove the top-level key and set `{key}` under `inference:` — the "
                f"nested block otherwise silently shadows it."
            )
        if inf_val is not None:
            return inf_val
        if top_keys:
            return cfg[top_keys[0]]
        return default

    return InferenceConfig(
        max_model_len=int(pick("max_model_len", default=defaults.max_model_len)),
        max_tokens=int(pick("max_tokens", "max_new_tokens", default=defaults.max_tokens)),
        gpu_memory_utilization=float(
            pick("gpu_memory_utilization", default=defaults.gpu_memory_utilization)
        ),
        tensor_parallel_size=int(
            pick("tensor_parallel_size", default=defaults.tensor_parallel_size)
        ),
        max_num_seqs=(
            int(seqs)
            if (seqs := pick("max_num_seqs", default=defaults.max_num_seqs)) is not None
            else None
        ),
    )
