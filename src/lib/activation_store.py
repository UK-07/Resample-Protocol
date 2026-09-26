"""Activation store: the on-disk / on-Hub layout of per-rollout activations, its
writer (model introspection, sequence assembly, capture, safetensors parts,
backends, background uploads), its validated reader (manifest checks, layer
resolution, ``rollout_id``-keyed tensor access, torch datasets) and the mirror
that copies a store folder to a local disk for training.

Store layout (root = ``<hf repo>/<folder>/`` or ``<local_dir>/<folder>/``)::

    manifest.csv                            one row per stored rollout
    manifest.meta.json                      identity + collection provenance
    seq_layer_<L>_part<NN>.safetensors      tensor key = rollout_id, bf16 [n_cot_tokens, hidden]
    points_layer_<L>_part<NN>.safetensors   tensor key = rollout_id, bf16 [len(POINTS), hidden]
"""

from __future__ import annotations

import abc
import io
import json
import os
import re
import shutil
import struct
import threading
import time
import uuid
import zlib
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from src.lib import hf_streaming
from src.lib.model_utils import build_chat_messages, get_model_config, get_text_config, template_kwargs
from src.lib.paths import resolve_data_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GB = 1024 ** 3
# Hugging Face Hub per-file limits: 500 GB hard, < 200 GB recommended.
HF_HARD_FILE_LIMIT_GB = 500.0
HF_RECOMMENDED_FILE_GB = 200.0
DEFAULT_MAX_FILE_GB = 20.0
DEFAULT_UPLOAD_WORKERS = 3
STAGING_MODES = ("memory", "disk")
SAFETENSORS_DTYPE = "BF16"
CGROUP_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")

STORE_DTYPE = torch.bfloat16
STORE_DTYPE_NAME = "bfloat16"
BYTES_PER_ELEMENT = 2

# Probe points, in the order they are stacked in the stored ``[4, hidden]`` tensor.
POINTS = ("pre_cot", "last_cot", "pre_answer", "mean_cot")

STORE_MANIFEST_NAME = "manifest.csv"
STORE_MANIFEST_META_NAME = "manifest.meta.json"
STORE_MANIFEST_COLUMNS = [
    "rollout_id", "subject_model", "hint_style", "case", "split",
    "n_cot_tokens", "n_seq_tokens", "pre_cot_idx", "last_cot_idx", "pre_answer_idx",
    "commit_form", "seq_file", "points_file", "collected_utc",
]
BACKENDS = ("hf", "local")
STORAGE_CONFIG_KEYS = {"backend", "hf_repo_id", "hf_private", "local_dir", "revision"}


def cgroup_memory_limit_bytes(path: Path = CGROUP_MEMORY_MAX) -> Optional[int]:
    """The container's memory cap in bytes; ``None`` when unreadable or unlimited."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Store file naming
# ---------------------------------------------------------------------------

STORE_FILE_KINDS = ("seq", "points")
STORE_FILES = {kind: kind + "_layer_{layer}_part{part:02d}.safetensors" for kind in STORE_FILE_KINDS}
STORE_FILE_COLUMNS = {"seq": "seq_file", "points": "points_file"}   # the manifest column naming the part
# Meta keys a mirror writes: per file kind, the layers whose files the copy holds.
MIRRORED_LAYERS_KEYS = {"seq": "mirrored_layers", "points": "mirrored_points_layers"}
# Meta key naming the layers a store holds seq files for (a subset of ``layers``;
# absent = every layer). ``STORED_LAYERS_KEYS[kind]`` is the source's layer list per kind.
SEQ_LAYERS_KEY = "seq_layers"
STORED_LAYERS_KEYS = {"seq": SEQ_LAYERS_KEY, "points": "layers"}
MIRROR_SOURCE_KEY = "mirror_source"
DEFAULT_MIRROR_WORKERS = 4
STORE_FILE_RE = re.compile(r"^(?P<kind>seq|points)_layer_(?P<layer>\d+)_part(?P<part>\d+)\.safetensors$")


def seq_file(layer: int, part: int) -> str:
    return STORE_FILES["seq"].format(layer=int(layer), part=int(part))


def points_file(layer: int, part: int) -> str:
    return STORE_FILES["points"].format(layer=int(layer), part=int(part))


def is_store_file(name: str) -> bool:
    """Whether ``name`` is a ``seq_*``/``points_*`` part file (folder prefix ignored)."""
    return STORE_FILE_RE.match(os.path.basename(str(name))) is not None


# A manifest's ``seq_file`` / ``points_file`` cell names the rollout's *part*, not one
# file; the canonical cell is the layer template ``<kind>_layer_{L}_partNN``.
STORE_PART_TEMPLATES = {kind: kind + "_layer_{{L}}_part{part:02d}" for kind in STORE_FILE_KINDS}
_PART_INDEX_RE = re.compile(r"part(?P<part>\d+)(?:\.safetensors)?$")


def part_template(kind: str, part: int) -> str:
    """The canonical manifest cell ``<kind>_layer_{L}_part<NN>`` (``{L}`` literal)."""
    if kind not in STORE_FILE_KINDS:
        raise ValueError(f"kind must be one of {STORE_FILE_KINDS}, got {kind!r}")
    return STORE_PART_TEMPLATES[kind].format(part=int(part))


def part_index(value) -> int:
    """The part index a manifest cell names: the canonical template, a concrete file
    name, a bare ``partNN`` or an integer (also as a digit string); anything else raises."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"cannot read a part index from {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) == int(value):
            return int(value)
        raise ValueError(f"cannot read a part index from {value!r}")
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    m = _PART_INDEX_RE.search(text)
    if m is None:
        raise ValueError(f"cannot read a part index from {value!r} (expected e.g. {part_template('seq', 0)!r})")
    return int(m.group("part"))


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from string parts (stable across processes)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

# Dotted paths to the decoder block stack, tried in order.
LAYER_STACK_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.text_model.layers",
    "model.model.layers",
    "model.decoder.layers",
    "transformer.h",
)


def _getattr_path(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _is_layer_stack(candidate) -> bool:
    try:
        return len(candidate) > 0 and isinstance(candidate[0], torch.nn.Module)
    except (TypeError, IndexError, KeyError):
        return False


def resolve_layer_stack(model, *, path: str | None = None) -> tuple[object, str]:
    """Locate the decoder block stack; returns ``(layers, resolved_path)``.

    ``path`` short-circuits the search and raises if it does not resolve; otherwise
    the first :data:`LAYER_STACK_PATHS` entry that is a non-empty module list wins.
    """
    if path:
        found = _getattr_path(model, path)
        if not _is_layer_stack(found):
            raise RuntimeError(
                f"layer_stack_path {path!r} does not resolve to a non-empty "
                f"module list on {type(model).__name__}."
            )
        return found, path

    for candidate_path in LAYER_STACK_PATHS:
        found = _getattr_path(model, candidate_path)
        if _is_layer_stack(found):
            return found, candidate_path

    named_modules = getattr(model, "named_modules", None)
    discovered = [
        f"{name} ({len(mod)})"
        for name, mod in (named_modules() if callable(named_modules) else [])
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8
    ]
    named_children = getattr(model, "named_children", None)
    children = [n for n, _ in (named_children() if callable(named_children) else [])]
    raise RuntimeError(
        f"Could not locate the decoder layer stack on {type(model).__name__}.\n"
        f"  tried: {', '.join(LAYER_STACK_PATHS)}\n"
        f"  children: {children}\n"
        f"  candidate ModuleLists: {discovered or 'none'}\n"
        "Set 'layer_stack_path' on the model's MODEL_CONFIGS entry to the right "
        "one (not auto-selected: picking a vision tower would silently produce "
        "mislabeled activations)."
    )


def resolve_hidden_size(model, model_name: str) -> Optional[int]:
    """Expected residual width: the registry hint, else the model's text config."""
    registered = get_model_config(model_name)["hidden_size"]
    if registered:
        return int(registered)
    config = getattr(model, "config", None)
    size = getattr(get_text_config(config), "hidden_size", None) if config else None
    return int(size) if size else None


def assert_layer_depth(model, model_name: str, layers, layer_path: str) -> None:
    """Fail fast if the resolved stack disagrees with the model's declared depth."""
    n_found = len(layers)
    config = getattr(model, "config", None)
    declared = (
        getattr(get_text_config(config), "num_hidden_layers", None) if config else None
    )
    if declared is not None and int(declared) != n_found:
        raise RuntimeError(
            f"Layer stack at {layer_path!r} has {n_found} layers but "
            f"{model_name} declares num_hidden_layers={declared}. The resolver "
            "most likely found the wrong stack — set 'layer_stack_path' explicitly."
        )
    registered = get_model_config(model_name)["n_layers"]
    if registered and int(registered) != n_found:
        raise RuntimeError(
            f"Layer stack at {layer_path!r} has {n_found} layers but "
            f"MODEL_CONFIGS[{model_name!r}]['n_layers'] is {registered}."
        )


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------


def parse_hinted_prompt(hinted_prompt: str):
    """A stored hinted prompt: plain user text, or a JSON message list (multi-turn hints)."""
    text = str(hinted_prompt)
    if text.lstrip().startswith("["):
        try:
            messages = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(messages, list) and all(isinstance(m, dict) and "role" in m for m in messages):
            return messages
    return text


def build_messages(hinted_prompt: str, system_prompt: str) -> list[dict]:
    """Rebuild the chat the rollout was generated from (mirrors build_hinted_items)."""
    parsed = parse_hinted_prompt(hinted_prompt)
    if isinstance(parsed, list):
        return [{"role": "system", "content": system_prompt}] + parsed
    return build_chat_messages(parsed, system_prompt)


def render_prompt_text(tokenizer, messages: list[dict], enable_thinking) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs(enable_thinking)
    )


def _tokenize(tokenizer, text: str) -> list[int]:
    if not text:
        return []
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# Forward pass with hooks
# ---------------------------------------------------------------------------


def mamba_chunk_sizes(model) -> dict[str, int]:
    """``{module name: chunk_size}`` for every submodule with an integer ``chunk_size``
    attribute (the Mamba-2 mixers of a hybrid model); empty otherwise."""
    out: dict[str, int] = {}
    for name, module in model.named_modules():
        value = getattr(module, "chunk_size", None)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    return out


def set_mamba_chunk_size(model, value: int) -> int:
    """Set ``chunk_size`` on every module of :func:`mamba_chunk_sizes`; returns the count."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {value!r}")
    names = mamba_chunk_sizes(model)
    modules = dict(model.named_modules())
    for name in names:
        modules[name].chunk_size = value
    return len(names)


class ActivationCapturer:
    """Runs single sequences through the model and returns the residual stream of the
    configured decoder blocks (bf16, on CPU), sliced inside the hook on the device."""

    def __init__(self, model, layers, layer_indices: list[int], *, expected_hidden: Optional[int]):
        self.model = model
        self.layers = layers
        self.layer_indices = list(layer_indices)
        self.expected_hidden = expected_hidden
        self._logits_to_keep_ok: Optional[bool] = None

    def _forward(self, ids: torch.Tensor) -> None:
        # `logits_to_keep=1` skips the LM head on all but the last position (full
        # logits over a long CoT would be several GB).
        if self._logits_to_keep_ok is not False:
            try:
                self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
                self._logits_to_keep_ok = True
                return
            except TypeError as e:
                if "logits_to_keep" not in str(e):
                    raise
                self._logits_to_keep_ok = False
        self.model(input_ids=ids, use_cache=False)

    def _run(self, input_ids: np.ndarray, extract: Callable[[torch.Tensor], object]) -> dict[int, object]:
        """One forward pass; ``extract(hidden[seq, hidden])`` is applied to every hooked
        layer's output (batch element 0) and the results returned per layer."""
        store: dict[int, object] = {}

        def make_hook(idx: int):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                store[idx] = extract(hidden[0])
            return hook

        handles = [self.layers[i].register_forward_hook(make_hook(i)) for i in self.layer_indices]
        try:
            device = getattr(self.model, "device", None) or next(self.model.parameters()).device
            ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=device).unsqueeze(0)
            with torch.inference_mode():
                self._forward(ids)
        finally:
            for h in handles:
                h.remove()

        missing = [i for i in self.layer_indices if i not in store]
        if missing:
            raise RuntimeError(f"Forward hooks on layers {missing} never fired — wrong layer stack?")
        return store

    def _check(self, store: dict[int, torch.Tensor], *, rows: int, what: str) -> None:
        for idx, t in store.items():
            if t.shape[0] != rows:
                raise RuntimeError(f"Layer {idx}: captured {t.shape[0]} {what}, expected {rows}.")
            if self.expected_hidden is not None and t.shape[-1] != self.expected_hidden:
                raise RuntimeError(
                    f"Layer {idx} produced hidden width {t.shape[-1]}, expected "
                    f"{self.expected_hidden}. The captured module is not the residual "
                    "stream of the text decoder — check 'layer_stack_path'."
                )

    def capture_points(
        self, input_ids: np.ndarray, *, cot_span: tuple[int, int], point_idx: tuple[int, int, int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """One forward pass → ``(seq, points)`` per layer: ``seq[layer]`` is
        ``Tensor[n_cot, hidden]`` over ``cot_span``, ``points[layer]`` is
        ``Tensor[len(POINTS), hidden]`` in :data:`POINTS` order (``mean_cot`` computed in
        float32 before the bf16 cast). ``last_cot`` must be the span's last position, so
        ``points[layer][1] == seq[layer][-1]`` exactly; every index must lie inside the sequence.
        """
        start, end = int(cot_span[0]), int(cot_span[1])
        n = len(input_ids)
        pre_cot, last_cot, pre_answer = (int(i) for i in point_idx)
        if not (0 <= start < end <= n):
            raise ValueError(f"cot_span {cot_span!r} is not a non-empty span inside {n} tokens")
        for name, idx in zip(POINTS[:3], (pre_cot, last_cot, pre_answer)):
            if not (0 <= idx < n):
                raise ValueError(f"{name} index {idx} is outside the {n}-token sequence")
        if last_cot != end - 1:
            raise ValueError(f"last_cot index {last_cot} must be the CoT span's last position {end - 1}")

        def extract(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = h.detach()
            seq = h[start:end, :]
            mean = seq.float().mean(dim=0)
            pts = torch.stack([
                h[pre_cot].to(STORE_DTYPE), h[last_cot].to(STORE_DTYPE), h[pre_answer].to(STORE_DTYPE),
                mean.to(STORE_DTYPE),
            ])
            return seq.to(STORE_DTYPE).cpu(), pts.cpu()

        store = self._run(input_ids, extract)
        seq = {idx: pair[0] for idx, pair in store.items()}   # type: ignore[index]
        points = {idx: pair[1] for idx, pair in store.items()}   # type: ignore[index]
        self._check(seq, rows=end - start, what="positions")
        self._check(points, rows=len(POINTS), what="points")
        return seq, points


# ---------------------------------------------------------------------------
# Safetensors part writing
# ---------------------------------------------------------------------------


def build_safetensors_header(
    entries: list[tuple[str, tuple[int, int]]], metadata: dict[str, str]
) -> tuple[bytes, dict[str, tuple[int, int]], int]:
    """Safetensors header for bf16 tensors laid out back to back in ``entries`` order:
    returns the on-disk header, each key's ``(start, end)`` data offsets and the data length."""
    header: dict = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    offsets: dict[str, tuple[int, int]] = {}
    cur = 0
    for key, shape in entries:
        if key in offsets or key == "__metadata__":
            raise ValueError(f"duplicate or reserved tensor key {key!r}")
        n = int(np.prod(shape)) * BYTES_PER_ELEMENT
        header[key] = {"dtype": SAFETENSORS_DTYPE, "shape": [int(d) for d in shape],
                       "data_offsets": [cur, cur + n]}
        offsets[key] = (cur, cur + n)
        cur += n
    body = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    body += b" " * ((8 - len(body) % 8) % 8)
    return struct.pack("<Q", len(body)) + body, offsets, cur


def tensor_bytes(t: torch.Tensor) -> memoryview:
    """Zero-copy byte view of a CPU bf16 tensor's storage."""
    t = t.detach()
    if t.dtype != STORE_DTYPE:
        t = t.to(STORE_DTYPE)
    return memoryview(t.contiguous().view(torch.uint8).numpy().reshape(-1))


class PartWriter:
    """Writes one part incrementally, in the planned tensor order: ``staging="memory"``
    fills a preallocated bytearray (``finish`` returns ``bytes``), ``staging="disk"``
    streams into ``staging_path`` (``finish`` returns the path)."""

    def __init__(self, entries: list[tuple[str, tuple[int, int]]], metadata: dict[str, str], *,
                 staging: str = "memory", staging_path: Optional[Path] = None):
        self.header, self.offsets, self.data_len = build_safetensors_header(entries, metadata)
        self.order = [k for k, _ in entries]
        self.shapes = {k: tuple(shape) for k, shape in entries}
        self.total = len(self.header) + self.data_len
        self.staging = staging
        self.path = staging_path
        self._i = 0
        self.pos = 0
        if staging == "disk":
            if staging_path is None:
                raise ValueError("disk staging needs a staging_path")
            self._f = open(staging_path, "wb")
            self._buf = None
            self._write(self.header)
        elif staging == "memory":
            self._f = None
            self._buf = bytearray(self.total)
            self._write(self.header)
        else:
            raise ValueError(f"unknown staging {staging!r}")

    def _write(self, data) -> None:
        n = len(data)
        if self._f is not None:
            self._f.write(data)
        else:
            self._buf[self.pos:self.pos + n] = data
        self.pos += n

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if self._i >= len(self.order) or key != self.order[self._i]:
            expected = self.order[self._i] if self._i < len(self.order) else None
            raise RuntimeError(f"tensor {key!r} written out of plan order (expected {expected!r})")
        if tuple(tensor.shape) != self.shapes[key]:
            raise RuntimeError(f"{key}: captured shape {tuple(tensor.shape)} != planned {self.shapes[key]}")
        start, _ = self.offsets[key]
        if self.pos != len(self.header) + start:
            raise RuntimeError(f"{key}: write position {self.pos} != planned {len(self.header) + start}")
        self._write(tensor_bytes(tensor))
        self._i += 1

    def skip(self, key: str) -> None:
        """Leave the planned slot of ``key`` zero-filled and move on (the header is fixed
        by the plan; the manifest never names a skipped key)."""
        if self._i >= len(self.order) or key != self.order[self._i]:
            expected = self.order[self._i] if self._i < len(self.order) else None
            raise RuntimeError(f"tensor {key!r} skipped out of plan order (expected {expected!r})")
        start, end = self.offsets[key]
        if self.pos != len(self.header) + start:
            raise RuntimeError(f"{key}: write position {self.pos} != planned {len(self.header) + start}")
        if self._f is not None:
            remaining = end - start
            zeros = bytes(min(remaining, 64 << 20))
            while remaining:
                n = min(remaining, len(zeros))
                self._f.write(zeros[:n])
                remaining -= n
        # (the memory bytearray is zero-initialised already)
        self.pos = len(self.header) + end
        self._i += 1

    def finish(self):
        if self._i != len(self.order) or self.pos != self.total:
            raise RuntimeError(f"part incomplete: {self._i}/{len(self.order)} tensors, {self.pos}/{self.total} bytes")
        if self._f is not None:
            self._f.close()
            return str(self.path)
        data = bytes(self._buf)
        self._buf = None
        return data

    def abort(self) -> None:
        if self._f is not None:
            self._f.close()
            Path(self.path).unlink(missing_ok=True)
        self._buf = None


# ---------------------------------------------------------------------------
# Part planning (two file families per part index)
# ---------------------------------------------------------------------------


def _item_key(item, i: int) -> str:
    for attr in ("rollout_id", "sample_key"):
        key = getattr(item, attr, None)
        if key is not None:
            return str(key)
    return f"item {i}"


@dataclass
class PartPlan:
    """The rollouts of one part index ``NN``; ``seq_bytes`` / ``points_bytes`` are the
    tensor bytes of **one layer's** file (every layer's pair holds the same items)."""
    index: int
    items: list
    n_tokens: int
    seq_bytes: int
    points_bytes: int

    def seq_file(self, layer: int) -> str:
        return seq_file(layer, self.index)

    def points_file(self, layer: int) -> str:
        return points_file(layer, self.index)

    def files(self, layers: Iterable[int], *, seq_layers: Optional[Iterable[int]] = None) -> list[str]:
        """Every file of this part in layer order: the points file of every layer in
        ``layers`` and the seq file of the layers in ``seq_layers`` (``None`` = every layer)."""
        layers = [int(l) for l in layers]
        seq = set(layers) if seq_layers is None else {int(l) for l in seq_layers}
        out = []
        for layer in layers:
            if layer in seq:
                out.append(self.seq_file(layer))
            out.append(self.points_file(layer))
        return out

    def total_bytes(self, n_layers: int, n_seq_layers: Optional[int] = None) -> int:
        """Tensor bytes of the part: ``n_seq_layers`` seq files (``None`` = ``n_layers``)
        plus ``n_layers`` points files."""
        n_seq = int(n_layers) if n_seq_layers is None else int(n_seq_layers)
        return n_seq * self.seq_bytes + int(n_layers) * self.points_bytes

    def entries(self, kind: str, *, hidden: int) -> list[tuple[str, tuple[int, int]]]:
        """The ``PartWriter`` entries ``(rollout_id, shape)`` of one file of this part."""
        if kind not in STORE_FILE_KINDS:
            raise ValueError(f"kind must be one of {STORE_FILE_KINDS}, got {kind!r}")
        rows = len(POINTS)
        return [
            (_item_key(it, i), ((int(it.n_tokens), int(hidden)) if kind == "seq" else (rows, int(hidden))))
            for i, it in enumerate(self.items)
        ]


def seq_item_bytes(n_tokens: int, hidden: int) -> int:
    return int(n_tokens) * int(hidden) * BYTES_PER_ELEMENT


def points_item_bytes(hidden: int) -> int:
    return len(POINTS) * int(hidden) * BYTES_PER_ELEMENT


def plan_parts(
    items: Sequence, *, hidden: int, n_layers: int, max_bytes: int,
    hard_limit_bytes: int = int(HF_HARD_FILE_LIMIT_GB * GB),
) -> list[PartPlan]:
    """Greedily pack ``items`` (anything with ``n_tokens`` and a ``rollout_id`` /
    ``sample_key``) into parts so that no single file exceeds ``max_bytes``.

    The cap applies to one layer's file (the seq file is the largest); an item alone
    above the cap gets its own part, one above ``hard_limit_bytes`` raises.
    ``n_layers`` only sizes the totals.
    """
    if int(n_layers) < 1:
        raise ValueError(f"n_layers must be ≥ 1, got {n_layers!r}")
    parts: list[PartPlan] = []
    cur: list = []
    cur_tokens = 0
    p_item = points_item_bytes(hidden)

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            parts.append(PartPlan(
                index=len(parts), items=cur, n_tokens=cur_tokens,
                seq_bytes=seq_item_bytes(cur_tokens, hidden), points_bytes=p_item * len(cur),
            ))
        cur, cur_tokens = [], 0

    for i, it in enumerate(items):
        size = seq_item_bytes(it.n_tokens, hidden)
        if size > hard_limit_bytes:
            raise ValueError(
                f"item {_item_key(it, i)} alone is {size / GB:.1f} GB per layer, above the "
                f"{hard_limit_bytes / GB:g} GB per-file hard limit."
            )
        if cur and (seq_item_bytes(cur_tokens, hidden) + size > max_bytes
                    or p_item * (len(cur) + 1) > max_bytes):
            flush()
        cur.append(it)
        cur_tokens += int(it.n_tokens)
    flush()
    return parts


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _payload_bytes_and_size(payload) -> tuple[Optional[Path], int]:
    """(staged path or None, size in bytes) of an upload payload."""
    if isinstance(payload, (str, os.PathLike)):
        path = Path(payload)
        return path, path.stat().st_size
    return None, len(payload)


class StoreBackend(abc.ABC):
    """Where a store's files live. Paths are relative to the store root; ``commit_message``
    is recorded by the Hub and ignored on disk."""

    #: one-line description for logs
    name: str = "backend"

    @abc.abstractmethod
    def preflight(self, *, dry_run: bool = False) -> str:
        """Verify the backend is writable before any data is read; create the
        repo/directory unless ``dry_run``. Returns a short identity (user, root)."""

    @abc.abstractmethod
    def exists(self, path: str) -> bool: ...

    @abc.abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        """Every file path under ``prefix`` (sorted; empty for a missing root)."""

    @abc.abstractmethod
    def upload_file(self, local_or_bytes, path: str, *, commit_message: Optional[str] = None) -> None:
        """Store ``bytes`` or a local file at ``path``; the local file is left in place."""

    @abc.abstractmethod
    def download_file(self, path: str) -> Path:
        """A local path holding ``path``'s content (the file itself for a local store)."""

    def download_to(self, path: str, dest: Path) -> Path:
        """Copy ``path``'s content to the local file ``dest`` without keeping a second
        copy elsewhere (``dest``'s parent must exist)."""
        src = self.download_file(path)
        if Path(src).resolve() != Path(dest).resolve():
            shutil.copyfile(src, dest)
        return Path(dest)

    @abc.abstractmethod
    def file_sizes(self, prefix: str = "") -> dict[str, int]:
        """``{path: size in bytes}`` of every file under ``prefix``."""

    @abc.abstractmethod
    def read_text(self, path: str) -> str: ...

    @abc.abstractmethod
    def write_text(self, path: str, text: str, *, commit_message: Optional[str] = None) -> None: ...

    @abc.abstractmethod
    def delete(self, paths: Iterable[str], *, commit_message: Optional[str] = None) -> None: ...

    @abc.abstractmethod
    def commit(self, adds: dict[str, bytes], deletes: Iterable[str] = (), *,
               commit_message: Optional[str] = None) -> None:
        """Write ``adds`` (path → bytes) and remove ``deletes`` — one commit on the Hub,
        file by file (atomic per file) on disk."""

    def describe(self) -> str:
        return self.name


class HfBackend(StoreBackend):
    """One Hugging Face dataset repo as the store root. ``revision`` pins the commit
    reads come from; ``api`` injects an ``HfApi``-compatible object (tests)."""

    def __init__(self, repo_id: str, token: Optional[str] = None, private: bool = False,
                 revision: Optional[str] = None, *, api=None):
        if not repo_id or "/" not in str(repo_id):
            raise ValueError("hf_repo_id must be '<namespace>/<name>'.")
        self.repo_id = str(repo_id)
        self.private = bool(private)
        self.revision = revision
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.api = api
        self.name = f"hf:{self.repo_id}" + (f"@{revision}" if revision else "")

    def _rev(self) -> dict:
        return {"revision": self.revision} if self.revision else {}

    def preflight(self, *, dry_run: bool = False) -> str:
        """Verify the token can write to the repo's namespace; create the repo."""
        try:
            info = self.api.whoami()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Hugging Face token check failed: {e}. Log in or set HF_TOKEN.") from e
        user = info.get("name", "?")
        allowed = {user} | {o.get("name") for o in info.get("orgs", [])}
        namespace = self.repo_id.split("/")[0]
        if namespace not in allowed:
            raise RuntimeError(
                f"The active Hugging Face token belongs to {user!r}, which cannot write to "
                f"{namespace!r} (memberships: {sorted(allowed)}). Run with a token of an "
                "account in that namespace, e.g. HF_TOKEN=<token> uv run ... — the "
                "environment variable takes precedence over `huggingface-cli login`."
            )
        print(f"HF token: {user} (write access to {namespace!r} confirmed)")
        if not dry_run:
            self.api.create_repo(self.repo_id, repo_type="dataset", private=self.private, exist_ok=True)
        return user

    def exists(self, path: str) -> bool:
        return path in set(self.list_files())

    def list_files(self, prefix: str = "") -> list[str]:
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type="dataset", **self._rev())
        except RepositoryNotFoundError:
            return []
        return sorted(f for f in files if f.startswith(prefix))

    def upload_file(self, local_or_bytes, path: str, *, commit_message: Optional[str] = None) -> None:
        payload = str(local_or_bytes) if isinstance(local_or_bytes, os.PathLike) else local_or_bytes
        self.api.upload_file(
            path_or_fileobj=payload, path_in_repo=path, repo_id=self.repo_id, repo_type="dataset",
            commit_message=commit_message or f"Add {path}", **self._rev(),
        )

    def download_file(self, path: str) -> Path:
        return Path(self.api.hf_hub_download(
            self.repo_id, path, repo_type="dataset", force_download=True, **self._rev(),
        ))

    def download_to(self, path: str, dest: Path) -> Path:
        """Download ``path`` straight into ``dest``'s directory (``hf_hub_download`` with
        a throw-away ``local_dir`` beside ``dest``, then ``os.replace``), so nothing
        lands in the HF cache."""
        dest = Path(dest)
        tmp_root = dest.parent / f".tmp-hf-{os.getpid()}.{uuid.uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=False)
        try:
            got = Path(self.api.hf_hub_download(
                self.repo_id, path, repo_type="dataset", local_dir=str(tmp_root), force_download=True,
                **self._rev(),
            ))
            os.replace(got, dest)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
        return dest

    def file_sizes(self, prefix: str = "") -> dict[str, int]:
        """Sizes from one ``list_repo_tree`` walk (folders skipped)."""
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
        folder = prefix.rstrip("/") or None
        try:
            entries = list(self.api.list_repo_tree(self.repo_id, path_in_repo=folder, recursive=True,
                                                   repo_type="dataset", **self._rev()))
        except (RepositoryNotFoundError, EntryNotFoundError):   # no repo / no such folder
            return {}
        out = {}
        for entry in entries:
            size = getattr(entry, "size", None)
            path = getattr(entry, "path", None)
            if size is None or path is None or not str(path).startswith(prefix):
                continue
            out[str(path)] = int(size)
        return dict(sorted(out.items()))

    def read_text(self, path: str) -> str:
        return self.download_file(path).read_text(encoding="utf-8")

    def write_text(self, path: str, text: str, *, commit_message: Optional[str] = None) -> None:
        self.upload_file(text.encode("utf-8"), path, commit_message=commit_message)

    def delete(self, paths: Iterable[str], *, commit_message: Optional[str] = None) -> None:
        paths = list(paths)
        if paths:
            self.commit({}, paths, commit_message=commit_message or f"Delete {', '.join(paths)}")

    def commit(self, adds: dict[str, bytes], deletes: Iterable[str] = (), *,
               commit_message: Optional[str] = None) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=bytes(b)) for p, b in adds.items()]
        ops += [CommitOperationDelete(path_in_repo=p) for p in deletes]
        if not ops:
            return
        self.api.create_commit(
            repo_id=self.repo_id, repo_type="dataset", operations=ops,
            commit_message=commit_message or f"Update {', '.join(adds)}", **self._rev(),
        )


class LocalBackend(StoreBackend):
    """A directory as the store root. Every write lands through a temp file in the
    target directory + ``os.replace``, so a reader never sees a half-written file."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.name = f"local:{self.root}"

    def _abs(self, path: str) -> Path:
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"store paths are relative to the root, got {path!r}")
        return self.root / rel

    def preflight(self, *, dry_run: bool = False) -> str:
        if dry_run:
            parent = self.root
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not os.access(parent, os.W_OK):
                raise RuntimeError(f"Cannot write under {parent} (needed to create {self.root}).")
            return str(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.root, os.W_OK):
            raise RuntimeError(f"Store directory {self.root} is not writable.")
        return str(self.root)

    def exists(self, path: str) -> bool:
        return self._abs(path).is_file()

    def list_files(self, prefix: str = "") -> list[str]:
        if not self.root.is_dir():
            return []
        out = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            rel_parts = p.relative_to(self.root).parts
            # ``.tmp-*`` entries are half-written writes or mirror downloads in flight.
            if any(part.startswith(".tmp-") for part in rel_parts):
                continue
            rel = "/".join(rel_parts)
            if rel.startswith(prefix):
                out.append(rel)
        return sorted(out)

    def _atomic_write(self, path: str, write: Callable[[Path], None]) -> None:
        target = self._abs(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".tmp-{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            write(tmp)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    def upload_file(self, local_or_bytes, path: str, *, commit_message: Optional[str] = None) -> None:
        if isinstance(local_or_bytes, (str, os.PathLike)):
            src = Path(local_or_bytes)
            self._atomic_write(path, lambda tmp: shutil.copyfile(src, tmp))
        else:
            data = local_or_bytes
            self._atomic_write(path, lambda tmp: tmp.write_bytes(data))

    def download_file(self, path: str) -> Path:
        target = self._abs(path)
        if not target.is_file():
            raise FileNotFoundError(f"{path!r} is not in the store at {self.root}")
        return target

    def file_sizes(self, prefix: str = "") -> dict[str, int]:
        return {f: self._abs(f).stat().st_size for f in self.list_files(prefix)}

    def read_text(self, path: str) -> str:
        return self.download_file(path).read_text(encoding="utf-8")

    def write_text(self, path: str, text: str, *, commit_message: Optional[str] = None) -> None:
        self.upload_file(text.encode("utf-8"), path)

    def delete(self, paths: Iterable[str], *, commit_message: Optional[str] = None) -> None:
        for p in paths:
            self._abs(p).unlink(missing_ok=True)

    def commit(self, adds: dict[str, bytes], deletes: Iterable[str] = (), *,
               commit_message: Optional[str] = None) -> None:
        for p, b in adds.items():
            self.upload_file(bytes(b), p)
        self.delete(deletes)


def make_backend(cfg: dict, *, token: Optional[str] = None, api=None) -> StoreBackend:
    """Build a backend from a ``storage:`` block
    ``{backend: hf|local, hf_repo_id, hf_private, local_dir, revision}``.

    Keys the chosen backend does not use may be present but must be null; an unknown
    key is an error. ``api`` injects the ``HfApi`` for tests.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"storage config must be a mapping, got {type(cfg).__name__}")
    unknown = sorted(set(cfg) - STORAGE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown storage keys {unknown}; accepted: {sorted(STORAGE_CONFIG_KEYS)}.")
    backend = cfg.get("backend")
    if backend not in BACKENDS:
        raise ValueError(f"storage.backend must be one of {BACKENDS}, got {backend!r}.")
    revision = cfg.get("revision")
    if backend == "hf":
        repo_id = cfg.get("hf_repo_id")
        if not repo_id or not isinstance(repo_id, str) or "/" not in repo_id:
            raise ValueError("storage.hf_repo_id must be '<namespace>/<name>' for backend: hf.")
        if cfg.get("local_dir") is not None:
            raise ValueError("storage.local_dir only applies to backend: local (set it to null for backend: hf).")
        return HfBackend(repo_id, token=token, private=bool(cfg.get("hf_private", False)),
                         revision=revision, api=api)
    local_dir = cfg.get("local_dir")
    if not local_dir or not isinstance(local_dir, str):
        raise ValueError("storage.local_dir is required for backend: local (e.g. ${DATA_ROOT}/probe_activations/<model>).")
    other = sorted(k for k in ("hf_repo_id", "hf_private", "revision") if cfg.get(k) is not None)
    if other:
        raise ValueError(f"storage.{'/'.join(other)} only appl{'ies' if len(other) == 1 else 'y'} to backend: hf "
                         "(set them to null for backend: local).")
    return LocalBackend(resolve_data_path(local_dir))


# ---------------------------------------------------------------------------
# Store state and writer
# ---------------------------------------------------------------------------


def store_path(folder: str, name: str) -> str:
    """``<folder>/<name>`` relative to the backend root (``name`` alone for an empty folder)."""
    folder = str(folder or "").strip("/")
    return f"{folder}/{name}" if folder else name


def load_store_state(backend: StoreBackend, folder: str, *, columns: Optional[list[str]] = None,
                     manifest_name: str = STORE_MANIFEST_NAME, meta_name: str = STORE_MANIFEST_META_NAME,
                     ) -> tuple[set[str], pd.DataFrame, dict]:
    """(files in the folder — names relative to it —, existing manifest, existing meta);
    empty when the folder is new. No validation beyond parsing."""
    prefix = store_path(folder, "")
    files = {f[len(prefix):] for f in backend.list_files(prefix) if f.startswith(prefix)}
    manifest = pd.DataFrame(columns=columns or [])
    meta: dict = {}
    if manifest_name in files:
        try:
            manifest = pd.read_csv(backend.download_file(store_path(folder, manifest_name)), low_memory=False)
        except pd.errors.EmptyDataError:
            pass
    if meta_name in files:
        meta = json.loads(backend.read_text(store_path(folder, meta_name)))
    return files, manifest, meta


@dataclass
class GroupResult:
    """What a finalized group ended up with: uploaded files, failed files, rows written."""
    key: str
    ok_files: list[str]
    failed_files: list[str]
    n_rows: int
    stale_deleted: list[str] = field(default_factory=list)


class StoreWriter:
    """Uploads finished parts in bounded background threads through a :class:`StoreBackend`
    and commits the merged manifest + meta per group once that group's files are done.

    ``submit_file(..., rows=)`` ties rows to one file (written iff it uploaded),
    ``finalize_group(..., rows=)`` to the whole group (written iff every file uploaded).
    Failed files go to ``failed_parts`` with their rows kept out; existing rows sharing a
    key (``key_columns``) with a planned row are replaced; stale files are deleted in
    the same commit.
    """

    def __init__(self, backend: StoreBackend, folder: str = "", *, manifest: Optional[pd.DataFrame] = None,
                 meta: Optional[dict] = None, columns: list[str] = STORE_MANIFEST_COLUMNS,
                 key_columns: Sequence[str] = ("rollout_id",),
                 upload_workers: int = DEFAULT_UPLOAD_WORKERS, staging: str = "memory",
                 staging_dir: Optional[Path] = None, max_file_gb: float = DEFAULT_MAX_FILE_GB,
                 manifest_name: str = STORE_MANIFEST_NAME, meta_name: str = STORE_MANIFEST_META_NAME):
        if staging not in STAGING_MODES:
            raise ValueError(f"staging must be one of {STAGING_MODES}, got {staging!r}")
        if staging == "disk" and staging_dir is None:
            raise ValueError("disk staging needs a staging_dir")
        if upload_workers < 1:
            raise ValueError("upload_workers must be ≥ 1")
        self.backend = backend
        self.folder = str(folder or "").strip("/")
        self.columns = list(columns)
        self.key_columns = list(key_columns)
        self.manifest = (manifest if manifest is not None and not manifest.empty
                         else pd.DataFrame(columns=self.columns)).reindex(columns=self.columns)
        self.meta = dict(meta or {})
        self.staging = staging
        self.staging_dir = Path(staging_dir) if staging_dir is not None else None
        self.max_file_gb = float(max_file_gb)
        self.manifest_name = manifest_name
        self.meta_name = meta_name
        self.upload_pool = ThreadPoolExecutor(max_workers=upload_workers)
        self.commit_pool = ThreadPoolExecutor(max_workers=1)
        self._slots = threading.BoundedSemaphore(upload_workers)
        self._lock = threading.Lock()
        self._futures: list[Future] = []
        self._group_futures: dict[str, list[Future]] = {}
        self.upload_workers = int(upload_workers)
        self._group_files: dict[str, list[str]] = {}
        self._file_rows: dict[tuple[str, str], list[dict]] = {}   # planned rows per (group, file)
        self._uploaded: set[str] = set()
        self.failed_parts: list[tuple[str, str]] = []
        self.results: dict[str, GroupResult] = {}
        self.uploaded_bytes = 0

    # -- sizing / staging ------------------------------------------------------

    @property
    def max_bytes(self) -> int:
        return int(self.max_file_gb * GB)

    def path(self, name: str) -> str:
        return store_path(self.folder, name)

    def prepare_staging(self) -> list[Path]:
        """Create the staging dir (disk staging) and return leftover staged files of an
        earlier interrupted run."""
        if self.staging != "disk":
            return []
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        return sorted(self.staging_dir.glob("*.safetensors"))

    def new_part_writer(self, filename: str, entries: list[tuple[str, tuple[int, int]]],
                        metadata: dict[str, str]) -> PartWriter:
        """A :class:`PartWriter` for ``filename`` under this writer's staging mode."""
        staged = (self.staging_dir / f"{os.getpid()}_{os.path.basename(filename)}") if self.staging == "disk" else None
        return PartWriter(entries, metadata, staging=self.staging, staging_path=staged)

    def staging_note(self, files_per_part: int = 1) -> str:
        """One line on the peak footprint for the plan printout."""
        if self.staging == "memory":
            return (f"staging: memory — peak RAM ≈ {(self.upload_workers + 1 + files_per_part) * self.max_file_gb:g} GB "
                    f"({self.upload_workers} uploads in flight + {files_per_part} full-size file(s) being written + a hand-off copy)")
        return (f"staging: disk under {self.staging_dir} — peak transient disk ≈ "
                f"{(self.upload_workers + files_per_part) * self.max_file_gb:g} GB, RAM stays small")

    # -- files -------------------------------------------------------------------

    def submit_file(self, group_key: str, filename: str, payload, *, rows: Sequence[dict] = ()) -> None:
        """Upload one finished file in the background. ``payload`` is ``bytes`` or the
        path of a staged file (deleted after the attempt); ``rows`` are manifest rows
        written iff this file uploads."""
        self._slots.acquire()
        with self._lock:
            self._group_files.setdefault(group_key, []).append(filename)
            self._file_rows[(group_key, filename)] = list(rows)
        fut = self.upload_pool.submit(self._upload, filename, payload, len(rows))
        fut.add_done_callback(lambda _f: self._slots.release())
        self._group_futures.setdefault(group_key, []).append(fut)
        self._futures.append(fut)

    def _upload(self, filename: str, payload, n_rows: int) -> None:
        staged_path, size = _payload_bytes_and_size(payload)
        try:
            self.backend.upload_file(payload, self.path(filename), commit_message=f"Add {self.path(filename)}")
        except Exception as e:  # noqa: BLE001 — a failed part must not kill the run
            with self._lock:
                self.failed_parts.append((filename, repr(e)))
            print(f"  [{self.backend.describe()}] upload FAILED for {filename} ({size / GB:.2f} GB): {e}")
            return
        finally:
            del payload
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
        with self._lock:
            self.uploaded_bytes += size
            self._uploaded.add(filename)
        print(f"  [{self.backend.describe()}] uploaded {filename} ({size / GB:.2f} GB"
              + (f", {n_rows} rows" if n_rows else "") + ")")

    # -- groups ------------------------------------------------------------------

    def finalize_group(self, group_key: str, *, rows: Sequence[dict] = (), keep: Iterable[str] = (),
                       stale_filter: Optional[Callable[[str], bool]] = None,
                       meta_update: Optional[Callable[[dict, GroupResult], None]] = None) -> None:
        """Commit the manifest + meta once every file of ``group_key`` finished.
        ``meta_update(meta, result)`` edits the meta in place first; ``stale_filter`` /
        ``keep`` select the files to delete (see :meth:`stale_files`; None = nothing)."""
        part_futures = self._group_futures.pop(group_key, [])
        fut = self.commit_pool.submit(
            self._finalize, group_key, list(rows), part_futures, list(keep), stale_filter, meta_update,
        )
        self._futures.append(fut)

    def _finalize(self, group_key: str, group_rows: list[dict], part_futures: list[Future],
                  keep: list[str], stale_filter, meta_update) -> None:
        # Commits are serialized (one worker); a failed upload is recorded, not raised.
        for fut in part_futures:
            fut.result()
        with self._lock:
            files = self._group_files.pop(group_key, [])
            ok_files = [f for f in files if f in self._uploaded]
            # Consumed per group, so a retried filename is judged on its own upload.
            self._uploaded.difference_update(files)
            per_file = {f: self._file_rows.pop((group_key, f), []) for f in files}
        failed = [f for f in files if f not in ok_files]
        rows = [r for f in ok_files for r in per_file[f]] + (list(group_rows) if not failed else [])
        planned = [r for f in files for r in per_file[f]] + list(group_rows)
        self.manifest = self._merge(self.manifest, rows, planned)
        result = GroupResult(key=group_key, ok_files=ok_files, failed_files=failed, n_rows=len(rows))
        if meta_update is not None:
            meta_update(self.meta, result)
        stale = self.stale_files(keep, stale_filter) if stale_filter is not None else []
        result.stale_deleted = stale
        self.results[group_key] = result
        self.write_manifest(self.manifest, self.meta, deletes=stale,
                            commit_message=f"Update manifest for {group_key}"
                            + (f" (remove stale {', '.join(stale)})" if stale else ""))
        print(f"  [{self.backend.describe()}] manifest updated for {group_key}: {len(rows)} rows"
              + (f"; removed stale {stale}" if stale else "")
              + (f"; FAILED {failed}" if failed else ""))

    def _merge(self, existing: pd.DataFrame, rows: list[dict], planned: list[dict]) -> pd.DataFrame:
        """Drop the existing rows keyed like any planned row, append ``rows``."""
        new = pd.DataFrame(rows, columns=self.columns)
        if existing is None or existing.empty:
            return new.reindex(columns=self.columns)
        kept = existing
        if planned and all(c in existing.columns for c in self.key_columns):
            planned_keys = {tuple(str(r.get(c, "")) for c in self.key_columns) for r in planned}
            existing_keys = [tuple(str(v) for v in t) for t in existing[self.key_columns].itertuples(index=False, name=None)]
            mask = np.array([k in planned_keys for k in existing_keys], dtype=bool)
            kept = existing[~mask]
        merged = pd.concat([kept, new], ignore_index=True) if not kept.empty else new
        return merged.reindex(columns=self.columns)

    def stale_files(self, keep: Iterable[str], stale_filter: Callable[[str], bool] = is_store_file) -> list[str]:
        """Files in the folder that ``stale_filter`` matches and ``keep`` does not name
        (relative to the folder). The default filter matches every part file, so ``keep``
        must then name every file the whole run keeps."""
        keep = set(keep)
        prefix = self.path("")
        names = [f[len(prefix):] for f in self.backend.list_files(prefix)]
        return sorted(f for f in names if f not in keep and stale_filter(f))

    def write_manifest(self, manifest: pd.DataFrame, meta: dict, *, deletes: Iterable[str] = (),
                       commit_message: Optional[str] = None) -> None:
        """One commit: ``manifest.csv`` + ``manifest.meta.json`` (+ deletions)."""
        adds = {
            self.path(self.manifest_name): manifest.reindex(columns=self.columns).to_csv(index=False).encode("utf-8"),
            self.path(self.meta_name): json.dumps(meta, indent=2).encode("utf-8"),
        }
        self.backend.commit(adds, [self.path(f) for f in deletes], commit_message=commit_message or "Update manifest")

    # -- lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        """Wait for every upload and commit; re-raise the first unexpected error."""
        self.upload_pool.shutdown(wait=True)
        self.commit_pool.shutdown(wait=True)
        for fut in self._futures:
            exc = fut.exception()
            if exc is not None:
                raise exc


# ---------------------------------------------------------------------------
# Reader side: validated manifest, layer resolution, rollout_id-keyed stores
# ---------------------------------------------------------------------------

# Concurrent Range requests per sample / per ``get_many`` (DataLoader workers multiply it).
DEFAULT_FETCH_THREADS = 8


def _check_layer_list(value, what: str) -> list[int]:
    if (not isinstance(value, (list, tuple))
            or any(isinstance(l, bool) or not isinstance(l, (int, np.integer)) for l in value)):
        raise ValueError(f"manifest meta {what!r} must be a list of ints, got {value!r}")
    layers = [int(l) for l in value]
    if layers != sorted(set(layers)):
        raise ValueError(f"manifest meta {what!r} must be sorted and unique, got {layers}")
    return layers


def check_store_meta(meta: dict) -> dict:
    """The validated copy of a store meta: ``points`` == :data:`POINTS`, ``layers`` a
    non-empty strictly increasing int list, ``hidden_size`` a positive int, ``seq_layers``
    (when present) a non-empty sorted subset of ``layers``, and each mirrored-layers key
    (when present) a sorted subset of the layers the source holds for that kind (``[]`` allowed)."""
    points = meta.get("points")
    if not isinstance(points, (list, tuple)) or [str(x) for x in points] != list(POINTS):
        raise ValueError(f"manifest meta 'points' {points!r} != {list(POINTS)} (the stored point order)")
    layers = meta.get("layers")
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError(f"manifest meta 'layers' must be a non-empty list of ints, got {layers!r}")
    layers = _check_layer_list(layers, "layers")
    hidden = meta.get("hidden_size")
    if isinstance(hidden, bool) or not isinstance(hidden, (int, np.integer)) or int(hidden) <= 0:
        raise ValueError(f"manifest meta 'hidden_size' must be a positive int, got {hidden!r}")
    meta = dict(meta)
    meta["layers"] = layers
    meta["hidden_size"] = int(hidden)
    if meta.get(SEQ_LAYERS_KEY) is not None:
        seq_layers = _check_layer_list(meta[SEQ_LAYERS_KEY], SEQ_LAYERS_KEY)
        extra = sorted(set(seq_layers) - set(layers))
        if not seq_layers or extra:
            raise ValueError(
                f"manifest meta {SEQ_LAYERS_KEY!r} must be a non-empty subset of 'layers' {layers}, got {seq_layers}"
                + (f" (unknown layers {extra})" if extra else "")
            )
        meta[SEQ_LAYERS_KEY] = seq_layers
    for kind, key in MIRRORED_LAYERS_KEYS.items():
        if key in meta and meta[key] is not None:
            mirrored = _check_layer_list(meta[key], key)
            stored = stored_layers(meta, kind)
            extra = sorted(set(mirrored) - set(stored))
            if extra:
                raise ValueError(
                    f"manifest meta {key!r} lists layers {extra} the store does not hold {kind} files for "
                    f"({STORED_LAYERS_KEYS[kind]}: {stored})"
                )
            meta[key] = mirrored
    return meta


def stored_layers(meta: dict, kind: str) -> list[int]:
    """The layers the **source** store holds ``<kind>`` files for: ``seq_layers`` (else
    every layer) for seq, ``layers`` for points."""
    if kind not in STORE_FILE_KINDS:
        raise ValueError(f"kind must be one of {STORE_FILE_KINDS}, got {kind!r}")
    if kind == "seq" and meta.get(SEQ_LAYERS_KEY) is not None:
        return [int(l) for l in meta[SEQ_LAYERS_KEY]]
    return [int(l) for l in meta["layers"]]


def available_layers(meta: dict, kind: str) -> list[int]:
    """The layers whose ``<kind>`` files the store as read holds: a mirror's mirrored
    list when the meta carries the key, else :func:`stored_layers`."""
    if kind not in STORE_FILE_KINDS:
        raise ValueError(f"kind must be one of {STORE_FILE_KINDS}, got {kind!r}")
    mirrored = meta.get(MIRRORED_LAYERS_KEYS[kind])
    if mirrored is None:
        return stored_layers(meta, kind)
    return [int(l) for l in mirrored]


def is_mirror(meta: dict) -> bool:
    return any(meta.get(key) is not None for key in MIRRORED_LAYERS_KEYS.values())


def resolve_kind_layers(requested, meta: dict, kind: str, *, allow_empty: bool = False) -> list[int]:
    """The layers a reader of ``<kind>`` files may use: ``requested`` (``None`` = every
    available layer; an empty mirror raises unless ``allow_empty``), which must be stored,
    stored for that kind and, on a mirror, present in the copy."""
    available = available_layers(meta, kind)
    stored = stored_layers(meta, kind)
    if requested is None:
        if not available and not allow_empty:
            raise LayerSelectionError(
                f"this mirror of the store holds no {kind} files (mirrored {kind} layers: []; layers the source "
                f"stores {kind} files for: {stored}) — re-run mirror_probe_store.py with "
                + ("--layers <L>" if kind == "seq" else "the points files (without --no-points)")
            )
        return available
    layers = resolve_store_layers(requested, meta["layers"])
    never_stored = sorted(set(layers) - set(stored))
    if never_stored:
        raise LayerSelectionError(
            f"layers {never_stored} have no {kind} files in this store: it was collected with "
            f"{STORED_LAYERS_KEYS[kind]}: {stored} (layers with points files: {meta['layers']}) — "
            f"pick a layer from {STORED_LAYERS_KEYS[kind]} or re-collect the store with it in `seq_layers`"
        )
    missing = sorted(set(layers) - set(available))
    if missing:
        raise LayerSelectionError(
            f"layers {missing} are not in this mirror of the store for its {kind} files (mirrored {kind} layers: "
            f"{available}; layers the source stores {kind} files for: {stored}) — re-run mirror_probe_store.py with "
            f"--layers {','.join(str(l) for l in missing)}"
            + ("" if kind == "seq" else " (points are mirrored for every stored layer unless --no-points)")
        )
    return layers


def read_store_manifest(backend: StoreBackend, folder: str, *, layers=None, kinds: Sequence[str] = STORE_FILE_KINDS,
                        manifest_name: str = STORE_MANIFEST_NAME,
                        meta_name: str = STORE_MANIFEST_META_NAME) -> tuple[pd.DataFrame, dict]:
    """The validated ``manifest.csv`` + ``manifest.meta.json`` of a store folder.

    Checks that both files exist, the frame passes :func:`check_store_manifest_frame`,
    the meta passes :func:`check_store_meta`, and every part a row names has its file
    present in the backend (one listing) for every layer the reader will use: per
    ``kind`` in ``kinds``, ``layers`` when given (resolved through
    :func:`resolve_kind_layers`) else every available layer of that kind. The file
    columns come back normalised to :func:`part_template`.
    """
    kinds = tuple(kinds)
    unknown_kinds = [k for k in kinds if k not in STORE_FILE_KINDS]
    if unknown_kinds or not kinds:
        raise ValueError(f"kinds must be a non-empty subset of {STORE_FILE_KINDS}, got {kinds!r}")
    files, manifest, meta = load_store_state(backend, folder, columns=STORE_MANIFEST_COLUMNS,
                                             manifest_name=manifest_name, meta_name=meta_name)
    root = store_path(folder, "")
    if manifest_name not in files:
        raise FileNotFoundError(f"no {manifest_name} under {root!r} on {backend.describe()}")
    if meta_name not in files:
        raise FileNotFoundError(f"no {meta_name} under {root!r} on {backend.describe()}")
    manifest = check_store_manifest_frame(manifest, manifest_name=manifest_name)
    meta = check_store_meta(meta)
    check_layers = {kind: resolve_kind_layers(layers, meta, kind, allow_empty=True) for kind in kinds}

    for kind in kinds:
        parts = [part_index(v) for v in manifest[STORE_FILE_COLUMNS[kind]].tolist()]
        needed = needed_files(parts, kind, check_layers[kind])
        missing = sorted(set(needed) - files)
        if missing:
            raise ValueError(
                f"store {root!r} on {backend.describe()} is missing {len(missing)} {kind} file(s) the "
                f"manifest needs for layers {check_layers[kind]}: {missing[:6]}{' …' if len(missing) > 6 else ''}"
                + (" — the mirror is incomplete; re-run mirror_probe_store.py" if is_mirror(meta) else "")
            )
    return manifest, meta


def check_store_manifest_frame(manifest: pd.DataFrame, *, manifest_name: str = STORE_MANIFEST_NAME) -> pd.DataFrame:
    """The validated copy of a parsed ``manifest.csv``: exactly :data:`STORE_MANIFEST_COLUMNS`,
    non-blank unique (stripped) ``rollout_id``, file cells normalised to :func:`part_template`."""
    columns = [str(c) for c in manifest.columns]
    if columns != list(STORE_MANIFEST_COLUMNS):
        raise ValueError(
            f"{manifest_name} columns {columns} != expected {list(STORE_MANIFEST_COLUMNS)}"
        )
    ids = manifest["rollout_id"]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise ValueError(f"{manifest_name} has blank rollout_id values")
    manifest = manifest.copy()
    manifest["rollout_id"] = ids.astype(str).str.strip()
    dupes = manifest["rollout_id"][manifest["rollout_id"].duplicated()].unique().tolist()
    if dupes:
        raise ValueError(f"{manifest_name} has duplicate rollout_id values, e.g. {dupes[:5]}")
    for kind, column in STORE_FILE_COLUMNS.items():
        try:
            parts = [part_index(v) for v in manifest[column].tolist()]
        except ValueError as e:
            raise ValueError(f"{manifest_name} column {column!r}: {e}") from e
        manifest[column] = [part_template(kind, n) for n in parts]
    return manifest


def needed_files(parts: Iterable[int], kind: str, layers: Iterable[int]) -> list[str]:
    """The ``<kind>`` file names (relative to the folder) ``parts`` need for ``layers``, sorted."""
    return sorted({STORE_FILES[kind].format(layer=int(layer), part=int(n)) for n in set(parts) for layer in layers})


class LayerSelectionError(ValueError):
    """A requested layer list the store (or mirror) cannot serve."""


def resolve_store_layers(requested, stored) -> list[int]:
    """The layers to read: ``requested`` (``None`` = every stored layer), which must all
    be stored; ``[]`` and duplicates raise."""
    stored = [int(l) for l in stored]
    if requested is None:
        return sorted(stored)
    if not isinstance(requested, (list, tuple)) or not requested:
        raise LayerSelectionError(f"data.layers must be a non-empty list or null, got {requested!r}")
    layers = [int(l) for l in requested]
    unknown = sorted(set(layers) - set(stored))
    if unknown:
        raise LayerSelectionError(
            f"data.layers {unknown} are not stored in the store; stored layers: {sorted(stored)}"
        )
    if len(set(layers)) != len(layers):
        raise LayerSelectionError(f"data.layers has duplicates: {layers}")
    return sorted(layers)


# An open safetensors handle is only an mmap, so keeping every file of a store open is cheap.
LOCAL_HANDLE_CACHE = 1024


@lru_cache(maxsize=LOCAL_HANDLE_CACHE)
def _open_local(path: str, pid: int):
    # The pid is part of the key so a forked DataLoader worker never reuses the parent's mmap.
    return safe_open(path, framework="pt", device="cpu")


def local_handle(path):
    """The cached ``safe_open`` handle of a local safetensors file (this process's)."""
    return _open_local(str(path), os.getpid())


def reset_read_caches() -> None:
    """Drop this process's cached local handles and ``hf_streaming``'s per-file URLs /
    headers (needed when a process rewrites a store file it has already read)."""
    _open_local.cache_clear()
    hf_streaming._resolve_url.cache_clear()
    hf_streaming._fetch_header.cache_clear()


class _Store:
    """Shared machinery of :class:`SequenceStore` and :class:`PointStore`: ``manifest`` /
    ``meta`` from :func:`read_store_manifest`, ``layers`` (``None`` = all available for the
    kind), an :class:`HfBackend` read through ``hf_streaming.stream_tensor``, any other
    backend through a cached ``safe_open`` handle; ``dtype=None`` keeps the stored bf16."""

    kind: str = ""          # "seq" or "points": which file family this store reads
    file_column: str = ""   # the manifest column naming the part

    def __init__(self, backend: StoreBackend, folder: str, manifest: pd.DataFrame, meta: dict, *,
                 layers=None, revision: Optional[str] = None, fetch_threads: Optional[int] = None,
                 dtype: Optional[torch.dtype] = None):
        if self.file_column not in manifest.columns or "rollout_id" not in manifest.columns:
            raise ValueError(f"manifest lacks {self.file_column!r}/'rollout_id' — pass read_store_manifest's frame")
        self.backend = backend
        self.folder = str(folder or "").strip("/")
        self.meta = meta
        self.layers = resolve_kind_layers(layers, meta, self.kind)
        self.hidden = int(meta["hidden_size"])
        self.dtype = dtype
        self.streaming = isinstance(backend, HfBackend)
        self.revision = revision if revision is not None else (getattr(backend, "revision", None) if self.streaming else None)
        self.fetch_threads = self._default_threads() if fetch_threads is None else max(1, int(fetch_threads))
        ids = manifest["rollout_id"].astype(str).tolist()
        parts = [part_index(v) for v in manifest[self.file_column].tolist()]
        self._part: dict[str, int] = dict(zip(ids, parts))

    def _default_threads(self) -> int:
        return min(len(self.layers), DEFAULT_FETCH_THREADS)

    # -- lookup ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._part)

    def __contains__(self, rollout_id) -> bool:
        return str(rollout_id) in self._part

    @property
    def rollout_ids(self) -> list[str]:
        return list(self._part)

    def part_of(self, rollout_id: str) -> int:
        try:
            return self._part[str(rollout_id)]
        except KeyError:
            raise KeyError(f"rollout_id {rollout_id!r} is not in the store ({len(self._part)} rows)") from None

    def file_of(self, rollout_id: str, layer: int) -> str:
        """The part file (relative to the folder) holding ``rollout_id`` at ``layer``."""
        return STORE_FILES[self.kind].format(layer=int(layer), part=self.part_of(rollout_id))

    def _check_layers(self, layers) -> list[int]:
        return self.layers if layers is None else resolve_store_layers(layers, self.layers)

    # -- one tensor --------------------------------------------------------------

    def _fetch(self, rollout_id: str, layer: int) -> torch.Tensor:
        """The stored tensor of one (rollout, layer): ``[n_cot, hidden]`` or ``[len(POINTS), hidden]``."""
        rollout_id = str(rollout_id)
        path = store_path(self.folder, self.file_of(rollout_id, layer))
        if self.streaming:
            # The backend's own token also signs the streamed Range URLs (None = the cached login).
            token = getattr(getattr(self.backend, "api", None), "token", None)
            return hf_streaming.stream_tensor(self.backend.repo_id, path, rollout_id, token=token,
                                              dtype=self.dtype, revision=self.revision)
        t = local_handle(self.backend.download_file(path)).get_tensor(rollout_id)
        return t if self.dtype is None else t.to(self.dtype)

    def _fetch_layers(self, rollout_id: str, layers: list[int]) -> dict[int, torch.Tensor]:
        rollout_id = str(rollout_id)
        self.part_of(rollout_id)   # unknown id raises before any thread starts
        if self.streaming and self.fetch_threads > 1 and len(layers) > 1:
            with ThreadPoolExecutor(max_workers=min(self.fetch_threads, len(layers))) as pool:
                tensors = list(pool.map(lambda l: self._fetch(rollout_id, l), layers))
        else:
            tensors = [self._fetch(rollout_id, l) for l in layers]
        return dict(zip(layers, tensors))


class SequenceStore(_Store):
    """``rollout_id`` → ``{layer: Tensor[n_cot_tokens, hidden]}`` from the ``seq_layer_*`` files."""

    kind = "seq"
    file_column = "seq_file"

    def get(self, rollout_id: str, layers=None) -> dict[int, torch.Tensor]:
        return self._fetch_layers(rollout_id, self._check_layers(layers))


class PointStore(_Store):
    """``rollout_id`` → one probe point per layer from the ``points_layer_*`` files:
    ``get(rollout_id, point)`` gives ``{layer: Tensor[hidden]}``, ``get_many(ids, point)``
    ``{layer: Tensor[n, hidden]}`` stacked in request order (each distinct id fetched once)."""

    kind = "points"
    file_column = "points_file"

    def _default_threads(self) -> int:
        return DEFAULT_FETCH_THREADS

    @staticmethod
    def point_index(point: str) -> int:
        if point not in POINTS:
            raise ValueError(f"point must be one of {POINTS}, got {point!r}")
        return POINTS.index(point)

    def get(self, rollout_id: str, point: str, layers=None) -> dict[int, torch.Tensor]:
        idx = self.point_index(point)
        full = self._fetch_layers(rollout_id, self._check_layers(layers))
        return {l: t[idx] for l, t in full.items()}

    def get_many(self, rollout_ids: Sequence[str], point: str, layers=None) -> dict[int, torch.Tensor]:
        idx = self.point_index(point)
        layers = self._check_layers(layers)
        ids = [str(r) for r in rollout_ids]
        for r in ids:
            self.part_of(r)
        n = len(ids)
        if n == 0:
            dtype = self.dtype or STORE_DTYPE
            return {l: torch.empty((0, self.hidden), dtype=dtype) for l in layers}
        positions: dict[str, list[int]] = {}
        for i, r in enumerate(ids):
            positions.setdefault(r, []).append(i)
        out: dict[int, Optional[torch.Tensor]] = {l: None for l in layers}

        def place(layer: int, rid: str, t: torch.Tensor) -> None:
            row = t[idx]
            if out[layer] is None:
                out[layer] = torch.empty((n, row.shape[-1]), dtype=row.dtype)
            out[layer][positions[rid]] = row

        if self.streaming:
            tasks = [(l, r) for l in layers for r in positions]
            # Bounded chunks: ``pool.map`` keeps every unconsumed result alive and, on a
            # failure, waits for every submitted task before the error propagates.
            chunk = self.fetch_threads * 4
            with ThreadPoolExecutor(max_workers=min(self.fetch_threads, len(tasks))) as pool:
                for start in range(0, len(tasks), chunk):
                    batch = tasks[start:start + chunk]
                    for (l, r), t in zip(batch, pool.map(lambda lr: self._fetch(lr[1], lr[0]), batch)):
                        place(l, r, t)
        else:
            # One handle per file: ids grouped by part, every layer of a part in turn.
            by_part: dict[int, list[str]] = {}
            for r in positions:
                by_part.setdefault(self.part_of(r), []).append(r)
            for part, rids in sorted(by_part.items()):
                for l in layers:
                    handle = local_handle(self.backend.download_file(
                        store_path(self.folder, STORE_FILES[self.kind].format(layer=l, part=part))))
                    for r in rids:
                        t = handle.get_tensor(r)
                        place(l, r, t if self.dtype is None else t.to(self.dtype))
        return out   # type: ignore[return-value]


def _as_numbers(values, name: str, n: int, cast) -> list:
    out = [cast(v) for v in list(values)]
    if len(out) != n:
        raise ValueError(f"{name} has {len(out)} entries for {n} rollout_ids")
    return out


class _RolloutDataset(Dataset):
    """Parallel ``rollout_ids`` / ``labels`` / ``weights`` (weights default to 1.0), every
    id checked against the store up front; items are ``(H, label, weight)`` with a 0-d
    ``long`` label and ``float32`` weight tensor."""

    def __init__(self, rollout_ids: Sequence[str], labels: Sequence[int], store: _Store, *,
                 weights: Optional[Sequence[float]] = None):
        self.rollout_ids = [str(r) for r in rollout_ids]
        n = len(self.rollout_ids)
        self.labels = _as_numbers(labels, "labels", n, int)
        self.weights = [1.0] * n if weights is None else _as_numbers(weights, "weights", n, float)
        self.store = store
        for r in self.rollout_ids:
            store.part_of(r)

    def __len__(self) -> int:
        return len(self.rollout_ids)

    def _item(self, idx: int, H: dict[int, torch.Tensor]):
        return H, torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(self.weights[idx], dtype=torch.float32)


class RolloutSequenceDataset(_RolloutDataset):
    """``torch`` dataset over a :class:`SequenceStore`: item ``i`` is
    ``(store.get(rollout_ids[i]), label, weight)``."""

    def __init__(self, rollout_ids: Sequence[str], labels: Sequence[int], store: SequenceStore, *,
                 weights: Optional[Sequence[float]] = None):
        super().__init__(rollout_ids, labels, store, weights=weights)

    def __getitem__(self, idx: int):
        return self._item(idx, self.store.get(self.rollout_ids[idx]))


# ---------------------------------------------------------------------------
# Mirror: a local copy of a store folder for training (points + chosen seq layers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MirrorFile:
    """One file of a mirror plan: ``name`` relative to the folder, ``kind``, ``layer``,
    the source ``size`` and the ``action`` (``download`` or ``skip``)."""

    name: str
    kind: str
    layer: int
    size: int
    action: str


@dataclass
class MirrorPlan:
    """What :func:`mirror_store` will do: the source manifest text + validated meta, the
    seq ``layers`` asked for, whether points come along, every file with its action, the
    destination's free bytes and the meta it already carries (``{}`` when new)."""

    folder: str
    dest: Path
    layers: list[int]
    include_points: bool
    manifest_text: str
    meta: dict
    files: list[MirrorFile]
    free_bytes: Optional[int]
    previous_meta: dict
    src_sizes: dict[str, int] = field(default_factory=dict)   # every source file of the folder (one listing)

    @property
    def to_download(self) -> list[MirrorFile]:
        return [f for f in self.files if f.action == "download"]

    @property
    def skipped(self) -> list[MirrorFile]:
        return [f for f in self.files if f.action == "skip"]

    @property
    def bytes_total(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def bytes_to_download(self) -> int:
        return sum(f.size for f in self.to_download)

    def bytes_by_layer(self, kind: str) -> dict[int, int]:
        out: dict[int, int] = {}
        for f in self.files:
            if f.kind == kind:
                out[f.layer] = out.get(f.layer, 0) + f.size
        return dict(sorted(out.items()))

    def summary(self) -> dict:
        return {"folder": self.folder, "dest": str(self.dest), "layers": list(self.layers),
                "include_points": self.include_points, "n_files": len(self.files),
                "n_download": len(self.to_download), "n_skip": len(self.skipped),
                "bytes_total": self.bytes_total, "bytes_to_download": self.bytes_to_download,
                "bytes_skipped": self.bytes_total - self.bytes_to_download, "free_bytes": self.free_bytes,
                "seq_bytes_by_layer": self.bytes_by_layer("seq"), "points_bytes_by_layer": self.bytes_by_layer("points")}


def _free_bytes(path: Path) -> Optional[int]:
    """Free bytes of the volume ``path`` is (or will be) on; None when unknown."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return None


def _relative_sizes(backend: StoreBackend, folder: str) -> dict[str, int]:
    prefix = store_path(folder, "")
    return {p[len(prefix):]: s for p, s in backend.file_sizes(prefix).items() if p.startswith(prefix)}


def plan_mirror(src: StoreBackend, folder: str, dst_dir: Path, *, layers=None, include_points: bool = True) -> MirrorPlan:
    """Plan the mirror of ``folder`` from ``src`` into ``dst_dir/<folder>``: the ``seq_*``
    files of ``layers`` (``None`` / ``[]`` = none; a layer the source lacks for seq raises)
    and, with ``include_points``, the ``points_*`` files of every available layer, for the
    parts the manifest names. A file present at the destination at the source's size is
    ``skip``, anything else ``download``; a manifest naming a file the listing lacks is an
    error. Nothing is written.
    """
    dst_dir = Path(dst_dir)
    src_prefix = store_path(folder, "")
    src_sizes = _relative_sizes(src, folder)   # one listing: existence, names and sizes
    for name in (STORE_MANIFEST_NAME, STORE_MANIFEST_META_NAME):
        if name not in src_sizes:
            raise FileNotFoundError(f"no {name} under {src_prefix!r} on {src.describe()}")
    manifest_text = src.read_text(store_path(folder, STORE_MANIFEST_NAME))
    meta = check_store_meta(json.loads(src.read_text(store_path(folder, STORE_MANIFEST_META_NAME))))
    manifest = check_store_manifest_frame(pd.read_csv(io.StringIO(manifest_text), low_memory=False))
    seq_layers = [] if not layers else resolve_kind_layers(list(layers), meta, "seq")
    points_layers = available_layers(meta, "points") if include_points else []
    if not seq_layers and not points_layers:
        raise ValueError("nothing to mirror: no seq layer requested and the points files excluded")
    local = LocalBackend(dst_dir)
    local_sizes = _relative_sizes(local, folder) if dst_dir.is_dir() else {}

    wanted: list[tuple[str, str, int]] = []
    for kind, kind_layers in (("seq", seq_layers), ("points", points_layers)):
        parts = {part_index(v) for v in manifest[STORE_FILE_COLUMNS[kind]].tolist()}
        for layer in kind_layers:
            for name in needed_files(parts, kind, [layer]):
                wanted.append((name, kind, layer))
    missing = sorted(name for name, _, _ in wanted if name not in src_sizes)
    if missing:
        raise ValueError(
            f"the source manifest of {src_prefix!r} on {src.describe()} names {len(missing)} file(s) the listing "
            f"lacks: {missing[:6]}{' …' if len(missing) > 6 else ''} — the source store is inconsistent"
        )
    files = [MirrorFile(name, kind, layer, src_sizes[name],
                        "skip" if local_sizes.get(name) == src_sizes[name] else "download")
             for name, kind, layer in wanted]
    previous_meta: dict = {}
    prev_meta_path = dst_dir / src_prefix / STORE_MANIFEST_META_NAME
    if prev_meta_path.is_file():
        try:
            previous_meta = json.loads(prev_meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_meta = {}
    return MirrorPlan(folder=str(folder).strip("/"), dest=dst_dir, layers=seq_layers, include_points=include_points,
                      manifest_text=manifest_text, meta=meta, files=files,
                      free_bytes=_free_bytes(dst_dir / src_prefix), previous_meta=previous_meta, src_sizes=src_sizes)


def _clear_tmp_entries(folder_dir: Path) -> int:
    """Remove ``.tmp-*`` files / directories a killed run left in the mirror folder."""
    n = 0
    if not folder_dir.is_dir():
        return 0
    for entry in folder_dir.iterdir():
        if entry.name.startswith(".tmp-"):
            shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
            n += 1
    return n


def _mirror_one(src: StoreBackend, folder: str, folder_dir: Path, f: MirrorFile, *, attempts: int = 2) -> int:
    """Download one planned file into ``folder_dir`` (temp file + ``os.replace``), verifying
    its size against the source listing (retried once, then raised). Returns the bytes written."""
    target = folder_dir / f.name
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        tmp = folder_dir / f".tmp-{f.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            src.download_to(store_path(folder, f.name), tmp)
            got = tmp.stat().st_size
            if got != f.size:
                raise RuntimeError(f"{f.name}: downloaded {got} bytes, the source listing says {f.size}")
            os.replace(tmp, target)
            return f.size
        except Exception as e:  # noqa: BLE001 — retried once, then reported per file
            last_error = e
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                break
    raise RuntimeError(f"{f.name}: {last_error}") from last_error


def _complete_layers(meta: dict, manifest: pd.DataFrame, kind: str, local_sizes: dict[str, int],
                     src_sizes: dict[str, int]) -> list[int]:
    """The stored layers whose every ``<kind>`` file the copy holds at the source's size."""
    parts = {part_index(v) for v in manifest[STORE_FILE_COLUMNS[kind]].tolist()}
    out = []
    for layer in available_layers(meta, kind):
        needed = needed_files(parts, kind, [layer])
        if needed and all(local_sizes.get(n) is not None and local_sizes.get(n) == src_sizes.get(n) for n in needed):
            out.append(layer)
    return out


def mirror_store(src: StoreBackend, folder: str, dst_dir: Path, *, layers=None, include_points: bool = True,
                 workers: int = DEFAULT_MIRROR_WORKERS, log: Optional[Callable[[str], None]] = None,
                 plan: Optional[MirrorPlan] = None, check_disk: bool = True) -> dict:
    """Mirror ``folder`` of ``src`` into ``dst_dir/<folder>``: download the plan's files
    (:func:`plan_mirror`, or ``plan``) with ``workers`` threads, size-verified, skipping
    files already present; then write the manifest and the meta (the source's plus
    ``mirrored_layers`` / ``mirrored_points_layers`` — the layers whose files are all
    present locally, a layer the source grew past reported as ``dropped`` — and
    ``mirror_source``) and validate the copy. A failed download keeps the previous
    manifest so a rerun resumes; ``check_disk`` refuses a plan above the free space.
    Returns a report.
    """
    log = log or (lambda msg: None)
    plan = plan if plan is not None else plan_mirror(src, folder, dst_dir, layers=layers, include_points=include_points)
    folder = plan.folder
    dst_dir = Path(plan.dest)
    if check_disk and plan.free_bytes is not None and plan.bytes_to_download > plan.free_bytes:
        raise RuntimeError(
            f"the mirror needs {plan.bytes_to_download / GB:.1f} GB but {dst_dir} has {plan.free_bytes / GB:.1f} GB free"
        )
    unrequested = [f for f in plan.to_download if f.kind == "seq" and f.layer not in plan.layers]
    if unrequested:   # cannot happen through plan_mirror; guards a hand-built plan
        raise ValueError(f"the plan downloads seq files of layers not requested: {sorted({f.layer for f in unrequested})}")
    folder_dir = dst_dir / folder
    folder_dir.mkdir(parents=True, exist_ok=True)
    cleared = _clear_tmp_entries(folder_dir)
    if cleared:
        log(f"removed {cleared} leftover .tmp-* entr{'y' if cleared == 1 else 'ies'} from {folder_dir}")

    started = time.monotonic()
    todo = plan.to_download
    log(f"Mirror {src.describe()}/{folder} → {folder_dir}: {len(todo)} file(s) to download "
        f"({plan.bytes_to_download / GB:.2f} GB), {len(plan.skipped)} already present "
        f"({(plan.bytes_total - plan.bytes_to_download) / GB:.2f} GB), {workers} worker(s)")
    failures: dict[str, str] = {}
    done_bytes = 0
    if todo:
        pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        try:
            futures = {pool.submit(_mirror_one, src, folder, folder_dir, f): f for f in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                f = futures[fut]
                try:
                    done_bytes += fut.result()
                    log(f"  [{i}/{len(todo)}] {f.name} ({f.size / GB:.2f} GB) ok")
                except Exception as e:  # noqa: BLE001
                    failures[f.name] = str(e)
                    log(f"  [{i}/{len(todo)}] {f.name} FAILED: {e}")
        except BaseException:   # Ctrl-C: drop the queued files
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
    if failures:
        names = sorted(failures)
        raise RuntimeError(
            f"{len(failures)} of {len(todo)} file(s) failed to mirror (the manifest was not updated; re-run to "
            f"resume): {names[:6]}{' …' if len(names) > 6 else ''} — first error: {failures[names[0]]}"
        )

    # Manifest + meta last: what the copy really holds, per file kind.
    manifest = check_store_manifest_frame(pd.read_csv(io.StringIO(plan.manifest_text), low_memory=False))
    src_sizes = dict(plan.src_sizes)
    src_sizes.update({f.name: f.size for f in plan.files})
    local = LocalBackend(dst_dir)
    local_sizes = _relative_sizes(local, folder)
    meta = {k: v for k, v in plan.meta.items() if k not in MIRRORED_LAYERS_KEYS.values() and k != MIRROR_SOURCE_KEY}
    mirrored = {kind: _complete_layers(plan.meta, manifest, kind, local_sizes, src_sizes) for kind in STORE_FILE_KINDS}
    dropped = {}
    for kind, key in MIRRORED_LAYERS_KEYS.items():
        before = plan.previous_meta.get(key) or []
        lost = sorted(set(int(l) for l in before) - set(mirrored[kind]))
        if lost:
            dropped[kind] = lost
            log(f"[warn] {kind} layers {lost} were mirrored before but are incomplete now (the source grew) — "
                f"re-run with --layers {','.join(str(l) for l in lost)}")
        meta[key] = mirrored[kind]
    source: dict = {"backend": "hf" if isinstance(src, HfBackend) else "local", "source": src.describe(),
                    "mirrored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if isinstance(src, HfBackend):
        source.update(repo_id=src.repo_id, revision=src.revision)
    else:
        source.update(root=str(getattr(src, "root", "")))
    meta[MIRROR_SOURCE_KEY] = source
    local.write_text(store_path(folder, STORE_MANIFEST_NAME), plan.manifest_text)
    local.write_text(store_path(folder, STORE_MANIFEST_META_NAME), json.dumps(meta, indent=2))
    read_store_manifest(local, folder)   # the copy validates with the reader's own checks
    seconds = time.monotonic() - started
    report = {**plan.summary(), "n_downloaded": len(todo), "bytes_downloaded": done_bytes,
              "n_skipped": len(plan.skipped), "seconds": seconds,
              "mirrored_layers": mirrored["seq"], "mirrored_points_layers": mirrored["points"],
              "dropped": dropped, "mirror_source": source}
    log(f"Mirrored {len(todo)} file(s), {done_bytes / GB:.2f} GB in {seconds:.0f} s; seq layers {mirrored['seq']}, "
        f"points layers {mirrored['points']} → {folder_dir}")
    return report
