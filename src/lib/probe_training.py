"""Probe training: config schema, fold plan, feature adapters, the training loop and evaluation.

One loop trains the three probe heads of :mod:`src.lib.probes` on a probe dataset
(:mod:`src.lib.probe_datasets`) with leave-one-hint-out folds. Every trial seeds torch with
``training.seed + fold_index``; sampler and prefix seeds derive from the same number, so two
runs of one config give identical folds and predictions on CPU.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from src.lib.activation_store import (
    BACKENDS,
    POINTS,
    STORAGE_CONFIG_KEYS,
    PointStore,
    RolloutSequenceDataset,
    SequenceStore,
    stable_seed,
)
from src.lib.paths import resolve_data_path
from src.lib.probe_datasets import fold_roles
from src.lib.probe_metrics import summarize
from src.lib.probes import PROBE_TYPES, allowed_probe_keys, build_probe, load_probe, p_detection_target, to_checkpoint
from src.lib.report import PREDICTION_COLUMNS, case_view
from src.lib.rollout_manifest import CASES
from src.lib.text_features import (
    TextPipelineConfig,
    VectorizerConfig,
    fit_vectorizer,
    n_features as vectorizer_n_features,
    to_torch_sparse,
    transform,
)

__all__ = [
    "Batch", "CASE_MODES", "DEFAULT_TRIAL_ID", "DataloaderConfig", "DatasetConfig", "ENSEMBLE_KEY",
    "Fold", "FoldResult", "ModelConfig", "NA", "OutputConfig", "PointSource", "SequenceSource", "TEST_SETS",
    "TfidfSource", "TokenBudgetBatchSampler", "TrainConfig", "TrainingConfig", "Trial", "UnitResult",
    "apply_trial", "class_weights", "collate_multi_layer", "ensemble_predictions", "evaluate", "expand_grid",
    "folds_from_json", "folds_to_json", "parse_train_config", "plan_folds", "sample_prefix_lengths", "train_fold",
]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Batches of sample indices bounded by padded tokens.

    Samples are shuffled (when ``shuffle``), cut into chunks of ``max_batch_size × bucket_mult``,
    sorted by length inside each chunk and filled so that ``len(batch) × longest ≤ max_batch_tokens``
    and ``len(batch) ≤ max_batch_size``; a single sample over the budget forms a batch of one.
    Each ``__iter__`` re-draws from ``seed + epoch``; :meth:`batches` is the deterministic plan.
    """

    def __init__(self, lengths, max_batch_tokens: int, max_batch_size: int, *,
                 shuffle: bool, seed: int = 0, bucket_mult: int = 64):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if max_batch_tokens < 1 or max_batch_size < 1:
            raise ValueError("max_batch_tokens and max_batch_size must be >= 1")
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.bucket_mult = max(1, int(bucket_mult))
        self.epoch = 0

    def _plan(self, rng: np.random.Generator | None) -> list[list[int]]:
        n = len(self.lengths)
        order = rng.permutation(n) if (self.shuffle and rng is not None) else np.arange(n)
        chunk = self.max_batch_size * self.bucket_mult
        batches: list[list[int]] = []
        for start in range(0, n, chunk):
            part = order[start:start + chunk]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            cur: list[int] = []
            cur_max = 0
            for i in part:
                L = int(self.lengths[i])
                new_max = max(cur_max, L)
                if cur and (len(cur) + 1 > self.max_batch_size
                            or new_max * (len(cur) + 1) > self.max_batch_tokens):
                    batches.append(cur)
                    cur, new_max = [], L
                cur.append(int(i))
                cur_max = new_max
            if cur:
                batches.append(cur)
        if self.shuffle and rng is not None:
            rng.shuffle(batches)
        return batches

    def batches(self) -> list[list[int]]:
        """Deterministic plan (no shuffling), for evaluation loaders."""
        return self._plan(None)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch) if self.shuffle else None
        self.epoch += 1
        return iter(self._plan(rng))

    def __len__(self) -> int:
        return len(self._plan(np.random.default_rng(self.seed) if self.shuffle else None))


def collate_multi_layer(batch):
    """Pad ``(H_dict, y, w)`` items (``H_dict = {layer: [seq_len, d]}``) in their stored dtype into
    ``(padded {layer: [B, T, d]}, labels [B], lengths [B])``."""
    Hs, ys, _ = zip(*batch)
    layer_nums = list(Hs[0].keys())
    lengths = [next(iter(h.values())).size(0) for h in Hs]
    max_t = max(lengths)
    first = next(iter(Hs[0].values()))
    padded = {l: torch.zeros(len(Hs), max_t, first.size(1), dtype=first.dtype) for l in layer_nums}
    for i, h in enumerate(Hs):
        for l in layer_nums:
            padded[l][i, :lengths[i]] = h[l]
    return padded, torch.stack(ys), torch.tensor(lengths, dtype=torch.long)


def sample_prefix_lengths(lengths: torch.Tensor, min_frac: float,
                          rng: np.random.Generator) -> torch.Tensor:
    """One random prefix length per sample in ``[max(1, L·min_frac), L]``."""
    out = []
    for L in lengths.tolist():
        lo = max(1, int(L * min_frac))
        out.append(int(rng.integers(lo, int(L) + 1)))
    return torch.tensor(out, dtype=lengths.dtype)


def class_weights(labels, mode: str) -> torch.Tensor | None:
    """Cross-entropy class weights: ``balanced`` = inverse class frequency
    normalised to mean 1; ``none`` = unweighted."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"class_weighting must be 'balanced' or 'none', got {mode!r}")
    y = np.asarray(labels).astype(int)
    counts = np.array([(y == 0).sum(), (y == 1).sum()], dtype=float)
    if (counts == 0).any():
        return None
    w = counts.sum() / (2 * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

CASE_MODES = ("both",) + CASES
CLASS_WEIGHTING_MODES = ("balanced", "none")
#: The unit slot a probe type does not use (the layer of a TF-IDF probe, the point of a sequence probe).
NA = "na"
DEFAULT_TRIAL_ID = "default"
#: ``probe:`` keys that select the feature *source*, not a hyperparameter — never swept.
UNSWEEPABLE_PROBE_KEYS = ("type", "points")
TOP_LEVEL_KEYS = ("dataset", "model", "probe", "training", "search", "dataloader", "output")
#: Loop-level ``probe:`` defaults per type (constructor keys take the class defaults in ``build_probe``).
PROBE_LOOP_DEFAULTS = {
    "attention": {"standardize": True, "norm_tokens": 500_000},
    "linear": {"points": list(POINTS), "standardize": True},
    "tfidf": {"text": {}, "vectorizer": {}},
}


def _check_keys(block: dict | None, allowed: Iterable[str], name: str, *, required: Iterable[str] = ()) -> dict:
    """``block`` as a dict with only ``allowed`` keys (an unknown key raises listing the allowed ones)."""
    block = {} if block is None else block
    if not isinstance(block, dict):
        raise ValueError(f"{name} must be a mapping, got {type(block).__name__}")
    allowed = list(allowed)
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise ValueError(f"{name}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")
    missing = [k for k in required if block.get(k) is None]
    if missing:
        raise ValueError(f"{name}: missing required key(s) {missing}")
    return dict(block)


def _int(value, name: str, *, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < low:
        raise ValueError(f"{name} must be an int >= {low}, got {value!r}")
    return int(value)


def _float(value, name: str, *, low: float, low_inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < low or (not low_inclusive and value == low):
        raise ValueError(f"{name} must be {'>=' if low_inclusive else '>'} {low}, got {value!r}")
    return value


def _str_list(value, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{name} must be null or a non-empty list of strings, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} has duplicates: {list(value)}")
    return tuple(value)


@dataclass(frozen=True)
class DatasetConfig:
    """``dataset:`` — the probe dataset parquet, the held-out styles, the training-row case filter."""

    path: Path
    folds: tuple[str, ...] | None = None
    cases: str = "both"

    def __post_init__(self):
        if self.cases not in CASE_MODES:
            raise ValueError(f"dataset.cases must be one of {CASE_MODES}, got {self.cases!r}")


#: The keys of ``model.activations``: ``make_backend``'s block plus ``folder`` (the store folder to read).
ACTIVATIONS_CONFIG_KEYS = frozenset(STORAGE_CONFIG_KEYS | {"folder"})


@dataclass(frozen=True)
class ModelConfig:
    """``model:`` — the subject model, its activation store (``activations`` = a ``make_backend`` block;
    ``folder`` = the store folder to read, ``None`` = the dataset's ``name``) and the layers to probe."""

    subject_model: str
    activations: dict
    layers: tuple[int, ...] | None = None
    folder: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """``training:`` — the optimiser, batching, prefix and early-stopping knobs."""

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 16
    max_batch_tokens: int = 65536
    min_prefix_frac: float = 1.0
    grad_clip: float = 1.0
    patience: int = 5
    class_weighting: str = "balanced"
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "epochs", _int(self.epochs, "training.epochs", low=1))
        object.__setattr__(self, "batch_size", _int(self.batch_size, "training.batch_size", low=1))
        object.__setattr__(self, "max_batch_tokens", _int(self.max_batch_tokens, "training.max_batch_tokens", low=1))
        object.__setattr__(self, "patience", _int(self.patience, "training.patience", low=0))
        object.__setattr__(self, "seed", _int(self.seed, "training.seed", low=0))
        object.__setattr__(self, "lr", _float(self.lr, "training.lr", low=0.0, low_inclusive=False))
        object.__setattr__(self, "weight_decay", _float(self.weight_decay, "training.weight_decay", low=0.0))
        object.__setattr__(self, "grad_clip", _float(self.grad_clip, "training.grad_clip", low=0.0))
        frac = _float(self.min_prefix_frac, "training.min_prefix_frac", low=0.0, low_inclusive=False)
        if frac > 1.0:
            raise ValueError(f"training.min_prefix_frac must be in (0, 1], got {frac!r}")
        object.__setattr__(self, "min_prefix_frac", frac)
        mode = self.class_weighting
        if isinstance(mode, bool):        # ``true`` / ``false`` in YAML
            mode = "balanced" if mode else "none"
        if mode not in CLASS_WEIGHTING_MODES:
            raise ValueError(f"training.class_weighting must be true/false or one of {CLASS_WEIGHTING_MODES}, "
                             f"got {self.class_weighting!r}")
        object.__setattr__(self, "class_weighting", mode)


@dataclass(frozen=True)
class DataloaderConfig:
    """``dataloader:`` — the sequence adapter's DataLoader settings."""

    num_workers: int = 8
    fetch_threads: int = 4
    prefetch_factor: int = 2

    def __post_init__(self):
        object.__setattr__(self, "num_workers", _int(self.num_workers, "dataloader.num_workers", low=0))
        object.__setattr__(self, "fetch_threads", _int(self.fetch_threads, "dataloader.fetch_threads", low=1))
        object.__setattr__(self, "prefetch_factor", _int(self.prefetch_factor, "dataloader.prefetch_factor", low=1))


@dataclass(frozen=True)
class OutputConfig:
    """``output:`` — where the run directory goes (``<dir>/<stamp>_<run_name>/``)."""

    dir: Path
    run_name: str | None = None


@dataclass(frozen=True)
class TrainConfig:
    """The whole validated config of a training run (see :func:`parse_train_config`)."""

    dataset: DatasetConfig
    model: ModelConfig | None
    probe: dict
    training: TrainingConfig
    search: dict
    dataloader: DataloaderConfig
    output: OutputConfig

    @property
    def probe_type(self) -> str:
        return str(self.probe["type"])

    def to_dict(self) -> dict:
        """A JSON-serialisable view (paths as strings) — what a run writes as its resolved config."""
        out = {
            "dataset": {**asdict(self.dataset), "path": str(self.dataset.path),
                        "folds": None if self.dataset.folds is None else list(self.dataset.folds)},
            "model": None if self.model is None else {
                "subject_model": self.model.subject_model,
                "activations": {**self.model.activations, "folder": self.model.folder},
                "layers": None if self.model.layers is None else list(self.model.layers)},
            "probe": json.loads(json.dumps(self.probe)),
            "training": asdict(self.training),
            "search": json.loads(json.dumps(self.search)),
            "dataloader": asdict(self.dataloader),
            "output": {"dir": str(self.output.dir), "run_name": self.output.run_name},
        }
        return out


def _parse_probe(block, probe_type_hint: str | None = None) -> dict:
    """The ``probe:`` block validated against :func:`allowed_probe_keys`, loop-level defaults filled,
    ``text`` / ``vectorizer`` / ``points`` checked. Returns a plain dict (what ``build_probe`` takes)."""
    if not isinstance(block, dict) or block.get("type") not in PROBE_TYPES:
        raise ValueError(f"probe.type must be one of {PROBE_TYPES}, got "
                         f"{block.get('type') if isinstance(block, dict) else block!r}")
    probe_type = str(block["type"])
    cfg = _check_keys(block, allowed_probe_keys(probe_type), "probe")
    for key, default in PROBE_LOOP_DEFAULTS[probe_type].items():
        if cfg.get(key) is None:
            cfg[key] = json.loads(json.dumps(default))
    if probe_type == "linear":
        points = cfg["points"]
        if not isinstance(points, (list, tuple)) or not points or any(p not in POINTS for p in points):
            raise ValueError(f"probe.points must be a non-empty subset of {list(POINTS)}, got {points!r}")
        if len(set(points)) != len(points):
            raise ValueError(f"probe.points has duplicates: {list(points)}")
        cfg["points"] = [str(p) for p in points]
    if probe_type == "tfidf":
        # Validated through the text-side dataclasses; kept as dicts so a trial's overrides merge into them.
        cfg["text"] = TextPipelineConfig.from_dict(cfg["text"]).to_dict()
        cfg["vectorizer"] = VectorizerConfig.from_dict(cfg["vectorizer"]).to_dict()
    if probe_type in ("attention", "linear") and not isinstance(cfg.get("standardize"), bool):
        raise ValueError(f"probe.standardize must be a bool, got {cfg.get('standardize')!r}")
    if probe_type == "attention":
        cfg["norm_tokens"] = _int(cfg["norm_tokens"], "probe.norm_tokens", low=1)
    if cfg.get("num_classes") is not None:
        # The labels, the class weights and ``p_detection_target`` are binary.
        if _int(cfg["num_classes"], "probe.num_classes", low=2) != 2:
            raise ValueError(f"probe.num_classes must be 2 (binary labels), got {cfg['num_classes']!r}")
        cfg["num_classes"] = 2
    return cfg


def _check_prefix_rule(training: TrainingConfig, probe: dict) -> None:
    if training.min_prefix_frac < 1.0 and probe["type"] != "attention":
        raise ValueError(f"training.min_prefix_frac < 1 (random CoT prefixes) is only valid for probe type "
                         f"'attention', not {probe['type']!r}")


def _parse_training(block: dict | None, probe_type: str) -> TrainingConfig:
    given = _check_keys(block, [f.name for f in fields(TrainingConfig)], "training")
    if given.get("min_prefix_frac") is None:
        # Only the sequence probe trains on random prefixes.
        given["min_prefix_frac"] = 0.1 if probe_type == "attention" else 1.0
    return TrainingConfig(**given)


def _search_key(key: str, probe: dict) -> str:
    """The canonical dotted form of a ``search:`` key — ``training.<k>``, ``probe.<k>`` or
    ``probe.vectorizer.<k>`` / ``probe.text.<k>`` — or a ``ValueError`` naming what exists."""
    if not isinstance(key, str) or not key:
        raise ValueError(f"search: keys must be non-empty strings, got {key!r}")
    training_keys = {f.name for f in fields(TrainingConfig)}
    probe_type = str(probe["type"])
    probe_keys = set(allowed_probe_keys(probe_type)) - set(UNSWEEPABLE_PROBE_KEYS) - {"text", "vectorizer"}
    nested = {"vectorizer": {f.name for f in fields(VectorizerConfig)},
              "text": {f.name for f in fields(TextPipelineConfig)}} if probe_type == "tfidf" else {}
    parts = key.split(".")
    if parts[0] in ("training", "probe") and len(parts) > 1:
        head, parts = parts[0], parts[1:]
    else:
        head = None
    if len(parts) == 1:
        leaf = parts[0]
        if head in (None, "training") and leaf in training_keys:
            return f"training.{leaf}"
        if head in (None, "probe") and leaf in probe_keys:
            return f"probe.{leaf}"
    elif len(parts) == 2 and head in (None, "probe") and parts[0] in nested and parts[1] in nested[parts[0]]:
        return f"probe.{parts[0]}.{parts[1]}"
    available = sorted(f"training.{k}" for k in training_keys) + sorted(f"probe.{k}" for k in probe_keys) + sorted(
        f"probe.{block}.{k}" for block, keys in nested.items() for k in keys)
    raise ValueError(f"search: key {key!r} is not a sweepable training/probe key for probe type {probe_type!r}; "
                     f"available: {available}")


def _parse_search(block, probe: dict) -> dict:
    block = {} if block is None else block
    if not isinstance(block, dict):
        raise ValueError(f"search must be a mapping, got {type(block).__name__}")
    out: dict[str, Any] = {}
    for key, value in block.items():
        canonical = _search_key(key, probe)
        if canonical in out:
            raise ValueError(f"search: {key!r} and another key both name {canonical}")
        if isinstance(value, list) and not value:
            raise ValueError(f"search: {key!r} is an empty list — a sweep needs at least one value")
        out[canonical] = value
    return out


def parse_train_config(cfg: dict) -> TrainConfig:
    """Validate the YAML of a training run (``extends:`` already resolved by ``load_config``).

    ``model`` is required for ``linear`` / ``attention`` and forbidden for ``tfidf``;
    ``training.min_prefix_frac`` defaults to 0.1 for ``attention`` and 1.0 otherwise, and must be
    1.0 for the non-sequence probes; a list-valued ``search:`` key is swept. Unknown keys anywhere
    raise, and every grid point must itself build a valid probe.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    top = _check_keys(cfg, TOP_LEVEL_KEYS, "config", required=("dataset", "probe", "output"))
    probe = _parse_probe(top["probe"])
    probe_type = str(probe["type"])

    ds = _check_keys(top["dataset"], ("path", "folds", "cases"), "dataset", required=("path",))
    dataset = DatasetConfig(path=resolve_data_path(str(ds["path"])), folds=_str_list(ds.get("folds"), "dataset.folds"),
                            cases=ds.get("cases") or "both")

    model = None
    if probe_type == "tfidf":
        if top.get("model") is not None:
            raise ValueError("model: must be absent for probe type 'tfidf' (it reads no activations)")
    else:
        if top.get("model") is None:
            raise ValueError(f"model: is required for probe type {probe_type!r} (the activation store to read)")
        m = _check_keys(top["model"], ("subject_model", "activations", "layers"), "model",
                        required=("subject_model", "activations"))
        acts = _check_keys(m["activations"], ACTIVATIONS_CONFIG_KEYS, "model.activations", required=("backend",))
        if acts["backend"] not in BACKENDS:
            raise ValueError(f"model.activations.backend must be one of {BACKENDS}, got {acts['backend']!r}")
        folder = acts.pop("folder", None)
        if folder is not None:
            name = folder.strip().strip("/") if isinstance(folder, str) else ""
            if not name or "/" in name or name in (".", ".."):
                raise ValueError("model.activations.folder must be null (= the dataset's name) or a non-empty "
                                 f"store folder name without '/' (not '.' or '..'), got {folder!r}")
            folder = name
        layers = m.get("layers")
        if layers is not None:
            if not isinstance(layers, (list, tuple)) or not layers:
                raise ValueError(f"model.layers must be null or a non-empty list of ints, got {layers!r}")
            layers = tuple(_int(l, "model.layers entry", low=0) for l in layers)
            if len(set(layers)) != len(layers):
                raise ValueError(f"model.layers has duplicates: {list(layers)}")
        model = ModelConfig(subject_model=str(m["subject_model"]), activations=acts, layers=layers, folder=folder)

    training = _parse_training(top.get("training"), probe_type)
    _check_prefix_rule(training, probe)
    search = _parse_search(top.get("search"), probe)
    loader = DataloaderConfig(**_check_keys(top.get("dataloader"), [f.name for f in fields(DataloaderConfig)],
                                            "dataloader"))
    out = _check_keys(top["output"], ("dir", "run_name"), "output", required=("dir",))
    run_name = out.get("run_name")
    if run_name is not None and (not isinstance(run_name, str) or not run_name):
        raise ValueError(f"output.run_name must be null or a non-empty string, got {run_name!r}")
    output = OutputConfig(dir=resolve_data_path(str(out["dir"])), run_name=run_name)
    config = TrainConfig(dataset=dataset, model=model, probe=probe, training=training, search=search,
                         dataloader=loader, output=output)
    for trial in expand_grid(config):       # every grid point must itself be a valid config
        _, pcfg = apply_trial(config.training, config.probe, trial)
        # Constructor values (heads, aggregation, ...) are only checked by the probe classes.
        try:
            build_probe(pcfg, hidden_dim=1, n_features=1)
        except (ValueError, TypeError) as e:
            raise ValueError(f"probe: invalid block for trial {trial.id!r}: {e}") from e
    return config


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """One grid point: ``id`` (the swept ``key=value`` pairs, sorted, joined by ``,``;
    :data:`DEFAULT_TRIAL_ID` when nothing is swept) and ``overrides`` (canonical dotted key → value)."""

    id: str
    overrides: dict


def _render_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (int, str)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def _short_key(canonical: str) -> str:
    return canonical.split(".", 1)[1]


def expand_grid(cfg: TrainConfig) -> list[Trial]:
    """The trials of a config: the Cartesian product of every list-valued ``search:`` key (the first
    key varying slowest); scalar keys override in every trial. A config without lists has one trial."""
    swept = [(k, list(v)) for k, v in cfg.search.items() if isinstance(v, list)]
    scalars = {k: v for k, v in cfg.search.items() if not isinstance(v, list)}
    if not swept:
        return [Trial(id=DEFAULT_TRIAL_ID, overrides=dict(scalars))]
    trials = []
    for combo in itertools.product(*(values for _, values in swept)):
        point = dict(zip((k for k, _ in swept), combo))
        pairs = sorted((_short_key(k), v) for k, v in point.items())
        trial_id = ",".join(f"{k}={_render_value(v)}" for k, v in pairs)
        trials.append(Trial(id=trial_id, overrides={**scalars, **point}))
    ids = [t.id for t in trials]
    if len(set(ids)) != len(ids):
        raise ValueError(f"search: the grid has duplicate trial ids (repeated values in a list?): {ids}")
    return trials


def apply_trial(training: TrainingConfig, probe: dict, trial: Trial) -> tuple[TrainingConfig, dict]:
    """``training`` / ``probe`` with the trial's overrides applied (re-validated)."""
    train_kwargs = asdict(training)
    probe_cfg = json.loads(json.dumps(probe))
    for key, value in trial.overrides.items():
        parts = key.split(".")
        if parts[0] == "training" and len(parts) == 2:
            train_kwargs[parts[1]] = value
        elif parts[0] == "probe" and len(parts) == 2:
            probe_cfg[parts[1]] = value
        elif parts[0] == "probe" and len(parts) == 3 and parts[1] in ("text", "vectorizer"):
            probe_cfg[parts[1]] = {**probe_cfg.get(parts[1], {}), parts[2]: value}
        else:
            raise ValueError(f"trial {trial.id!r}: override key {key!r} is not canonical (expand_grid builds them)")
    new_training = TrainingConfig(**train_kwargs)
    new_probe = _parse_probe(probe_cfg)
    _check_prefix_rule(new_training, new_probe)
    return new_training, new_probe


# ---------------------------------------------------------------------------
# Fold plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One leave-one-hint-out fold: sorted ``rollout_id`` tuples per role."""

    held_out_style: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_id_ids: tuple[str, ...]
    test_ood_ids: tuple[str, ...]

    def ids(self, test_set: str) -> tuple[str, ...]:
        """The ids of a test set (``test_id`` / ``test_ood``)."""
        if test_set not in TEST_SETS:
            raise ValueError(f"test_set must be one of {TEST_SETS}, got {test_set!r}")
        return getattr(self, f"{test_set}_ids")


TEST_SETS = ("test_id", "test_ood")
_FOLD_ROLE_FIELDS = {"train": "train_ids", "val": "val_ids", "test_id": "test_id_ids", "test_ood": "test_ood_ids"}


def plan_folds(rows: pd.DataFrame, folds: Sequence[str] | None, *, cases: str = "both") -> list[Fold]:
    """One :class:`Fold` per held-out style (``folds`` null = every ``hint_style`` in ``rows``, sorted)
    from :func:`~src.lib.probe_datasets.fold_roles`. ``cases`` restricts ``train_ids`` / ``val_ids``
    only; the two test sets keep every case. Unknown styles raise listing the styles present."""
    if cases not in CASE_MODES:
        raise ValueError(f"cases must be one of {CASE_MODES}, got {cases!r}")
    for column in ("rollout_id", "hint_style", "case"):
        if column not in rows.columns:
            raise ValueError(f"rows lack the {column!r} column")
    present = sorted(set(rows["hint_style"].astype(str)))
    if folds is None:
        styles = present
    else:
        styles = list(folds)
        unknown = sorted(set(styles) - set(present))
        if unknown:
            raise ValueError(f"held-out style(s) {unknown} have no row; styles present: {present}")
        if len(set(styles)) != len(styles):
            raise ValueError(f"dataset.folds has duplicates: {styles}")
    ids = rows["rollout_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("rows carry duplicate rollout_ids")
    # ``case_view`` is the one sanctioned case filter (``both`` = its ``pooled`` view).
    trainable = set(case_view(rows, "pooled" if cases == "both" else cases)["rollout_id"].astype(str))
    plan = []
    for style in styles:
        roles = fold_roles(rows, style)
        picked = {}
        for role, attr in _FOLD_ROLE_FIELDS.items():
            mask = (roles == role).fillna(False).astype(bool)
            if role in ("train", "val"):
                mask &= ids.isin(trainable)
            picked[attr] = tuple(sorted(ids[mask.to_numpy()].tolist()))
        plan.append(Fold(held_out_style=str(style), **picked))
    return plan


def folds_to_json(plan: Sequence[Fold]) -> list[dict]:
    """``folds.json`` content: one dict per fold with the role id lists."""
    return [{"held_out_style": f.held_out_style, **{attr: list(getattr(f, attr)) for attr in _FOLD_ROLE_FIELDS.values()}}
            for f in plan]


def folds_from_json(data: Sequence[dict]) -> list[Fold]:
    """The inverse of :func:`folds_to_json` (ids re-sorted, unknown keys rejected)."""
    plan = []
    for entry in data:
        entry = _check_keys(entry, ("held_out_style", *_FOLD_ROLE_FIELDS.values()), "folds.json entry",
                            required=("held_out_style", *_FOLD_ROLE_FIELDS.values()))
        plan.append(Fold(held_out_style=str(entry["held_out_style"]),
                         **{attr: tuple(sorted(str(i) for i in entry[attr])) for attr in _FOLD_ROLE_FIELDS.values()}))
    return plan


# ---------------------------------------------------------------------------
# Feature adapters
# ---------------------------------------------------------------------------

ENSEMBLE_KEY = "ensemble"


class Batch(NamedTuple):
    """What every adapter yields: per unit the features (``[B, T, d]`` bf16 sequences, ``[B, d]``
    bf16 vectors or a ``[B, F]`` ``torch.sparse_csr`` block), the valid token counts (sequences
    only, else ``None``), the labels ``[B]`` (long), the weights ``[B]`` (float32) and the ids."""

    features: dict
    lengths: torch.Tensor | None
    labels: torch.Tensor
    weights: torch.Tensor
    ids: list


def _labels_of(labels: Mapping, ids: Sequence[str]) -> list[int]:
    out = []
    for i in ids:
        try:
            out.append(int(labels[str(i)]))
        except KeyError:
            raise KeyError(f"rollout_id {i!r} has no label") from None
    return out


def _as_label_map(labels) -> Mapping[str, int]:
    if isinstance(labels, pd.Series):
        return {str(k): int(v) for k, v in labels.items()}
    return {str(k): int(v) for k, v in dict(labels).items()}


def _row_batches(n: int, *, shuffle: bool, seed: int, batch_size: int) -> list[list[int]]:
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    return [order[s:s + batch_size].tolist() for s in range(0, n, batch_size)]


def _collate_sequence(items):
    """``collate_multi_layer`` plus the stacked weights."""
    H, y, lengths = collate_multi_layer(items)
    w = torch.stack([it[2] for it in items])
    return H, y, lengths, w


class _LayerSubsetSequenceDataset(RolloutSequenceDataset):
    """:class:`RolloutSequenceDataset` fetching only ``layers`` (the base class streams every layer
    the store reads)."""

    def __init__(self, rollout_ids, labels, store: SequenceStore, *, layers: Sequence[int]):
        super().__init__(rollout_ids, labels, store)
        self.layers = list(layers)

    def __getitem__(self, idx: int):
        return self._item(idx, self.store.get(self.rollout_ids[idx], self.layers))


class SequenceSource:
    """The CoT token sequences of a :class:`~src.lib.activation_store.SequenceStore`, one unit
    ``(layer, "na")`` per layer. ``labels`` maps ``rollout_id`` → label, ``n_tokens`` maps
    ``rollout_id`` → ``n_cot_tokens`` (what :class:`TokenBudgetBatchSampler` budgets on). Batches
    come from a never-persistent ``DataLoader``; ``fetch_threads`` is only recorded here."""

    input_kind = "sequence"

    def __init__(self, store: SequenceStore, layers: Sequence[int] | None, *, labels, n_tokens,
                 num_workers: int = 0, fetch_threads: int | None = None, prefetch_factor: int = 2):
        self.store = store
        self.layers = list(store.layers) if layers is None else [int(l) for l in layers]
        unknown = sorted(set(self.layers) - set(store.layers))
        if unknown:
            raise ValueError(f"layers {unknown} are not read by the store (its layers: {list(store.layers)})")
        self.labels = _as_label_map(labels)
        self.n_tokens = {str(k): int(v) for k, v in (n_tokens.items() if hasattr(n_tokens, "items") else n_tokens)}
        self.num_workers = int(num_workers)
        self.fetch_threads = fetch_threads
        self.prefetch_factor = int(prefetch_factor)

    @property
    def units(self) -> list:
        return [(l, NA) for l in self.layers]

    def feature_dim(self, unit) -> int:
        return int(self.store.hidden)

    def fit(self, train_ids: Sequence[str], *, probe_cfg: dict | None = None) -> None:
        """Nothing to fit."""

    def lengths(self, ids: Sequence[str]) -> list[int]:
        out = []
        for i in ids:
            try:
                out.append(self.n_tokens[str(i)])
            except KeyError:
                raise KeyError(f"rollout_id {i!r} has no n_tokens entry") from None
        return out

    def dataset(self, ids: Sequence[str]) -> RolloutSequenceDataset:
        ids = [str(i) for i in ids]
        return _LayerSubsetSequenceDataset(ids, _labels_of(self.labels, ids), self.store, layers=self.layers)

    def batches(self, ids: Sequence[str], *, shuffle: bool, seed: int, max_batch_tokens: int,
                batch_size: int) -> Iterator[Batch]:
        ids = [str(i) for i in ids]
        if not ids:
            return
        ds = self.dataset(ids)
        sampler = TokenBudgetBatchSampler(self.lengths(ids), max_batch_tokens, batch_size, shuffle=shuffle, seed=seed)
        plan = list(iter(sampler)) if shuffle else sampler.batches()
        kwargs: dict = {"collate_fn": _collate_sequence, "num_workers": self.num_workers}
        if self.num_workers > 0:
            kwargs.update(persistent_workers=False, prefetch_factor=self.prefetch_factor)
        loader = DataLoader(ds, batch_sampler=plan, **kwargs)
        for indices, (H, y, lengths, w) in zip(plan, loader):
            yield Batch({(l, NA): H[l] for l in self.layers}, lengths, y, w, [ids[i] for i in indices])


class PointSource:
    """One residual vector per (rollout, layer, point) from a :class:`~src.lib.activation_store.PointStore`;
    units ``(layer, point)``. ``materialise(ids)`` fetches ``get_many`` once per point and caches the
    host tensors per ``(point, layer, tuple(ids))``; ``clear()`` releases them between folds. Batches
    are ``batch_size`` row slices (``max_batch_tokens`` does not apply)."""

    input_kind = "vector"

    def __init__(self, store: PointStore, layers: Sequence[int] | None, points: Sequence[str], *, labels):
        self.store = store
        self.layers = list(store.layers) if layers is None else [int(l) for l in layers]
        unknown = sorted(set(self.layers) - set(store.layers))
        if unknown:
            raise ValueError(f"layers {unknown} are not read by the store (its layers: {list(store.layers)})")
        self.points = [str(p) for p in points]
        if not self.points or any(p not in POINTS for p in self.points) or len(set(self.points)) != len(self.points):
            raise ValueError(f"points must be a non-empty, duplicate-free subset of {list(POINTS)}, got {points!r}")
        self.labels = _as_label_map(labels)
        self._cache: dict[tuple, torch.Tensor] = {}
        self.n_fetches = 0

    @property
    def units(self) -> list:
        return [(l, p) for l in self.layers for p in self.points]

    def feature_dim(self, unit) -> int:
        return int(self.store.hidden)

    def fit(self, train_ids: Sequence[str], *, probe_cfg: dict | None = None) -> None:
        """Nothing to fit."""

    def clear(self) -> None:
        self._cache.clear()

    def materialise(self, ids: Sequence[str]) -> dict:
        """``{(layer, point): Tensor[n, hidden]}`` for ``ids`` in order, from the cache or the store."""
        ids = tuple(str(i) for i in ids)
        out = {}
        for point in self.points:
            missing = [l for l in self.layers if (point, l, ids) not in self._cache]
            if missing:
                fetched = self.store.get_many(list(ids), point, missing)
                self.n_fetches += 1
                for l in missing:
                    self._cache[(point, l, ids)] = fetched[l]
            for l in self.layers:
                out[(l, point)] = self._cache[(point, l, ids)]
        return out

    def batches(self, ids: Sequence[str], *, shuffle: bool, seed: int, max_batch_tokens: int,
                batch_size: int) -> Iterator[Batch]:
        ids = [str(i) for i in ids]
        if not ids:
            return
        features = self.materialise(ids)
        labels = torch.tensor(_labels_of(self.labels, ids), dtype=torch.long)
        for indices in _row_batches(len(ids), shuffle=shuffle, seed=seed, batch_size=batch_size):
            idx = torch.tensor(indices, dtype=torch.long)
            yield Batch({u: t[idx] for u, t in features.items()}, None, labels[idx],
                        torch.ones(len(indices)), [ids[i] for i in indices])


class TfidfSource:
    """Sparse TF-IDF rows over ``corpus`` (a ``rollout_id``-indexed series of normalised text), one
    unit ``("na", "na")``. :meth:`fit` fits the vectorizer on the training ids' texts only, cached per
    (vectorizer settings, text settings, train ids); a trial sweeping ``probe.text.*`` needs
    ``text_cfg`` (the pipeline ``corpus`` came from) and ``corpus_for`` (text settings → corpus).
    Batches are ``batch_size`` CSR row blocks as ``torch.sparse_csr``."""

    input_kind = "sparse"

    def __init__(self, corpus: pd.Series, vectorizer_cfg: VectorizerConfig | dict | None = None, *, labels,
                 text_cfg: TextPipelineConfig | dict | None = None,
                 corpus_for: Callable[[dict], pd.Series] | None = None):
        if corpus_for is not None and text_cfg is None:
            raise ValueError("corpus_for needs text_cfg (the pipeline the given corpus was normalised with)")
        self.base_corpus = self._check_corpus(corpus)
        self.vectorizer_cfg = (vectorizer_cfg if isinstance(vectorizer_cfg, VectorizerConfig)
                               else VectorizerConfig.from_dict(vectorizer_cfg))
        self.text_cfg = (text_cfg if isinstance(text_cfg, TextPipelineConfig) or text_cfg is None
                         else TextPipelineConfig.from_dict(text_cfg))
        self.corpus_for = corpus_for
        self.labels = _as_label_map(labels)
        self.corpus = self.base_corpus
        self.vectorizer = None
        self.fitted_on: tuple[str, ...] | None = None
        self.fit_calls = 0
        self._fit_key = None
        self._fits: dict[tuple, tuple] = {}            # fit key → (vectorizer, corpus)
        self._matrices: dict[tuple, sp.csr_matrix] = {}   # (fit key, ids) → the transformed rows
        self._corpora: dict[str, pd.Series] = {}

    @staticmethod
    def _check_corpus(corpus: pd.Series) -> pd.Series:
        if not isinstance(corpus, pd.Series):
            raise TypeError(f"corpus must be a pandas Series indexed by rollout_id, got {type(corpus).__name__}")
        corpus = corpus.copy()
        corpus.index = corpus.index.astype(str)
        if corpus.index.duplicated().any():
            raise ValueError("corpus index (rollout_id) has duplicates")
        if corpus.isna().any():
            raise ValueError("corpus holds null texts")
        return corpus.astype(str)

    @property
    def units(self) -> list:
        return [(NA, NA)]

    def feature_dim(self, unit) -> int:
        if self.vectorizer is None:
            raise RuntimeError("TfidfSource.fit(train_ids) must run before feature_dim")
        return vectorizer_n_features(self.vectorizer)

    def _texts(self, ids: Sequence[str]) -> list[str]:
        ids = [str(i) for i in ids]
        missing = [i for i in ids if i not in self.corpus.index]
        if missing:
            raise KeyError(f"{len(missing)} rollout_id(s) have no text in the corpus, e.g. {missing[:5]}")
        return self.corpus.loc[ids].tolist()

    def fit(self, train_ids: Sequence[str], *, probe_cfg: dict | None = None) -> None:
        probe_cfg = probe_cfg or {}
        vec_cfg = self.vectorizer_cfg if probe_cfg.get("vectorizer") is None else VectorizerConfig.from_dict(
            probe_cfg["vectorizer"])
        text_cfg = probe_cfg.get("text")
        text_key = "base"
        if text_cfg is not None and self.text_cfg is not None and self.corpus_for is not None:
            # A text sweep: any setting other than the base pipeline's gets its own corpus.
            wanted = TextPipelineConfig.from_dict(text_cfg).to_dict()
            if wanted != self.text_cfg.to_dict():
                text_key = json.dumps(wanted, sort_keys=True)
                if text_key not in self._corpora:
                    self._corpora[text_key] = self._check_corpus(self.corpus_for(dict(text_cfg)))
        corpus = self.base_corpus if text_key == "base" else self._corpora[text_key]
        train_ids = tuple(str(i) for i in train_ids)
        key = (json.dumps(vec_cfg.to_dict(), sort_keys=True), text_key, train_ids)
        if key == self._fit_key:
            return
        if key not in self._fits:
            self.corpus = corpus
            self._fits[key] = (fit_vectorizer(self._texts(train_ids), vec_cfg), corpus)
            self.fit_calls += 1
        self.vectorizer, self.corpus = self._fits[key]
        self.fitted_on = train_ids
        self._fit_key = key

    def clear(self) -> None:
        """Drop every fitted vectorizer and transformed matrix (between folds)."""
        self._fits.clear()
        self._matrices.clear()
        self.vectorizer, self.corpus, self.fitted_on, self._fit_key = None, self.base_corpus, None, None

    def matrix(self, ids: Sequence[str]) -> sp.csr_matrix:
        """The TF-IDF rows of ``ids`` under the current fit, transformed once per (fit, ids) and cached."""
        if self.vectorizer is None:
            raise RuntimeError("TfidfSource.fit(train_ids) must run before features are requested")
        key = (self._fit_key, tuple(str(i) for i in ids))
        if key not in self._matrices:
            self._matrices[key] = transform(self.vectorizer, self._texts(ids))
        return self._matrices[key]

    def batches(self, ids: Sequence[str], *, shuffle: bool, seed: int, max_batch_tokens: int,
                batch_size: int) -> Iterator[Batch]:
        ids = [str(i) for i in ids]
        if not ids:
            return
        X = self.matrix(ids)
        labels = torch.tensor(_labels_of(self.labels, ids), dtype=torch.long)
        for indices in _row_batches(len(ids), shuffle=shuffle, seed=seed, batch_size=batch_size):
            yield Batch({(NA, NA): to_torch_sparse(X[indices])}, None, labels[indices],
                        torch.ones(len(indices)), [ids[i] for i in indices])


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _log(log, message: str) -> None:
    if log is None:
        return
    if isinstance(log, logging.Logger):
        log.info(message)
    else:
        log(message)


def _to_device(x: torch.Tensor, device) -> torch.Tensor:
    """bf16 host tensors widened to float32 on the device; sparse inputs stay sparse."""
    if x.layout in (torch.sparse_csr, torch.sparse_coo):
        return x.to(device)
    return x.to(device, non_blocking=True).float()


def _selection_score(metrics: dict | None) -> float:
    """Validation AUROC, ``-val_loss`` when val holds one class, NaN without a val set."""
    if metrics is None:
        return math.nan
    auc = metrics.get("auroc", math.nan)
    return float(auc) if auc == auc else -float(metrics["loss"])


def _better(score: float, best: float) -> bool:
    """Strictly better (NaN never wins, so the earlier candidate keeps ties and no-val runs)."""
    return score == score and (best != best or score > best)


@dataclass
class UnitResult:
    """One unit's outcome over every trial of a fold: the best trial / epoch / score,
    ``history[trial_id]`` (one dict per epoch), ``trials[trial_id]`` (best epoch / score / val
    metrics), ``val_metrics`` at the best epoch of the best trial and ``state`` (its checkpoint)."""

    unit: tuple
    best_trial: str | None = None
    best_epoch: int = 0
    best_score: float = math.nan
    history: dict = field(default_factory=dict)
    trials: dict = field(default_factory=dict)
    val_metrics: dict | None = None
    state: dict | None = None


@dataclass
class FoldResult:
    """Everything :func:`train_fold` produced for one fold, plus what :meth:`predict` needs;
    ``predictions`` collects what :func:`evaluate` scored, keyed by test set."""

    fold: Fold
    fold_index: int
    units: dict
    trials: list
    training: TrainingConfig
    probe: dict
    source: Any
    device: str = "cpu"
    predictions: dict = field(default_factory=dict)

    @property
    def held_out_style(self) -> str:
        return self.fold.held_out_style

    def probe_cfg_of(self, unit) -> dict:
        """The probe block of the unit's best trial."""
        result = self.units[unit]
        trial = next(t for t in self.trials if t.id == result.best_trial)
        return apply_trial(self.training, self.probe, trial)[1]

    def probes(self, device=None) -> dict:
        """The best probe of every unit, rebuilt from its checkpoint (on ``device``)."""
        device = self.device if device is None else device
        return {u: load_probe(r.state).to(device).eval() for u, r in self.units.items()}

    def predict(self, ids: Sequence[str], *, source=None, device=None) -> dict:
        """``{unit: DataFrame(rollout_id, p_unfaithful)}`` over ``ids`` (in the given order), full
        sequences, the best probe of every unit. A TF-IDF source is switched to the best trial's fit."""
        source = self.source if source is None else source
        device = self.device if device is None else device
        ids = [str(i) for i in ids]
        if source.input_kind == "sparse":
            source.fit(self.fold.train_ids, probe_cfg=self.probe_cfg_of(next(iter(self.units))))
        probes = self.probes(device)
        scores = {u: {} for u in probes}
        if ids:
            with torch.no_grad():
                for batch in source.batches(ids, shuffle=False, seed=0, max_batch_tokens=self.training.max_batch_tokens,
                                            batch_size=self.training.batch_size):
                    lengths = None if batch.lengths is None else batch.lengths.to(device)
                    for unit, probe in probes.items():
                        p = p_detection_target(probe(_to_device(batch.features[unit], device), lengths)).cpu().numpy()
                        scores[unit].update(zip(batch.ids, p.tolist()))
        return {u: pd.DataFrame({PREDICTION_COLUMNS[0]: ids, PREDICTION_COLUMNS[1]: [s[i] for i in ids]})
                for u, s in scores.items()}


def ensemble_predictions(per_unit: Mapping) -> pd.DataFrame:
    """The mean score over the units' ``(rollout_id, p_unfaithful)`` frames (aligned on id)."""
    frames = [f for u, f in per_unit.items() if u != ENSEMBLE_KEY]
    if not frames:
        raise ValueError("no unit predictions to ensemble")
    ids = frames[0][PREDICTION_COLUMNS[0]].tolist()
    stacked = np.stack([f.set_index(PREDICTION_COLUMNS[0])[PREDICTION_COLUMNS[1]].reindex(ids).to_numpy(dtype=float)
                        for f in frames])
    return pd.DataFrame({PREDICTION_COLUMNS[0]: ids, PREDICTION_COLUMNS[1]: stacked.mean(axis=0)})


def _standardise(probes: dict, source, train_ids, *, seed: int, training: TrainingConfig, norm_tokens: int | None,
                 device, cache: dict, log) -> None:
    """Per-unit mean / std over the training features into every probe's buffers: up to
    ``norm_tokens`` valid tokens of a shuffled pass for sequences, every row for vectors.
    Statistics are cached across trials."""
    key = (source.input_kind, norm_tokens, seed)
    if key not in cache:
        units = list(probes)
        s = {u: None for u in units}
        ss = {u: None for u in units}
        n = 0
        with torch.no_grad():
            for batch in source.batches(train_ids, shuffle=True, seed=seed, max_batch_tokens=training.max_batch_tokens,
                                        batch_size=training.batch_size):
                if batch.lengths is not None:
                    T = int(batch.lengths.max())
                    valid = (torch.arange(T).unsqueeze(0) < batch.lengths.unsqueeze(1)).to(device)
                for u in units:
                    X = _to_device(batch.features[u], device)
                    X = X[valid] if batch.lengths is not None else X
                    part_s, part_ss = X.sum(0, dtype=torch.float64), (X.double() ** 2).sum(0)
                    s[u] = part_s if s[u] is None else s[u] + part_s
                    ss[u] = part_ss if ss[u] is None else ss[u] + part_ss
                    del X
                n += int(valid.sum()) if batch.lengths is not None else len(batch.ids)
                if norm_tokens is not None and n >= norm_tokens:
                    break
        if n == 0:
            raise RuntimeError("no training features seen while estimating standardisation statistics")
        stats = {}
        for u in units:
            mean = s[u] / n
            std = (ss[u] / n - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
            stats[u] = (mean.float().cpu(), std.float().cpu())
        cache[key] = (n, stats)
        _log(log, f"standardisation statistics from {n:,} training {'tokens' if source.input_kind == 'sequence' else 'rows'}")
    _, stats = cache[key]
    for u, probe in probes.items():
        mean, std = stats[u]
        probe.set_standardization(mean.to(device), std.to(device))


def _pass(probes: dict, source, ids, *, training: TrainingConfig, device, optimizers: dict | None,
          class_weight: torch.Tensor | None, min_prefix_frac: float, seed: int, rng: np.random.Generator | None) -> dict:
    """One pass over ``ids`` for every probe: training when ``optimizers`` is given (random prefixes,
    weighted CE, clipped AdamW steps), else evaluation (full sequences, no gradients). Returns per
    unit ``{"loss", "labels", "p"}`` in pass order."""
    train = optimizers is not None
    for p in probes.values():
        p.train(train)
    totals = {u: None for u in probes}          # summed loss on the device; one host sync per pass
    probs = {u: [] for u in probes}
    labels: list[int] = []
    n_seen = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in source.batches(ids, shuffle=train, seed=seed, max_batch_tokens=training.max_batch_tokens,
                                    batch_size=training.batch_size):
            y = batch.labels.to(device)
            lengths = batch.lengths
            if lengths is not None:
                if train and min_prefix_frac < 1.0:
                    lengths = sample_prefix_lengths(lengths, min_prefix_frac, rng)
                lengths = lengths.to(device)
            for u, probe in probes.items():
                X = _to_device(batch.features[u], device)
                logits = probe(X, lengths)
                loss = F.cross_entropy(logits, y, weight=class_weight)
                if train:
                    optimizers[u].zero_grad(set_to_none=True)
                    loss.backward()
                    if training.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(probe.parameters(), training.grad_clip)
                    optimizers[u].step()
                part = loss.detach() * len(y)
                totals[u] = part if totals[u] is None else totals[u] + part
                probs[u].append(p_detection_target(logits.detach()).cpu().numpy())
                del X, logits
            labels.extend(batch.labels.tolist())
            n_seen += len(y)
    lab = np.asarray(labels, dtype=np.int64)
    return {u: {"loss": (float(totals[u].item()) if totals[u] is not None else 0.0) / max(n_seen, 1), "labels": lab,
                "p": np.concatenate(probs[u]) if probs[u] else np.array([], dtype=float)} for u in probes}


def train_fold(fold: Fold, source, probe_cfg: dict, training_cfg: TrainingConfig, trials: Sequence[Trial] | None = None,
               *, device: str = "cpu", log=None, fold_index: int = 0) -> FoldResult:
    """Train every unit of ``source`` on ``fold`` over every trial.

    Per trial: one probe per unit, the standardisation pass when ``probe.standardize`` (never for
    TF-IDF), AdamW with class-weighted CE; after each epoch the validation score (AUROC, or
    ``-val loss`` when val holds one class) picks the best epoch's checkpoint, and a unit that has
    not improved for ``patience`` epochs stops (``patience: 0`` never stops early). Without a val
    set the last epoch is kept. The best trial per unit is the highest score, ties to the earlier one.
    """
    trials = list(trials) if trials else [Trial(id=DEFAULT_TRIAL_ID, overrides={})]
    if len({t.id for t in trials}) != len(trials):
        raise ValueError(f"trials carry duplicate ids (their history / records would overwrite each other): "
                         f"{[t.id for t in trials]}")
    units = list(source.units)
    if not units:
        raise ValueError("the feature source has no units")
    if not fold.train_ids:
        raise ValueError(f"fold {fold.held_out_style!r} has no training rows")
    results = {u: UnitResult(unit=u) for u in units}
    if any(k.startswith("probe.text.") for t in trials for k in t.overrides) and getattr(source, "corpus_for", None) is None:
        raise ValueError("the search sweeps probe.text.* but the TfidfSource has no corpus_for — a text sweep needs "
                         "the normalised corpus per setting (build the source with text_cfg= and corpus_for=)")
    stats_cache: dict = {}
    _log(log, f"fold {fold.held_out_style!r} (#{fold_index}): {len(fold.train_ids)} train / {len(fold.val_ids)} val / "
              f"{len(fold.test_id_ids)} test_id / {len(fold.test_ood_ids)} test_ood rows; "
              f"{len(units)} unit(s), {len(trials)} trial(s)")
    for trial in trials:
        tcfg, pcfg = apply_trial(training_cfg, probe_cfg, trial)
        # The trial's own seed, so a ``seed`` sweep takes effect and same-seed trials share shuffles.
        seed_base = tcfg.seed + int(fold_index)
        torch.manual_seed(seed_base)
        rng = np.random.default_rng(stable_seed("prefix", seed_base))
        source.fit(fold.train_ids, probe_cfg=pcfg)
        probes = {}
        for u in units:
            dim = source.feature_dim(u)
            probes[u] = build_probe(pcfg, hidden_dim=dim, n_features=dim).to(device)
        if pcfg.get("standardize", False) and source.input_kind != "sparse":
            _standardise(probes, source, fold.train_ids, seed=stable_seed("norm", seed_base), training=tcfg,
                         norm_tokens=pcfg.get("norm_tokens") if source.input_kind == "sequence" else None,
                         device=device, cache=stats_cache, log=log)
        optimizers = {u: torch.optim.AdamW(p.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
                      for u, p in probes.items()}
        cw = class_weights(_labels_of(source.labels, fold.train_ids), tcfg.class_weighting)
        cw = cw.to(device) if cw is not None else None
        has_val = len(fold.val_ids) > 0
        best = {u: {"epoch": 0, "score": math.nan, "val": None, "state": None} for u in units}
        bad = {u: 0 for u in units}
        active = set(units)
        history = {u: [] for u in units}
        _log(log, f"trial {trial.id!r}: {trial.overrides or 'no overrides'}")
        for epoch in range(1, tcfg.epochs + 1):
            live = {u: probes[u] for u in units if u in active}
            tr = _pass(live, source, fold.train_ids, training=tcfg, device=device,
                       optimizers={u: optimizers[u] for u in live}, class_weight=cw, min_prefix_frac=tcfg.min_prefix_frac,
                       seed=stable_seed("epoch", seed_base, epoch), rng=rng)
            va = _pass(live, source, fold.val_ids, training=tcfg, device=device, optimizers=None, class_weight=cw,
                       min_prefix_frac=1.0, seed=0, rng=None) if has_val else None
            for u in list(live):
                trm = summarize(tr[u]["labels"], tr[u]["p"], loss=tr[u]["loss"])
                vam = summarize(va[u]["labels"], va[u]["p"], loss=va[u]["loss"]) if va else None
                history[u].append({"epoch": epoch, "train_loss": trm["loss"], "train_auroc": trm.get("auroc", math.nan),
                                   "val_loss": vam["loss"] if vam else math.nan,
                                   "val_auroc": vam.get("auroc", math.nan) if vam else math.nan})
                improved = False
                if vam is not None:
                    score = _selection_score(vam)
                    if _better(score, best[u]["score"]):
                        best[u] = {"epoch": epoch, "score": score, "val": vam, "state": to_checkpoint(probes[u])}
                        bad[u], improved = 0, True
                    else:
                        bad[u] += 1
                        if tcfg.patience > 0 and bad[u] >= tcfg.patience:
                            active.discard(u)
                else:
                    best[u] = {"epoch": epoch, "score": math.nan, "val": None, "state": to_checkpoint(probes[u])}
                _log(log, f"  unit {u} epoch {epoch}: train loss {trm['loss']:.4f} auroc {trm.get('auroc', math.nan):.3f} | "
                          f"val loss {history[u][-1]['val_loss']:.4f} auroc {history[u][-1]['val_auroc']:.3f}"
                          f"{' *' if improved else ''}")
            if not active:
                _log(log, f"  early stop at epoch {epoch}: every unit out of patience ({tcfg.patience})")
                break
        for u in units:
            r = results[u]
            r.history[trial.id] = history[u]
            r.trials[trial.id] = {"best_epoch": best[u]["epoch"], "best_score": best[u]["score"],
                                  "val_metrics": best[u]["val"]}
            if r.best_trial is None or _better(best[u]["score"], r.best_score):
                r.best_trial, r.best_epoch, r.best_score = trial.id, best[u]["epoch"], best[u]["score"]
                r.val_metrics, r.state = best[u]["val"], best[u]["state"]
        del probes, optimizers
    for u, r in results.items():
        _log(log, f"unit {u}: best trial {r.best_trial!r} epoch {r.best_epoch} score {r.best_score:.4f}")
    return FoldResult(fold=fold, fold_index=int(fold_index), units=results, trials=trials, training=training_cfg,
                      probe=probe_cfg, source=source, device=device)


def evaluate(fold_result: FoldResult, source, ids: Sequence[str], test_set: str, *, device=None) -> dict:
    """Score ``ids`` (``test_id`` / ``test_ood``) with every unit's best probe over full sequences:
    ``{unit: DataFrame(rollout_id, p_unfaithful), "ensemble": the mean over units}``. Also stored
    under ``fold_result.predictions[test_set]``."""
    per_unit = fold_result.predict(ids, source=source, device=device)
    per_unit[ENSEMBLE_KEY] = ensemble_predictions(per_unit)
    fold_result.predictions[str(test_set)] = per_unit
    return per_unit
