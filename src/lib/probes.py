"""Probe heads behind one checkpoint contract.

Three heads — ``TfidfLogReg`` (``type: tfidf``, sparse CSR input), ``LinearProbe``
(``type: linear``, one vector per rollout) and ``AttentionProbe`` (``type: attention``,
a token sequence per rollout) — each expose ``input_kind``, ``probe_config`` (``type``
plus constructor kwargs), ``from_state(probe_config, state_dict)``,
``set_standardization(mean, std)`` and ``forward(features, lengths=None)``.
:func:`p_detection_target` is the one definition of the score: the probability of
label 0, the detection target.
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.attention_probe import AttentionProbe as _BaseAttentionProbe

__all__ = [
    "AttentionProbe",
    "LinearProbe",
    "PROBE_CLASSES",
    "PROBE_TYPES",
    "TfidfLogReg",
    "allowed_probe_keys",
    "build_probe",
    "load_probe",
    "p_detection_target",
    "to_checkpoint",
]

PROBE_TYPES = ("tfidf", "linear", "attention")
DETECTION_TARGET_LABEL = 0


def _strip_type(probe_config: dict, expected: str) -> dict:
    """Return ``probe_config`` without ``type``; a present ``type`` must match."""
    cfg = dict(probe_config)
    found = cfg.pop("type", None)
    if found is not None and found != expected:
        raise ValueError(f"probe_config type is {found!r}, expected {expected!r}")
    return cfg


class AttentionProbe(_BaseAttentionProbe):
    """The base attention probe with ``"type": "attention"`` in ``probe_config``;
    ``from_state`` accepts a config with or without it."""

    input_kind: ClassVar[str] = "sequence"

    @property
    def probe_config(self) -> dict:
        return {"type": "attention", **super().probe_config}

    @classmethod
    def from_state(cls, probe_config: dict, state_dict: dict) -> "AttentionProbe":
        return super().from_state(_strip_type(probe_config, "attention"), state_dict)


class LinearProbe(nn.Module):
    """Linear probe over one activation vector: ``[batch, hidden] -> [batch, num_classes]``.

    Inputs are standardised by the ``mean``/``std`` buffers (identity until
    :meth:`set_standardization`); with ``standardize=False`` the call warns and is a no-op.
    """

    input_kind: ClassVar[str] = "vector"

    def __init__(self, hidden_dim: int, num_classes: int = 2, *, standardize: bool = True):
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if int(num_classes) < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.standardize = bool(standardize)
        self.linear = nn.Linear(self.hidden_dim, self.num_classes)
        self.register_buffer("mean", torch.zeros(self.hidden_dim))
        self.register_buffer("std", torch.ones(self.hidden_dim))

    @property
    def probe_config(self) -> dict:
        return {
            "type": "linear",
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "standardize": self.standardize,
        }

    @classmethod
    def from_state(cls, probe_config: dict, state_dict: dict) -> "LinearProbe":
        probe = cls(**_strip_type(probe_config, "linear"))
        probe.load_state_dict(state_dict)
        return probe

    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if not self.standardize:
            warnings.warn("LinearProbe built with standardize=False: set_standardization ignored",
                          stacklevel=2)
            return
        if mean.shape != (self.hidden_dim,) or std.shape != (self.hidden_dim,):
            raise ValueError(
                f"mean/std must have shape ({self.hidden_dim},), got {tuple(mean.shape)} / {tuple(std.shape)}"
            )
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("mean must be finite and std finite and strictly positive in every dimension")
        self.mean.copy_(mean.to(self.mean.dtype))
        self.std.copy_(std.to(self.std.dtype))

    def forward(self, H: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``H``: ``[batch, hidden_dim]``; ``lengths`` is ignored."""
        if H.dim() != 2 or H.size(1) != self.hidden_dim:
            raise ValueError(f"expected [batch, {self.hidden_dim}] features, got {tuple(H.shape)}")
        H = H.to(self.linear.weight.dtype)
        return self.linear((H - self.mean) / self.std)


class TfidfLogReg(nn.Module):
    """Logistic regression over sparse TF-IDF features: ``[batch, n_features] -> [batch, num_classes]``.

    ``n_features`` is the fold's vocabulary size, so a checkpoint only pairs with its own
    vectorizer. No in-model regulariser: the optimiser's ``weight_decay`` is the L2 penalty
    (sklearn's ``C = 1 / (weight_decay * n_train)`` for a coupled penalty). Inputs are never
    standardised.
    """

    input_kind: ClassVar[str] = "sparse"

    def __init__(self, n_features: int, num_classes: int = 2):
        super().__init__()
        if int(n_features) < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if int(num_classes) < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.linear = nn.Linear(self.n_features, self.num_classes)

    @property
    def probe_config(self) -> dict:
        return {"type": "tfidf", "n_features": self.n_features, "num_classes": self.num_classes}

    @classmethod
    def from_state(cls, probe_config: dict, state_dict: dict) -> "TfidfLogReg":
        probe = cls(**_strip_type(probe_config, "tfidf"))
        probe.load_state_dict(state_dict)
        return probe

    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        warnings.warn("TfidfLogReg does not standardise its inputs: set_standardization ignored",
                      stacklevel=2)

    def forward(self, X: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``X``: ``[batch, n_features]``, sparse CSR/COO or dense; ``lengths`` is ignored."""
        if X.dim() != 2 or X.size(1) != self.n_features:
            raise ValueError(f"expected [batch, {self.n_features}] features, got {tuple(X.shape)}")
        weight, bias = self.linear.weight, self.linear.bias
        if X.layout in (torch.sparse_csr, torch.sparse_coo):
            X = X.to(weight.dtype)
            return torch.sparse.mm(X, weight.t().contiguous()) + bias
        return F.linear(X.to(weight.dtype), weight, bias)


PROBE_CLASSES: dict[str, type[nn.Module]] = {
    "tfidf": TfidfLogReg,
    "linear": LinearProbe,
    "attention": AttentionProbe,
}

# ``constructor`` keys reach ``__init__``; ``loop`` keys belong to the training loop
# and are accepted but never forwarded.
_CONSTRUCTOR_KEYS = {
    "attention": ("heads", "aggregation", "window", "position_bias"),
    "linear": ("standardize",),
    "tfidf": (),
}
_LOOP_KEYS = {
    "attention": ("standardize", "norm_tokens"),
    "linear": ("points",),
    "tfidf": ("text", "vectorizer"),
}
_COMMON_KEYS = ("type", "num_classes")


def allowed_probe_keys(probe_type: str) -> tuple[str, ...]:
    """Every key ``build_probe`` accepts in a ``probe:`` block of this type."""
    if probe_type not in PROBE_TYPES:
        raise ValueError(f"probe type must be one of {PROBE_TYPES}, got {probe_type!r}")
    return _COMMON_KEYS + _CONSTRUCTOR_KEYS[probe_type] + _LOOP_KEYS[probe_type]


def build_probe(cfg: dict, *, hidden_dim: int | None = None, n_features: int | None = None) -> nn.Module:
    """Build an untrained probe from the YAML ``probe:`` block.

    Unknown keys raise. ``hidden_dim`` is required for ``attention`` / ``linear``,
    ``n_features`` for ``tfidf``; the other dimension is ignored.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"probe config must be a dict, got {type(cfg).__name__}")
    probe_type = cfg.get("type")
    if probe_type not in PROBE_TYPES:
        raise ValueError(f"probe.type must be one of {PROBE_TYPES}, got {probe_type!r}")
    allowed = allowed_probe_keys(probe_type)
    unknown = sorted(k for k in cfg if k not in allowed)
    if unknown:
        raise ValueError(f"unknown keys for probe type {probe_type!r}: {unknown}; allowed: {list(allowed)}")
    kwargs = {k: cfg[k] for k in _CONSTRUCTOR_KEYS[probe_type] if k in cfg}
    num_classes = int(cfg["num_classes"]) if cfg.get("num_classes") is not None else 2

    if probe_type == "tfidf":
        if n_features is None:
            raise ValueError("build_probe: n_features is required for probe type 'tfidf'")
        return TfidfLogReg(int(n_features), num_classes)
    if hidden_dim is None:
        raise ValueError(f"build_probe: hidden_dim is required for probe type {probe_type!r}")
    if probe_type == "linear":
        return LinearProbe(int(hidden_dim), num_classes, **kwargs)
    return AttentionProbe(int(hidden_dim), num_classes, **kwargs)


def to_checkpoint(probe: nn.Module) -> dict:
    """``{"probe_config", "model_state_dict"}`` with CPU tensor copies (a plain
    ``.cpu()`` would alias the live parameters of a CPU probe)."""
    return {
        "probe_config": dict(probe.probe_config),
        "model_state_dict": {k: v.detach().to("cpu", copy=True) for k, v in probe.state_dict().items()},
    }


def load_probe(checkpoint: dict) -> nn.Module:
    """Rebuild a probe from a checkpoint dict, dispatching on ``probe_config["type"]``."""
    try:
        probe_config = checkpoint["probe_config"]
        state_dict = checkpoint["model_state_dict"]
    except (KeyError, TypeError) as e:
        raise ValueError("checkpoint must carry 'probe_config' and 'model_state_dict'") from e
    probe_type = probe_config.get("type") if isinstance(probe_config, dict) else None
    if probe_type not in PROBE_CLASSES:
        raise ValueError(f"checkpoint probe_config.type must be one of {PROBE_TYPES}, got {probe_type!r}")
    return PROBE_CLASSES[probe_type].from_state(probe_config, state_dict)


def p_detection_target(logits: torch.Tensor) -> torch.Tensor:
    """``softmax(logits)[:, 0]`` in float32: the probability of label 0, the detection target."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be [batch, num_classes], got {tuple(logits.shape)}")
    return F.softmax(logits.float(), dim=-1)[:, DETECTION_TARGET_LABEL]
