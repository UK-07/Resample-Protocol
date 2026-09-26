"""Attention-pooling probe over per-token residual-stream activations.

Single-head form (Kantamneni et al. 2025): ``z = sum_t softmax_t(q . h_t) * (W_v h_t)``,
extended with input standardisation, multiple summed heads, a zero-initialised
query (mean pooling at init), an optional relative-position slope, an output
bias and a rolling-max aggregation (Kramár et al. 2026).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

AGGREGATIONS = ("softmax", "rolling_max")


class AttentionProbe(nn.Module):
    """``[batch, T, hidden] -> [batch, num_classes]``.

    ``aggregation``: ``"softmax"`` over the whole sequence, or ``"rolling_max"``
    (softmax inside every ``window``-token window, max over windows).
    ``position_bias`` adds a learned per-head slope on the relative position.
    Padding positions (``>= lengths``) get no attention weight, so a padded
    sample gives the same logits as the sample forwarded alone.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        *,
        heads: int = 1,
        aggregation: str = "softmax",
        window: int = 16,
        position_bias: bool = False,
    ):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.heads = int(heads)
        self.aggregation = aggregation
        self.window = int(window)

        self.Wq = nn.Linear(hidden_dim, heads, bias=False)
        self.Wv = nn.Linear(hidden_dim, heads * num_classes, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.pos_slope = nn.Parameter(torch.zeros(heads)) if position_bias else None
        self.register_buffer("mean", torch.zeros(hidden_dim))
        self.register_buffer("std", torch.ones(hidden_dim))
        nn.init.zeros_(self.Wq.weight)  # uniform attention at init == mean pooling

    @property
    def probe_config(self) -> dict:
        """Constructor kwargs, stored in checkpoints so a probe can be rebuilt."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "heads": self.heads,
            "aggregation": self.aggregation,
            "window": self.window,
            "position_bias": self.pos_slope is not None,
        }

    @classmethod
    def from_state(cls, probe_config: dict, state_dict: dict) -> "AttentionProbe":
        probe = cls(**probe_config)
        probe.load_state_dict(state_dict)
        return probe

    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the per-dimension input standardisation (train-set statistics)."""
        if mean.shape != (self.hidden_dim,) or std.shape != (self.hidden_dim,):
            raise ValueError(
                f"mean/std must have shape ({self.hidden_dim},), got {tuple(mean.shape)} / {tuple(std.shape)}"
            )
        if bool((std <= 0).any()):
            raise ValueError("std must be strictly positive in every dimension")
        self.mean.copy_(mean.to(self.mean.dtype))
        self.std.copy_(std.to(self.std.dtype))

    def token_scores(self, H: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Per-token, per-head scores ``[batch, T, heads]`` before softmax and masking."""
        scores = self.Wq(H)
        if self.pos_slope is not None:
            T = H.size(1)
            pos = torch.arange(T, device=H.device, dtype=scores.dtype).unsqueeze(0)
            denom = (lengths.to(scores.dtype) - 1).clamp(min=1).unsqueeze(1)
            rel = (pos / denom).unsqueeze(-1)
            scores = scores + rel * self.pos_slope
        return scores

    def attention(self, H: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Attention weights ``[batch, T, heads]`` (zero on padding); softmax aggregation only."""
        if self.aggregation != "softmax":
            raise ValueError("attention() is only defined for softmax aggregation")
        H, lengths, pad = self._prepare(H, lengths)
        scores = self.token_scores(H, lengths).masked_fill(pad.unsqueeze(-1), float("-inf"))
        return F.softmax(scores, dim=1)

    def _prepare(self, H: torch.Tensor, lengths: torch.Tensor | None):
        B, T, _ = H.shape
        if lengths is None:
            lengths = torch.full((B,), T, device=H.device, dtype=torch.long)
        else:
            lengths = lengths.to(H.device)
        H = (H - self.mean) / self.std
        pos = torch.arange(T, device=H.device).unsqueeze(0)
        pad = pos >= lengths.unsqueeze(1)
        return H, lengths, pad

    def forward(self, H: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``H``: ``[batch, T, hidden]``; ``lengths``: valid tokens per sample (None = all)."""
        H = H.to(self.Wq.weight.dtype)
        H, lengths, pad = self._prepare(H, lengths)
        B, T, _ = H.shape
        scores = self.token_scores(H, lengths)
        values = self.Wv(H).view(B, T, self.heads, self.num_classes)

        if self.aggregation == "softmax":
            scores = scores.masked_fill(pad.unsqueeze(-1), float("-inf"))
            attn = F.softmax(scores, dim=1)
            z = torch.einsum("bth,bthc->bc", attn, values)
        else:
            z = self._rolling_max(scores, values, lengths, pad)
        return z + self.bias

    def _rolling_max(self, scores, values, lengths, pad) -> torch.Tensor:
        B, T, _ = scores.shape
        w = min(self.window, T)
        # Finite fill keeps a fully padded window NaN-free.
        scores = scores.masked_fill(pad.unsqueeze(-1), torch.finfo(scores.dtype).min)
        S = scores.unfold(1, w, 1)
        V = values.unfold(1, w, 1)
        A = F.softmax(S, dim=-1)
        out = (A.unsqueeze(3) * V).sum(-1)
        n_windows = out.size(1)
        starts = torch.arange(n_windows, device=scores.device).unsqueeze(0)
        # Only windows fully inside the sample count; a sample shorter than the window keeps window 0.
        invalid = starts > (lengths - w).clamp(min=0).unsqueeze(1)
        out = out.masked_fill(invalid[:, :, None, None], float("-inf"))
        return out.max(dim=1).values.sum(dim=1)

    def forward_all_prefixes(self, H: torch.Tensor) -> torch.Tensor:
        """Logits at every prefix of one sequence ``[T, d]`` -> ``[T, num_classes]`` (quadratic in T)."""
        T = H.size(0)
        logits = []
        for t in range(1, T + 1):
            prefix = H[:t].unsqueeze(0)
            logits.append(self.forward(prefix))
        return torch.cat(logits, dim=0)
