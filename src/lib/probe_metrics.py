"""Classification metrics for the faithfulness probes (numpy only).

Every function takes ``labels`` (1 = faithful, 0 = unfaithful) and ``p_unfaithful``
and treats unfaithful as the positive class; accuracy-style metrics threshold
``p_unfaithful`` at 0.5.
"""

from __future__ import annotations

import math

import numpy as np


def _as_arrays(labels, p_unfaithful) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(int).ravel()
    p = np.asarray(p_unfaithful, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"labels and scores differ in length: {y.shape} vs {p.shape}")
    return y, p


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    _, inverse, counts = np.unique(sx, return_inverse=True, return_counts=True)
    first = np.cumsum(np.concatenate([[0], counts[:-1]])) + 1
    avg = first + (counts - 1) / 2.0
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = avg[inverse]
    return ranks


def auroc(labels, p_unfaithful) -> float:
    """Area under the ROC curve for detecting unfaithful traces (NaN when a
    class is absent). Rank-based, ties count half."""
    y, p = _as_arrays(labels, p_unfaithful)
    pos = y == 0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _average_ranks(p)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_at_fpr(labels, p_unfaithful, fpr: float) -> float:
    """Smallest score threshold such that flagging ``p > threshold`` marks at
    most ``fpr`` of the faithful traces. ``-inf`` when every faithful trace
    may be flagged; NaN without faithful traces."""
    y, p = _as_arrays(labels, p_unfaithful)
    neg = np.sort(p[y == 1])[::-1]
    if len(neg) == 0:
        return math.nan
    allowed = int(math.floor(fpr * len(neg)))
    if allowed >= len(neg):
        return -math.inf
    return float(neg[allowed])


def recall_at_fpr(labels, p_unfaithful, fpr: float) -> float:
    """Share of unfaithful traces flagged at the threshold of :func:`threshold_at_fpr`."""
    y, p = _as_arrays(labels, p_unfaithful)
    thr = threshold_at_fpr(y, p, fpr)
    pos = p[y == 0]
    if len(pos) == 0 or math.isnan(thr):
        return math.nan
    return float((pos > thr).mean())


def expected_calibration_error(labels, p_unfaithful, n_bins: int = 10) -> float:
    """ECE over the predicted class's confidence, equal-width bins."""
    y, p = _as_arrays(labels, p_unfaithful)
    if len(y) == 0:
        return math.nan
    pred_unf = p > 0.5
    conf = np.where(pred_unf, p, 1 - p)
    correct = pred_unf == (y == 0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def summarize(labels, p_unfaithful, *, loss: float | None = None) -> dict:
    """Every metric the training script reports for one set of predictions."""
    y, p = _as_arrays(labels, p_unfaithful)
    n = len(y)
    out: dict = {"n": int(n), "n_unfaithful": int((y == 0).sum()), "n_faithful": int((y == 1).sum())}
    if loss is not None:
        out["loss"] = float(loss)
    if n == 0:
        return out
    pred_unf = p > 0.5
    is_unf = y == 0
    tp = int((pred_unf & is_unf).sum())
    fp = int((pred_unf & ~is_unf).sum())
    fn = int((~pred_unf & is_unf).sum())
    tn = int((~pred_unf & ~is_unf).sum())
    rec_unf = tp / (tp + fn) if tp + fn else math.nan
    rec_fai = tn / (tn + fp) if tn + fp else math.nan
    out.update({
        "accuracy": float((pred_unf == is_unf).mean()),
        "balanced_accuracy": float(np.nanmean([rec_unf, rec_fai])),
        "auroc": auroc(y, p),
        "precision_unfaithful": tp / (tp + fp) if tp + fp else math.nan,
        "recall_unfaithful": rec_unf,
        "precision_faithful": tn / (tn + fn) if tn + fn else math.nan,
        "recall_faithful": rec_fai,
        "recall_at_1pct_fpr": recall_at_fpr(y, p, 0.01),
        "recall_at_5pct_fpr": recall_at_fpr(y, p, 0.05),
        "threshold_at_1pct_fpr": threshold_at_fpr(y, p, 0.01),
        "ece": expected_calibration_error(y, p),
    })
    return out
