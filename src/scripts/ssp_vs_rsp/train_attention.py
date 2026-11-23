#!/usr/bin/env python3
"""Attention probes of the SSP-vs-RSP comparison: every (variant, arm, hyper-parameter) probe of one model
trained in ONE shared pass over the activation store (the disk reads, not the probes, dominate the cost).

Each probe is the repo's ``AttentionProbe`` trained on random CoT prefixes in ``[0.1 L, L]`` with its loss
masked to its own train rows (class-balanced weights), AdamW, gradient clipping and early stopping on its own
val rows (the mean AUROC over the 10 CoT fractions). Inputs are standardised with per-dimension statistics of
train-split tokens, shared by every probe. The hyper-parameters of each (variant, arm) are picked by val
score and every test row is scored at every fraction (``predictions/<model>_attention.parquet``).

    DATA_ROOT=/workspace/data python -u -m src.scripts.ssp_vs_rsp.train_attention --model qwen3-8b
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.lib import ssp_vs_rsp as X
from src.lib.activation_store import local_handle
from src.lib.probes import AttentionProbe, p_detection_target

HP_GRID = [{"lr": lr, "aggregation": agg} for agg in ("softmax", "rolling_max") for lr in (3e-4, 1e-3)]
PROBE_KW = dict(heads=4, window=16, position_bias=True)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EPOCHS = 20
PATIENCE = 5
MIN_PREFIX_FRAC = 0.1
NORM_TOKENS = 500_000
MAX_BATCH_TOKENS = 65536
MAX_BATCH_ROWS = 16
WORKERS = 8


class Prefixes(Dataset):
    """``(row, n)`` → ``(row, Tensor[n, hidden])``: the first ``n`` CoT tokens of a row."""

    def __init__(self, paths: list[str], ids: list[str]):
        self.paths, self.ids = paths, ids

    def __getitem__(self, key):
        row, n = key
        return row, local_handle(self.paths[row]).get_slice(self.ids[row])[:n]


def collate(items):
    rows, seqs = zip(*items)
    return torch.tensor(rows), pad_sequence(seqs, batch_first=True), torch.tensor([len(s) for s in seqs])


def loader(ds: Prefixes, rows: np.ndarray, lengths: np.ndarray, rng=None) -> DataLoader:
    """Length-bucketed batches of ``rows`` cut to ``lengths``, ``rows × longest ≤ MAX_BATCH_TOKENS``; shuffled
    with ``rng``."""
    batches, cur = [], []
    for r in rows[np.argsort(lengths[rows], kind="stable")]:
        if cur and (lengths[r] * (len(cur) + 1) > MAX_BATCH_TOKENS or len(cur) == MAX_BATCH_ROWS):
            batches.append(cur)
            cur = []
        cur.append((int(r), int(lengths[r])))
    batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return DataLoader(ds, batch_sampler=batches, collate_fn=collate, num_workers=WORKERS, prefetch_factor=4,
                      pin_memory=True)


def token_stats(ds: Prefixes, rows: np.ndarray, n_cot: np.ndarray, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean and std over the CoT tokens of random ``rows``, about ``NORM_TOKENS`` of them."""
    rows = rng.permutation(rows)
    rows = rows[: np.searchsorted(np.cumsum(n_cot[rows]), NORM_TOKENS) + 1]
    total = squares = n = 0
    for r in rows:
        seq = ds[int(r), int(n_cot[r])][1].double()
        total, squares, n = total + seq.sum(0), squares + (seq * seq).sum(0), n + len(seq)
    mean = total / n
    return mean.float(), (squares / n - mean * mean).clamp_min(0).sqrt().clamp_min(1e-4).float()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(X.MODELS))
    a = ap.parse_args(argv)
    torch.manual_seed(X.SEED)
    rng = np.random.default_rng(X.SEED)
    dev = torch.device("cuda")

    store, manifest = X.open_store(a.model)
    uni = X.load_universe(a.model)
    ids = uni["rollout_id"].tolist()
    n_cot = manifest.loc[ids, "n_cot_tokens"].to_numpy(int)
    folder = X.path(f"activations/{a.model}_ssp_rsp")
    ds = Prefixes([str(folder / store.file_of(r, X.MODELS[a.model].layer)) for r in ids], ids)

    masks, te = X.split_masks(uni)
    units = [(unit, hp) for unit in masks for hp in HP_GRID]
    tr, va = (np.stack([masks[unit][i] for unit, _ in units]) for i in (0, 1))
    y = uni[X.LABEL].to_numpy(int)
    n_class = np.stack([(tr & (y == c)).sum(1) for c in (0, 1)])
    weights = tr * (tr.sum(1, keepdims=True) / (2.0 * n_class[y].T))    # class-balanced, zero off the train rows
    train_rows, val_rows = np.flatnonzero(tr.any(0)), np.flatnonzero(va.any(0))

    mean, std = (t.to(dev) for t in token_stats(ds, train_rows, n_cot, rng))
    probes = [AttentionProbe(store.hidden, 2, aggregation=hp["aggregation"], **PROBE_KW).to(dev) for _, hp in units]
    opts = [torch.optim.AdamW(p.parameters(), lr=hp["lr"], weight_decay=WEIGHT_DECAY) for p, (_, hp) in zip(probes, units)]
    y_dev = torch.tensor(y, device=dev)
    w_dev = torch.tensor(weights, dtype=torch.float32, device=dev)

    def standardise(H: torch.Tensor) -> torch.Tensor:
        return (H.to(dev, non_blocking=True).float() - mean) / std

    @torch.no_grad()
    def score(rows: np.ndarray, which) -> np.ndarray:
        """P(unfaithful) ``[n_units, n_rows, n_fractions]`` of the probes ``which`` on ``rows`` (NaN elsewhere)."""
        out = np.full((len(units), len(ids), len(X.FRACTIONS)), np.nan, dtype=np.float32)
        for batch, H, lengths in loader(ds, rows, n_cot):
            H = standardise(H)
            prefixes = torch.tensor([X.prefix_lengths(int(n)) for n in lengths], device=dev).T
            for k in which:
                for j, prefix in enumerate(prefixes):
                    out[k, batch.numpy(), j] = p_detection_target(probes[k](H, prefix)).cpu().numpy()
        return out

    best_score = np.full(len(units), -np.inf)
    best_state = [None] * len(units)
    stale = np.zeros(len(units), dtype=int)
    for epoch in range(EPOCHS):
        active = np.flatnonzero(stale < PATIENCE)
        if not len(active):
            break
        prefix = rng.integers(np.maximum(1, (n_cot * MIN_PREFIX_FRAC).astype(int)), n_cot + 1)
        for batch, H, lengths in loader(ds, train_rows, prefix, rng):
            H, batch, lengths = standardise(H), batch.to(dev), lengths.to(dev)
            for k in active:
                w = w_dev[k, batch]
                if w.sum() == 0:
                    continue
                loss = (F.cross_entropy(probes[k](H, lengths), y_dev[batch], reduction="none") * w).sum() / w.sum()
                opts[k].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probes[k].parameters(), GRAD_CLIP)
                opts[k].step()
        scores = score(val_rows, active)
        for k in active:
            val = np.mean([X.auroc(y[va[k]] == 0, scores[k, va[k], j]) for j in range(len(X.FRACTIONS))])
            if val > best_score[k] + 1e-4:
                best_score[k], stale[k] = val, 0
                best_state[k] = {name: t.clone() for name, t in probes[k].state_dict().items()}
            else:
                stale[k] += 1
        print(f"epoch {epoch}: {len(active)} probes training, best val score mean {best_score.mean():.3f}", flush=True)

    n_hp = len(HP_GRID)
    chosen = {unit: i * n_hp + int(np.argmax(best_score[i * n_hp:(i + 1) * n_hp])) for i, unit in enumerate(masks)}
    for k in chosen.values():
        probes[k].load_state_dict(best_state[k])
    scores = score(np.flatnonzero(te), chosen.values())
    X.write_predictions(a.model, "attention", {unit: scores[k, te] for unit, k in chosen.items()}, uni.loc[te, "rollout_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
