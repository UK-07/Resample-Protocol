#!/usr/bin/env python3
"""Attention probes of the SSP-vs-RSP comparison: every (task, variant, arm, hyper-parameter) probe of one
model trained in ONE shared pass over the activation store.

Each probe is the repo's ``AttentionProbe`` (4 heads, relative-position slope, softmax or rolling-max
aggregation) trained on random CoT prefixes in ``[0.1 L, L]`` (the repo's attention default), with its loss
masked to its own training rows (class-balanced weights), AdamW, gradient clipping and early stopping on
its own val rows: the mean AUROC over the 10 CoT fractions. Inputs are standardised once with per-dimension
statistics from train-split tokens (shared by every probe; written into each checkpoint). The disk reads, not
the probes, dominate the cost, so one pass serves all 32 probes.

The hyper-parameter per (task, variant, arm) is picked by val score; every test-split row of the task's two
classes is scored at every fraction.

Writes ``predictions/<model>_attention.parquet`` (+ ``_selection.csv``, ``_history.csv``) and
``probes/<model>_attention/<task>__<variant>__<arm>.pt``.

    DATA_ROOT=/workspace/data python -u -m src.scripts.ssp_vs_rsp.train_attention --model qwen3-8b
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.lib import ssp_vs_rsp as X
from src.lib.activation_store import local_handle, make_backend, read_store_manifest, store_path, STORE_FILES
from src.lib.paths import resolve_data_path
from src.lib.probes import AttentionProbe, p_detection_target, to_checkpoint

ROOT = "${DATA_ROOT}/ssp_vs_rsp"
HP_GRID = [{"lr": lr, "aggregation": agg} for agg in ("softmax", "rolling_max") for lr in (3e-4, 1e-3)]
PROBE_KW = dict(heads=4, window=16, position_bias=True)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MIN_PREFIX_FRAC = 0.1
NORM_TOKENS = 500_000


class Rows(Dataset):
    """Item ``i`` → (i, ``Tensor[length_i, hidden]`` bf16): the first ``length_i`` CoT tokens of row ``i``."""

    def __init__(self, paths: list[str], ids: list[str]):
        self.paths, self.ids = paths, ids
        self.lengths: np.ndarray | None = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, key):
        i, n = key
        return i, local_handle(self.paths[i]).get_slice(self.ids[i])[:n]


def collate(items):
    idx = torch.tensor([i for i, _ in items], dtype=torch.long)
    lengths = torch.tensor([t.shape[0] for _, t in items], dtype=torch.long)
    H = torch.zeros((len(items), int(lengths.max()), items[0][1].shape[1]), dtype=items[0][1].dtype)
    for k, (_, t) in enumerate(items):
        H[k, : t.shape[0]] = t
    return idx, H, lengths


def make_batches(rows: np.ndarray, lengths: np.ndarray, *, max_tokens: int, max_rows: int,
                 rng: np.random.Generator | None) -> list[list[tuple[int, int]]]:
    """Length-bucketed batches of ``(row, length)`` with ``rows × longest ≤ max_tokens``; shuffled when ``rng``."""
    order = rows[np.argsort(lengths[rows], kind="stable")]
    batches, cur, longest = [], [], 0
    for r in order:
        n = int(lengths[r])
        if cur and (max(longest, n) * (len(cur) + 1) > max_tokens or len(cur) >= max_rows):
            batches.append(cur)
            cur, longest = [], 0
        cur.append((int(r), n))
        longest = max(longest, n)
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def loader(ds, batches, workers):
    return DataLoader(ds, batch_sampler=batches, collate_fn=collate, num_workers=workers,
                      prefetch_factor=4 if workers else None, persistent_workers=False, pin_memory=True)


def standardization(ds: Rows, rows: np.ndarray, n_cot: np.ndarray, rng) -> tuple[torch.Tensor, torch.Tensor]:
    picked, total = [], 0
    for r in rng.permutation(rows):
        picked.append(int(r))
        total += int(n_cot[r])
        if total >= NORM_TOKENS:
            break
    s = s2 = None
    n = 0
    for r in picked:
        t = ds[(r, int(n_cot[r]))][1].double()
        s = t.sum(0) if s is None else s + t.sum(0)
        s2 = (t * t).sum(0) if s2 is None else s2 + (t * t).sum(0)
        n += t.shape[0]
    mean = s / n
    std = (s2 / n - mean * mean).clamp_min(0).sqrt().clamp_min(1e-4)
    return mean.float(), std.float()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(X.MODELS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-batch-tokens", type=int, default=65536)
    ap.add_argument("--max-rows", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=X.DEFAULT_SEED)
    ap.add_argument("--limit-epochs-debug", type=int, default=None, help="cap batches per epoch (smoke test)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    layer = X.MODELS[a.model].layer

    backend = make_backend({"backend": "local", "local_dir": resolve_data_path(f"{ROOT}/activations").as_posix()})
    folder = f"{a.model}_ssp_rsp"
    manifest, meta = read_store_manifest(backend, folder)
    hidden = int(meta["hidden_size"])
    man = manifest.set_index(manifest["rollout_id"].astype(str))
    uni = pd.read_parquet(resolve_data_path(f"{ROOT}/universe/{a.model}.parquet"),
                          columns=["rollout_id", "question_id", "dataset_split", "original_index", "subject_model",
                                   "hint_style", "case", "split", "cls", "to_hint", "judge_label_final", "role",
                                   "reliance_label", "exclude_reason", "provenance"])
    stored = uni["rollout_id"].astype(str).isin(man.index)
    if (~stored).any():
        print(f"  {int((~stored).sum())} universe rows not in the store (collector skips) — dropped")
    uni = uni[stored.to_numpy()].reset_index(drop=True)
    ids = uni["rollout_id"].astype(str).tolist()
    n_cot = man.loc[ids, "n_cot_tokens"].astype(int).to_numpy()
    paths = [backend.download_file(store_path(folder, STORE_FILES["seq"].format(layer=layer, part=int(p))))
             for p in man.loc[ids, "seq_file"].map(lambda v: int(str(v).split("part")[-1].split(".")[0]))]
    paths = [str(p) for p in paths]
    ds = Rows(paths, ids)
    split = uni["split"].astype(str).to_numpy()

    # Units: one probe per (task, variant, arm, hp).
    units, tr_masks, va_masks, labels = [], [], [], {}
    for task in X.TASKS:
        labels[task] = X.task_labels(uni, task).fillna(-1).astype(int).to_numpy()
    for task, variant, arm in itertools.product(X.TASKS, X.VARIANTS, X.ARMS):
        member = X.membership(uni, task, arm, variant).to_numpy()
        for h, hp in enumerate(HP_GRID):
            units.append({"task": task, "variant": variant, "arm": arm, "hp": h, **hp})
            tr_masks.append(member & (split == "train"))
            va_masks.append(member & (split == "val"))
    tr_masks, va_masks = np.stack(tr_masks), np.stack(va_masks)
    Y = np.stack([labels[u["task"]] for u in units])            # [n_units, n_rows]
    W = np.zeros_like(Y, dtype=np.float32)                      # balanced class weights on train rows
    for k in range(len(units)):
        y = Y[k][tr_masks[k]]
        for c in (0, 1):
            n_c = int((y == c).sum())
            W[k][tr_masks[k] & (Y[k] == c)] = len(y) / (2.0 * max(n_c, 1))
    print(f"{a.model}: {len(ids)} rows, {len(units)} probes; train rows per probe "
          f"{int(tr_masks.sum(1).min())}–{int(tr_masks.sum(1).max())}", flush=True)

    train_rows = np.flatnonzero(tr_masks.any(0))
    val_rows = np.flatnonzero(va_masks.any(0))
    test_rows = np.flatnonzero((split == "test") & (np.stack(list(labels.values())) >= 0).any(0))
    mean, std = standardization(ds, train_rows, n_cot, rng)
    mean_d, std_d = mean.to(dev), std.to(dev)

    probes, opts = [], []
    for u in units:
        p = AttentionProbe(hidden, 2, aggregation=u["aggregation"], **PROBE_KW).to(dev)
        probes.append(p)
        opts.append(torch.optim.AdamW(p.parameters(), lr=u["lr"], weight_decay=WEIGHT_DECAY))
    Yd, Wd = torch.tensor(Y, device=dev).clamp_min(0), torch.tensor(W, device=dev)
    best = [{"score": -np.inf, "epoch": -1, "state": None} for _ in units]
    stale = np.zeros(len(units), dtype=int)
    active = np.ones(len(units), dtype=bool)
    history = []

    def evaluate(rows: np.ndarray, which: np.ndarray) -> np.ndarray:
        """Scores ``[n_units, len(rows), n_frac]`` (NaN where unused) for the probes in ``which``."""
        pos = {int(r): j for j, r in enumerate(rows)}
        out = np.full((len(units), len(rows), len(X.FRACTIONS)), np.nan, dtype=np.float32)
        batches = make_batches(rows, n_cot, max_tokens=a.max_batch_tokens, max_rows=a.max_rows, rng=None)
        for p in probes:
            p.eval()
        with torch.no_grad():
            for idx, H, lengths in loader(ds, batches, a.workers):
                H = (H.to(dev, non_blocking=True).float() - mean_d) / std_d
                cols = [pos[int(i)] for i in idx]
                L = [torch.tensor([X.prefix_lengths(int(n))[j] for n in lengths], device=dev)
                     for j in range(len(X.FRACTIONS))]
                for k in np.flatnonzero(which):
                    for j in range(len(X.FRACTIONS)):
                        out[k, cols, j] = p_detection_target(probes[k](H, L[j])).cpu().numpy()
        return out

    t_start = time.time()
    for epoch in range(a.epochs):
        if not active.any():
            break
        t0 = time.time()
        lo = np.maximum(1, (n_cot * MIN_PREFIX_FRAC).astype(int))
        pref = np.where(np.isin(np.arange(len(ids)), train_rows), rng.integers(lo, n_cot + 1), n_cot)
        batches = make_batches(train_rows, pref, max_tokens=a.max_batch_tokens, max_rows=a.max_rows, rng=rng)
        if a.limit_epochs_debug:
            batches = batches[: a.limit_epochs_debug]
        for p in probes:
            p.train()
        losses = np.zeros(len(units))
        nb = 0
        for idx, H, lengths in loader(ds, batches, a.workers):
            H = (H.to(dev, non_blocking=True).float() - mean_d) / std_d
            lengths = lengths.to(dev)
            idx_d = idx.to(dev)
            for k in np.flatnonzero(active):
                w = Wd[k, idx_d]
                if float(w.sum()) == 0.0:
                    continue
                logits = probes[k](H, lengths)
                loss = (F.cross_entropy(logits, Yd[k, idx_d], reduction="none") * w).sum() / w.sum()
                opts[k].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probes[k].parameters(), GRAD_CLIP)
                opts[k].step()
                losses[k] += float(loss)
            nb += 1
        scores = evaluate(val_rows, active)
        for k in np.flatnonzero(active):
            cols = np.searchsorted(val_rows, np.flatnonzero(va_masks[k]))
            per_f = [X.auroc(Y[k][val_rows[cols]], scores[k, cols, j]) for j in range(len(X.FRACTIONS))]
            v = float(np.nanmean(per_f))
            history.append({**units[k], "epoch": epoch, "train_loss": losses[k] / max(nb, 1), "val_score": v,
                            **{f"val_auroc_{f}": x for f, x in zip(X.FRACTIONS, per_f)}})
            if v > best[k]["score"] + 1e-4:
                best[k] = {"score": v, "epoch": epoch,
                           "state": {n: t.detach().cpu().clone() for n, t in probes[k].state_dict().items()}}
                stale[k] = 0
            else:
                stale[k] += 1
                if stale[k] >= a.patience:
                    active[k] = False
        h = pd.DataFrame(history)
        last = h[h["epoch"] == epoch]
        print(f"epoch {epoch}: {nb} batches, {time.time() - t0:.0f}s, active {int(active.sum())}/{len(units)}, "
              f"val score mean {last['val_score'].mean():.3f} (max {last['val_score'].max():.3f})", flush=True)

    for k, p in enumerate(probes):
        p.load_state_dict(best[k]["state"])
    scores = evaluate(test_rows, np.ones(len(units), dtype=bool))
    sel = pd.DataFrame([{**u, "val_score": best[k]["score"], "best_epoch": best[k]["epoch"]}
                        for k, u in enumerate(units)])
    chosen = sel.loc[sel.groupby(["task", "variant", "arm"])["val_score"].idxmax()]
    sel["chosen"] = sel.index.isin(chosen.index)
    preds = []
    ckpt_dir = resolve_data_path(f"{ROOT}/probes/{a.model}_attention")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for k in chosen.index:
        u = units[k]
        te = test_rows[Y[k][test_rows] >= 0]
        cols = np.searchsorted(test_rows, te)
        for j, f in enumerate(X.FRACTIONS):
            preds.append(pd.DataFrame({"rollout_id": np.asarray(ids)[te], "task": u["task"], "variant": u["variant"],
                                       "arm": u["arm"], "probe": "attention", "fraction": f,
                                       "score": scores[k, cols, j]}))
        probe = probes[k].cpu()
        probe.set_standardization(mean, std)
        torch.save({**to_checkpoint(probe), "unit": u, "layer": layer, "val_score": best[k]["score"]},
                   ckpt_dir / f"{u['task']}__{u['variant']}__{u['arm']}.pt")
    out = resolve_data_path(f"{ROOT}/predictions")
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(out / f"{a.model}_attention.parquet", index=False)
    sel.to_csv(out / f"{a.model}_attention_selection.csv", index=False)
    pd.DataFrame(history).to_csv(out / f"{a.model}_attention_history.csv", index=False)
    print(f"done in {time.time() - t_start:.0f}s")
    print(chosen[["task", "variant", "arm", "lr", "aggregation", "val_score", "best_epoch"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
