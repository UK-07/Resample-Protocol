#!/usr/bin/env python3
"""Per universe row, the residual at the LAST token of each CoT prefix (10%, 20%, ..., 100%).

Reads the model's SSP-vs-RSP activation store (one seq layer) and writes
``features/<model>_prefix_points.pt``: ``{"rollout_ids", "n_cot", "fractions", "layer", "X"}`` with
``X`` bf16 ``[n_rows, len(FRACTIONS), hidden]``; position ``prefix_lengths(n_cot)[j] - 1`` of the CoT span.

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.extract_prefix_points --model qwen3-8b
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from src.lib import ssp_vs_rsp as X
from src.lib.activation_store import SequenceStore, make_backend, read_store_manifest
from src.lib.paths import resolve_data_path

ROOT = "${DATA_ROOT}/ssp_vs_rsp"


def open_store(model: str, root: str = ROOT):
    backend = make_backend({"backend": "local", "local_dir": f"{root}/activations"})
    folder = f"{model}_ssp_rsp"
    manifest, meta = read_store_manifest(backend, folder)
    return SequenceStore(backend, folder, manifest, meta), manifest, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(X.MODELS))
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    store, manifest, meta = open_store(a.model)
    layer = X.MODELS[a.model].layer
    ids = manifest["rollout_id"].astype(str).tolist()
    n_cot = [int(v) for v in manifest["n_cot_tokens"]]
    hidden = int(meta["hidden_size"])
    out = torch.empty((len(ids), len(X.FRACTIONS), hidden), dtype=torch.bfloat16)

    def one(i: int) -> None:
        t = store.get(ids[i], layers=[layer])[layer]
        if t.shape[0] != n_cot[i]:
            raise ValueError(f"{ids[i]}: seq has {t.shape[0]} tokens, manifest says {n_cot[i]}")
        pos = [k - 1 for k in X.prefix_lengths(n_cot[i])]
        out[i] = t[pos].to(torch.bfloat16)

    with ThreadPoolExecutor(a.threads) as pool:
        for j, _ in enumerate(pool.map(one, range(len(ids)))):
            if j % 500 == 0:
                print(f"  {j}/{len(ids)}", flush=True)
    dest = resolve_data_path(f"{ROOT}/features/{a.model}_prefix_points.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rollout_ids": ids, "n_cot": n_cot, "fractions": list(X.FRACTIONS), "layer": layer, "X": out}, dest)
    print(f"wrote {dest} {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
