#!/usr/bin/env python3
"""Per-prefix linear and TF-IDF probes of the SSP-vs-RSP comparison (one probe per CoT fraction).

``--probe linear``: the residual at the last token of the prefix, standardised on the train rows.
``--probe tfidf``: the prefix text (cut at the model-token boundary, so 30% means the tokens the activation
probes see) through the repo's spaCy normalisation and TF-IDF vectorizer. Both: L2 logistic regression
fitted on the train rows of each (variant, arm), C picked by val AUROC, every test row scored
(``predictions/<model>_<probe>.parquet``). The features are cached under ``features/``; the linear ones are
read from the model's activation store.

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.train_flat_probes --model qwen3-8b --probe tfidf
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.lib import ssp_vs_rsp as X
from src.lib.text_features import TextPipelineConfig, VectorizerConfig, fit_vectorizer, normalise_prefixes

C_GRID = {"linear": (1e-4, 1e-3, 1e-2, 1e-1, 1.0), "tfidf": (0.1, 1.0, 10.0, 100.0)}


def prefix_points(model: str, uni: pd.DataFrame) -> np.ndarray:
    """The residual at the last token of each CoT prefix: ``[n_rows, len(FRACTIONS), hidden]``."""
    import torch

    cache = X.path(f"features/{model}_prefix_points.pt")
    if not cache.exists():
        store, _ = X.open_store(model)
        layer = X.MODELS[model].layer

        def points(rollout_id: str):
            seq = store.get(rollout_id, layers=[layer])[layer]
            return seq[[n - 1 for n in X.prefix_lengths(len(seq))]]

        ids = uni["rollout_id"].tolist()
        with ThreadPoolExecutor(16) as pool:
            feats = torch.stack(list(pool.map(points, ids)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rollout_ids": ids, "X": feats}, cache)
    blob = torch.load(cache)
    rows = pd.Series(range(len(blob["rollout_ids"])), index=blob["rollout_ids"])[uni["rollout_id"]]
    return blob["X"][rows.tolist()].float().numpy()


def prefix_texts(model: str, uni: pd.DataFrame, n_process: int) -> np.ndarray:
    """The normalised text of each CoT prefix: ``[n_rows, len(FRACTIONS)]`` strings."""
    cache = X.path(f"features/{model}_prefix_texts.parquet")
    if not cache.exists():
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(X.MODELS[model].name, trust_remote_code=True)
        cots = X.load_universe(model, "reasoning")["reasoning"].tolist()
        cutoffs = []
        for cot in cots:
            ends = [end for _, end in tok(cot, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
            cutoffs.append([ends[n - 1] for n in X.prefix_lengths(len(ends))])
        texts = normalise_prefixes(cots, cutoffs, TextPipelineConfig(n_process=n_process))
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(texts, index=uni["rollout_id"].to_numpy(), columns=[str(f) for f in X.FRACTIONS]).to_parquet(cache)
    return pd.read_parquet(cache).loc[uni["rollout_id"]].to_numpy()


def fit(probe: str, feats: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> np.ndarray:
    """One (variant, arm, fraction): fit on the train rows, pick C by val AUROC, score the test rows
    (P(unfaithful))."""
    pre = StandardScaler().fit(feats[tr]) if probe == "linear" else fit_vectorizer(feats[tr], VectorizerConfig())
    # float64: a float32 lbfgs solve changes with the number of BLAS threads, i.e. with --jobs
    f_tr, f_va, f_te = (pre.transform(feats[rows]).astype(np.float64) for rows in (tr, va, te))
    clfs = [LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(f_tr, y[tr]) for C in C_GRID[probe]]
    best = max(clfs, key=lambda clf: X.auroc(y[va] == 0, clf.predict_proba(f_va)[:, 0]))
    return best.predict_proba(f_te)[:, 0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(X.MODELS))
    ap.add_argument("--probe", required=True, choices=list(C_GRID))
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args(argv)

    uni = X.load_universe(a.model)
    feats = prefix_points(a.model, uni) if a.probe == "linear" else prefix_texts(a.model, uni, a.jobs)
    y = uni[X.LABEL].to_numpy(int)
    masks, te = X.split_masks(uni)
    n = len(X.FRACTIONS)
    fits = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fit)(a.probe, feats[:, j], y, *masks[unit], te)
                                              for unit in masks for j in range(n))
    scores = {unit: np.column_stack(fits[i * n:(i + 1) * n]) for i, unit in enumerate(masks)}
    X.write_predictions(a.model, a.probe, scores, uni.loc[te, "rollout_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
