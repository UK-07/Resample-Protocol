#!/usr/bin/env python3
"""Per-prefix linear and TF-IDF probes of the SSP-vs-RSP comparison (one probe per CoT fraction).

``--probe linear``: the residual at the last token of the prefix (extract_prefix_points.py), standardised on
the train rows, L2 logistic regression. ``--probe tfidf``: the CoT prefix text through the repo's spaCy
normalisation (``text_features.normalise_prefixes``; the prefix is cut at the model-token boundary, so 30%
means the same tokens the activation probes see) and the repo's TF-IDF vectorizer settings, L2 logistic
regression. Per (task, variant, arm, fraction) the train split fits, C is picked by val AUROC, and every
test-split row of the task's two classes is scored.

Writes ``predictions/<model>_<probe>.parquet`` (rollout_id, task, variant, arm, probe, fraction, score) and
``predictions/<model>_<probe>_selection.csv`` (chosen C, val AUROC per unit).

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.train_flat_probes --model qwen3-8b --probe tfidf
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from src.lib import ssp_vs_rsp as X
from src.lib.paths import resolve_data_path

ROOT = "${DATA_ROOT}/ssp_vs_rsp"
C_GRID = {"linear": (1e-4, 1e-3, 1e-2, 1e-1, 1.0), "tfidf": (0.1, 1.0, 10.0, 100.0)}
VECTORIZER = dict(ngram_range=(1, 2), min_df=5, max_df=1.0, max_features=100000, sublinear_tf=True, norm="l2",
                  binary=False, token_pattern=r"\S+", lowercase=False)


def load_universe(model: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_data_path(f"{ROOT}/universe/{model}.parquet"))


def tfidf_prefix_texts(model: str, uni: pd.DataFrame, n_process: int) -> pd.DataFrame:
    """``rollout_id × fraction → normalised prefix text`` (cached under features/)."""
    cache = resolve_data_path(f"{ROOT}/features/{model}_prefix_texts.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    from transformers import AutoTokenizer

    from src.lib.text_features import TextPipelineConfig, load_nlp, normalise_prefixes

    tok = AutoTokenizer.from_pretrained(X.MODELS[model].model_name, trust_remote_code=True)
    cots = uni["reasoning"].astype(str).tolist()
    cutoffs, n_tokens = [], []
    for cot in cots:
        enc = tok(cot, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        n = len(enc["input_ids"])
        n_tokens.append(n)
        cutoffs.append([int(offs[k - 1][1]) for k in X.prefix_lengths(max(n, 1))] if n else [0] * len(X.FRACTIONS))
    cfg = TextPipelineConfig(lowercase=True, remove_stopwords=True, lemmatize=True, spacy_model="en_core_web_sm",
                             min_token_chars=2, keep_numbers=False, n_process=n_process, batch_size=64)
    t0 = time.time()
    texts = normalise_prefixes(cots, cutoffs, cfg, nlp=load_nlp(cfg))
    print(f"  spaCy over {len(cots)} CoTs in {time.time() - t0:.0f}s", flush=True)
    rows = [{"rollout_id": rid, "fraction": f, "n_cot": n, "text": t}
            for rid, n, ts in zip(uni["rollout_id"].astype(str), n_tokens, texts) for f, t in zip(X.FRACTIONS, ts)]
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def fit_select(Xtr, ytr, Xva, yva, Cs, *, scale: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = None
    if scale:
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xva = scaler.transform(Xtr), scaler.transform(Xva)
    best = (-np.inf, None, None)
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(Xtr, ytr)
        s = clf.predict_proba(Xva)[:, list(clf.classes_).index(0)]
        v = X.auroc(yva, s)
        if v > best[0]:
            best = (v, C, clf)
    return best[0], best[1], best[2], scaler


def unit_masks(uni: pd.DataFrame) -> dict:
    """Per (task, variant, arm): train / val row masks, per task: test mask and labels (0 = unfaithful)."""
    split = uni["split"].astype(str)
    out = {}
    for task in X.TASKS:
        labels = X.task_labels(uni, task)
        out[task] = {"y": labels.fillna(-1).astype(int).to_numpy(),
                     "te": (labels.notna() & split.eq("test")).to_numpy()}
        for variant in X.VARIANTS:
            for arm in X.ARMS + ("ssp_sized",):
                member = (X.size_matched(uni, task, variant) if arm == "ssp_sized"
                          else X.membership(uni, task, arm, variant))
                out[(task, variant, arm)] = ((member & split.eq("train")).to_numpy(),
                                             (member & split.eq("val")).to_numpy())
    return out


def run_unit(probe, block, ids, tr, va, te, y, task, variant, arm, f):
    """One (task, variant, arm, fraction): fit on ``tr``, pick C on ``va``, score ``te``. ``block`` is the
    fraction's feature matrix (linear) or its prefix texts (tfidf), aligned with ``ids``."""
    if probe == "linear":
        val, C, clf, scaler = fit_select(block[tr], y[tr], block[va], y[va], C_GRID[probe], scale=True)
        Xte = scaler.transform(block[te])
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(**VECTORIZER)
        Xtr = vec.fit_transform([block[i] for i in np.flatnonzero(tr)])
        Xva = vec.transform([block[i] for i in np.flatnonzero(va)])
        Xte = vec.transform([block[i] for i in np.flatnonzero(te)])
        val, C, clf, _ = fit_select(Xtr, y[tr], Xva, y[va], C_GRID[probe], scale=False)
    score = clf.predict_proba(Xte)[:, list(clf.classes_).index(0)]
    pred = pd.DataFrame({"rollout_id": ids[te], "task": task, "variant": variant, "arm": arm, "probe": probe,
                         "fraction": f, "score": score})
    sel = {"task": task, "variant": variant, "arm": arm, "probe": probe, "fraction": f, "C": C, "val_auroc": val,
           "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_train_pos": int((y[tr] == 0).sum())}
    return pred, sel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(X.MODELS))
    ap.add_argument("--probe", required=True, choices=["linear", "tfidf"])
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--spacy-procs", type=int, default=10)
    ap.add_argument("--sized-only", action="store_true",
                    help="train only the size-matched control arm ssp_sized (written to <model>_<probe>_sized.*)")
    a = ap.parse_args(argv)
    from joblib import Parallel, delayed

    uni = load_universe(a.model)
    if a.probe == "linear":
        import torch
        blob = torch.load(resolve_data_path(f"{ROOT}/features/{a.model}_prefix_points.pt"), weights_only=False)
        pos = {r: i for i, r in enumerate(blob["rollout_ids"])}
        keep = uni["rollout_id"].astype(str).map(pos)
        missing = int(keep.isna().sum())
        if missing:
            print(f"  {missing} universe rows have no activations (collector skips) — dropped")
        uni = uni[keep.notna().to_numpy()].reset_index(drop=True)
        feats = blob["X"][keep.dropna().astype(int).to_numpy()].float().numpy()
    else:
        texts = tfidf_prefix_texts(a.model, uni, a.spacy_procs)
        wide = texts.pivot(index="rollout_id", columns="fraction", values="text")
        feats = {f: wide.loc[uni["rollout_id"].astype(str), f].tolist() for f in X.FRACTIONS}
    masks = unit_masks(uni)
    ids = uni["rollout_id"].astype(str).to_numpy()
    arms = ("ssp_sized",) if a.sized_only else X.ARMS
    units = list(itertools.product(X.TASKS, X.VARIANTS, arms, enumerate(X.FRACTIONS)))

    def block(j, f):
        return feats[:, j, :] if a.probe == "linear" else feats[f]

    t0 = time.time()
    results = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(run_unit)(a.probe, block(j, f), ids, *masks[(t, v, arm)], masks[t]["te"], masks[t]["y"], t, v, arm, f)
        for t, v, arm, (j, f) in units)
    preds = pd.concat([p for p, _ in results], ignore_index=True)
    sel = pd.DataFrame([s for _, s in results])
    out = resolve_data_path(f"{ROOT}/predictions")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.model}_{a.probe}" + ("_sized" if a.sized_only else "")
    preds.to_parquet(out / f"{stem}.parquet", index=False)
    sel.to_csv(out / f"{stem}_selection.csv", index=False)
    print(f"{a.model} {a.probe}: {len(units)} units in {time.time() - t0:.0f}s")
    print(sel.groupby(["task", "variant", "arm"])["val_auroc"].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
