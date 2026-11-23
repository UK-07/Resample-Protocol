#!/usr/bin/env python3
"""AUROC of every SSP-vs-RSP probe on every test set, with question-clustered bootstrap intervals.

Per (model, variant, test set, probe, fraction): the test rows of the set (``ssp_all`` / ``rsp_survivors``
memberships on the test split, or ``gold``) and every arm's AUROC on those same rows with a 95% percentile
interval, from one shared cluster bootstrap (clusters = questions). The model means average the per-model
bootstrap draws, so the differences RSP − ssp_all and RSP − ssp_sized are paired.

Writes under ``--results``: ``auroc.csv`` (one row per cell and arm), ``delta_model_mean.csv`` (RSP − ssp_all
per fraction, mean over models) and ``summary_fracavg.csv`` / ``sized_fracavg.csv`` (RSP against ssp_all /
ssp_sized, mean over models and fractions).

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.evaluate
"""

from __future__ import annotations

import argparse
import itertools
import zlib

import numpy as np
import pandas as pd

from src.lib import ssp_vs_rsp as X
from src.lib.paths import resolve_data_path

PROBES = ("attention", "linear", "tfidf")
RSP = "rsp_survivors"
N_BOOT = 1000


def interval(draws: np.ndarray) -> tuple[float, float]:
    return tuple(np.nanquantile(draws, [0.025, 0.975]))


def mean_delta(cells: dict, base: str, n_keys: int) -> pd.DataFrame:
    """Per group — the first ``n_keys`` of (variant, test, probe, fraction): the AUROC of ``base`` and RSP
    averaged over the group's cells and their paired difference with its 95% interval."""
    rows = []
    for key in sorted({k[1:1 + n_keys] for k in cells}):
        boots, points = zip(*(cell for k, cell in cells.items() if k[1:1 + n_keys] == key))
        (boot_base, point_base), (boot_rsp, point_rsp) = (
            [np.mean([part[arm] for part in parts], axis=0) for parts in (boots, points)] for arm in (base, RSP))
        lo, hi = interval(boot_rsp - boot_base)
        rows.append({**dict(zip(("variant", "test", "probe", "fraction"), key)), f"auroc_{base}": point_base,
                     f"auroc_{RSP}": point_rsp, "delta": point_rsp - point_base, "delta_lo": lo, "delta_hi": hi})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(X.MODELS))
    ap.add_argument("--results", default=f"{X.ROOT}/results", help="results directory (default %(default)s)")
    a = ap.parse_args(argv)
    rows, cells = [], {}
    for model in a.models:
        uni = X.load_universe(model, gold=True)
        preds = pd.concat([pd.read_parquet(X.path(f"predictions/{model}_{probe}.parquet")) for probe in PROBES])
        wide = preds.pivot(index=["rollout_id", "variant", "probe", "fraction"], columns="arm", values="score").reset_index()
        for variant, test in itertools.product(X.VARIANTS, X.TEST_SETS):
            tset = uni[X.membership(uni, test, variant) & uni["split"].eq("test")].set_index("rollout_id")
            in_set = wide[wide["variant"].eq(variant) & wide["rollout_id"].isin(tset.index)]
            for (probe, f), sub in in_set.groupby(["probe", "fraction"]):
                pos = tset.loc[sub["rollout_id"], X.LABEL].eq(0).to_numpy()
                codes, uniq = pd.factorize(tset.loc[sub["rollout_id"], "question_id"])
                members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
                scores = {arm: sub[arm].to_numpy() for arm in X.ARMS}
                point = {arm: X.auroc(pos, scores[arm]) for arm in X.ARMS}
                # Same seed for every fraction: the rows are the same, so the draws pair up across fractions.
                rng = np.random.default_rng([X.SEED, zlib.crc32(f"{model}|{variant}|{test}|{probe}".encode())])
                boot = {arm: np.empty(N_BOOT) for arm in X.ARMS}
                for b in range(N_BOOT):
                    idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
                    for arm in X.ARMS:
                        boot[arm][b] = X.auroc(pos[idx], scores[arm][idx])
                for arm in X.ARMS:
                    lo, hi = interval(boot[arm])
                    rows.append({"model": model, "variant": variant, "test": test, "probe": probe, "fraction": f,
                                 "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()), "arm": arm,
                                 "auroc": point[arm], "lo": lo, "hi": hi})
                cells[(model, variant, test, probe, f)] = (boot, point)
        print(f"[{model}] evaluated", flush=True)

    out = resolve_data_path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "auroc.csv", index=False)
    mean_delta(cells, "ssp_all", 4).to_csv(out / "delta_model_mean.csv", index=False)
    mean_delta(cells, "ssp_all", 3).to_csv(out / "summary_fracavg.csv", index=False)
    mean_delta(cells, "ssp_sized", 3).to_csv(out / "sized_fracavg.csv", index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
