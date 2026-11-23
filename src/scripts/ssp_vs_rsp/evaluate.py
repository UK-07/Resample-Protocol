#!/usr/bin/env python3
"""AUROC of every SSP-vs-RSP probe on every test set, with question-clustered bootstrap intervals.

Per (model, task, variant, test set, probe, fraction): the test rows of the set (``ssp_all`` /
``rsp_survivors`` memberships on the test split, or ``gold``), both arms' scores on those same rows, AUROC
per arm with a 95% percentile interval and the paired difference RSP − SSP, all from one shared cluster
bootstrap (clusters = questions). The model-mean difference averages the per-model bootstrap draws.

Writes ``results/auroc.csv`` (one row per cell and arm), ``results/delta.csv`` (one row per cell) and
``results/delta_model_mean.csv``.

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.evaluate
"""

from __future__ import annotations

import argparse
import itertools
import zlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.lib import ssp_vs_rsp as X
from src.lib.paths import resolve_data_path

ROOT = "${DATA_ROOT}/ssp_vs_rsp"
PROBES = ("attention", "linear", "tfidf")
UNI_COLS = ["rollout_id", "question_id", "dataset_split", "hint_style", "case", "split", "cls", "to_hint",
            "judge_label_final", "role", "reliance_label", "exclude_reason", "provenance", "in_gold"]


def fast_auroc(pos: np.ndarray, s: np.ndarray) -> float:
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    uni = pd.read_parquet(resolve_data_path(f"{ROOT}/universe/{model}.parquet"), columns=UNI_COLS)
    gold_path = resolve_data_path(f"{ROOT}/gold/{model}_gold_labels.parquet")
    if gold_path.exists():
        gold = pd.read_parquet(gold_path)[["rollout_id", "gold_used", "gold_ignored", "n_gold", "n_target"]]
        uni = uni.merge(gold, on="rollout_id", how="left")
    else:
        uni["gold_used"] = pd.NA
        uni["gold_ignored"] = pd.NA
    preds = []
    for probe in PROBES:
        p = resolve_data_path(f"{ROOT}/predictions/{model}_{probe}.parquet")
        if p.exists():
            preds.append(pd.read_parquet(p))
    return uni, (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def frac_avg(arm_draws: dict, out) -> None:
    """``summary_fracavg.csv``: AUROC averaged over the 10 fractions per arm and the paired difference, per
    model and as the model mean (draws averaged over fractions, then over models), with 95% intervals."""
    keys = sorted({k[:5] for k in arm_draws})
    per_model = {}
    rows = []
    for key in keys:
        fs = sorted(k[5] for k in arm_draws if k[:5] == key)
        if len(fs) != len(X.FRACTIONS):
            continue
        ssp = np.nanmean(np.stack([arm_draws[key + (f,)][0] for f in fs]), axis=0)
        rsp = np.nanmean(np.stack([arm_draws[key + (f,)][1] for f in fs]), axis=0)
        p_ssp = float(np.mean([arm_draws[key + (f,)][2] for f in fs]))
        p_rsp = float(np.mean([arm_draws[key + (f,)][3] for f in fs]))
        per_model[key] = (ssp, rsp, p_ssp, p_rsp)
        rows.append(_summary_row(dict(zip(("model", "task", "variant", "test", "probe"), key)), ssp, rsp, p_ssp, p_rsp))
    for sub in sorted({k[1:] for k in per_model}):
        ks = [k for k in per_model if k[1:] == sub]
        ssp = np.nanmean(np.stack([per_model[k][0] for k in ks]), axis=0)
        rsp = np.nanmean(np.stack([per_model[k][1] for k in ks]), axis=0)
        rows.append(_summary_row({"model": "mean(" + ",".join(sorted(k[0] for k in ks)) + ")",
                                  **dict(zip(("task", "variant", "test", "probe"), sub))}, ssp, rsp,
                                 float(np.mean([per_model[k][2] for k in ks])),
                                 float(np.mean([per_model[k][3] for k in ks]))))
    pd.DataFrame(rows).to_csv(out / "summary_fracavg.csv", index=False)


def _summary_row(key, ssp, rsp, p_ssp, p_rsp) -> dict:
    d = rsp - ssp
    q = lambda v: np.nanquantile(v, [0.025, 0.975])  # noqa: E731
    return {**key, "auroc_ssp_all": p_ssp, "ssp_lo": q(ssp)[0], "ssp_hi": q(ssp)[1],
            "auroc_rsp_survivors": p_rsp, "rsp_lo": q(rsp)[0], "rsp_hi": q(rsp)[1],
            "delta": p_rsp - p_ssp, "delta_lo": q(d)[0], "delta_hi": q(d)[1], "p_le_0": float(np.nanmean(d <= 0))}


def sized_analysis(models, out, *, n_boot: int, seed: int) -> None:
    """``sized_fracavg.csv``: rsp_survivors vs the size-matched control ``ssp_sized`` (linear, TF-IDF), AUROC
    averaged over fractions, paired cluster bootstrap; per model and model mean."""
    rows, per_model = [], {}
    for model in models:
        uni, preds = load(model)
        sized = [pd.read_parquet(p) for probe in ("linear", "tfidf")
                 if (p := resolve_data_path(f"{ROOT}/predictions/{model}_{probe}_sized.parquet")).exists()]
        if preds.empty or not sized:
            continue
        allp = pd.concat([preds[preds["arm"].eq("rsp_survivors")]] + sized, ignore_index=True)
        wide = allp.pivot_table(index=["rollout_id", "task", "variant", "probe", "fraction"], columns="arm",
                                values="score").reset_index().dropna(subset=["rsp_survivors", "ssp_sized"])
        tests = [t for t in X.TEST_SETS if t != "gold" or uni["gold_used"].notna().any()]
        for task, variant, test in itertools.product(X.TASKS, X.VARIANTS, tests):
            member = X.membership(uni, task, test, variant) & uni["split"].astype(str).eq("test")
            tset = uni[member.to_numpy()]
            lab = pd.Series(X.task_labels(uni, task)[member].astype(int).to_numpy(),
                            index=tset["rollout_id"].astype(str).to_numpy())
            q = pd.Series(tset["question_id"].astype(str).to_numpy(), index=lab.index)
            for probe, sub_p in wide[(wide["task"] == task) & (wide["variant"] == variant)].groupby("probe"):
                b_r, b_s, p_r, p_s = [], [], [], []
                for f, sub in sub_p.groupby("fraction"):
                    sub = sub[sub["rollout_id"].isin(lab.index)]
                    pos = lab.loc[sub["rollout_id"]].to_numpy() == 0
                    codes, uniq = pd.factorize(q.loc[sub["rollout_id"]].to_numpy())
                    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
                    r, z = sub["rsp_survivors"].to_numpy(float), sub["ssp_sized"].to_numpy(float)
                    rng = np.random.default_rng([seed, zlib.crc32(f"{model}|{task}|{variant}|{test}|{probe}".encode())])
                    br, bs = np.empty(n_boot), np.empty(n_boot)
                    for b in range(n_boot):
                        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
                        br[b], bs[b] = fast_auroc(pos[idx], r[idx]), fast_auroc(pos[idx], z[idx])
                    b_r.append(br); b_s.append(bs); p_r.append(fast_auroc(pos, r)); p_s.append(fast_auroc(pos, z))
                key = (model, task, variant, test, probe)
                per_model[key] = (np.nanmean(b_s, axis=0), np.nanmean(b_r, axis=0), float(np.mean(p_s)), float(np.mean(p_r)))
    for key, (bs, br, ps, pr) in per_model.items():
        rows.append(_summary_row(dict(zip(("model", "task", "variant", "test", "probe"), key)), bs, br, ps, pr))
    for sub in sorted({k[1:] for k in per_model}):
        ks = [k for k in per_model if k[1:] == sub]
        rows.append(_summary_row({"model": "mean(" + ",".join(sorted(k[0] for k in ks)) + ")",
                                  **dict(zip(("task", "variant", "test", "probe"), sub))},
                                 np.nanmean(np.stack([per_model[k][0] for k in ks]), axis=0),
                                 np.nanmean(np.stack([per_model[k][1] for k in ks]), axis=0),
                                 float(np.mean([per_model[k][2] for k in ks])),
                                 float(np.mean([per_model[k][3] for k in ks]))))
    df = pd.DataFrame(rows).rename(columns={"auroc_ssp_all": "auroc_ssp_sized", "ssp_lo": "sized_lo", "ssp_hi": "sized_hi"})
    df.to_csv(out / "sized_fracavg.csv", index=False)


def extra_analyses(models, out) -> None:
    """``gold_precision.csv``: per (model, class, reliance label) the share of test pairs whose 12 fresh re-rolls
    confirm them (switches: gold_used; controls: gold_ignored). ``by_reliance.csv``: per arm, AUROC of the
    unfaithful test rows of each reliance group (survivors = robust_used, non-survivors = weak_used + mixed)
    against the task's ssp_all negatives, no_posthoc variant."""
    gp, br = [], []
    for model in models:
        uni, preds = load(model)
        if uni["gold_used"].notna().any():
            g = uni[uni["in_gold"].astype(bool) & uni["n_gold"].eq(len(X.GOLD_SEEDS))]
            for (cls, rel), sub in g.groupby(["cls", "reliance_label"]):
                conf = sub["gold_ignored"] if cls == "C" else sub["gold_used"]
                gp.append({"model": model, "cls": cls, "reliance_label": rel, "n": len(sub),
                           "gold_confirmed": float(conf.astype(bool).mean()),
                           "mean_target_share": float((sub["n_target"] / len(X.GOLD_SEEDS)).mean())})
        if preds.empty:
            continue
        test = uni[uni["split"].astype(str).eq("test")]
        for task in X.TASKS:
            pos_cls, neg_cls = X.TASK_CLASSES[task]
            neg_ids = set(test.loc[test["cls"].eq(neg_cls) & test["hint_style"].ne(X.POST_HOC), "rollout_id"])
            u = test[test["cls"].eq(pos_cls) & test["hint_style"].ne(X.POST_HOC)]
            groups = {"survivors": set(u.loc[u["reliance_label"].eq("robust_used"), "rollout_id"]),
                      "non_survivors": set(u.loc[u["reliance_label"].isin(["weak_used", "mixed"]), "rollout_id"])}
            pv = preds[(preds["task"] == task) & (preds["variant"] == "no_posthoc")]
            for (probe, arm, f), sub in pv.groupby(["probe", "arm", "fraction"]):
                sc = sub.set_index("rollout_id")["score"]
                for gname, ids in groups.items():
                    keep = [r for r in sc.index if r in ids or r in neg_ids]
                    pos = np.array([r in ids for r in keep])
                    br.append({"model": model, "task": task, "probe": probe, "arm": arm, "fraction": f,
                               "group": gname, "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
                               "auroc": fast_auroc(pos, sc.loc[keep].to_numpy())})
    pd.DataFrame(gp).to_csv(out / "gold_precision.csv", index=False)
    pd.DataFrame(br).to_csv(out / "by_reliance.csv", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(X.MODELS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=X.DEFAULT_SEED)
    ap.add_argument("--results", default=f"{ROOT}/results", help="results directory (default %(default)s)")
    a = ap.parse_args(argv)
    rows, deltas, draws, arm_draws = [], [], {}, {}
    for model in a.models:
        uni, preds = load(model)
        if preds.empty:
            print(f"[{model}] no predictions yet")
            continue
        has_gold = uni["gold_used"].notna().any()
        tests = [t for t in X.TEST_SETS if t != "gold" or has_gold]
        wide = preds.pivot_table(index=["rollout_id", "task", "variant", "probe", "fraction"], columns="arm",
                                 values="score").reset_index()
        for task, variant, test in itertools.product(X.TASKS, X.VARIANTS, tests):
            member = X.membership(uni, task, test, variant) & uni["split"].astype(str).eq("test")
            tset = uni[member.to_numpy()]
            labels = X.task_labels(uni, task)[member]
            lab = pd.Series(labels.astype(int).to_numpy(), index=tset["rollout_id"].astype(str).to_numpy())
            q = pd.Series(tset["question_id"].astype(str).to_numpy(), index=lab.index)
            sub_all = wide[(wide["task"] == task) & (wide["variant"] == variant)]
            for (probe, f), sub in sub_all.groupby(["probe", "fraction"]):
                sub = sub[sub["rollout_id"].isin(lab.index)].dropna(subset=list(X.ARMS))
                if sub.empty:
                    continue
                y = lab.loc[sub["rollout_id"]].to_numpy()
                pos = y == 0
                clusters = q.loc[sub["rollout_id"]].to_numpy()
                codes, uniq = pd.factorize(clusters)
                members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
                sc = {arm: sub[arm].to_numpy(dtype=np.float64) for arm in X.ARMS}
                point = {arm: fast_auroc(pos, sc[arm]) for arm in X.ARMS}
                # Same seed for every fraction: the rows are the same, so the draws pair up across fractions.
                rng = np.random.default_rng([a.seed, zlib.crc32(f"{model}|{task}|{variant}|{test}|{probe}".encode())])
                boot = {arm: np.empty(a.n_boot) for arm in X.ARMS}
                for b in range(a.n_boot):
                    idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
                    for arm in X.ARMS:
                        boot[arm][b] = fast_auroc(pos[idx], sc[arm][idx])
                key = dict(model=model, task=task, variant=variant, test=test, probe=probe, fraction=f,
                           n_pos=int(pos.sum()), n_neg=int((~pos).sum()), n_questions=len(uniq))
                for arm in X.ARMS:
                    lo, hi = np.nanquantile(boot[arm], [0.025, 0.975])
                    rows.append({**key, "arm": arm, "auroc": point[arm], "lo": lo, "hi": hi})
                d = boot["rsp_survivors"] - boot["ssp_all"]
                lo, hi = np.nanquantile(d, [0.025, 0.975])
                deltas.append({**key, "delta": point["rsp_survivors"] - point["ssp_all"], "lo": lo, "hi": hi,
                               "p_le_0": float(np.nanmean(d <= 0))})
                draws[(model, task, variant, test, probe, f)] = d
                arm_draws[(model, task, variant, test, probe, f)] = (boot["ssp_all"], boot["rsp_survivors"],
                                                                     point["ssp_all"], point["rsp_survivors"])
        print(f"[{model}] evaluated", flush=True)
    out = resolve_data_path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "auroc.csv", index=False)
    dd = pd.DataFrame(deltas)
    dd.to_csv(out / "delta.csv", index=False)
    mean_rows = []
    if len(dd):
        for (task, variant, test, probe, f), g in dd.groupby(["task", "variant", "test", "probe", "fraction"]):
            ms = sorted(g["model"])
            stack = np.stack([draws[(m, task, variant, test, probe, f)] for m in ms])
            md = np.nanmean(stack, axis=0)
            lo, hi = np.nanquantile(md, [0.025, 0.975])
            mean_rows.append({"task": task, "variant": variant, "test": test, "probe": probe, "fraction": f,
                              "models": ",".join(ms), "delta": float(g["delta"].mean()), "lo": lo, "hi": hi,
                              "p_le_0": float(np.nanmean(md <= 0))})
    pd.DataFrame(mean_rows).to_csv(out / "delta_model_mean.csv", index=False)
    frac_avg(arm_draws, out)
    extra_analyses(a.models, out)
    sized_analysis(a.models, out, n_boot=min(a.n_boot, 500), seed=a.seed)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
