#!/usr/bin/env python3
"""Build the SSP-vs-RSP row universe per model (originals only) and the activation collector's dataset.

Per model: every unfaithful switch original plus, per (split, style, dataset, case) stratum, at most
NEG_RATIO × as many faithful switch originals and as many matched-control originals (hash order), with the
texts joined from the judged hinted-rollout CSVs. ``in_gold`` marks the test-split rows whose pair gets the
12 fresh gold re-rolls: every unfaithful test row and GOLD_NEG_RATIO × as many of each negative class.

Writes under ``--out``:
    universe/<model>.parquet                        the rows + cls + in_gold + texts
    probe_datasets/<model>_ssp_rsp_universe.parquet  (+ .meta.json, provenances [hinted_once]) for the collector
    universe/report.json                            counts per class / split / reliance and per membership

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.build_universe \
        --manifest /workspace/data/ursa/resample/resample_manifest.parquet --out /workspace/data/ssp_vs_rsp
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.lib import probe_datasets as pdx
from src.lib import selection
from src.lib import ssp_vs_rsp as X


def build(manifest_path: Path, out: Path, models: list[str], *, ratio: float, gold_ratio: float, seed: int,
          source_dir: Path, items_dir: Path) -> dict:
    frame, mmeta = pdx.load_dataset_manifest(manifest_path)
    frame = selection.with_provenance(frame)
    frame = frame[frame["subject_model"].astype(str).isin(models)].copy()
    frame["cls"] = X.row_class(frame)
    frame = frame[frame["cls"].notna()].copy()
    report: dict = {"manifest": str(manifest_path), "manifest_written_utc": mmeta.get("written_utc"),
                    "ratio": ratio, "gold_ratio": gold_ratio, "seed": seed, "models": {}}
    (out / "universe").mkdir(parents=True, exist_ok=True)
    (out / "probe_datasets").mkdir(parents=True, exist_ok=True)
    for model in models:
        df = frame[frame["subject_model"].astype(str).eq(model)].copy()
        keep = X.sample_universe(df, ratio=ratio, seed=seed)
        uni = df[keep.to_numpy()].copy()
        test = uni[uni["split"].astype(str).eq("test")]
        gold = X.sample_universe(test, ratio=gold_ratio, seed=seed, salt="gold")
        uni["in_gold"] = False
        uni.loc[gold.index[gold.to_numpy()], "in_gold"] = True
        uni = uni.sort_values("rollout_id", kind="stable").reset_index(drop=True)
        print(f"[{model}] universe {len(uni)} rows of {len(df)} ({uni['cls'].value_counts().to_dict()}); "
              f"gold pairs {int(uni['in_gold'].sum())}")
        joined = pdx.join_texts(uni.drop(columns=["sample_seed"]), source_dir=source_dir, items_dir=items_dir,
                                log=print)
        joined["sample_seed"] = uni["sample_seed"].to_numpy()
        uni_path = out / "universe" / f"{model}.parquet"
        joined.to_parquet(uni_path, index=False)

        ds = joined.copy()
        ds["label"] = pd.array(ds["cls"].ne("U").astype(int).to_numpy(), dtype="Int8")
        ds["balance_kept"] = True
        ds = ds[list(pdx.DATASET_COLUMNS)].sort_values("rollout_id", kind="stable").reset_index(drop=True)
        provenances = (selection.HINTED_ONCE,)
        pdx.validate_dataset(ds, provenances=provenances)
        name = f"{model}_ssp_rsp_universe"
        ds_path = out / "probe_datasets" / f"{name}.parquet"
        ds.to_parquet(ds_path, engine="pyarrow", compression="zstd", index=False)
        meta = {
            "schema_version": pdx.DATASET_SCHEMA_VERSION, "name": name, "predicate": "ssp_rsp_universe",
            "label_col": "cls (U = 0, F / C = 1; informational, the experiment relabels per task)",
            "provenances": list(provenances), "manifest_path": str(manifest_path),
            "manifest_written_utc": mmeta.get("written_utc"), "split": mmeta.get("split"),
            "sampling": {"ratio": ratio, "stratum": list(X.STRATUM), "seed": seed},
            "hint_styles": sorted(ds["hint_style"].astype(str).unique()), "n_rows": int(len(ds)),
            "columns": list(pdx.DATASET_COLUMNS), "fingerprint": pdx.dataset_fingerprint(ds),
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": pdx.git_sha(),
        }
        meta_file, _ = pdx.dataset_sidecars(ds_path)
        meta_file.write_text(json.dumps(meta, indent=2) + "\n")
        pdx.read_dataset(ds_path)   # the collector's reader accepts it

        rep = {
            "n_candidates": int(len(df)), "n_universe": int(len(uni)), "n_gold_pairs": int(uni["in_gold"].sum()),
            "by_class_split": {f"{c}|{s}": int(n) for (c, s), n in uni.groupby(["cls", "split"]).size().items()},
            "by_class_reliance": {f"{c}|{r}": int(n) for (c, r), n in
                                  uni.groupby(["cls", "reliance_label"]).size().items()},
            "cot_tokens_sum": int(pd.to_numeric(uni["trace_token_len"]).sum()),
            "memberships": {},
        }
        for task in X.TASKS:
            for which in X.ARMS:
                for variant in X.VARIANTS:
                    m = X.membership(uni, task, which, variant, seed=seed)
                    lab = X.task_labels(uni, task)
                    sub = uni[m.to_numpy()]
                    rep["memberships"][f"{task}|{which}|{variant}"] = {
                        f"{s}|{int(l)}": int(n) for (s, l), n in
                        sub.groupby([sub["split"].astype(str), lab[m].astype(int)]).size().items()}
        report["models"][model] = rep
    (out / "universe" / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=list(X.MODELS))
    ap.add_argument("--ratio", type=float, default=X.NEG_RATIO)
    ap.add_argument("--gold-ratio", type=float, default=X.GOLD_NEG_RATIO)
    ap.add_argument("--seed", type=int, default=X.DEFAULT_SEED)
    ap.add_argument("--source-dir", type=Path, default=None, help="default: <manifest>/../../hinted_rollouts")
    ap.add_argument("--items-dir", type=Path, default=None, help="default: <manifest dir>/items")
    a = ap.parse_args(argv)
    source_dir = a.source_dir or a.manifest.parent.parent / "hinted_rollouts"
    items_dir = a.items_dir or a.manifest.parent / "items"
    build(a.manifest, a.out, a.models, ratio=a.ratio, gold_ratio=a.gold_ratio, seed=a.seed,
          source_dir=source_dir, items_dir=items_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
