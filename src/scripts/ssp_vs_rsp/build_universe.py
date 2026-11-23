#!/usr/bin/env python3
"""Build the SSP-vs-RSP row universe per model (originals only) and the activation collector's dataset.

Per model: every unfaithful switch original plus, per (split, style, dataset, case) stratum, at most
NEG_RATIO × as many faithful ones (hash order), with the texts joined from the judged hinted-rollout CSVs.
``in_gold`` marks the test-split rows whose pair gets the 12 fresh gold re-rolls: every unfaithful test row
and GOLD_NEG_RATIO × as many faithful ones.

Writes ``universe/<model>.parquet`` (the rows + in_gold + texts) and, for the collector,
``probe_datasets/<model>_ssp_rsp_universe.parquet`` (+ .meta.json, provenances [hinted_once]).

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.build_universe \
        --manifest /workspace/data/ursa/resample/resample_manifest.parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.lib import probe_datasets as pdx
from src.lib import selection
from src.lib import ssp_vs_rsp as X


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    a = ap.parse_args(argv)
    frame = selection.select_rows("ssp_all", pdx.load_dataset_manifest(a.manifest)[0])
    for folder in ("universe", "probe_datasets"):
        X.path(folder).mkdir(parents=True, exist_ok=True)
    for model in X.MODELS:
        df = frame[frame["subject_model"].eq(model)]
        uni = df[X.sample_universe(df)].copy()
        test = uni[uni["split"].eq("test")]
        uni["in_gold"] = uni.index.isin(test.index[X.sample_universe(test, ratio=X.GOLD_NEG_RATIO, salt="gold")])
        uni = uni.sort_values("rollout_id", kind="stable").reset_index(drop=True)
        print(f"[{model}] universe {len(uni)} rows of {len(df)} ({int(uni[X.LABEL].eq(0).sum())} unfaithful); "
              f"gold pairs {int(uni['in_gold'].sum())}")
        joined = pdx.join_texts(uni.drop(columns=["sample_seed"]), source_dir=a.manifest.parent.parent / "hinted_rollouts",
                                items_dir=a.manifest.parent / "items", log=print)
        joined["sample_seed"] = uni["sample_seed"].to_numpy()
        joined.to_parquet(X.path(f"universe/{model}.parquet"), index=False)

        ds = joined.assign(label=joined[X.LABEL].astype("Int8"), balance_kept=True)[list(pdx.DATASET_COLUMNS)]
        ds_path = X.path(f"probe_datasets/{model}_ssp_rsp_universe.parquet")
        ds.to_parquet(ds_path, engine="pyarrow", compression="zstd", index=False)
        meta = {"name": ds_path.stem, "provenances": [selection.HINTED_ONCE], "n_rows": len(ds),
                "fingerprint": pdx.dataset_fingerprint(ds),
                "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        pdx.dataset_sidecars(ds_path)[0].write_text(json.dumps(meta, indent=2) + "\n")
        pdx.read_dataset(ds_path)   # the collector's reader accepts it
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
