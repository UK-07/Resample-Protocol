#!/usr/bin/env python3
"""Write the gold re-roll items: the existing re-sampling items of every ``in_gold`` universe pair.

The gold test set relabels the test-split pairs from GOLD_SEEDS (12 fresh sampling seeds, independent of
the k=4 re-rolls of seeds 43-46). The items are the rows of the original re-sampling items files (same
rollout_id, same stored hinted_prompt), so ``resample_hinted_rollouts`` rebuilds the byte-identical cued
prompt. One items file per source, sidecar copied with ``sample_seeds`` set to GOLD_SEEDS.

    python -m src.scripts.ssp_vs_rsp.make_gold_items --universe /workspace/data/ssp_vs_rsp/universe \
        --items-dir /workspace/data/ursa/resample/items --out /workspace/data/ssp_vs_rsp/gold/items
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.lib import ssp_vs_rsp as X
from src.lib.resample import items_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", required=True, type=Path)
    ap.add_argument("--items-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=list(X.MODELS))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for model in a.models:
        uni = pd.read_parquet(a.universe / f"{model}.parquet", columns=["rollout_id", "source_csv", "in_gold"])
        gold = uni[uni["in_gold"].astype(bool)]
        found = 0
        for source_csv, sub in gold.groupby("source_csv"):
            items_csv = Path(items_path(a.items_dir, str(source_csv)))
            items = pd.read_csv(items_csv, dtype={"rollout_id": str}, keep_default_na=False)
            hit = items[items["rollout_id"].isin(set(sub["rollout_id"].astype(str)))]
            missing = sorted(set(sub["rollout_id"].astype(str)) - set(hit["rollout_id"]))
            if missing:
                raise ValueError(f"{items_csv.name}: {len(missing)} gold pair(s) have no item, e.g. {missing[:3]}")
            if hit["rollout_id"].duplicated().any():
                raise ValueError(f"{items_csv.name}: duplicate item rows for a gold pair")
            meta = json.loads(items_csv.with_suffix(".meta.json").read_text())
            meta.update({"n_items": int(len(hit)), "n_used": int(hit["role"].eq("used_candidate").sum()),
                         "n_control": int(hit["role"].eq("control").sum()),
                         "sample_seeds": list(X.GOLD_SEEDS),
                         "ssp_vs_rsp_gold": {"source_items": str(items_csv),
                                             "note": "test-split pairs of the SSP-vs-RSP universe (in_gold)"}})
            hit.to_csv(a.out / items_csv.name, index=False)
            (a.out / items_csv.name).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            found += len(hit)
            print(f"[{model}] {items_csv.name}: {len(hit)} gold items")
        print(f"[{model}] {found} gold items of {len(gold)} gold pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
