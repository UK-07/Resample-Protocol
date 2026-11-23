#!/usr/bin/env python3
"""Write the gold re-roll items: the existing re-sampling items of every ``in_gold`` universe pair.

The gold test set relabels the test-split pairs from GOLD_SEEDS (12 fresh sampling seeds, independent of
the k=4 re-rolls of seeds 43-46). The items are the rows of the original re-sampling items files (same
rollout_id, same stored hinted_prompt), so ``resample_hinted_rollouts`` rebuilds the byte-identical cued
prompt. One items file per source, with the source's sidecar.

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.make_gold_items \
        --items-dir /workspace/data/ursa/resample/items
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.lib import ssp_vs_rsp as X
from src.lib.resample import items_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    out = X.path("gold/items")
    out.mkdir(parents=True, exist_ok=True)
    for model in X.MODELS:
        uni = X.load_universe(model, "source_csv")
        gold = uni[uni["in_gold"]]
        found = 0
        for source_csv, pairs in gold.groupby("source_csv"):
            items_csv = items_path(a.items_dir, source_csv)
            items = pd.read_csv(items_csv, dtype={"rollout_id": str}, keep_default_na=False)
            items = items[items["rollout_id"].isin(pairs["rollout_id"])]
            items.to_csv(out / items_csv.name, index=False)
            shutil.copy(items_csv.with_suffix(".meta.json"), out)
            found += len(items)
        print(f"[{model}] {found} gold items of {len(gold)} gold pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
