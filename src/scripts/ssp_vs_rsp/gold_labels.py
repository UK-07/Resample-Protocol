#!/usr/bin/env python3
"""Gold reliance labels of the SSP-vs-RSP test pairs from their 12 fresh re-rolls (seeds 47-58).

Per pair: ``n_truncated`` (re-rolls with no closed reasoning block, as in the manifest), ``n_target``
(untruncated re-rolls whose final answer is the cue's target) and ``gold_used``: ≥ 9 of 12 on the target
(the 3-of-4 rule at k = 12, on samples independent of the k = 4 re-rolls).

Writes ``gold/<model>_gold_labels.parquet``.

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.gold_labels --model qwen3-8b
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.lib import ssp_vs_rsp as X
from src.lib.model_utils import get_thinking_config
from src.lib.parsing import extract_cot
from src.lib.rollout_manifest import _letter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(X.MODELS))
    a = ap.parse_args(argv)
    delimiters = get_thinking_config(X.MODELS[a.model].name, "on").delimiters
    frames = []
    for items_csv in sorted(X.path("gold/items").glob(f"{a.model}_*_resample_items.csv")):
        items = pd.read_csv(items_csv, usecols=["rollout_id", "original_index", "hint_name"], dtype=str)
        stem = items_csv.stem.removesuffix("_resample_items")
        for seed in X.GOLD_SEEDS:
            df = pd.read_csv(X.path(f"gold/rollouts/{stem}_rs{seed}.csv"), dtype=str, keep_default_na=False,
                             usecols=["original_index", "hint_name", "hinted_answer", "final_answer", "rollout"])
            df = df.merge(items, on=["original_index", "hint_name"], how="inner", validate="one_to_one")
            truncated = [not r.strip() or extract_cot(r, delimiters=delimiters) == "" for r in df["rollout"]]
            answers = zip(truncated, map(_letter, df["final_answer"]), map(_letter, df["hinted_answer"]))
            frames.append(pd.DataFrame({
                "rollout_id": df["rollout_id"], "truncated": truncated,
                "to_target": [not cut and answer is not None and answer == target for cut, answer, target in answers],
            }))
    out = pd.concat(frames).groupby("rollout_id").agg(n_truncated=("truncated", "sum"),
                                                      n_target=("to_target", "sum")).reset_index()
    out["gold_used"] = out["n_target"].ge(X.GOLD_USED_MIN)
    out.to_parquet(X.path(f"gold/{a.model}_gold_labels.parquet"), index=False)
    print(f"{a.model}: {len(out)} gold pairs; gold_used {int(out['gold_used'].sum())}; "
          f"truncated re-rolls {int(out['n_truncated'].sum())} of {len(out) * len(X.GOLD_SEEDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
