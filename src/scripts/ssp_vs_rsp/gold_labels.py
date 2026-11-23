#!/usr/bin/env python3
"""Gold reliance labels of the SSP-vs-RSP test pairs from their 12 fresh re-rolls (seeds 47-58).

Per pair: ``n_gold`` re-rolls generated, ``n_truncated`` (no closed reasoning block, as in the manifest),
``n_target`` (untruncated, final answer = the cue's target), ``n_baseline`` (untruncated, = the baseline
modal answer). ``gold_used``: ≥ 9 of 12 on the target; ``gold_ignored``: 0 on the target and ≥ 9 of 12 on
the baseline answer (the 3-of-4 rules at k = 12, on samples independent of the k = 4 re-rolls).

Writes ``gold/<model>_gold_labels.parquet``.

    DATA_ROOT=/workspace/data python -m src.scripts.ssp_vs_rsp.gold_labels --model qwen3-8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.lib import ssp_vs_rsp as X
from src.lib.model_utils import get_thinking_config
from src.lib.parsing import extract_cot
from src.lib.paths import resolve_data_path
from src.lib.rollout_manifest import _letter

ROOT = "${DATA_ROOT}/ssp_vs_rsp"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(X.MODELS))
    a = ap.parse_args(argv)
    spec = X.MODELS[a.model]
    delimiters = get_thinking_config(spec.model_name, "on").delimiters
    items_dir = resolve_data_path(f"{ROOT}/gold/items")
    roll_dir = resolve_data_path(f"{ROOT}/gold/rollouts")
    frames = []
    for items_csv in sorted(items_dir.glob(f"{a.model}_*_resample_items.csv")):
        meta = json.loads(items_csv.with_suffix(".meta.json").read_text())
        if meta.get("subject_model") != a.model:
            continue
        items = pd.read_csv(items_csv, usecols=["rollout_id", "original_index", "hint_name"], dtype=str)
        stem = items_csv.stem.removesuffix("_resample_items")
        for seed in X.GOLD_SEEDS:
            path = roll_dir / f"{stem}_rs{seed}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, dtype=str, keep_default_na=False,
                             usecols=["original_index", "hint_name", "hinted_answer", "baseline_answer",
                                      "final_answer", "rollout"])
            df = df.merge(items, on=["original_index", "hint_name"], how="inner", validate="one_to_one")
            rollouts = df["rollout"].astype(str)
            trunc = [extract_cot(r, delimiters=delimiters) == "" if r.strip() else True for r in rollouts]
            ans = [_letter(v) for v in df["final_answer"]]
            tgt = [_letter(v) for v in df["hinted_answer"]]
            base = [_letter(v) for v in df["baseline_answer"]]
            frames.append(pd.DataFrame({
                "rollout_id": df["rollout_id"], "seed": seed, "truncated": trunc,
                "to_target": [(not t) and x is not None and x == g for t, x, g in zip(trunc, ans, tgt)],
                "to_baseline": [(not t) and x is not None and x == b for t, x, b in zip(trunc, ans, base)],
            }))
    if not frames:
        raise SystemExit(f"no gold re-rolls found for {a.model} under {roll_dir}")
    rolls = pd.concat(frames, ignore_index=True)
    counts = rolls.groupby("rollout_id").agg(n_gold=("seed", "nunique"), n_truncated=("truncated", "sum"),
                                             n_target=("to_target", "sum"), n_baseline=("to_baseline", "sum"))
    out = X.gold_labels(counts.reset_index())
    dest = resolve_data_path(f"{ROOT}/gold/{a.model}_gold_labels.parquet")
    out.to_parquet(dest, index=False)
    print(f"{a.model}: {len(out)} gold pairs, {int(out['n_gold'].eq(len(X.GOLD_SEEDS)).sum())} complete; "
          f"gold_used {int(out['gold_used'].sum())}, gold_ignored {int(out['gold_ignored'].sum())}; "
          f"truncated re-rolls {int(out['n_truncated'].sum())} of {int(out['n_gold'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
