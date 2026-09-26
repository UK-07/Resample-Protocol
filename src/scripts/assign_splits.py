#!/usr/bin/env python3
"""The split stage: populate ``split`` on the per-rollout manifest(s), once.

GPU-free. Assigns every ``question_id`` of ``rollout_manifest.parquet`` to
train / val / test (:mod:`src.lib.splits`: seeded, stratified on hint_style x case,
one split per question) in place, records the assignment in the ``.meta.json``
sidecar, and propagates it by question to the re-sampling manifest when present.
Refuses an already split manifest: ``--extend`` assigns only the questions without
a split, ``--force`` re-assigns everything (never after activations were extracted).

    uv run python -m src.scripts.assign_splits --seed 42
    uv run python -m src.scripts.assign_splits --dry-run      # report only, writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.lib.paths import resolve_data_path
from src.lib.rollout_manifest import (
    DEFAULT_MANIFEST_PATH,
    meta_path,
    read_manifest,
    read_manifest_meta,
    write_manifest,
)
from src.lib.splits import (
    SPLIT_FRACTIONS,
    assert_assigned,
    assign_splits,
    propagate_splits,
    split_meta,
    split_report,
)

DEFAULT_RESAMPLE_MANIFEST = "${DATA_ROOT}/resample/resample_manifest.parquet"


def _fractions(text: str | None) -> dict[str, float]:
    if not text:
        return dict(SPLIT_FRACTIONS)
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise SystemExit("--fractions takes three comma-separated numbers: train,val,test")
    return dict(zip(SPLIT_FRACTIONS, parts))


def _report_lines(report: pd.DataFrame) -> str:
    cols = ["hint_style", "case", "n_questions", "frac_q_train", "frac_q_val", "frac_q_test", "n_rows"]
    return report[[c for c in cols if c in report.columns]].to_string(index=False, float_format="{:.3f}".format)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--resample-manifest", default=DEFAULT_RESAMPLE_MANIFEST,
                        help="the re-sampling manifest to propagate the split to (skipped when absent)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", default=None, help="train,val,test (default 0.70,0.15,0.15)")
    parser.add_argument("--force", action="store_true", help="re-assign an already split manifest")
    parser.add_argument("--extend", action="store_true",
                        help="keep the stored assignment and assign only the questions without a split "
                             "(rows added by a manifest rebuild since the split stage ran)")
    parser.add_argument("--dry-run", action="store_true", help="print the report without writing")
    args = parser.parse_args(argv)

    fractions = _fractions(args.fractions)
    path = resolve_data_path(args.manifest)
    manifest = read_manifest(path)
    meta = read_manifest_meta(path)
    if args.force and args.extend:
        parser.error("--force and --extend are exclusive")
    stored = meta.get("split") or {}
    if manifest["split"].notna().any() and not (args.force or args.extend):
        print(f"{path}: split is already assigned on {int(manifest['split'].notna().sum())} rows — "
              "refusing to re-assign (--extend assigns only the unassigned questions; --force re-assigns "
              "everything, never after activations were extracted)", file=sys.stderr)
        return 1
    if stored and manifest["split"].isna().all() and not args.force:
        print(f"{path}: the sidecar records a split assignment (seed {stored.get('seed')}) but the column is "
              "empty — the manifest was rebuilt without carrying it over; restore it (or --force to re-assign, "
              "only if no activation was extracted under it)", file=sys.stderr)
        return 1
    if args.extend and stored.get("seed") is not None and int(stored["seed"]) != args.seed:
        print(f"{path}: --extend with seed {args.seed}, but the stored assignment used seed {stored['seed']}",
              file=sys.stderr)
        return 1
    n_before = int(manifest["split"].notna().sum())
    assigned = assign_splits(manifest, args.seed, fractions=fractions, force=args.force, extend=args.extend)
    assert_assigned(assigned)
    report = split_report(assigned)
    block = split_meta(assigned, args.seed, fractions=fractions)
    if args.extend:
        block["extended"] = {"n_rows_before": n_before, "n_rows_added": int(len(assigned) - n_before),
                             "previous": {k: v for k, v in stored.items() if k != "report"}}
    print(f"{path}: {block['n_questions']} questions → {block['questions_per_split']} (seed {args.seed}"
          + (f"; {block['extended']['n_rows_added']} rows newly assigned)" if args.extend else ")"))
    print(_report_lines(report))

    resample_path = resolve_data_path(args.resample_manifest)
    resample = None
    if resample_path.exists():
        resample = pd.read_parquet(resample_path)
        resample = propagate_splits(resample, assigned)
        assert_assigned(resample)
        r_meta = split_meta(resample, args.seed, fractions=fractions)
        print(f"{resample_path}: {r_meta['n_questions']} questions → {r_meta['questions_per_split']} (propagated)")
    else:
        print(f"{resample_path}: not found — no re-sampling manifest to propagate to")

    if args.dry_run:
        print("dry run — nothing written")
        return 0
    write_manifest(assigned, path, {**{k: v for k, v in meta.items()
                                      if k not in ("schema_version", "written_utc", "n_rows", "columns")},
                                   "split": {**block, "report": report.to_dict(orient="records")}})
    print(f"wrote {path} (+ {meta_path(path).name})")
    if resample is not None:
        resample.to_parquet(resample_path, index=False)
        r_meta_path = Path(meta_path(resample_path))
        sidecar = json.loads(r_meta_path.read_text()) if r_meta_path.exists() else {}
        sidecar["split"] = {**r_meta, "propagated_from": str(path)}
        r_meta_path.write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"wrote {resample_path} (+ {r_meta_path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
