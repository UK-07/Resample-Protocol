#!/usr/bin/env python3
"""Build one probe dataset parquet from a spec YAML (GPU-free, no API key).

Selects the spec predicate's rows from the re-sampling manifest, applies the
balance policy, derives the leave-one-hint-out fold report, joins the text
columns from the per-seed re-roll CSVs (``<manifest dir>/rollouts/<source_csv>``;
``choices`` / ``option_letters`` from the baseline CSV named by the recipe
sidecar when the CSV lacks them) and writes ``<output_dir>/<name>.parquet`` +
``<name>.meta.json`` + ``<name>_report.json`` atomically.

    uv run python -m src.scripts.build_probe_dataset --config configs/probe_datasets/<spec>.yaml [--dry-run] [--force]
        [--manifest <parquet>] [--chunk-rows N]

``--dry-run`` prints the fold report, the balance summary and the fingerprint
without joining texts or writing anything. Exit 1 with a message on any
validation error.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.lib.config import load_config
from src.lib.paths import resolve_data_path
from src.lib.probe_datasets import (
    DEFAULT_CHUNK_ROWS,
    FOLD_ROLES,
    build_dataset,
    dataset_paths,
    load_dataset_manifest,
    parse_spec,
    plan_dataset,
    spec_with,
    write_dataset,
)


def fold_table(fold_report: dict) -> str:
    """One line per (held-out style, role): rows, questions, label counts."""
    records = []
    for style, entry in fold_report.items():
        for role in FOLD_ROLES:
            r = entry["roles"][role]
            records.append({"held_out": style, "role": role, "rows": r["n_rows"], "questions": r["n_questions"],
                            "label_0": r["n_label_0"], "label_1": r["n_label_1"]})
    return pd.DataFrame(records).to_string(index=False)


def fold_flags(fold_report: dict) -> str:
    """One line per held-out style: the question checks and whether test_ood holds both labels."""
    records = [{"held_out": style, "test_ood_rows": entry["roles"]["test_ood"]["n_rows"],
                "ood_has_both_labels": entry["ood_has_both_labels"],
                "question_consistency": entry["question_consistency"],
                "n_ood_only_questions": entry["n_ood_only_questions"], "n_dropped": entry["n_dropped"]}
               for style, entry in fold_report.items()]
    return pd.DataFrame(records).to_string(index=False)


def balance_summary(report: dict) -> str:
    bal = report["balance"]
    before, after = report["labels_selected"], report["labels_after_balance"]
    if bal["method"] == "none":
        return (f"balance: none — {report['n_after_balance']} rows kept "
                f"(label 0: {after['0']}, label 1: {after['1']})")
    return (f"balance: ratio r={bal['r']} over {bal['n_groups']} groups {tuple(bal['groups'])} — "
            f"{bal['n_before']} rows (0: {before['0']}, 1: {before['1']}) → {bal['n_after']} "
            f"(0: {after['0']}, 1: {after['1']}), {bal['n_dropped']} dropped, "
            f"{bal['n_groups_missing_class']} one-class group(s) kept whole")


def print_plan(spec, report: dict) -> None:
    print(f"dataset {spec.name}: predicate {spec.predicate} (label column {spec.label_col}), cases {spec.cases}")
    print(f"manifest {report['manifest']['path']} ({report['manifest']['n_rows']} rows, written "
          f"{report['manifest']['written_utc']})")
    print(f"selected {report['n_selected']} rows (label 0: {report['labels_selected']['0']}, "
          f"label 1: {report['labels_selected']['1']}); {len(report['hint_styles'])} styles: "
          f"{', '.join(report['hint_styles'])}")
    print(balance_summary(report))
    print()
    print("fold report (leave-one-hint-out):")
    print(fold_table(report["fold_report"]))
    print()
    print(fold_flags(report["fold_report"]))
    lacking = [s for s, e in report["fold_report"].items() if e["roles"]["test_ood"]["n_label_0"] == 0]
    if lacking:
        print(f"WARNING: test_ood without a label-0 row (no OOD AUROC) for: {', '.join(lacking)}")
    print()
    print(f"fingerprint {report['fingerprint']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the dataset spec YAML (probe_datasets.parse_spec)")
    parser.add_argument("--dry-run", action="store_true", help="select, balance and report; no text join, no write")
    parser.add_argument("--force", action="store_true", help="overwrite an existing parquet")
    parser.add_argument("--manifest", default=None, help="override the spec's manifest path")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help=f"CSV rows per streamed chunk of the text join (default {DEFAULT_CHUNK_ROWS}; "
                             "lower it on small-memory machines)")
    args = parser.parse_args(argv)
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be ≥ 1")

    try:
        spec = parse_spec(load_config(args.config))
        if args.manifest:
            spec = spec_with(spec, manifest=args.manifest)
        manifest_path = resolve_data_path(spec.manifest)
        parquet, _, _ = dataset_paths(spec)
        if not args.dry_run and parquet.exists() and not args.force:
            print(f"{parquet} exists — pass --force to overwrite it", file=sys.stderr)
            return 1
        if args.dry_run:
            manifest, meta = load_dataset_manifest(manifest_path)
            _, report = plan_dataset(spec, manifest, meta, manifest_path=manifest_path)
            print_plan(spec, report)
            print("dry run — nothing written")
            return 0
        rows, report = build_dataset(spec, log=print, chunk_rows=args.chunk_rows)
        print_plan(spec, report)
        written = write_dataset(rows, report, spec, force=args.force)
        print(f"wrote {written} ({len(rows)} rows) + {written.with_suffix('.meta.json').name} + "
              f"{written.stem}_report.json")
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
