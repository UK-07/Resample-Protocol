"""Reproduce the revised paper figures from the verified numeric data bundle.

Run --dry-run first; every real run writes to a fresh output directory. This
command uses no GPU, model credentials, paper checkout, or code from the bundle.
"""
from __future__ import annotations
import argparse
import json

from src.lib.paths import resolve_data_path
from src.lib.paper_revision.workflow import reproduce, worker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Absolute or ${DATA_ROOT} path to the released .tar.gz or extracted revision_analysis directory")
    parser.add_argument("--out-root", required=True, help="Fresh absolute or ${DATA_ROOT} output directory, outside the input bundle")
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and print the plan without writing files")
    parser.add_argument("--_phase", choices=("recompute", "survival", "rates", "alpha_roles", "crosscheck", "dose", "judge_kappa"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    bundle, output = resolve_data_path(args.bundle), resolve_data_path(args.out_root)
    if args._phase and args.dry_run:
        parser.error("--dry-run cannot be combined with an internal worker phase")
    if args._phase:
        worker(args._phase, bundle, output)
    else:
        report = reproduce(bundle, output, dry_run=args.dry_run)
        print(json.dumps(report if args.dry_run else {k: report[k] for k in ("status", "output", "pdf_count", "latex_table_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
