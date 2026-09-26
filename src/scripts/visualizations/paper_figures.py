"""Build paper figures from a released data tree or two prepared numeric tables."""
from __future__ import annotations

import argparse
import json

from src.lib.paths import resolve_data_path
from src.lib.paper_figures.workflow import FAMILIES, build, worker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cueball-dir", help="Released data tree; paths may use a quoted ${DATA_ROOT} prefix")
    inputs.add_argument("--data", help="Directory containing pairs_master.parquet and rerolls_long.parquet")
    parser.add_argument("--out-root", required=True, help="Fresh output directory outside the input data tree")
    parser.add_argument("--dry-run", action="store_true", help="Check required inputs and print the steps without writing files")
    parser.add_argument("--_phase", choices=("prepare", "compute", *FAMILIES), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    source = resolve_data_path(args.cueball_dir or args.data)
    output = resolve_data_path(args.out_root)
    if args._phase:
        if args.dry_run:
            parser.error("--dry-run cannot be combined with an internal worker phase")
        if (args._phase == "prepare") != bool(args.cueball_dir):
            parser.error("The prepare phase requires --cueball-dir; other phases require --data")
        worker(args._phase, source, output)
    else:
        print(json.dumps(build(source, output, from_tree=bool(args.cueball_dir), dry_run=args.dry_run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
