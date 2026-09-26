"""Benchmark table — data. One cells CSV per key set: every count, rate and Wilson CI per cell.

All logic lives in ``common.rates_common`` (SSP stitching via ``common.ssp_common``, dataset_B via
``src.lib.selection``); both modules' sha256 are recorded in gather_meta.json.

Run:
    python -m src.scripts.visualizations.table2_benchmark.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/table2_benchmark/data/): cells_model_dataset_style.csv,
cells_model_style.csv, cells_model_case.csv, ssp_rows.csv.gz, rsp_rows.csv.gz, join_checks.csv,
robust_used_rerolls_not_in_dataset_B.csv, exclusions.md, gather_meta.json. ``--only`` (runs whose
<model>_<run> stem contains it; debug) never touches the cache and, without --out-dir, writes under
<plot dir>/_debug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import rates_common as rc

PLOT = "table2_benchmark"
KEYSETS = {"cells_model_dataset_style": ["subject_model", "dataset", "hint_style", "case"],
           "cells_model_style": ["subject_model", "hint_style", "case"],
           "cells_model_case": ["subject_model", "case"]}
WRITE_ROWS = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args(argv)
    cueball = P.cueball_dir(a.cueball_dir)
    if a.out_dir is None:
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if a.only else P.plot_dir(PLOT, cueball)
        if a.only:
            print(f"--only without --out-dir: writing to the debug dir {out_dir}")
    else:
        out_dir = Path(a.out_dir)
    if a.no_cache or a.only:
        cache: str | Path | None = None
    else:
        cache = Path(a.cache_dir) if a.cache_dir else P.cache_dir(cueball)
    rc.gather_common(out_dir / "data", a.only, str(Path(__file__).resolve()), KEYSETS, write_rows=WRITE_ROWS,
                     cueball=cueball, cache_dir=cache, refresh=a.refresh_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
