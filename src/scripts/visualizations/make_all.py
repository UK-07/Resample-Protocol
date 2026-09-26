#!/usr/bin/env python3
"""Run every plot's ``gather`` then ``plot`` in a fixed order (GPU-free, no API key).

For the paper figures, ``--paper-data D --out-root O`` reads prepared numeric tables.
The released-tree plotting commands below remain available for the other figures.

Each step is a subprocess ``python -m src.scripts.visualizations.<plot>.<gather|plot>`` run from the
repository root with the interpreter's ``bin/`` on ``PATH``. A failed step is reported and the driver
moves on (a failed gather skips that plot's plot step); the exit code is the number of failed steps.

Run:
    python -m src.scripts.visualizations.make_all [--only a,b] [--gather-only | --plot-only]
        [--cueball-dir D] [--out-root D] [--cache-dir D] [--dry-run]

``--out-root`` (default ``<cueball>/plots``) holds one ``<plot>/`` folder per plot (``data/`` + figures);
the four plots that re-aggregate the alpha_vs_measured_noise tables are pointed at
``<out-root>/alpha_vs_measured_noise/data`` when ``--out-root`` is given, so a run never mixes trees.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from src.lib.paths import REPO_ROOT
from src.scripts.visualizations.common import paths as P

PACKAGE = "src.scripts.visualizations"

# Fixed order: the alpha follow-ups read alpha_vs_measured_noise's data/, everything else is independent.
PLOTS = (
    # survival (single-sample vs resampled protocol)
    "survival_vs_stability",
    "survival_by_style",
    "survival_by_dataset",
    "reliance_composition",
    "reroll_distribution",
    "survival_slice_case",
    "survival_slice_cue_family",
    "survival_slice_post_hoc",
    # alpha noise model
    "alpha_vs_measured_noise",
    "distractor_uniformity",
    "alpha_bias_vs_stability",
    "alpha_negative_case_check",
    "noise_crosscheck",
    # claims vs behaviour
    "role_reliance_heatmap",
    "dose_response",
    "ssp_vs_rsp_false_rejections",
    # rates under the two protocols
    "rates_dumbbell",
    "rates_rank_change",
    "susceptibility_vs_unfaithfulness",
    "table2_benchmark",
    # yield and cost
    "funnel",
    "yield_per_style",
)

ALPHA_PLOT = "alpha_vs_measured_noise"
ALPHA_CONSUMERS = ("distractor_uniformity", "alpha_bias_vs_stability", "alpha_negative_case_check",
                   "noise_crosscheck")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper-data", default=None, help="Build paper figures from a directory containing the two numeric tables; requires --out-root")
    ap.add_argument("--only", default=None, help="comma-separated plot names (a subset of PLOTS, kept in order)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--gather-only", action="store_true", help="run only the gather steps")
    mode.add_argument("--plot-only", action="store_true", help="run only the plot steps (tables must exist)")
    ap.add_argument("--cueball-dir", default=None, help=f"the paper tree (default {P.DEFAULT_CUEBALL_DIR})")
    ap.add_argument("--out-root", default=None, help="root of the per-plot output folders (default <cueball>/plots)")
    ap.add_argument("--cache-dir", default=None, help="the stitched-rows cache (default <cueball>/plots/_cache)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    return ap.parse_args(argv)


def select_plots(only: str | None) -> list[str]:
    if not only:
        return list(PLOTS)
    wanted = [s.strip() for s in only.split(",") if s.strip()]
    unknown = sorted(set(wanted) - set(PLOTS))
    if unknown:
        raise SystemExit(f"unknown plot(s) {unknown}; known: {', '.join(PLOTS)}")
    return [p for p in PLOTS if p in wanted]


def commands(plot: str, args: argparse.Namespace, out_root: Path | None) -> list[tuple[str, list[str]]]:
    """The (label, argv) steps of one plot under the driver's flags."""
    out_dir = (out_root / plot) if out_root is not None else None
    steps: list[tuple[str, list[str]]] = []
    if not args.plot_only:
        cmd = [sys.executable, "-m", f"{PACKAGE}.{plot}.gather"]
        if args.cueball_dir:
            cmd += ["--cueball-dir", args.cueball_dir]
        if out_dir is not None:
            cmd += ["--out-dir", str(out_dir)]
        if args.cache_dir:
            cmd += ["--cache-dir", args.cache_dir]
        if plot in ALPHA_CONSUMERS and out_root is not None:
            cmd += ["--alpha-data", str(out_root / ALPHA_PLOT / "data")]
        steps.append(("gather", cmd))
    if not args.gather_only:
        cmd = [sys.executable, "-m", f"{PACKAGE}.{plot}.plot"]
        if out_dir is not None:
            cmd += ["--out-dir", str(out_dir), "--data", str(out_dir / "data")]
        steps.append(("plot", cmd))
    return steps


def run_step(cmd: list[str], *, dry_run: bool) -> tuple[bool, float]:
    print(f"  $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return True, 0.0
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return proc.returncode == 0, time.time() - t0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.paper_data:
        if not args.out_root:
            raise SystemExit("--paper-data requires a fresh --out-root directory")
        if args.only or args.gather_only or args.plot_only or args.cueball_dir or args.cache_dir:
            raise SystemExit("The paper figure workflow cannot be combined with legacy gather/plot filters or tree paths")
        from src.scripts.visualizations.paper_figures import main as figures_main
        return figures_main(["--data", args.paper_data, "--out-root", args.out_root]
                             + (["--dry-run"] if args.dry_run else []))
    plots = select_plots(args.only)
    out_root = Path(args.out_root) if args.out_root else None
    results: list[tuple[str, str, str, float]] = []   # (plot, step, status, seconds)
    for plot in plots:
        print(f"== {plot}", flush=True)
        gather_failed = False
        for label, cmd in commands(plot, args, out_root):
            if label == "plot" and gather_failed:
                results.append((plot, label, "skipped (gather failed)", 0.0))
                continue
            ok, secs = run_step(cmd, dry_run=args.dry_run)
            status = "ok" if ok else "FAILED"
            if label == "gather" and not ok:
                gather_failed = True
            results.append((plot, label, status, secs))
            print(f"  -> {label}: {status} ({secs:.0f}s)", flush=True)
    width = max(len(p) for p, *_ in results) if results else 4
    print("\nsummary")
    for plot, label, status, secs in results:
        print(f"  {plot:<{width}}  {label:<6}  {status:<24} {secs:7.0f}s")
    n_failed = sum(status == "FAILED" for *_, status, _ in results)
    print(f"{len(results) - n_failed} of {len(results)} steps succeeded" + (f", {n_failed} failed" if n_failed else ""))
    return n_failed


if __name__ == "__main__":
    sys.exit(main())
