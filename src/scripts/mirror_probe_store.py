#!/usr/bin/env python3
"""Mirror an activation store folder to a local disk for probe training.

Copies the store's ``manifest.csv``, ``manifest.meta.json``, every ``points_*``
file and the ``seq_*`` files of ``--layers`` (a subset of the store's
``seq_layers``) straight into ``<dest>/<store folder>``, never through the HF
cache; the trainer reads the copy with ``activations.backend: local``. The
source is the collector config's ``storage`` + ``store_folder``. ``--dry-run``
prints the plan and the disk check; a plan larger than the free disk is refused
in both modes. Reruns are idempotent (only missing or wrong-size files are
downloaded) and the meta's ``mirrored_layers`` / ``mirrored_points_layers``
record what the copy holds.
"""

from __future__ import annotations

import argparse
import shutil
from typing import Optional

from src.lib.activation_store import (
    DEFAULT_MIRROR_WORKERS,
    GB,
    MirrorPlan,
    make_backend,
    mirror_store,
    plan_mirror,
)
from src.lib.config import load_config
from src.lib.paths import resolve_data_path
from src.scripts.collect_probe_activations import dataset_folder, parse_config


def parse_layers(value: Optional[str]) -> Optional[list[int]]:
    """``"31,39"`` → ``[31, 39]``; ``None`` / blank → ``None`` (no seq file — a points-only mirror)."""
    if value is None or not str(value).strip():
        return None
    out = []
    for piece in str(value).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if not piece.isdigit():
            raise ValueError(f"--layers must be comma-separated non-negative integers, got {value!r}")
        out.append(int(piece))
    if not out:
        return None
    if len(set(out)) != len(out):
        raise ValueError(f"--layers has duplicates: {out}")
    return sorted(out)


def plan_text(plan: MirrorPlan) -> list[str]:
    """The printed plan: per layer and kind the files / GB, the totals and the disk check."""
    lines = [f"Plan — mirror {plan.folder!r} → {plan.dest / plan.folder}",
             f"  seq layers: {plan.layers or 'none (points only)'}; points: "
             f"{'layers ' + str(list(plan.bytes_by_layer('points'))) if plan.include_points else 'excluded'}",
             f"  source stores layers {plan.meta.get('layers')} (points), seq files for "
             f"{plan.meta.get('seq_layers', plan.meta.get('layers'))}"
             f"{' (seq_layers)' if plan.meta.get('seq_layers') is not None else ' (every layer; no seq_layers in the meta)'}, "
             f"hidden {plan.meta.get('hidden_size')}, "
             f"{sum(1 for _ in plan.manifest_text.splitlines()) - 1} manifest rows"]
    for kind in ("seq", "points"):
        by_layer = plan.bytes_by_layer(kind)
        for layer, n_bytes in by_layer.items():
            files = [f for f in plan.files if f.kind == kind and f.layer == layer]
            todo = [f for f in files if f.action == "download"]
            lines.append(f"  {kind:6s} layer {layer:3d}: {len(files):4d} file(s), {n_bytes / GB:9.2f} GB "
                         f"({len(todo)} to download, {sum(f.size for f in todo) / GB:.2f} GB)")
    lines.append(f"  total: {len(plan.files)} file(s), {plan.bytes_total / GB:.2f} GB — to download "
                 f"{len(plan.to_download)} file(s), {plan.bytes_to_download / GB:.2f} GB; already present "
                 f"{len(plan.skipped)} file(s), {(plan.bytes_total - plan.bytes_to_download) / GB:.2f} GB")
    free = plan.free_bytes
    lines.append(f"  free disk at {plan.dest}: {'unknown' if free is None else f'{free / GB:.2f} GB'}")
    prev = plan.previous_meta
    if prev:
        lines.append(f"  existing mirror: seq layers {prev.get('mirrored_layers')}, points layers "
                     f"{prev.get('mirrored_points_layers')}")
    return lines


def main(argv: Optional[list[str]] = None, *, api=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="The collector's YAML config (its `storage` + `store_folder` name the source).")
    parser.add_argument("--dest", required=True,
                        help="Local root directory of the mirror (the copy lives at <dest>/<store folder>); "
                             "a local disk (not a network volume), resolved through resolve_data_path.")
    parser.add_argument("--layers", default=None, metavar="L[,L...]",
                        help="Layers whose seq_* files to mirror (comma-separated). Omitted = none (points only).")
    parser.add_argument("--no-points", action="store_true", help="Do not mirror the points_* files.")
    parser.add_argument("--workers", type=int, default=DEFAULT_MIRROR_WORKERS,
                        help=f"Concurrent downloads (default {DEFAULT_MIRROR_WORKERS}).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the disk check; download nothing.")
    args = parser.parse_args(argv)

    layers = parse_layers(args.layers)
    if layers is None and args.no_points:
        raise ValueError("nothing to mirror: give --layers, or drop --no-points (the points files alone serve the linear probes).")
    if args.workers < 1:
        raise ValueError(f"--workers must be ≥ 1, got {args.workers}")
    cc = parse_config(load_config(args.config))
    backend = make_backend(cc.storage, api=api)
    folder = dataset_folder(cc.dataset, cc.store_folder)
    dest = resolve_data_path(args.dest)
    print(f"Source: {backend.describe()}/{folder} ({'store_folder' if cc.store_folder else 'the dataset name'}); "
          f"destination {dest}")

    plan = plan_mirror(backend, folder, dest, layers=layers, include_points=not args.no_points)
    for line in plan_text(plan):
        print(line)
    if plan.free_bytes is not None and plan.bytes_to_download > plan.free_bytes:
        print(f"[refused] the plan needs {plan.bytes_to_download / GB:.2f} GB but only "
              f"{plan.free_bytes / GB:.2f} GB are free at {dest} — mirror fewer layers or free the disk.")
        return 1
    if args.dry_run:
        print("Dry run — nothing downloaded.")
        return 0
    report = mirror_store(backend, folder, dest, plan=plan, workers=args.workers, log=print)
    print(f"Done: {report['n_downloaded']} downloaded ({report['bytes_downloaded'] / GB:.2f} GB), "
          f"{report['n_skipped']} skipped, {report['seconds']:.0f} s; mirrored seq layers {report['mirrored_layers']}, "
          f"points layers {report['mirrored_points_layers']}"
          + (f"; DROPPED {report['dropped']} (re-run with those layers)" if report["dropped"] else ""))
    usage = shutil.disk_usage(dest)
    print(f"Free disk at {dest} now: {usage.free / GB:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
