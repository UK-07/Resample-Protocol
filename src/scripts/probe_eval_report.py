#!/usr/bin/env python3
"""Render a probe's predictions as the three-way (pooled / positive / negative) report.

GPU-free. Joins a predictions CSV (``rollout_id``, ``p_unfaithful``, optionally
``predicted_label``) to the manifest on ``rollout_id`` and writes
``<stem>_report.json`` + ``<stem>_report.md`` beside it (or under ``--output-dir``).
``--compare`` scores a second predictions file on the same rows and adds the
ΔAUROC on the primary view.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.lib.paths import resolve_data_path
from src.lib.report import (
    DEFAULT_LABEL_COL,
    build_report,
    delta_auroc,
    join_predictions,
    load_predictions,
    render_markdown,
    three_way_table,
)
from src.lib.rollout_manifest import DEFAULT_MANIFEST_PATH


def _path(arg: str) -> Path:
    """``${DATA_ROOT}``-style arguments go through the data-tree resolver; anything else is a plain path."""
    return resolve_data_path(arg) if "$" in arg else Path(arg).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", required=True, help="CSV with rollout_id, p_unfaithful[, predicted_label]")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH,
                        help="the manifest the rollout_ids live in (the re-sampling manifest for Datasets A/B)")
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--name", default=None, help="report title (default: the predictions file stem)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compare", default=None,
                        help="a second predictions CSV over the same rows: reports ΔAUROC (first − second) on the primary view")
    args = parser.parse_args(argv)

    pred_path = _path(args.predictions)
    manifest_path = _path(args.manifest)
    manifest = pd.read_parquet(manifest_path)
    pred = load_predictions(pred_path)
    name = args.name or pred_path.stem
    report = build_report(pred, manifest, label_col=args.label_col, name=name)
    report["predictions"] = str(pred_path)
    report["manifest"] = str(manifest_path)
    if args.compare:
        other = load_predictions(_path(args.compare))
        joined_a = join_predictions(pred, manifest, label_col=args.label_col)
        joined_b = join_predictions(other, manifest, label_col=args.label_col)
        report["ablation"] = {**delta_auroc(joined_a, joined_b), "a": str(pred_path), "b": args.compare}

    out_dir = _path(args.output_dir) if args.output_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pred_path).stem
    (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    text = render_markdown(report)
    if "ablation" in report:
        ab = report["ablation"]
        text += (f"\n## Ablation (ΔAUROC on `{ab['view']}`, n = {ab['n']})\n\n"
                 f"a = `{ab['a']}`: {ab['auroc_a']:.3f}; b = `{ab['b']}`: {ab['auroc_b']:.3f}; "
                 f"Δ = {ab['delta_auroc']:+.3f}\n")
    (out_dir / f"{stem}_report.md").write_text(text)
    print(three_way_table(report["overall"]).to_string(float_format="{:.3f}".format))
    print(f"\nwrote {out_dir / f'{stem}_report.json'} and {out_dir / f'{stem}_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
