#!/usr/bin/env python3
"""Step 1 of the re-sampling design: the noise model of the hinted-once rollouts.

Reads the per-rollout manifest (smoke runs and dropped hint styles removed by
:func:`src.lib.resample.filter_manifest`) and writes under ``--output-dir``
(default ``${DATA_ROOT}/resample/``): ``noise_model_cells.csv`` (per cell:
counts, ``noise_flip_estimate`` = (changed − to_hint) / (n_options − 2),
observed vs chance ``to_hint`` rates, ``prior_target_rate``),
``noise_model_by_stability.csv`` (the same per baseline-stability bin),
``noise_model_stability_summary.csv`` (pooled over runs) and ``noise_model.md``.
The option count per dataset comes from :data:`N_OPTIONS_BY_DATASET` and is
checked against every source baseline CSV's ``choices`` width.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.lib.paths import resolve_data_path
from src.lib.resample import (
    N_OPTIONS_BY_DATASET,
    filter_manifest,
    noise_model,
    noise_model_by_stability,
    stability_summary,
)
from src.lib.rollout_manifest import DEFAULT_MANIFEST_PATH, read_manifest, read_manifest_meta

DEFAULT_OUTPUT_DIR = "${DATA_ROOT}/resample"

CELL_TABLE_COLUMNS = [
    "n_rollouts", "n_changed", "n_to_hint", "n_off_target", "n_options",
    "noise_flip_estimate", "to_hint_rate", "chance_to_hint_rate", "noise_share_of_to_hint",
    "to_hint_rate_excess", "prior_target_rate", "prior_to_hint_estimate",
    "n_unfaithful", "n_faithful", "unfaithful_rate", "n_noise_flagged",
]
STABILITY_TABLE_COLUMNS = [
    "n_rollouts", "n_changed", "n_to_hint", "n_off_target", "to_hint_rate",
    "off_target_frac_of_changed", "noise_share_of_to_hint", "prior_target_rate",
]


def check_option_widths(meta: dict, runs: set[tuple[str, str]]) -> list[str]:
    """Problems found comparing each source baseline's choice width with the registry."""
    problems = []
    for source in meta.get("sources", []):
        key = (source.get("subject_model"), source.get("run"))
        if key not in runs:
            continue
        expected = N_OPTIONS_BY_DATASET.get(source.get("dataset"))
        path = source.get("baseline_csv")
        if expected is None or not path or not Path(path).exists():
            problems.append(f"{key}: no registered width or baseline CSV ({source.get('dataset')}, {path})")
            continue
        widths = {len(json.loads(c)) for c in pd.read_csv(path, usecols=["choices"], dtype=str)["choices"]}
        if widths != {expected}:
            problems.append(f"{key}: baseline choice widths {sorted(widths)} != registered {expected}")
    return problems


def _fmt(df: pd.DataFrame, floats: int = 3) -> str:
    return df.to_markdown(floatfmt=f".{floats}f")


def render_markdown(cells: pd.DataFrame, stratified: pd.DataFrame, pooled: pd.DataFrame, n_rows: int) -> str:
    lines = [
        "# Noise model of the hinted-once rollouts",
        "",
        f"{n_rows} rollouts (smoke-test runs and the dropped hint styles removed). "
        "`noise_flip_estimate` = (changed − to_hint) / (n_options − 2): off-target flips spread over the "
        "options that are neither the baseline answer nor the target estimate the per-option spontaneous "
        "flip rate, so this many `to_hint` switches are noise-shaped. `to_hint_rate_excess` = "
        "(to_hint_rate − chance) / (1 − chance). `prior_target_rate` = no-hint votes for the target over "
        "the baseline samples — an independent chance estimate. Negative cases point at the correct "
        "option, where a spontaneous flip is likelier to land, so their estimate is a lower bound.",
        "",
        "## Pooled over runs, by baseline stability",
        "",
        "Rates by (hint style, case, stability bin): noise flips concentrate in the low-stability strata.",
        "",
        _fmt(pooled[STABILITY_TABLE_COLUMNS]),
        "",
        "## Per cell",
        "",
        _fmt(cells[CELL_TABLE_COLUMNS]),
        "",
        "## Per cell × stability bin",
        "",
        _fmt(stratified[STABILITY_TABLE_COLUMNS]),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", default=None, help="only runs whose subject_model or run tag contains this")
    parser.add_argument("--skip-width-check", action="store_true",
                        help="do not open the baseline CSVs to verify the option widths")
    args = parser.parse_args(argv)

    manifest = read_manifest(args.manifest)
    df = filter_manifest(manifest)
    if args.only:
        df = df[df["subject_model"].str.contains(args.only) | df["run"].str.contains(args.only)]
    if df.empty:
        print("no rows left after filtering", file=sys.stderr)
        return 1
    runs = set(zip(df["subject_model"].astype(str), df["run"].astype(str)))
    if not args.skip_width_check:
        problems = check_option_widths(read_manifest_meta(args.manifest), runs)
        if problems:
            print("option width check failed:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1

    cells = noise_model(df)
    stratified = noise_model_by_stability(df)
    pooled = stability_summary(df)

    out = resolve_data_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "noise_model_cells.csv")
    stratified.to_csv(out / "noise_model_by_stability.csv")
    pooled.to_csv(out / "noise_model_stability_summary.csv")
    (out / "noise_model.md").write_text(render_markdown(cells, stratified, pooled, len(df)))

    print(f"{len(df)} rollouts, {len(runs)} runs, {len(cells)} cells → {out}")
    print(pooled[STABILITY_TABLE_COLUMNS].to_string(float_format="{:.3f}".format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
