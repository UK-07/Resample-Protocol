"""Survival slice: positive vs negative case (manifest case) — data, per model (styles and datasets pooled).

Survival = P(robust_used | SSP-unfaithful), stitched by ``common.ssp_common`` (sha recorded in gather_meta.json).

Run:
    python -m src.scripts.visualizations.survival_slice_case.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/survival_slice_case/data/).
"""
import sys

from src.scripts.visualizations.common import ssp_common as S

PLOT = "survival_slice_case"


def build(df):
    t = S.survival_table(df, ["subject_model", "case"])
    t.insert(1, "slice", t["case"])
    return {"survival_by_model_slice.csv": t}, S.rows_for_audit(df), {"slice": "case", "levels": ["positive", "negative"]}


if __name__ == "__main__":
    sys.exit(S.gather_cli(__doc__, PLOT, build))
