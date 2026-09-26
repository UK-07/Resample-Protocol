"""Reliance-label composition — data. Per manifest case (positive = main figures, negative = companions),
per model x style, two pools:
  (a) SSP-unfaithful rows (SSP flip AND binary label 0)
  (b) all SSP flips (regardless of the binary verdict; label-missing flips included)
Shares of robust_used / weak_used / mixed (0/4 re-rolls on target) over rows with a reliance label.

Stitched by ``common.ssp_common`` (sha recorded in gather_meta.json).

Run:
    python -m src.scripts.visualizations.reliance_composition.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/reliance_composition/data/).
"""
import sys

from src.scripts.visualizations.common import ssp_common as S

PLOT = "reliance_composition"
CASES = ["positive", "negative"]
POOLS = {"a_ssp_unfaithful": "ssp_unfaithful", "b_ssp_flips": "ssp_flip"}


def build(df):
    tables = {}
    for name, flag in POOLS.items():
        pool = df[flag].astype(bool)
        tables[f"composition_{name}_by_model_style.csv"] = S.composition_table(
            df, pool, ["subject_model", "case", "hint_style"])
        tables[f"composition_{name}_by_model.csv"] = S.composition_table(df, pool, ["subject_model", "case"])
    tables["exclusions_by_model_style.csv"] = S.exclusion_summary(df, ["subject_model", "case", "hint_style"])
    return tables, S.rows_for_audit(df), {"cases": CASES, "pools": POOLS}


if __name__ == "__main__":
    sys.exit(S.gather_cli(__doc__, PLOT, build))
