"""Survival vs stability — data. Positive case (manifest), pooled over styles and datasets.

Survival = P(robust_used | SSP-unfaithful), stitched by ``common.ssp_common`` (sha recorded in gather_meta.json).
Stability bin = manifest ``baseline_stability`` (binning only).

Run:
    python -m src.scripts.visualizations.survival_vs_stability.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/survival_vs_stability/data/).
"""
import sys

from src.scripts.visualizations.common import ssp_common as S

PLOT = "survival_vs_stability"
CASE = "positive"


def build(df):
    d = df[df.case.astype(object) == CASE]
    tables = {
        "survival_by_model_bin.csv": S.survival_table(d, ["subject_model", "case", "stability_bin"]),
        "survival_by_model.csv": S.survival_table(d, ["subject_model", "case"]),
        "survival_pooled.csv": S.survival_table(d, ["case"]),
    }
    return tables, S.rows_for_audit(d), {"case": CASE, "stability_bins": S.STABILITY_BINS}


if __name__ == "__main__":
    sys.exit(S.gather_cli(__doc__, PLOT, build))
