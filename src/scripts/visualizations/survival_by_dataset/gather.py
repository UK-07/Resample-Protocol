"""Survival by dataset — data. Both manifest cases, per model x dataset x case (styles pooled).

Survival = P(robust_used | SSP-unfaithful), stitched by ``common.ssp_common`` (sha recorded in gather_meta.json).

Run:
    python -m src.scripts.visualizations.survival_by_dataset.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/survival_by_dataset/data/).
"""
import sys

from src.scripts.visualizations.common import ssp_common as S

PLOT = "survival_by_dataset"


def build(df):
    tables = {
        "survival_by_model_dataset_case.csv": S.survival_table(df, ["subject_model", "dataset", "case"]),
        "survival_by_model_case.csv": S.survival_table(df, ["subject_model", "case"]),
    }
    return tables, S.rows_for_audit(df), {"cases": ["positive", "negative"]}


if __name__ == "__main__":
    sys.exit(S.gather_cli(__doc__, PLOT, build))
