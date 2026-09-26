"""Survival slice: post_hoc vs all other styles — data, per model x case x slice (datasets pooled).

post_hoc is the only cue in the assistant turn; "other styles" pools the other 7 styles.
Survival = P(robust_used | SSP-unfaithful), stitched by ``common.ssp_common`` (sha recorded in gather_meta.json).

Run:
    python -m src.scripts.visualizations.survival_slice_post_hoc.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/survival_slice_post_hoc/data/).
"""
import sys

from src.scripts.visualizations.common import ssp_common as S

PLOT = "survival_slice_post_hoc"


def build(df):
    d = df.copy()
    d["slice"] = (d.hint_style.astype(object) == "post_hoc").map({True: "post_hoc", False: "other styles"})
    t = S.survival_table(d, ["subject_model", "case", "slice"])
    return ({"survival_by_model_slice.csv": t}, S.rows_for_audit(d, ["slice"]),
            {"slice": "post_hoc", "levels": ["post_hoc", "other styles"]})


if __name__ == "__main__":
    sys.exit(S.gather_cli(__doc__, PLOT, build))
