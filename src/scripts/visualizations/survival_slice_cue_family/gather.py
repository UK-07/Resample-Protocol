"""Survival slice: social vs artifact cue family — data, per model x case x family (datasets pooled).

social   = expert_opinion, consensus
artifact = metadata, grader_hacking, tool_output, answer_key_artifact, unethical_info
post_hoc belongs to neither family: its rows are left out of this slice (counted in excluded_post_hoc.csv).

Survival = P(robust_used | SSP-unfaithful), stitched by ``common.ssp_common`` (sha recorded in gather_meta.json).

Run:
    python -m src.scripts.visualizations.survival_slice_cue_family.gather [--cueball-dir D] [--out-dir D]
        [--only SUBSTR] [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/survival_slice_cue_family/data/).
"""
import sys

from src.scripts.visualizations.common import ssp_common as S

PLOT = "survival_slice_cue_family"


def build(df):
    fam = df[df.cue_family.notna()].copy()
    fam["slice"] = fam.cue_family
    t = S.survival_table(fam, ["subject_model", "case", "slice"])
    ph = S.survival_table(df[df.cue_family.isna()], ["subject_model", "case", "hint_style"])
    return ({"survival_by_model_slice.csv": t, "excluded_post_hoc.csv": ph},
            S.rows_for_audit(df), {"slice": "cue_family", "families": S.CUE_FAMILY,
                                   "levels": ["social", "artifact"], "post_hoc": "excluded from this slice"})


if __name__ == "__main__":
    sys.exit(S.gather_cli(__doc__, PLOT, build))
