"""Re-roll distribution — data. Over ALL SSP flips (any binary verdict), both manifest cases:
the number of the 4 re-rolls that land on the target (k_to_hint_count, 0-4), per model x case.
Share = flips at k / flips with a reliance label (k_n = 4), Wilson 95 % CI.

SSP rows stitched by ``common.ssp_common`` (sha recorded in gather_meta.json).

Run:
    python -m src.scripts.visualizations.reroll_distribution.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/reroll_distribution/data/).
"""
import sys

import pandas as pd

from src.lib.resample import wilson_interval
from src.scripts.visualizations.common import ssp_common as S

PLOT = "reroll_distribution"
KS = [0, 1, 2, 3, 4]


def build(df):
    flips = df[df.ssp_flip]
    rows = []
    for (m, c), g in flips.groupby(["subject_model", "case"], observed=True, sort=True):
        lab = g[g.has_reliance]
        k = pd.to_numeric(lab.k_to_hint_count, errors="coerce").astype(int)
        n = len(lab)
        for kk in KS:
            cnt = int((k == kk).sum())
            lo, hi = wilson_interval(cnt, n)
            rows.append({"subject_model": m, "case": c, "k_to_target": kk, "count": cnt, "denominator": n,
                         "share": cnt / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi,
                         "n_ssp_flip": len(g), "n_flip_no_reliance": int((~g.has_reliance).sum()),
                         "n_flip_binary_unfaithful": int(g.ssp_unfaithful.sum()),
                         "n_flip_binary_faithful": int(g.ssp_faithful.sum()),
                         "n_flip_binary_missing": int(g.ssp_label_missing.sum()),
                         "faded_n_lt_20": n < S.SMALL_N})
    hist = pd.DataFrame(rows)
    tables = {
        "reroll_hist_by_model_case.csv": hist,
        "exclusions_by_model_case.csv": S.exclusion_summary(df, ["subject_model", "case"]),
    }
    return tables, S.rows_for_audit(df), {"cases": ["positive", "negative"], "k_values": KS}


if __name__ == "__main__":
    sys.exit(S.gather_cli(__doc__, PLOT, build))
