"""Role x reliance heatmap — data gathering.

Population: every judged on-target rollout of a (question, hint) pair — the hinted-once original and the k=4
re-rolls. Row category = the v2 judge_role, or "incoherent" for a v2 −1 rollout (its own row, whatever role it
also carries). Unjudged rollouts, 0/1 verdicts without a role and truncated rollouts are excluded and counted per
model. Column = the pair's reliance_label (question_reliance.csv); value = share of each category within the
column, per model. Also writes the per-model −1 x role counts table.

Columns written (column_key):
  robust_used, weak_used, mixed      — used_candidate pairs (the original switched); mixed = 0/4 re-rolls on target
  mixed_control                      — control pairs labelled mixed (the original did NOT switch, but some re-rolls
                                       went to the target) — tabulated only, never drawn
  robust_ignored                     — control pairs, 0/4 on target: no on-target rollouts, never judged
                                       (pair count only)

Run:
    python -m src.scripts.visualizations.role_reliance_heatmap.gather [--cueball-dir D] [--out-dir D]
        [--only SUBSTR | --models a,b] [--minus1 own_category|exclude]
Tables land in <out-dir>/data/ (default <cueball>/plots/role_reliance_heatmap/data/). ``--only`` / ``--models``
are debug subsets (models whose name contains SUBSTR / an exact comma list); without --out-dir they write under
<plot dir>/_debug. ``--minus1 exclude`` is the superseded sensitivity rule (−1 excluded) and must go to a
separate --out-dir. The cache flags (--no-cache, --refresh-cache, --cache-dir) are accepted for uniformity with
the other gathers; this gather reads the resample manifest directly and keeps no cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import section3_claims as s3
from src.scripts.visualizations.common import ssp_common as sc

PLOT = "role_reliance_heatmap"
COLUMN_KEYS = ["robust_used", "weak_used", "mixed", "mixed_control", "robust_ignored"]


def column_key(df: pd.DataFrame) -> pd.Series:
    used = df.sel_role.astype(str) == "used_candidate"
    lab = df.rel_label.astype(object)
    ctrl = lab.map({"mixed": "mixed_control", "robust_ignored": "robust_ignored"})
    return lab.where(used, ctrl)


def shares(kept: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Long table: per (by..., column_key, category) n, column total, share and Wilson CI."""
    keys = by + ["column_key"]
    n = kept.groupby(keys + ["category"], observed=True).size().rename("n").reset_index()
    tot = kept.groupby(keys, observed=True).size().rename("n_column").reset_index()
    grid = tot.merge(pd.DataFrame({"category": s3.CATEGORIES}), how="cross")
    out = grid.merge(n, on=keys + ["category"], how="left").fillna({"n": 0})
    out["n"] = out.n.astype(int)
    return s3.add_wilson(out, "n", "n_column")


def model_filter(a: argparse.Namespace):
    """The debug subset: ``--models`` (exact comma list) and/or ``--only`` (substring of the model name)."""
    if not a.models and not a.only:
        return None
    models = set(a.models.split(",")) if a.models else set(s3.MODELS)
    if a.only:
        models = {m for m in models if a.only in m}
    return sorted(models)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None, help="debug subset: models whose name contains this substring")
    ap.add_argument("--models", default=None, help="debug subset: exact comma list of models")
    ap.add_argument("--minus1", choices=s3.MINUS1_RULES, default="own_category",
                    help="own_category (the definition) | exclude (SENSITIVITY ONLY; write to a separate --out-dir)")
    ap.add_argument("--no-cache", action="store_true", help="accepted for uniformity; this gather keeps no cache")
    ap.add_argument("--refresh-cache", action="store_true", help="accepted for uniformity; no cache")
    ap.add_argument("--cache-dir", default=None, help="accepted for uniformity; no cache")
    a = ap.parse_args(argv)
    cueball = P.cueball_dir(a.cueball_dir)
    models = model_filter(a)
    if a.out_dir is None:
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if models is not None else P.plot_dir(PLOT, cueball)
        if models is not None:
            s3.log(f"debug subset without --out-dir: writing to {out_dir}")
    else:
        out_dir = Path(a.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)

    checks: dict = {}
    pool = s3.load_role_pool(checks, minus1=a.minus1, cueball=cueball)
    if models is not None:
        pool = pool[pool.subject_model.isin(models)]
    pool["column_key"] = column_key(pool)
    kept = pool[pool.s3_status == "ok"]

    # the pairs themselves (incl. robust_ignored, which has no on-target rollouts)
    rel = s3.read_reliance(cueball=cueball)
    rel = rel.merge(pd.read_csv(P.reliance_path(cueball),
                                usecols=["rollout_id", "subject_model", "run", "hint_style", "case"])
                    .rename(columns={"rollout_id": "pair_id"}), on="pair_id")
    rel = s3.keep_scope(rel)
    if models is not None:
        rel = rel[rel.subject_model.isin(models)]
    rel["column_key"] = column_key(rel)
    pairs = rel.groupby(["subject_model", "column_key"]).size().rename("n_pairs").reset_index()
    on_target = pool.groupby(["subject_model", "column_key"]).size().rename("n_on_target_rollouts").reset_index()
    kept_n = kept.groupby(["subject_model", "column_key"]).size().rename("n_kept_rollouts").reset_index()
    col_summary = pairs.merge(on_target, how="left").merge(kept_n, how="left").fillna(0)
    col_summary.to_csv(out / "column_summary.csv", index=False)
    k = ["subject_model", "case", "column_key"]
    cs_case = (rel.groupby(k).size().rename("n_pairs").reset_index()
               .merge(pool.groupby(k).size().rename("n_on_target_rollouts").reset_index(), how="left")
               .merge(kept.groupby(k).size().rename("n_kept_rollouts").reset_index(), how="left").fillna(0))
    cs_case.to_csv(out / "column_summary_by_case.csv", index=False)
    s3.exclusion_counts(pool, ["subject_model", "case"]).to_csv(out / "exclusions_by_model_case.csv", index=False)

    shares(kept, ["subject_model"]).to_csv(out / "role_share_by_model.csv", index=False)
    shares(kept, ["subject_model", "case"]).to_csv(out / "role_share_by_model_case.csv", index=False)
    shares(kept, ["subject_model", "provenance"]).to_csv(out / "role_share_by_model_provenance.csv", index=False)
    s3.exclusion_counts(pool, ["subject_model"]).to_csv(out / "exclusions_by_model.csv", index=False)
    s3.exclusion_counts(pool, ["subject_model", "column_key"]).to_csv(out / "exclusions_by_model_column.csv",
                                                                      index=False)
    s3.exclusion_counts(pool, ["subject_model", "case", "column_key"]).to_csv(
        out / "exclusions_by_model_case_column.csv", index=False)
    s3.exclusion_counts(pool, ["subject_model", "provenance"]).to_csv(out / "exclusions_by_model_provenance.csv",
                                                                      index=False)
    # the −1 x role table (independent of --minus1: counts every v2 −1 rollout of the pool)
    t_m = s3.minus1_role_table(pool, ["subject_model"])
    t_mp = s3.minus1_role_table(pool, ["subject_model", "provenance"])
    t_m.to_csv(out / "minus1_by_role_by_model.csv", index=False)
    t_mp.to_csv(out / "minus1_by_role_by_model_provenance.csv", index=False)
    (out / "minus1_by_role.md").write_text(
        "# v2 −1 (incoherent) rollouts by the role the judge also emitted\n\n"
        "Pool: judged on-target rollouts (originals + re-rolls) of the re-sampled pairs. `no_role` = −1 without a "
        "role. `n_judged` = rollouts with any v2 verdict.\n\n## per model\n\n" + s3.md_table(t_m)
        + "\n## per model × provenance\n\n" + s3.md_table(t_mp))
    inputs = {"resample_manifest": sc.file_info(P.resample_manifest_path(cueball), sha=True),
              "reliance": sc.file_info(P.reliance_path(cueball), sha=True)}
    script = P.repo_path(f"src/scripts/visualizations/{PLOT}/gather.py")
    meta = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plot": PLOT, "script": f"src.scripts.visualizations.{PLOT}.gather",
        "script_sha256": sc.sha256_file(script) if script.exists() else None,
        "cueball_dir": str(cueball), "inputs": inputs, "code": s3.code_shas(),
        "checks": checks, "models_filter": models, "only": a.only, "minus1_rule": a.minus1,
        "column_keys": COLUMN_KEYS, "categories": s3.CATEGORIES,
    }
    (out / "gather_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    s3.log("\ncolumn_summary:\n" + col_summary.to_string(index=False))
    t = pd.read_csv(out / "role_share_by_model.csv")
    s3.log("\nrole shares (model × column):\n" + t.pivot_table(index=["subject_model", "category"],
                                                              columns="column_key", values="rate").round(3)
           .to_string())
    s3.log("\nexclusions:\n" + pd.read_csv(out / "exclusions_by_model.csv").to_string(index=False))
    s3.log("\n−1 × role:\n" + t_m.to_string(index=False))
    s3.log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
