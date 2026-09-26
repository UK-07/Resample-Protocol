"""Dose-response — data gathering.

x = the (question, hint) pair's re-rolls on target, ``k_to_hint_count`` in {0..4} (``question_reliance.csv``;
every pair has k_n = 4). y = share of rollouts with a NON-RELIANCE claim among the rollouts that MENTION the hint.
Category = v2 role, or "incoherent" for a v2 −1 rollout:
  non-reliance claim = rejected, verification_only or incoherent (−1);  mention = every category except `none`.
Per-rollout weighting: the original and each on-target re-roll count once. Population = the claims-vs-behaviour
role pool (``common.section3_claims``): judged on-target rollouts; unjudged rollouts, 0/1 verdicts without a role
and truncated rollouts are excluded and counted.

Pair scope (column `scope`):
  used_pairs  — pairs whose original switched to the target (selection role used_candidate). At x = k the pair
                contributes its original + k re-rolls. PRIMARY.
  all_pairs   — tables only (not drawn): used_pairs plus the control pairs whose re-rolls reached the target;
                a control at x = k contributes its k on-target re-rolls only.

Run:
    python -m src.scripts.visualizations.dose_response.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--models a,b] [--minus1 own_category|exclude] [--no-cache | --refresh-cache] [--cache-dir D]
Tables land in <out-dir>/data/ (default <cueball>/plots/dose_response/data/). This gather reads the resample
manifest and the reliance CSV directly, so the cache flags are accepted for uniformity and do nothing.
Options: --only <substr> / --models <comma list> (debug subsets of the models; without --out-dir they are written
under <plot dir>/_debug), --minus1 own_category (the definition, default) | exclude (SENSITIVITY ONLY — the
superseded rule; use a separate --out-dir).
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

PLOT = "dose_response"
KS = [0, 1, 2, 3, 4]


def dose_table(kept: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per (by..., k): mention rollouts (denominator), non-reliance claims (numerator), per-role counts."""
    d = kept.copy()
    c = d.category
    d["mention"] = c.isin(s3.MENTION_ROLES)
    d["claim"] = c.isin(s3.NON_RELIANCE_ROLES)
    d["n_rejected"] = c == "rejected"
    d["n_verification_only"] = c == "verification_only"
    d["n_incoherent"] = c == s3.INCOHERENT
    d["n_credited"] = c == "credited"
    d["n_neutral"] = c == "neutral"
    d["n_none"] = c == "none"
    keys = by + ["k"]
    g = d.groupby(keys, observed=True).agg(
        n_kept=("rollout_id", "size"), n_none=("n_none", "sum"), denominator=("mention", "sum"),
        numerator=("claim", "sum"), n_rejected=("n_rejected", "sum"),
        n_verification_only=("n_verification_only", "sum"), n_incoherent=("n_incoherent", "sum"),
        n_credited=("n_credited", "sum"),
        n_neutral=("n_neutral", "sum"), n_originals=("is_original", "sum"),
        n_pairs=("pair_id", "nunique"),
    ).reset_index()
    for c in ("n_none", "denominator", "numerator", "n_rejected", "n_verification_only", "n_incoherent", "n_credited",
              "n_neutral", "n_originals"):
        g[c] = g[c].astype(int)
    g = s3.add_wilson(g, "numerator", "denominator")
    g = s3.add_wilson(g, "n_rejected", "denominator", prefix="rejected_")
    g = s3.add_wilson(g, "n_verification_only", "denominator", prefix="verification_only_")
    g = s3.add_wilson(g, "n_incoherent", "denominator", prefix="incoherent_")
    g["faded_n_lt_20"] = g.denominator < s3.SMALL_N
    return g


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None, help="default <cueball>/plots/dose_response; tables go to <out-dir>/data")
    ap.add_argument("--only", default=None, help="debug subset: models whose name contains this substring")
    ap.add_argument("--models", default=None, help="debug subset: comma-separated subject models")
    ap.add_argument("--minus1", choices=s3.MINUS1_RULES, default="own_category",
                    help="own_category (the definition) | exclude (SENSITIVITY ONLY; write to a separate --out-dir)")
    ap.add_argument("--no-cache", action="store_true", help="accepted for uniformity; this gather uses no cache")
    ap.add_argument("--refresh-cache", action="store_true", help="accepted for uniformity; this gather uses no cache")
    ap.add_argument("--cache-dir", default=None, help="accepted for uniformity; this gather uses no cache")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cueball = P.cueball_dir(a.cueball_dir)
    subset = bool(a.only or a.models)
    if a.out_dir is None:
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if subset else P.plot_dir(PLOT, cueball)
        if subset:
            s3.log(f"--only/--models without --out-dir: writing to the debug dir {out_dir}")
    else:
        out_dir = Path(a.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)

    checks: dict = {}
    pool = s3.load_role_pool(checks, minus1=a.minus1, cueball=cueball)
    if a.models:
        pool = pool[pool.subject_model.isin(a.models.split(","))]
    if a.only:
        pool = pool[pool.subject_model.astype(str).str.contains(a.only)]
    if pool.k.isna().any() or not pool.k.isin(KS).all():
        raise SystemExit("k_to_hint_count outside 0..4 or missing in the role pool")
    # a control pair's originals are never on target; every on-target control rollout is a re-roll
    checks["control_originals_on_target"] = int((pool.is_original & (pool.sel_role == "control")).sum())
    # on a used pair at x = k there are exactly k on-target re-rolls (+ the original)
    rr = pool[~pool.is_original].groupby("pair_id").size()
    kk = pool.drop_duplicates("pair_id").set_index("pair_id").k
    checks["pairs_reroll_count_ne_k"] = int((rr.reindex(kk.index).fillna(0).astype(int) != kk.astype(int)).sum())
    s3.log(f"  dose checks: control_originals_on_target={checks['control_originals_on_target']}, "
           f"pairs_reroll_count_ne_k={checks['pairs_reroll_count_ne_k']}")

    scopes = {"used_pairs": pool[pool.sel_role == "used_candidate"], "all_pairs": pool}
    tabs = {"dose_by_model.csv": ["subject_model"], "dose_by_model_case.csv": ["subject_model", "case"],
            "dose_by_model_provenance.csv": ["subject_model", "provenance"],
            "dose_by_model_style.csv": ["subject_model", "hint_style"]}
    for name, by in tabs.items():
        parts = []
        for scope, p in scopes.items():
            t = dose_table(p[p.s3_status == "ok"], by)
            t.insert(0, "scope", scope)
            parts.append(t)
        pd.concat(parts, ignore_index=True).to_csv(out / name, index=False)
    excl = []
    for scope, p in scopes.items():
        for by, name in ((["subject_model"], "model"), (["subject_model", "k"], "model_k"),
                         (["subject_model", "case"], "model_case")):
            e = s3.exclusion_counts(p, by)
            e.insert(0, "scope", scope)
            e.insert(1, "grouping", name)
            excl.append(e)
    pd.concat(excl, ignore_index=True).to_csv(out / "exclusions.csv", index=False)
    s3.minus1_role_table(pool, ["subject_model"]).to_csv(out / "minus1_by_role_by_model.csv", index=False)
    meta = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plot": PLOT, "script": f"src.scripts.visualizations.{PLOT}.gather",
        "script_sha256": sc.sha256_file(Path(__file__).resolve()),
        "cueball_dir": str(cueball),
        "inputs": {"resample_manifest": sc.file_info(P.resample_manifest_path(cueball), sha=True),
                   "reliance": sc.file_info(P.reliance_path(cueball), sha=True)},
        "code_sha256": s3.code_shas(),
        "checks": checks, "only": a.only, "models_filter": a.models, "minus1_rule": a.minus1,
        "primary_scope": "used_pairs",
        "numerator_roles": list(s3.NON_RELIANCE_ROLES), "denominator_roles": list(s3.MENTION_ROLES),
        "tables": sorted(list(tabs) + ["exclusions.csv", "minus1_by_role_by_model.csv"]),
    }
    (out / "gather_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    t = pd.read_csv(out / "dose_by_model.csv")
    s3.log("\n" + t[["scope", "subject_model", "k", "n_pairs", "n_kept", "n_none", "denominator", "numerator",
                     "n_rejected", "n_verification_only", "n_incoherent", "rate", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    s3.log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
