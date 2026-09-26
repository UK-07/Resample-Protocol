"""SSP vs RSP false rejections — data gathering.

SSP false rejection: an SSP flip (sample-0 rules, manifest case — ``common.ssp_common``) whose ORIGINAL
(hinted-once) rollout makes a non-reliance claim: v2 ``judge_role`` in {rejected, verification_only}, or v2
verdict −1 (category "incoherent", whatever its role). Truncated hinted rollouts are never SSP flips; they are
counted. RSP confirmation: the pair's ``reliance_label`` (question_reliance.csv) == robust_used (≥ 3/4 re-rolls
on target). Rate = confirmed / SSP false rejections with a reliance label.

Exclusions (counted per model and per slice, never in a denominator): SSP flips whose original rollout has no v2
verdict, or a 0/1 verdict without a role (older caches). Sensitivity only (``--minus1 exclude``, the superseded
rule): −1 originals excluded too.
Reference (not the headline quantity, drawn as a grey tick): share robust_used over ALL kept SSP flips (any role).

Slices: model (bar), model × style (primary slice), model × case, model × stability bin, model × claim role.
Counts tables (CSV + markdown) are written and printed BEFORE the rates.

Run:
    python -m src.scripts.visualizations.ssp_vs_rsp_false_rejections.gather [--cueball-dir D] [--out-dir D]
        [--only SUBSTR] [--no-cache | --refresh-cache] [--cache-dir D] [--minus1 own_category|exclude]
Tables land in <out-dir>/data/ (default <cueball>/plots/ssp_vs_rsp_false_rejections/data/). ``--only`` is a
debug subset (never cached; written under <plot dir>/_debug unless --out-dir is given). ``--minus1 exclude``
is the sensitivity rule only — write it to a separate --out-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import section3_claims as s3
from src.scripts.visualizations.common import ssp_common as ssp

PLOT = "ssp_vs_rsp_false_rejections"
V2_COLS = ["rollout_id", "judge_label_final", "judge_label_final_model", "judge_role"]
SLICES = {
    "model": ["subject_model"],
    "model_style": ["subject_model", "hint_style"],
    "model_case": ["subject_model", "case"],
    "model_stability": ["subject_model", "stability_bin"],
    "model_claim_role": ["subject_model", "category"],
}


def attach_v2(rows: pd.DataFrame, minus1: str = "exclude", *, cueball: str | Path | None = None) -> pd.DataFrame:
    v2 = pd.read_parquet(P.manifest_path(cueball), columns=V2_COLS)
    v2 = v2[v2.rollout_id.isin(rows.rollout_id)]
    n = len(rows)
    df = rows.merge(v2, on="rollout_id", how="left", validate="one_to_one", indicator="_v2")
    assert len(df) == n
    if (df._v2 != "both").any():
        raise SystemExit(f"{int((df._v2 != 'both').sum())} SSP rows without a rollout_manifest row")
    df = df.drop(columns="_v2")
    df["v2_label"] = pd.to_numeric(df.judge_label_final, errors="coerce")
    df["s3_status"], df["category"], df["role_v2"] = s3.s3_status(df, minus1)
    return df


def counts(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Counts only (printed before any rate). `df` = all SSP rows; flips are selected here so the truncated
    to-target rows (never flips) can be counted per slice too."""
    d = df
    flip = d.ssp_flip.astype(bool)
    rl = d.reliance_label.astype(object)
    cat = d.category
    ok = flip & (d.s3_status == s3.KEPT)
    fr = ok & cat.isin(s3.NON_RELIANCE_ROLES)
    lab = pd.to_numeric(d.judge_label_final, errors="coerce")
    w = pd.DataFrame({k: d[k] for k in by})
    cols = {
        "n_truncated_to_target_excluded": d.ssp_flip_truncated.astype(bool),
        "n_ssp_flips": flip,
        "n_excl_missing_role": flip & (d.s3_status == "missing_role"),
        "n_excl_unjudged": flip & (d.s3_status == "unjudged"),
        "n_excl_minus1_sensitivity": flip & (d.s3_status == "minus1_excluded"),
        "n_kept": ok,
        "n_cat_credited": ok & (cat == "credited"),
        "n_cat_neutral": ok & (cat == "neutral"),
        "n_cat_none": ok & (cat == "none"),
        "n_cat_rejected": ok & (cat == "rejected"),
        "n_cat_verification_only": ok & (cat == "verification_only"),
        "n_cat_incoherent": ok & (cat == s3.INCOHERENT),
        "n_false_rejections": fr,
        "n_fr_no_reliance": fr & rl.isna(),
        "n_fr_with_reliance": fr & rl.notna(),
        "n_fr_robust_used": fr & (rl == "robust_used"),
        "n_fr_weak_used": fr & (rl == "weak_used"),
        "n_fr_mixed_0of4": fr & (rl == "mixed"),
        "n_kept_with_reliance": ok & rl.notna(),
        "n_kept_robust_used": ok & (rl == "robust_used"),
        "n_flip_minus1_role_rejected": flip & (lab == -1) & (d.role_v2 == "rejected"),
        "n_flip_minus1_role_verification_only": flip & (lab == -1) & (d.role_v2 == "verification_only"),
    }
    for k, v in cols.items():
        w[k] = v.fillna(False).astype(int) if hasattr(v, "fillna") else v.astype(int)
    return w.groupby(by, observed=True, dropna=False)[list(cols)].sum().reset_index()


def rates(c: pd.DataFrame) -> pd.DataFrame:
    r = s3.add_wilson(c, "n_fr_robust_used", "n_fr_with_reliance")
    r = s3.add_wilson(r, "n_kept_robust_used", "n_kept_with_reliance", prefix="ref_all_flips_")
    r["faded_n_lt_20"] = r.n_fr_with_reliance < s3.SMALL_N
    return r


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None, help=f"default {P.DEFAULT_CUEBALL_DIR}/plots/{PLOT}")
    ap.add_argument("--only", default=None, help="ssp_common source filter (debug subset, never cached)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="ssp_common row cache; default <cueball>/plots/_cache")
    ap.add_argument("--minus1", choices=s3.MINUS1_RULES, default="own_category",
                    help="own_category (the definition) | exclude (SENSITIVITY ONLY; write to a separate --out-dir)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cueball = P.cueball_dir(a.cueball_dir)
    if a.out_dir is None:
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if a.only else P.plot_dir(PLOT, cueball)
        if a.only:
            ssp.log(f"--only without --out-dir: writing to the debug dir {out_dir}")
    else:
        out_dir = Path(a.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    if a.no_cache or a.only:
        cache: str | Path | None = None
    else:
        cache = Path(a.cache_dir) if a.cache_dir else P.cache_dir(cueball)

    rows, ssp_meta = ssp.load_rows(a.only, cache_dir=cache, refresh=a.refresh_cache, cueball=cueball)
    df = attach_v2(rows, a.minus1, cueball=cueball)
    df["stability_bin"] = df.stability_bin.astype("Int64")
    flips = df[df.ssp_flip.astype(bool)].copy()
    checks = {
        "ssp_flips": int(len(flips)),
        "ssp_flips_without_reliance": int(flips.reliance_label.isna().sum()),
        "ssp_flips_rel_role_not_used_candidate": int((flips.rel_role.astype(object) != "used_candidate").sum()),
    }
    s3.log("  checks: " + ", ".join(f"{k}={v}" for k, v in checks.items()))

    # SSP funnel per model × case (the sample-0 exclusions of ssp_common, for NOTES)
    ssp.exclusion_summary(df, ["subject_model", "case"]).to_csv(out / "ssp_funnel_by_model_case.csv", index=False)

    md = ["# SSP vs RSP false rejections — counts, then rates\n",
          "SSP false rejection = SSP flip whose original rollout's v2 role ∈ {rejected, verification_only} or whose "
          "v2 verdict is −1 (incoherent). Confirmed = the pair is robust_used. Excluded: unjudged, 0/1 verdict "
          "without a role. Truncated to-target rollouts are not SSP flips (counted in the first column).\n"
          f"minus1 rule: {a.minus1}\n"]
    count_tabs, rate_tabs = {}, {}
    for name, by in SLICES.items():
        if name == "model_claim_role":
            c = counts(flips[flips.s3_status == s3.KEPT], by)
            c = c[c.category.isin(s3.NON_RELIANCE_ROLES)]
        else:
            c = counts(df, by)
        count_tabs[name] = c
        c.to_csv(out / f"counts_by_{name}.csv", index=False)
        r = rates(c)
        if name == "model_claim_role":      # the reference is only meaningful over all kept flips
            r[["ref_all_flips_rate", "ref_all_flips_ci_lo", "ref_all_flips_ci_hi"]] = np.nan
        rate_tabs[name] = r
        r.to_csv(out / f"rates_by_{name}.csv", index=False)

    ccols = ["n_truncated_to_target_excluded", "n_ssp_flips", "n_excl_missing_role", "n_excl_unjudged",
             "n_excl_minus1_sensitivity", "n_kept", "n_cat_rejected", "n_cat_verification_only", "n_cat_incoherent",
             "n_false_rejections", "n_fr_robust_used", "n_fr_weak_used", "n_fr_mixed_0of4"]
    rcols = ["n_fr_robust_used", "n_fr_with_reliance", "rate", "ci_lo", "ci_hi", "ref_all_flips_rate"]
    md.append("## Counts\n")
    for name, by in SLICES.items():
        md.append(f"### counts by {name}\n")
        cc = [c for c in ccols if c in count_tabs[name].columns]
        md.append(s3.md_table(count_tabs[name], by + cc))
        s3.log(f"\nCOUNTS by {name}:\n" + count_tabs[name][by + cc].to_string(index=False))
    md.append("## Rates (share of SSP false rejections that are robust_used; Wilson 95 %)\n")
    for name, by in SLICES.items():
        md.append(f"### rates by {name}\n")
        md.append(s3.md_table(rate_tabs[name], by + rcols))
        s3.log(f"\nRATES by {name}:\n" + rate_tabs[name][by + rcols].round(3).to_string(index=False))
    m1 = s3.minus1_role_table(flips, ["subject_model"])
    m1.to_csv(out / "minus1_by_role_ssp_flips_by_model.csv", index=False)
    md.append("## v2 −1 SSP flips by the role the judge also emitted (n_judged = SSP flips with a v2 verdict)\n")
    md.append(s3.md_table(m1))
    (out / "counts_then_rates.md").write_text("\n".join(md))

    manifest = P.manifest_path(cueball)
    meta = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plot": PLOT, "script": f"src.scripts.visualizations.{PLOT}.gather",
        "script_sha256": ssp.sha256_file(Path(__file__).resolve()),
        "code_sha256": s3.code_shas(),
        "ssp_common": ssp.lib_provenance(), "ssp_rows_meta": ssp_meta,
        "inputs": {"rollout_manifest_v2": ssp.file_info(manifest, sha=True)},
        "cueball_dir": str(cueball),
        "checks": checks, "only": a.only, "minus1_rule": a.minus1, "slices": SLICES,
        "false_rejection_roles": list(s3.NON_RELIANCE_ROLES),
    }
    (out / "gather_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    s3.log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
