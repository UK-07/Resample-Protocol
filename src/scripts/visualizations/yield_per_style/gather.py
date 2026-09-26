"""Yield per cue style — data: clean unfaithful examples per 1,000 hinted rollouts, by style x model.

Numerator   = RSP-unfaithful re-rolls (``yield_common.rsp_unfaithful_pool``): ``src.lib.selection``'s
              'dataset_B' rows of resample/resample_manifest.parquet with judge_label_final == 0, plus the
              re-rolls meeting dataset_B's other conditions whose label is −1 (counted as unfaithful; reported apart).
Denominator = hinted rollouts = rows of hinted_rollouts/rollout_manifest.parquet (one per question x style pair,
              both manifest cases), same model x style.
Yield       = 1000 x numerator / denominator (may exceed 1,000: up to 4 re-rolls per pair). 95 % CI:
              question-clustered percentile bootstrap (questions resampled with replacement within the cell, ratio
              of sums; ``--reps`` reps, seed 0).

Filters as in the resample scripts (no smoke runs, the 8 styles) and the 5 models (Qwen3.6-27B excluded).

Run:
    python -m src.scripts.visualizations.yield_per_style.gather [--cueball-dir D] [--out-dir D] [--only SUBSTR]
        [--no-cache | --refresh-cache] [--cache-dir D] [--reps N]
Tables land in <out-dir>/data/ (default <cueball>/plots/yield_per_style/data/). This gather reads the two manifests
and question_reliance.csv directly (no stitched-rows cache): the cache flags are accepted for uniformity and ignored.
``--only`` keeps the rows whose run, subject model or hint style contains the substring (debug); without
``--out-dir`` the subset is written under <plot dir>/_debug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import yield_common as yc

PLOT = "yield_per_style"
RM_COLS = ["rollout_id", "source_rollout_id", "subject_model", "run", "dataset", "hint_style", "case", "provenance",
           "reliance_label", "to_hint", "judge_label_final", "exclude_reason", "truncated", "sample_seed"]
GROUPINGS = {
    "yield_by_model_style.csv": ["subject_model", "hint_style"],
    "yield_by_model.csv": ["subject_model"],
    "yield_by_model_style_case.csv": ["subject_model", "hint_style", "case"],
    "yield_by_model_style_dataset.csv": ["subject_model", "hint_style", "dataset"],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _only(df: pd.DataFrame, only: str | None) -> pd.DataFrame:
    if not only:
        return df
    keep = pd.Series(False, index=df.index)
    for col in ("run", "subject_model", "hint_style"):
        if col in df.columns:
            keep |= df[col].astype(str).str.contains(only, regex=False)
    return df[keep].copy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap reps (default 2000)")
    args = ap.parse_args(argv)
    cueball = P.cueball_dir(args.cueball_dir)
    if args.out_dir is None:
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if args.only else P.plot_dir(PLOT, cueball)
        if args.only:
            log(f"--only without --out-dir: writing to the debug dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)

    man = _only(yc.load_rollout_manifest(
        ["rollout_id", "question_id", "dataset", "case", "to_hint", "exclude_reason"], cueball=cueball), args.only)
    rm = _only(yc.load_resample_manifest(RM_COLS, cueball=cueball), args.only)
    B = yc.dataset_B(rm)
    P_ = yc.rsp_unfaithful_pool(rm)
    checks = {"hinted_pairs": len(man), "resample_rows": len(rm),
              "resample_k4_rows": int((rm.provenance.astype(str) == "resample_k4").sum()),
              "dataset_B_rows": len(B), "dataset_B_label0": int((B.judge_label_final == 0).sum()),
              "dataset_B_label1": int((B.judge_label_final == 1).sum()),
              "admitted_v2_label_-1_rows": int((P_.v2_label == -1).sum()),
              "pool_rows (dataset_B + -1)": len(P_)}
    orphan = ~P_.source_rollout_id.isin(man.rollout_id)
    checks["pool_rows_whose_original_is_not_in_rollout_manifest"] = int(orphan.sum())
    if orphan.any():
        raise SystemExit("pool rows without their hinted-once original in the rollout manifest")
    rel = _only(yc.load_reliance(None, cueball=cueball), args.only)[["rollout_id", "reliance_label"]].rename(
        columns={"rollout_id": "source_rollout_id", "reliance_label": "qr_reliance"}).drop_duplicates("source_rollout_id")
    chk = P_[["source_rollout_id", "reliance_label"]].merge(rel, on="source_rollout_id", how="left")
    checks["pool_reliance_mismatch_vs_question_reliance"] = int(
        (chk.reliance_label.astype(object).fillna("<NA>") != chk.qr_reliance.astype(object).fillna("<NA>")).sum())
    rr = rm[(rm.provenance.astype(str) == "resample_k4") & (rm.reliance_label.astype(object) == "robust_used")
            & rm.to_hint.fillna(False).astype(bool)]
    notP = rr[~rr.rollout_id.isin(P_.rollout_id)]
    lab = pd.to_numeric(notP.judge_label_final, errors="coerce")
    checks["robust_used_to_target_rerolls"] = len(rr)
    checks["robust_used_to_target_rerolls_not_in_pool"] = {
        **{f"label={int(l) if pd.notna(l) else 'NA'},exclude_reason={e}": int(n) for (l, e), n in
           notP.assign(l=lab, e=notP.exclude_reason.astype(object)).groupby(["l", "e"], dropna=False).size().items()},
    }

    # per hinted pair: RSP-unfaithful re-rolls (0-4), of which −1
    U = P_[P_.rsp_unfaithful]
    per_pair = pd.DataFrame({
        "n_clean_unfaithful": U.groupby("source_rollout_id").size(),
        "n_clean_unfaithful_label_-1": U[U.v2_label == -1].groupby("source_rollout_id").size(),
        "n_pool": P_.groupby("source_rollout_id").size(),
        "n_rerolls": rm[rm.provenance.astype(str) == "resample_k4"].groupby("source_rollout_id").size(),
    })
    pairs = man.merge(per_pair, left_on="rollout_id", right_index=True, how="left")
    for c in per_pair.columns:
        pairs[c] = pairs[c].fillna(0).astype(int)
    checks["unfaithful_rerolls_attached_to_pairs"] = int(pairs.n_clean_unfaithful.sum())
    assert checks["unfaithful_rerolls_attached_to_pairs"] == len(U)

    for name, keys in GROUPINGS.items():
        rows = []
        for key, d in pairs.groupby(keys, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            n = len(d)
            k = int(d.n_clean_unfaithful.sum())
            k_inc = int(d["n_clean_unfaithful_label_-1"].sum())
            q = d.groupby("question_id").agg(k=("n_clean_unfaithful", "sum"), n=("rollout_id", "size"))
            lo, hi = yc.cluster_bootstrap_ratio_ci(q.k.values, q.n.values, reps=args.reps, seed=0)
            npool = int(d.n_pool.sum())
            rows.append({**dict(zip(keys, key)),
                         "hinted_rollouts": n, "clean_unfaithful": k, "clean_unfaithful_label_0": k - k_inc,
                         "clean_unfaithful_label_-1": k_inc,
                         "yield_per_1000": 1000 * k / n, "ci_lo": 1000 * lo, "ci_hi": 1000 * hi,
                         "yield_per_1000_label0_only": 1000 * (k - k_inc) / n,
                         "pairs_with_any": int((d.n_clean_unfaithful > 0).sum()), "n_questions": len(q),
                         "pool_rows": npool, "rsp_unfaithful_rate": k / npool if npool else np.nan,
                         "rerolls": int(d.n_rerolls.sum()),
                         "small_n": n < yc.SMALL_N})
        pd.DataFrame(rows).to_csv(out / name, index=False)
    t = pd.read_csv(out / "yield_by_model_style.csv")
    piv = t.pivot(index="hint_style", columns="subject_model", values="yield_per_1000").reindex(
        index=yc.STYLES, columns=yc.MODELS)
    (out / "yield_table.md").write_text(
        "Clean unfaithful examples (dataset_B conditions, v2 label 0 or -1) per 1,000 hinted rollouts\n\n"
        + piv.round(1).to_markdown() + "\n")
    yc.json_dump(checks, out / "checks.json")
    script = P.repo_path(f"src/scripts/visualizations/{PLOT}/gather.py")
    yc.json_dump({"written_utc": yc.now_utc(), "plot": PLOT,
                  "script": f"src.scripts.visualizations.{PLOT}.gather",
                  "script_sha256": yc.sha256_file(script) if script.exists() else None,
                  "cueball_dir": str(cueball), "only": args.only,
                  "shared_code": yc.shared_code_meta(), "models": yc.MODELS, "styles": yc.STYLES,
                  "bootstrap_reps": args.reps, "bootstrap_seed": 0,
                  "inputs": {"manifest": yc.file_info(P.manifest_path(cueball), sha=True),
                             "resample_manifest": yc.file_info(P.resample_manifest_path(cueball), sha=True),
                             "question_reliance": yc.file_info(P.reliance_path(cueball), sha=True)}},
                 out / "gather_meta.json")
    log(json.dumps(checks, indent=1))
    log("\n" + piv.round(1).to_string())
    log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
