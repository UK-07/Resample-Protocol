"""Claims vs behaviour — shared helpers for the judge-role / reliance plots
(role_reliance_heatmap, dose_response, ssp_vs_rsp_false_rejections).

The single-sample (SSP) quantities (sample-0 baseline, SSP flip) do NOT live here: they come from
``common.ssp_common``.

Definitions:
  * role = the v2 judge's `judge_role` (none / neutral / verification_only / rejected / credited).
  * The ROLE POOL = every judged on-target rollout of a (question, hint) pair: the hinted-once original
    (provenance hinted_once) and the k=4 re-rolls (provenance resample_k4), `to_hint` True, from
    resample/resample_manifest.parquet (which holds the originals of every selected pair plus the re-rolls).
  * v2 verdict −1 (`judge_label_final == -1`) is NOT excluded: such a rollout gets the category "incoherent" in
    place of its role (heatmap: its own row; dose-response / false rejections: a non-reliance claim and a hint
    mention). `category` = "incoherent" for −1, else the role.
  * Excluded from every denominator (counted per model): verdict missing (`judge_label_final` null — never
    judged), verdict 0/1 without a role (judged before the role-emitting prompt), and truncated rollouts
    (status "truncated", checked after the verdict so it wins over the other statuses).
  * Sensitivity only (`minus1="exclude"`, the superseded rule): −1 rollouts excluded as well.
  * reliance_label / k_to_hint_count / selection role come from resample/question_reliance.csv, joined on the
    pair key (original: rollout_id; re-roll: source_rollout_id) and asserted equal to the manifest's copy.

Nothing here writes files; the gather scripts do.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib.resample import DROPPED_HINTS, SMOKE_RUN_MARKER, wilson_interval
from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import ssp_common as sc

MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
STYLES = ["unethical_info", "metadata", "grader_hacking", "expert_opinion", "tool_output",
          "answer_key_artifact", "post_hoc", "consensus"]
ROLES = ["credited", "verification_only", "rejected", "neutral", "none"]   # = paper_plots.ROLE_ORDER
INCOHERENT = "incoherent"
CATEGORIES = ROLES + [INCOHERENT]      # heatmap rows: the 5 roles + v2 −1
NON_RELIANCE_ROLES = ("rejected", "verification_only", INCOHERENT)            # −1 is a non-reliance claim
MENTION_ROLES = ("credited", "verification_only", "rejected", "neutral", INCOHERENT)   # every category but `none`
RELIANCE_LABELS = ["robust_used", "weak_used", "mixed", "robust_ignored"]
STABILITY_BINS = [5, 6, 7, 8]          # manifest baseline_stability; <= 4 folds into 5 (none exist in the tree)
SMALL_N = 20

RM_COLS = ["rollout_id", "question_id", "subject_model", "run", "dataset", "hint_style", "case",
           "original_index", "baseline_stability", "target_option", "model_answer", "to_hint", "truncated",
           "judge_label_final", "judge_label_final_model", "judge_role", "exclude_reason", "provenance",
           "sample_seed", "source_rollout_id", "role", "reliance_label"]
REL_COLS = ["rollout_id", "role", "reliance_label", "k_n", "k_to_hint_count", "k_modal_count", "k_judged",
            "k_unfaithful", "k_faithful", "k_incoherent"]


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_shas() -> dict[str, str]:
    """{name: sha256} of this module, ssp_common and the repo code the numbers depend on (gather_meta.json)."""
    out = {"section3_claims.py": sha256(Path(__file__).resolve()), "ssp_common.py": sha256(sc.LIB_PATH)}
    out.update(sc.code_deps_provenance(sc.CODE_DEPS + ["src/lib/selection.py"]))
    out["git_sha"] = sc.repo_git_sha()
    return out


def keep_scope(df: pd.DataFrame) -> pd.DataFrame:
    """The 5 models, no smoke runs, the 8 styles."""
    keep = df.subject_model.astype(str).isin(MODELS)
    keep &= ~df.run.astype(str).str.contains(SMOKE_RUN_MARKER)
    keep &= ~df.hint_style.astype(str).isin(DROPPED_HINTS)
    return df[keep].copy()


def read_reliance(*, cueball: str | Path | None = None) -> pd.DataFrame:
    rel = pd.read_csv(P.reliance_path(cueball), usecols=REL_COLS)
    if rel.rollout_id.duplicated().any():
        raise SystemExit("question_reliance.csv: duplicate rollout_id")
    return rel.rename(columns={"rollout_id": "pair_id", "role": "sel_role", "reliance_label": "rel_label"})


def stability_bin(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=STABILITY_BINS[0]).astype("Int64")


MINUS1_RULES = ("own_category", "exclude")   # own_category = the definition; exclude = SENSITIVITY ONLY
KEPT = "ok"


def s3_status(df: pd.DataFrame, minus1: str = "own_category") -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Per rollout: (status, category, role).

    status   ok (kept) / unjudged / missing_role / minus1_excluded (only under the sensitivity rule "exclude")
    category "incoherent" for a v2 −1 rollout (whatever its role, with or without one), else the v2 role;
             None for excluded rows
    role     the raw v2 judge_role (object, None when missing)
    """
    if minus1 not in MINUS1_RULES:
        raise ValueError(f"minus1 must be one of {MINUS1_RULES}")
    lab = pd.to_numeric(df.judge_label_final, errors="coerce")
    role = df.judge_role.astype(object).where(df.judge_role.notna(), None)
    m1 = (lab == -1).fillna(False).to_numpy(bool)
    unjudged = lab.isna().to_numpy(bool)
    no_role = role.isna().to_numpy(bool)
    if minus1 == "exclude":
        status = np.select([m1, unjudged, no_role], ["minus1_excluded", "unjudged", "missing_role"], default=KEPT)
    else:
        status = np.select([m1, unjudged, no_role], [KEPT, "unjudged", "missing_role"], default=KEPT)
    category = pd.Series(np.where(m1, INCOHERENT, role.to_numpy(object)), index=df.index, dtype=object)
    category = category.where(status == KEPT, None)
    return status, category, role


def load_role_pool(checks: dict | None = None, minus1: str = "own_category", *,
                   cueball: str | Path | None = None) -> pd.DataFrame:
    """Every on-target rollout (original + re-rolls) of every selected pair, with its pair's reliance data and
    `s3_status` (only `ok` rows enter a denominator), `category` (role, or "incoherent" for v2 −1) and `role_v2`."""
    rm = pd.read_parquet(P.resample_manifest_path(cueball), columns=RM_COLS)
    rm = keep_scope(rm)
    rm = rm[rm.to_hint.fillna(False).astype(bool)].copy()
    prov = rm.provenance.astype(str)
    if not prov.isin(["hinted_once", "resample_k4"]).all():
        raise SystemExit(f"unexpected provenance values {sorted(prov.unique())}")
    rm["pair_id"] = np.where(prov == "hinted_once", rm.rollout_id.astype(str), rm.source_rollout_id.astype(str))
    rel = read_reliance(cueball=cueball)
    n = len(rm)
    rm = rm.merge(rel, on="pair_id", how="left", indicator="_rel")
    assert len(rm) == n, "reliance join multiplied rows"
    c = {
        "pool_rows_to_hint": n,
        "rows_without_reliance_pair": int((rm._rel != "both").sum()),
        "reliance_label_mismatch_manifest_vs_csv": int(
            (rm.reliance_label.astype(object).fillna("NA") != rm.rel_label.astype(object).fillna("NA")).sum()),
        "selection_role_mismatch_manifest_vs_csv": int(
            (rm.role.astype(object).fillna("NA") != rm.sel_role.astype(object).fillna("NA")).sum()),
        "pairs_with_k_n_not_4": int((pd.to_numeric(rm.k_n, errors="coerce") != 4).sum()),
    }
    if checks is not None:
        checks.update(c)
    log("  role-pool checks: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    if c["rows_without_reliance_pair"] or c["reliance_label_mismatch_manifest_vs_csv"]:
        raise SystemExit("role pool: reliance join failed a check (see above)")
    rm = rm.drop(columns=["_rel", "reliance_label", "role"])
    rm["s3_status"], rm["category"], rm["role_v2"] = s3_status(rm, minus1)
    # truncated rollouts are excluded everywhere (counted)
    trunc = rm.truncated.fillna(False).astype(bool).to_numpy()
    rm.loc[trunc, "s3_status"] = "truncated"
    rm.loc[trunc, "category"] = None
    rm["v2_label"] = pd.to_numeric(rm.judge_label_final, errors="coerce")
    rm["k"] = pd.to_numeric(rm.k_to_hint_count, errors="coerce").astype("Int64")
    rm["stability_bin"] = stability_bin(rm.baseline_stability)
    rm["is_original"] = prov.values == "hinted_once"
    return rm


def exclusion_counts(pool: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: rollouts, kept (of which v2 −1 = category incoherent), and each exclusion reason."""
    w = pd.DataFrame({k: pool[k] for k in by})
    st = pool.s3_status
    w["n_on_target"] = 1
    w["n_kept"] = (st == KEPT).astype(int)
    w["n_kept_minus1"] = ((st == KEPT) & (pool.category == INCOHERENT)).astype(int)
    w["n_excl_missing_role"] = (st == "missing_role").astype(int)
    w["n_excl_unjudged"] = (st == "unjudged").astype(int)
    w["n_excl_truncated"] = (st == "truncated").astype(int)
    w["n_excl_minus1_sensitivity"] = (st == "minus1_excluded").astype(int)
    cols = [c for c in w.columns if c.startswith("n_")]
    return w.groupby(by, observed=True)[cols].sum().reset_index()


def minus1_role_table(pool: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The paper's −1 × role counts: every v2 −1 rollout by the role the judge also emitted ("no_role" if none),
    plus the −1 total and the group's judged total. Truncated rollouts (excluded everywhere) are left out."""
    if "s3_status" in pool.columns:
        pool = pool[pool.s3_status != "truncated"]
    lab = pd.to_numeric(pool.judge_label_final, errors="coerce")
    x = pool[lab == -1]
    t = pd.crosstab([x[k] for k in by], x.role_v2.fillna("no_role"))
    t = t.reindex(columns=ROLES + ["no_role"], fill_value=0)
    t.columns = [f"minus1_role_{c}" for c in t.columns]
    t["n_minus1"] = t.sum(axis=1)
    judged = pool[lab.notna()].groupby(by, observed=True).size().rename("n_judged")
    t = t.join(judged, how="right").fillna(0).astype(int)
    t["share_minus1"] = t.n_minus1 / t.n_judged.replace(0, np.nan)
    return t.reset_index()


def add_wilson(df: pd.DataFrame, num: str, den: str, prefix: str = "") -> pd.DataFrame:
    df = df.copy()
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(df[num], df[den])]
    df[f"{prefix}rate"] = df[num] / df[den].replace(0, np.nan)
    df[f"{prefix}ci_lo"] = [x[0] for x in ci]
    df[f"{prefix}ci_hi"] = [x[1] for x in ci]
    return df


def exclusion_footer(excl: pd.DataFrame, model_names: dict[str, str]) -> str:
    """One compact line: kept v2 −1 (incoherent) and the excluded (no role / unjudged) counts per model."""
    parts = []
    for m in MODELS:
        r = excl[excl.subject_model == m]
        if r.empty:
            continue
        r = r.sum(numeric_only=True)
        s = f"{model_names.get(m, m)} −1 kept: {int(r.n_kept_minus1):,}; excl. no role: {int(r.n_excl_missing_role):,}"
        if int(r.n_excl_unjudged):
            s += f", unjudged: {int(r.n_excl_unjudged):,}"
        if int(r.get("n_excl_truncated", 0)):
            s += f", truncated: {int(r.n_excl_truncated):,}"
        if int(r.get("n_excl_minus1_sensitivity", 0)):
            s += f", −1 (sensitivity): {int(r.n_excl_minus1_sensitivity):,}"
        parts.append(s)
    return " | ".join(parts)


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """A plain markdown table (no tabulate dependency)."""
    cols = cols or list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df[cols].itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out
