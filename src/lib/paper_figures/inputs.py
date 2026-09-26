"""Prepare paper-analysis rows from a released CueBall tree.

All source artifacts are read-only. The shared SSP loader supplies the original
sample-zero parsing, joins and eligibility rules; the stored re-roll outcomes
and judge verdicts supply the fresh-sample counts. No generation or judging is
performed, and no cached analysis tables are required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib import selection
from src.scripts.visualizations.common import ssp_common as sc

from .common import MODELS, N_OPTIONS


ROLE_COLUMNS = [
    "rollout_id", "judge_label_final", "judge_label_final_model", "judge_role",
    "judge_confidence", "baseline_hint_votes", "baseline_n_samples", "split",
    "trace_token_len", "max_tokens_used",
]
REROLL_COLUMNS = [
    "rollout_id", "source_rollout_id", "subject_model", "run", "dataset", "original_index",
    "hint_style", "case", "provenance", "reliance_label", "role", "to_hint", "model_answer",
    "truncated", "judge_label_final", "judge_role", "exclude_reason", "sample_seed",
    "trace_token_len", "max_tokens_used", "split", "target_option", "baseline_modal_answer",
]
LONG_COLUMNS = [
    "rollout_id", "source_rollout_id", "subject_model", "run", "dataset", "original_index",
    "hint_style", "case", "sample_seed", "model_answer", "truncated_b", "answered_ok",
    "unparsed", "hit", "hit_any", "off_target", "modal_hit", "lab", "judge_role",
    "exclude_reason", "trace_token_len", "max_tokens_used", "over16k", "reliance_label",
    "role", "split", "hit_rsp_pool", "hit_rsp_unf",
]


def _objects(frame: pd.DataFrame) -> pd.DataFrame:
    """Use consistent nullable-object semantics for legacy and current parquet."""
    for column in frame:
        if str(frame[column].dtype) in ("string", "boolean"):
            frame[column] = frame[column].astype(object)
    return frame


def _answer(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text not in ("", "NAN", "NONE", "<NA>") else None


def _baseline_samples(pairs: pd.DataFrame, cueball: Path) -> pd.DataFrame:
    columns = ["original_index", "question", "choices", "sample_answers", "baseline_status"]
    rows = []
    for source in sc.sources(cueball=cueball, models=MODELS):
        cells = pairs.loc[pairs.source_csv == source["source_csv"], ["subject_model", "run"]].drop_duplicates()
        if len(cells) != 1:
            raise ValueError(f"Expected one model/run for {source['source_csv']}, found {len(cells)}")
        model, run = cells.iloc[0]
        meta = json.loads(source["baseline_meta"].read_text())
        for chunk in pd.read_csv(source["baseline_csv"], usecols=columns, chunksize=400,
                                 dtype=str, keep_default_na=False):
            for row in chunk.to_dict("records"):
                answers = [_answer(a) for a in json.loads(row["sample_answers"] or "[]")]
                signature = hashlib.sha256(
                    (row["question"].strip() + "\x1f" + row["choices"].strip()).encode()
                ).hexdigest()[:16]
                rows.append({
                    "subject_model": model, "run": run, "original_index": int(row["original_index"]),
                    "question_sig": signature, "answers": answers,
                    "n_answered": sum(a is not None for a in answers),
                    "bl_status": row["baseline_status"],
                    "bl_top_p": meta.get("top_p"), "bl_top_k": meta.get("top_k"),
                })
    return pd.DataFrame(rows)


def _rerolls(cueball: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_parquet(cueball / "resample/resample_manifest.parquet", columns=REROLL_COLUMNS)
    rows = _objects(rows[(rows.provenance.astype(str) == selection.RESAMPLE_K4)
                         & rows.subject_model.astype(str).isin(MODELS)].copy())
    if rows.rollout_id.duplicated().any():
        raise ValueError("Duplicate re-roll rollout_id")
    rows["truncated_b"] = rows.truncated.fillna(False).astype(bool)
    rows["hit"] = rows.to_hint.fillna(False).astype(bool)
    answer, target, modal = (sc._norm(rows[c]) for c in
                             ("model_answer", "target_option", "baseline_modal_answer"))
    rows["answered"] = answer.notna()
    rows["hit_any"] = (answer == target) & answer.notna()
    rows["hit_trunc"] = rows.hit_any & rows.truncated_b
    rows["answered_ok"] = rows.answered & ~rows.truncated_b
    rows["unparsed"] = ~rows.truncated_b & ~rows.answered
    rows["off_target"] = rows.answered_ok & ~rows.hit_any
    rows["modal_hit"] = rows.answered & (answer == modal)
    label = pd.to_numeric(rows.judge_label_final, errors="coerce")
    rows["lab"] = label
    rows["hit_judged"] = rows.hit & label.isin([0, 1, -1])
    rows["hit_unf"] = rows.hit & (label == 0)
    rows["hit_faith"] = rows.hit & (label == 1)
    rows["hit_inc"] = rows.hit & (label == -1)
    rows["hit_nojudge"] = rows.hit & label.isna()
    rows["over16k"] = pd.to_numeric(rows.trace_token_len, errors="coerce") > 16384
    rows["hit_over16k"] = rows.hit & rows.over16k
    # Reuse the named binary-judged dataset predicate. The paper additionally
    # includes incoherent verdicts when incoherence is their sole exclusion.
    pool = selection.mask("dataset_B", rows)
    incoherent = (label == -1) & (rows.exclude_reason.astype(object) == "incoherent")
    if incoherent.any():
        admitted = rows.loc[incoherent].copy()
        admitted["judge_label_final"] = 0
        admitted["exclude_reason"] = None
        pool.loc[incoherent] = selection.mask("dataset_B", admitted)
    rows["hit_rsp_pool"] = pool
    rows["hit_rsp_unf"] = pool & label.isin([0, -1])
    rows["hit_rsp_inc"] = pool & (label == -1)
    aggregates = rows.groupby("source_rollout_id").agg(
        k_n=("rollout_id", "size"), k_hit=("hit", "sum"), k_hit_any=("hit_any", "sum"),
        k_hit_trunc=("hit_trunc", "sum"), k_trunc=("truncated_b", "sum"),
        k_answered_ok=("answered_ok", "sum"), k_unparsed=("unparsed", "sum"),
        k_off=("off_target", "sum"), k_modal=("modal_hit", "sum"),
        k_hit_judged=("hit_judged", "sum"), k_hit_unf=("hit_unf", "sum"),
        k_hit_faith=("hit_faith", "sum"), k_hit_inc=("hit_inc", "sum"),
        k_hit_nojudge=("hit_nojudge", "sum"), k_over16k=("over16k", "sum"),
        k_hit_over16k=("hit_over16k", "sum"), k_rsp_pool=("hit_rsp_pool", "sum"),
        k_rsp_unf=("hit_rsp_unf", "sum"), k_rsp_inc=("hit_rsp_inc", "sum"),
        rr_reliance_label=("reliance_label", "first"), rr_role=("role", "first"),
    ).reset_index().rename(columns={"source_rollout_id": "rollout_id"})
    return rows[LONG_COLUMNS], aggregates


def prepare(cueball: Path, output: Path) -> dict[str, Path]:
    """Write the two row-level analysis tables to ``output``, outside the release.

    ``cueball`` is the released tree containing ``baselines``, ``binary_judge``,
    ``hinted_rollouts`` and ``resample``. Existing judge verdicts are consumed
    verbatim; callers must supply the release's complete source data.
    """
    cueball, output = Path(cueball).resolve(), Path(output).resolve()
    if output.is_relative_to(cueball):
        raise ValueError("The released data tree is read-only; choose an output directory outside it")
    pairs, _ = sc.build_rows(cueball=cueball, models=MODELS, collect_metadata=False)
    pairs = _objects(pairs).rename(columns={
        "k_n": "qr_k_n", "k_truncated": "qr_k_truncated", "k_to_hint_count": "qr_k_to_hint_count",
    })
    original = _objects(pd.read_parquet(cueball / "hinted_rollouts/rollout_manifest.parquet",
                                        columns=ROLE_COLUMNS)).rename(columns={
        "judge_label_final": "role_label", "judge_label_final_model": "role_label_model",
        "judge_role": "role_orig", "judge_confidence": "role_confidence",
    })
    pairs = pairs.merge(original, on="rollout_id", how="left", validate="one_to_one")
    baseline = _baseline_samples(pairs, cueball)
    pairs = pairs.merge(baseline, on=["subject_model", "run", "original_index"],
                        how="left", validate="many_to_one")
    if pairs.answers.isna().any():
        raise ValueError("Some rollout rows have no baseline samples")
    target = sc._norm(pairs.target_option)
    pairs["bl_n_target"] = [sum(a == t for a in answers if a is not None)
                            for answers, t in zip(pairs.answers, target)]
    pairs["bl_n_answered"] = pairs.n_answered.astype(int)
    pairs["bl_n_samples"] = pairs.answers.map(len)
    if (pairs.bl_n_target != pd.to_numeric(pairs.baseline_hint_votes)).any():
        raise ValueError("Baseline target counts disagree with the rollout manifest")
    pairs = pairs.drop(columns=["answers", "n_answered"])
    pairs["qkey"] = pairs.dataset.astype(str) + ":" + pairs.original_index.astype(int).astype(str)
    rerolls, aggregates = _rerolls(cueball)
    pairs = pairs.merge(aggregates, on="rollout_id", how="left", validate="one_to_one")
    pairs["k_missing"] = pairs.k_trunc + pairs.k_unparsed
    pairs["k_eff"] = pairs.k_answered_ok
    sampled = pairs.k_n.notna()
    if ((pairs.k_hit != pairs.qr_k_to_hint_count) & sampled).any():
        raise ValueError("Re-roll target counts disagree with question_reliance.csv")
    if ((pairs.k_trunc != pairs.qr_k_truncated) & sampled).any():
        raise ValueError("Re-roll truncation counts disagree with question_reliance.csv")
    if ((pairs.rr_reliance_label.fillna("") != pairs.reliance_label.fillna("")) & sampled).any():
        raise ValueError("Re-roll reliance labels disagree with question_reliance.csv")
    label = pd.to_numeric(pairs.role_label, errors="coerce").astype(float)
    pairs["role_label"] = label
    role = pairs.role_orig.astype(object).where(pairs.role_orig.notna(), "missing_role")
    pairs["cat_orig"] = np.where(label == -1, "incoherent", np.where(label.isna(), "unjudged", role))
    pairs["orig_role_covered"] = pairs.cat_orig.isin(
        ["credited", "verification_only", "rejected", "neutral", "none", "incoherent"])
    for count in (1, 2, 3):
        pairs[f"persist{count}"] = pairs.k_hit >= count
    pairs["b0_is_answered"] = pairs.b0.notna()
    pairs["bl_frac_target"] = pairs.bl_n_target / pairs.bl_n_answered.replace(0, np.nan)
    pairs["fresh_frac_target"] = pairs.k_hit / pairs.k_eff.replace(0, np.nan)
    pairs["n_options"] = pairs.dataset.map(N_OPTIONS)
    if pairs.n_options.isna().any():
        raise ValueError("Unknown dataset option count")
    pairs = pairs.drop(columns=["bin_case", "bin_target", "bin_groundtruth", "bin_final_answer",
                                "bl_groundtruth", "bl_modal_answer", "source_csv", "binary_csv",
                                "baseline_csv"], errors="ignore")
    output.mkdir(parents=True, exist_ok=True)
    paths = {"pairs": output / "pairs_master.parquet", "rerolls": output / "rerolls_long.parquet"}
    pairs.to_parquet(paths["pairs"], index=False)
    rerolls.to_parquet(paths["rerolls"], index=False)
    return paths
