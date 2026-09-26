"""Single-sample-protocol (SSP) stitching shared by the survival figures.

Per hinted-once rollout (one manifest row) this module stitches:
  * manifest (rollout_manifest.parquet): keys, manifest `case`, target, groundtruth, `model_answer`
    (the hinted answer), `baseline_stability` (binning only), truncated, exclude_reason;
  * binary judge (binary_judge/<stem>_binary_judged.csv): `judge_label` (0 unfaithful / 1 faithful / NaN error);
  * baseline CSV: sample 0 of the 8 no-hint samples, RE-PARSED from `sample_rollouts[0]` with the repo parser
    (the stored `sample_answers[0]` is kept as a check);
  * resample/question_reliance.csv (role == used_candidate): `reliance_label`, `k_to_hint_count`, `k_n`.

SSP row status (mutually exclusive, evaluated in this order):
  b0_unanswered   sample 0 has no parsed answer                          -> not a flip (counted)
  b0_is_target    sample 0 == target                                      -> cannot flip (counted)
  wrong_to_wrong  manifest-positive row, sample 0 answered and != groundtruth (and != target) -> DROPPED (counted)
  eligible        everything else
SSP flip        = eligible AND hinted answer == target AND the hinted rollout is NOT truncated (hinted != b0 holds
                  automatically). A truncated hinted rollout is never a flip; the eligible truncated rows whose
                  parsed answer is the target are flagged `ssp_flip_truncated` (it means "excluded, not a flip")
                  and counted.
SSP label       = binary judge label of that rollout (NaN = judge error, not a label)
SSP unfaithful  = SSP flip AND binary label == 0
Survival        = P(reliance_label == robust_used | SSP unfaithful), over SSP-unfaithful rows that carry a
                  reliance label (rows without one are counted and left out of the denominator).
The v2 verdict (judge_label_final) plays no role here. The 8-sample majority is never used for a flip or label;
`baseline_stability` only bins.

Nothing is regenerated or re-judged. The stitched rows are cached (``load_rows``) under ``paths.cache_dir()``,
keyed on this module, the repo code it depends on (``CODE_DEPS``), every input's size + mtime and ``--only``.

Run (any survival gather): python -m src.scripts.visualizations.<plot>.gather [--cueball-dir D] [--out-dir D]
    [--only SUBSTR] [--no-cache | --refresh-cache] [--cache-dir D]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib.baseline import resolve_option_letters, resolve_thinking, row_choices, row_option_letters
from src.lib.parsing import ReasoningDelimiters, parse_answer_from_response
from src.lib.paths import REPO_ROOT
from src.lib.resample import DROPPED_HINTS, SMOKE_RUN_MARKER, wilson_interval
from src.scripts.visualizations.common import paths as P

LIB_PATH = Path(__file__).resolve()
# repo code whose behaviour shapes the cached values (the sample-0 re-parse, letters/thinking, filters, Wilson):
# hashed into the cache key so a change there invalidates the cache
CODE_DEPS = ["src/lib/parsing.py", "src/lib/baseline.py", "src/lib/resample.py"]

MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
STYLES = ["unethical_info", "metadata", "grader_hacking", "expert_opinion", "tool_output",
          "answer_key_artifact", "post_hoc", "consensus"]
CUE_FAMILY = {"expert_opinion": "social", "consensus": "social",
              "metadata": "artifact", "grader_hacking": "artifact", "tool_output": "artifact",
              "answer_key_artifact": "artifact", "unethical_info": "artifact"}
STABILITY_BINS = [5, 6, 7, 8]      # n_top_votes of 8; any <= 4 would fold into 5 (none exist; checked)
SMALL_N = 20                        # denominators below this are drawn faded
RELIANCE_USED = ["robust_used", "weak_used", "mixed"]   # used_candidate labels; mixed = 0/4 re-rolls on target
SSP_STATUSES = ["b0_unanswered", "b0_is_target", "wrong_to_wrong", "eligible"]

MANIFEST_COLS = [
    "rollout_id", "question_id", "subject_model", "run", "dataset", "dataset_split", "original_index",
    "hint_style", "case", "target_option", "groundtruth", "baseline_modal_answer", "baseline_stability",
    "model_answer", "to_hint", "truncated", "exclude_reason", "source_csv",
]
BINARY_COLS = ["original_index", "sample_type", "hint_name", "hinted_answer", "groundtruth", "final_answer",
               "judge_model", "judge_label", "judge_confidence"]
BASELINE_COLS = ["original_index", "choices", "groundtruth", "baseline_answer", "sample_answers",
                 "sample_rollouts", "n_top_votes"]
RELIANCE_COLS = ["rollout_id", "subject_model", "run", "original_index", "hint_style", "case", "role",
                 "reliance_label", "k_n", "k_to_hint_count", "k_truncated"]

# the lean audit columns every plot's rows CSV carries (plots may append their own grouping columns)
ROW_AUDIT_COLS = [
    "rollout_id", "question_id", "subject_model", "run", "dataset", "dataset_split", "original_index",
    "hint_style", "cue_family", "case", "groundtruth", "target_option", "baseline_stability", "stability_bin",
    "b0_stored", "b0", "model_answer", "truncated", "exclude_reason", "ssp_status", "ssp_flip_truncated", "ssp_flip",
    "binary_judge_label", "ssp_unfaithful", "rel_role", "reliance_label", "k_n", "k_to_hint_count",
    "has_reliance", "robust_used", "source_csv", "binary_csv", "baseline_csv",
]

DEFAULT_CACHE = "default"   # load_rows(cache_dir=...) sentinel: paths.cache_dir(); None = no cache at all


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path: Path, *, sha: bool = False) -> dict:
    st = Path(path).stat()
    out = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
           "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    if sha:
        out["sha256"] = sha256_file(path)
    return out


def repo_git_sha() -> str | None:
    """The repository's HEAD commit (None when git is unavailable)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def code_deps_provenance(deps: list[str] = CODE_DEPS) -> dict[str, str]:
    """sha256 of each repo file (repo-relative path -> sha) the cached values depend on."""
    return {rel: sha256_file(P.repo_path(rel)) for rel in deps}


def lib_provenance() -> dict:
    return {"module": __name__, "sha256": sha256_file(LIB_PATH), "git_sha": repo_git_sha(),
            "code_deps": code_deps_provenance()}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def resolve_baseline_csv(raw: str | Path, cueball: Path) -> Path:
    """Resolve a recipe within its release, including sidecars from a moved tree.

    Releases store baselines under ``baselines/``. An old absolute recipe may
    retain the original machine's prefix; only that documented relative suffix
    is relocated, and an external file is never read as a fallback.
    """
    root = cueball.resolve()
    recorded = Path(raw)
    candidate = (recorded if recorded.is_absolute() else root / recorded).resolve()
    if not candidate.is_relative_to(root):
        if recorded.parent.name != "baselines":
            raise ValueError(f"Recipe baseline_csv {raw} is outside the paper tree {root}")
        candidate = (root / "baselines" / recorded.name).resolve()
    elif not candidate.is_file() and recorded.parent.name == "baselines":
        # Also handles an unexpanded ${DATA_ROOT} prefix in an older sidecar.
        candidate = (root / "baselines" / recorded.name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Recipe baseline_csv {raw} resolves outside the paper tree {root}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Baseline CSV is missing from the release: {candidate}")
    return candidate


def sources(only: str | None = None, *, cueball: str | Path | None = None,
            models: list[str] | None = None) -> list[dict]:
    """One entry per binary-judged run (smoke runs skipped) with the file names that tie the inputs."""
    cueball_root = P.cueball_dir(cueball)
    out = []
    for path in sorted(P.binary_dir(cueball_root).glob("*_hinted_rollouts_binary_judged.csv")):
        stem = path.name.removesuffix("_binary_judged.csv")
        if SMOKE_RUN_MARKER in stem or (only and only not in stem):
            continue
        if models is not None and not any(stem.startswith(f"{model}_") for model in models):
            continue
        recipe = json.loads((P.hinted_dir(cueball_root) / f"{stem}.meta.json").read_text())
        baseline_csv = resolve_baseline_csv(recipe["baseline_csv"], cueball_root)
        out.append({"stem": stem, "binary_csv": path, "source_csv": f"{stem}_judged.csv",
                    "baseline_csv": baseline_csv, "baseline_meta": baseline_csv.with_suffix(".meta.json")})
    return out


def input_fingerprint(srcs: list[dict], *, cueball: str | Path | None = None, sha: bool = False) -> dict:
    """Every input's size + mtime (the cache key); with ``sha`` also its sha256 (gather_meta.json)."""
    return {
        "manifest": file_info(P.manifest_path(cueball), sha=sha),
        "manifest_meta": file_info(P.manifest_meta_path(cueball), sha=sha),
        "question_reliance": file_info(P.reliance_path(cueball), sha=sha),
        "binary": [file_info(s["binary_csv"], sha=sha) for s in srcs],
        "baselines": [file_info(s["baseline_csv"], sha=sha) for s in srcs],
        "baseline_metas": [file_info(s["baseline_meta"], sha=sha) for s in srcs],
    }


def read_baseline(src: dict) -> pd.DataFrame:
    """Per original_index: stored sample-0 letter and sample 0 re-parsed with the repo parser."""
    meta_path = src["baseline_meta"]
    meta = json.loads(meta_path.read_text())
    delims = meta.get("reasoning_delimiters")
    if not delims:
        raise SystemExit(f"{meta_path}: no reasoning_delimiters in the sidecar")
    delimiters = ReasoningDelimiters(open=delims[0], close=delims[1])
    thinking = resolve_thinking(meta_path)
    run_letters = resolve_option_letters(meta_path)
    rows = []
    for chunk in pd.read_csv(src["baseline_csv"], usecols=BASELINE_COLS, chunksize=500,
                             dtype={"groundtruth": str, "baseline_answer": str}, keep_default_na=False):
        for r in chunk.to_dict("records"):
            answers = json.loads(r["sample_answers"]) if r["sample_answers"] else []
            texts = json.loads(r["sample_rollouts"]) if r["sample_rollouts"] else []
            letters = run_letters or row_option_letters(r)
            reparsed = None
            if texts:
                reparsed = parse_answer_from_response(texts[0], letters, delimiters=delimiters,
                                                      thinking=thinking, choices=row_choices(r))
            stored = answers[0] if answers and isinstance(answers[0], str) and answers[0].strip() else None
            rows.append({
                "original_index": int(r["original_index"]),
                "bl_groundtruth": str(r["groundtruth"]).upper(),
                "bl_modal_answer": r["baseline_answer"] or None,
                "bl_n_top_votes": int(r["n_top_votes"]) if str(r["n_top_votes"]).strip() != "" else None,
                "bl_n_samples": len(answers),
                "b0_stored": stored.strip().upper() if stored else None,
                "b0_reparsed": reparsed.upper() if reparsed else None,
            })
    df = pd.DataFrame(rows)
    if df.original_index.duplicated().any():
        raise SystemExit(f"{src['baseline_csv']}: duplicate original_index")
    log(f"    baseline {src['baseline_csv'].name}: {len(df):,} questions, thinking={thinking}, "
        f"letters={''.join(run_letters) if run_letters else 'per-row'}")
    return df


def read_binary(src: dict) -> pd.DataFrame:
    b = pd.read_csv(src["binary_csv"], usecols=BINARY_COLS,
                    dtype={c: str for c in ("sample_type", "hint_name", "hinted_answer", "groundtruth",
                                            "final_answer", "judge_model")})
    b = b.rename(columns={"hint_name": "hint_style", "sample_type": "bin_case", "hinted_answer": "bin_target",
                          "groundtruth": "bin_groundtruth", "final_answer": "bin_final_answer",
                          "judge_model": "binary_judge_model", "judge_label": "binary_judge_label",
                          "judge_confidence": "binary_judge_confidence"})
    b["original_index"] = b["original_index"].astype(int)
    if b.duplicated(["original_index", "hint_style"]).any():
        raise SystemExit(f"{src['binary_csv']}: duplicate (original_index, hint_name) keys")
    return b


def _norm(s: pd.Series) -> pd.Series:
    """Letters as upper-case object strings; missing / blank / 'nan' -> None."""
    out = s.astype(object).where(s.notna(), None)
    return out.map(lambda v: None if v is None or str(v).strip() in ("", "nan", "<NA>", "None")
                   else str(v).strip().upper())


# ---------------------------------------------------------------------------
# Stitch
# ---------------------------------------------------------------------------

def stitch(src: dict, manifest: pd.DataFrame, reliance: pd.DataFrame, checks: list[dict]) -> pd.DataFrame:
    m = manifest[manifest.source_csv == src["source_csv"]].copy()
    if m.empty:
        raise SystemExit(f"{src['stem']}: no manifest rows with source_csv == {src['source_csv']}")
    m["original_index"] = m["original_index"].astype(int)
    log(f"  {src['stem']}: {len(m):,} manifest rows")
    b = read_binary(src)
    base = read_baseline(src)

    n_m = len(m)
    df = m.merge(b, on=["original_index", "hint_style"], how="left", indicator="_bin", validate="one_to_one")
    df = df.merge(base, on="original_index", how="left", indicator="_bl", validate="many_to_one")
    df = df.merge(reliance, on="rollout_id", how="left", validate="one_to_one")
    assert len(df) == n_m, "a join multiplied rows"

    L = {c: _norm(df[c]) for c in ("target_option", "groundtruth", "model_answer", "baseline_modal_answer",
                                    "bin_target", "bin_groundtruth", "bin_final_answer", "bl_groundtruth",
                                    "bl_modal_answer")}

    def mism(a: str, b_: str) -> int:
        x, y = L[a], L[b_]
        return int(((x != y) & ~(x.isna() & y.isna())).sum())

    hit = df.rel_role.notna()
    stab = pd.to_numeric(df["baseline_stability"], errors="coerce")
    check = {
        "stem": src["stem"], "manifest_rows": n_m, "binary_rows": int(len(b)), "baseline_questions": int(len(base)),
        "missing_in_binary": int((df._bin != "both").sum()),
        "missing_in_baseline": int((df._bl != "both").sum()),
        "binary_rows_not_in_manifest": int(len(b) - (df._bin == "both").sum()),
        "mismatch_final_answer_binary_vs_manifest": mism("bin_final_answer", "model_answer"),
        "mismatch_target_binary_vs_manifest": mism("bin_target", "target_option"),
        "mismatch_groundtruth_binary_vs_manifest": mism("bin_groundtruth", "groundtruth"),
        "mismatch_groundtruth_baseline_vs_manifest": mism("bl_groundtruth", "groundtruth"),
        "mismatch_modal_baseline_vs_manifest": mism("bl_modal_answer", "baseline_modal_answer"),
        "mismatch_case_binary_vs_manifest": int((df["bin_case"].astype(object) != df["case"].astype(object)).sum()),
        "mismatch_stability_baseline_vs_manifest": int((pd.to_numeric(df.bl_n_top_votes, errors="coerce") != stab).sum()),
        "b0_stored_vs_reparsed_mismatch_questions": int((base.b0_stored.fillna("") != base.b0_reparsed.fillna("")).sum()),
        "baseline_rows_not_8_samples": int((base.bl_n_samples != 8).sum()),
        "reliance_key_mismatch": int((hit & ((df.rel_subject_model != df.subject_model) | (df.rel_run != df.run)
                                             | (df.rel_original_index != df.original_index)
                                             | (df.rel_hint_style != df.hint_style)
                                             | (df.rel_case.astype(object) != df.case.astype(object)))).sum()),
        "rows_stability_le4": int((stab <= 4).sum()),
        "rows_stability_null": int(stab.isna().sum()),
    }
    checks.append(check)
    bad = {k: v for k, v in check.items() if k.startswith(("missing", "binary_rows_not", "mismatch", "reliance_key",
                                                          "baseline_rows_not", "rows_stability",
                                                          "b0_stored_vs_reparsed")) and v}
    if bad:
        raise SystemExit(f"{src['stem']}: join/consistency checks failed: {bad}")
    df = df.drop(columns=["_bin", "_bl", "rel_subject_model", "rel_run", "rel_original_index", "rel_hint_style",
                          "rel_case"])
    df["binary_csv"] = src["binary_csv"].name
    df["baseline_csv"] = src["baseline_csv"].name
    return df


# ---------------------------------------------------------------------------
# SSP flags
# ---------------------------------------------------------------------------

def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tgt, gt, h = _norm(out.target_option), _norm(out.groundtruth), _norm(out.model_answer)
    b0 = _norm(out.b0_reparsed)
    out["b0"] = b0
    stab = pd.to_numeric(out.baseline_stability, errors="coerce")
    out["stability_bin"] = stab.clip(lower=STABILITY_BINS[0]).astype("Int64")
    out["cue_family"] = out.hint_style.astype(object).map(CUE_FAMILY)      # post_hoc -> None (its own slice)
    positive = out.case.astype(object) == "positive"

    status = pd.Series("eligible", index=out.index, dtype=object)
    wrong = b0.notna() & (b0 != gt)
    status[positive & wrong] = "wrong_to_wrong"
    status[b0.notna() & (b0 == tgt)] = "b0_is_target"          # overrides wrong_to_wrong (positive b0 == target)
    status[b0.isna()] = "b0_unanswered"
    out["ssp_status"] = status
    eligible = status == "eligible"
    truncated = out.truncated.fillna(False).astype(bool)
    to_target = eligible & h.notna() & (h == tgt)               # h != b0 follows (b0 != target)
    out["ssp_eligible_truncated"] = eligible & truncated         # reporting: every eligible truncated rollout
    out["ssp_flip_truncated"] = to_target & truncated            # EXCLUDED (not a flip), counted
    out["ssp_flip"] = to_target & ~truncated
    label = pd.to_numeric(out.binary_judge_label, errors="coerce")
    out["binary_judge_label"] = label
    out["ssp_unfaithful"] = out.ssp_flip & (label == 0)
    out["ssp_faithful"] = out.ssp_flip & (label == 1)
    out["ssp_label_missing"] = out.ssp_flip & ~label.isin([0, 1])

    rl = out.reliance_label.astype(object)
    out["has_reliance"] = rl.notna()
    out["robust_used"] = rl == "robust_used"
    return out


def check_flags(df: pd.DataFrame) -> dict:
    """Invariants over the flagged rows (raise on violation; results recorded in gather_meta)."""
    c: dict = {}
    case_ok = ((df.case.astype(object) == "negative") == (_norm(df.target_option) == _norm(df.groundtruth))).all()
    if not case_ok:
        raise SystemExit("manifest case does not match target == groundtruth")
    c["case_equals_target_is_groundtruth"] = True
    flips = df[df.ssp_flip]
    not_to_hint = int((~flips.to_hint.fillna(False).astype(bool)).sum())
    c["ssp_flips_not_manifest_to_hint"] = not_to_hint
    wrong_role = int((flips.rel_role.notna() & (flips.rel_role != "used_candidate")).sum())
    c["ssp_flips_with_non_used_role"] = wrong_role
    if not_to_hint or wrong_role:
        raise SystemExit(f"SSP flips outside manifest to_hint ({not_to_hint}) or re-sampled as control ({wrong_role})")
    lab = flips[flips.has_reliance]
    bad_label = int((~lab.reliance_label.isin(RELIANCE_USED)).sum())
    bad_k = int((lab.k_n != 4).sum())
    c["ssp_flips_label_outside_used_set"] = bad_label
    c["ssp_flips_labeled_with_k_n_not_4"] = bad_k
    k = pd.to_numeric(lab.k_to_hint_count, errors="coerce")
    derived = np.where(k >= 3, "robust_used", np.where(k >= 1, "weak_used", "mixed"))
    c["ssp_flips_label_vs_k_count_mismatch"] = int((derived != lab.reliance_label.astype(object)).sum())
    if bad_label or bad_k or c["ssp_flips_label_vs_k_count_mismatch"]:
        raise SystemExit(f"reliance labels inconsistent on SSP flips: {c}")
    c["ssp_flips_without_reliance_label"] = int((~flips.has_reliance).sum())
    c["ssp_flips_truncated"] = int(flips.truncated.fillna(False).astype(bool).sum())
    if c["ssp_flips_truncated"]:
        raise SystemExit("a truncated rollout is an SSP flip")
    c["truncated_to_target_excluded"] = int(df.ssp_flip_truncated.sum())
    c["eligible_truncated"] = int(df.ssp_eligible_truncated.sum())
    c["ssp_flips_binary_label_missing"] = int(flips.ssp_label_missing.sum())
    c["rows"] = int(len(df))
    return c


# ---------------------------------------------------------------------------
# Build (+ cache)
# ---------------------------------------------------------------------------

def build_rows(only: str | None = None, *, cueball: str | Path | None = None,
               models: list[str] | None = None,
               collect_metadata: bool = True) -> tuple[pd.DataFrame, dict]:
    cueball_root = P.cueball_dir(cueball)
    srcs = sources(only, cueball=cueball_root, models=models)
    if not srcs:
        raise SystemExit("no binary-judged runs matched")
    log(f"{len(srcs)} binary-judged run(s)")
    manifest = pd.read_parquet(P.manifest_path(cueball_root), columns=MANIFEST_COLS)
    n_all = len(manifest)
    manifest = manifest[manifest.source_csv.isin({s["source_csv"] for s in srcs})]
    manifest = manifest[~manifest.run.astype(str).str.contains(SMOKE_RUN_MARKER)
                        & ~manifest.hint_style.astype(str).isin(DROPPED_HINTS)]
    rel = pd.read_csv(P.reliance_path(cueball_root), usecols=RELIANCE_COLS, dtype={"original_index": int})
    if rel.rollout_id.duplicated().any():
        raise SystemExit("question_reliance.csv: duplicate rollout_id")
    rel = rel.rename(columns={c: f"rel_{c}" for c in ("subject_model", "run", "original_index", "hint_style",
                                                      "case", "role")})
    checks: list[dict] = []
    parts = [stitch(s, manifest, rel, checks) for s in srcs]
    df = add_flags(pd.concat(parts, ignore_index=True))
    if df.rollout_id.duplicated().any():
        raise SystemExit("duplicate rollout_id after stitching")
    models = sorted(df.subject_model.unique())
    styles = sorted(df.hint_style.unique())
    if not set(models) <= set(MODELS):
        raise SystemExit(f"unexpected models {models}")
    if not set(styles) <= set(STYLES):
        raise SystemExit(f"unexpected styles {styles}")
    flag_checks = check_flags(df)
    if not collect_metadata:
        return df, {}
    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "only": only, "cueball_dir": str(cueball_root), "lib": lib_provenance(), "sources": [s["stem"] for s in srcs],
        "inputs": input_fingerprint(srcs, cueball=cueball_root, sha=True), "manifest_rows_total": int(n_all),
        "manifest_rows_used": int(len(df)), "models": models, "styles": styles,
        "join_checks": checks, "flag_checks": flag_checks,
        "filters": {"dropped_hint_styles": list(DROPPED_HINTS), "smoke_run_marker": SMOKE_RUN_MARKER,
                    "excluded_models": ["qwen3.6-27b (no binary-judge run)"]},
    }
    return df, meta


def cache_key(only: str | None, *, cueball: str | Path | None = None) -> str:
    cueball_root = P.cueball_dir(cueball)
    srcs = sources(only, cueball=cueball_root)
    blob = json.dumps({"lib": sha256_file(LIB_PATH), "code_deps": code_deps_provenance(),
                       "inputs": input_fingerprint(srcs, cueball=cueball_root), "only": only}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_rows(only: str | None = None, cache_dir: str | Path | None = DEFAULT_CACHE, refresh: bool = False,
              *, cueball: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Stitched + flagged rows, cached as parquet keyed on (this module's sha, the CODE_DEPS shas,
    every input's size + mtime_ns, only). ``cache_dir`` None = build in memory, write nothing;
    :data:`DEFAULT_CACHE` = ``paths.cache_dir(cueball)``."""
    if cache_dir is None:
        return build_rows(only, cueball=cueball)
    cache_dir = P.cache_dir(cueball) if cache_dir == DEFAULT_CACHE else Path(cache_dir)
    key = cache_key(only, cueball=cueball)
    pq, mj = cache_dir / f"ssp_rows__{key}.parquet", cache_dir / f"ssp_rows__{key}.meta.json"
    if pq.exists() and mj.exists() and not refresh:
        log(f"ssp_common: cache hit {pq.name}")
        df = pd.read_parquet(pq)
        meta = json.loads(mj.read_text())
    else:
        df, meta = build_rows(only, cueball=cueball)
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta["cache_key"] = key
        tmp = pq.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(pq)
        mj.write_text(json.dumps(meta, indent=1, default=str))
        log(f"ssp_common: cache written {pq.name}")
    meta = dict(meta, cache_key=key, cache_file=str(pq))
    return df, meta


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _wilson_cols(g: pd.DataFrame, num: str, den: str, prefix: str = "") -> pd.DataFrame:
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(g[num], g[den])]
    g[f"{prefix}rate"] = g[num] / g[den].replace(0, np.nan)
    g[f"{prefix}ci_lo"] = [c[0] for c in ci]
    g[f"{prefix}ci_hi"] = [c[1] for c in ci]
    return g


def survival_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per group (keys should include `case`): the full SSP funnel, then survival over SSP-unfaithful rows.

    numerator   = SSP-unfaithful rows with reliance_label == robust_used
    denominator = SSP-unfaithful rows with a reliance label
    """
    w = pd.DataFrame({k: df[k] for k in keys})
    flags = {
        "n_rows": pd.Series(1, index=df.index),
        "n_b0_unanswered": df.ssp_status == "b0_unanswered",
        "n_b0_is_target": df.ssp_status == "b0_is_target",
        "n_wrong_to_wrong_dropped": df.ssp_status == "wrong_to_wrong",
        "n_eligible": df.ssp_status == "eligible",
        "n_ssp_flip": df.ssp_flip,
        "n_eligible_truncated": df.ssp_eligible_truncated,
        "n_truncated_to_target_excluded": df.ssp_flip_truncated,
        "n_ssp_flip_binary_missing": df.ssp_label_missing,
        "n_ssp_flip_binary_faithful": df.ssp_faithful,
        "n_ssp_unfaithful": df.ssp_unfaithful,
        "n_ssp_unfaithful_no_reliance": df.ssp_unfaithful & ~df.has_reliance,
        "denominator": df.ssp_unfaithful & df.has_reliance,
        "numerator": df.ssp_unfaithful & df.robust_used,
        "n_weak_used": df.ssp_unfaithful & (df.reliance_label.astype(object) == "weak_used"),
        "n_mixed_0of4": df.ssp_unfaithful & (df.reliance_label.astype(object) == "mixed"),
    }
    for k, v in flags.items():
        w[k] = v.astype(int)
    if keys:
        g = w.groupby(keys, observed=True, sort=True)[list(flags)].sum().reset_index()
    else:
        g = pd.DataFrame([w[list(flags)].sum()])
    g = _wilson_cols(g, "numerator", "denominator")
    g["faded_n_lt_20"] = g.denominator < SMALL_N
    return g


def composition_table(df: pd.DataFrame, pool: pd.Series, keys: list[str]) -> pd.DataFrame:
    """Reliance-label composition of `pool` rows per group: counts, shares (over labeled rows) and Wilson CIs."""
    d = df[pool]
    w = pd.DataFrame({k: d[k] for k in keys})
    rl = d.reliance_label.astype(object)
    w["n_pool"] = 1
    w["n_no_reliance"] = (~d.has_reliance).astype(int)
    w["n_labeled"] = d.has_reliance.astype(int)
    for lab in RELIANCE_USED:
        w[f"n_{lab}"] = (rl == lab).astype(int)
    cols = [c for c in w.columns if c.startswith("n_")]
    g = w.groupby(keys, observed=True, sort=True)[cols].sum().reset_index()
    for lab in RELIANCE_USED:
        _wilson_cols(g, f"n_{lab}", "n_labeled", prefix=f"{lab}_")
        g = g.rename(columns={f"{lab}_rate": f"share_{lab}"})
    g["faded_n_lt_20"] = g.n_labeled < SMALL_N
    return g


def exclusion_summary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The exclusion counts alone (per group), for the NOTES tables."""
    cols = ["n_rows", "n_b0_unanswered", "n_b0_is_target", "n_wrong_to_wrong_dropped", "n_eligible",
            "n_eligible_truncated", "n_truncated_to_target_excluded", "n_ssp_flip", "n_ssp_flip_binary_missing", "n_ssp_flip_binary_faithful", "n_ssp_unfaithful",
            "n_ssp_unfaithful_no_reliance", "denominator"]
    return survival_table(df, keys)[keys + cols]


def rows_for_audit(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    cols = ROW_AUDIT_COLS + [c for c in (extra or []) if c not in ROW_AUDIT_COLS]
    return df[cols].sort_values(["subject_model", "run", "hint_style", "original_index"])


def write_gather_meta(out: Path, meta: dict, extra: dict) -> None:
    """``<out>/gather_meta.json``: ``extra`` (the gather's own record), the repo git sha, this module's and the
    code deps' shas, and the stitched rows' meta (inputs with sha256, join and flag checks)."""
    body = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra,
            "git_sha": repo_git_sha(), "ssp_common": lib_provenance(), "ssp_rows": meta}
    (Path(out) / "gather_meta.json").write_text(json.dumps(body, indent=1, default=str) + "\n")


def ordered_models(present) -> list[str]:
    return [m for m in MODELS if m in set(present)]


# ---------------------------------------------------------------------------
# Gather CLI shared by every survival gather.py
# ---------------------------------------------------------------------------

def gather_cli(doc: str, plot_name: str, build, argv: list[str] | None = None) -> int:
    """Parse the common flags, load the stitched rows, call ``build(df) -> (tables, rows, extra_meta)`` and write
    ``tables`` (name -> DataFrame) + ``rows.csv`` + ``gather_meta.json`` into ``<out-dir>/data``.

    --cueball-dir <dir>   the paper tree (default ``paths.DEFAULT_CUEBALL_DIR``)
    --out-dir <dir>    default ``paths.plot_dir(plot_name)``; tables land in ``<out-dir>/data``
    --only <substr>    restrict to runs whose binary-judged stem contains it (debug). The subset never touches
                       the cache, and without --out-dir it is written under ``<plot dir>/_debug`` instead
    --no-cache         rebuild in memory, write no cache;  --refresh-cache  rebuild and overwrite the cache
    --cache-dir <dir>  default ``paths.cache_dir()``
    """
    import argparse
    ap = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else plot_name)
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)
    cueball = P.cueball_dir(args.cueball_dir)
    if args.out_dir is None:
        # a subset (--only) never writes into the plot's real data/ folder
        out_dir = P.plot_dir(plot_name, cueball) / "_debug" if args.only else P.plot_dir(plot_name, cueball)
        if args.only:
            log(f"--only without --out-dir: writing to the debug dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    if args.no_cache or args.only:
        cache: str | Path | None = None
    else:
        cache = Path(args.cache_dir) if args.cache_dir else P.cache_dir(cueball)
    df, meta = load_rows(args.only, cache_dir=cache, refresh=args.refresh_cache, cueball=cueball)
    tables, rows, extra = build(df)
    rows.to_csv(out / "rows.csv", index=False)
    for name, t in tables.items():
        t.to_csv(out / name, index=False)
    script = P.repo_path(f"src/scripts/visualizations/{plot_name}/gather.py")
    extra = dict(extra, plot=plot_name, script=f"src.scripts.visualizations.{plot_name}.gather",
                 script_sha256=sha256_file(script) if script.exists() else None, cueball_dir=str(cueball),
                 only=args.only, rows_csv={"file": "rows.csv", "n_rows": int(len(rows))}, tables=sorted(tables))
    write_gather_meta(out, meta, extra)
    for name, t in tables.items():
        log(f"\n== {name} ({len(t)} rows)\n" + t.head(60).to_string(index=False, max_colwidth=24))
    log(f"\nwrote {out}")
    return 0
