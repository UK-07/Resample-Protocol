"""Shared helpers for the yield figures (funnel, yield_per_style).

What lives here:
  * the fixed entity lists (5 models, 8 styles, re-roll seeds); the input locations are ``common.paths``
    (``manifest_path``, ``resample_manifest_path``, ``reliance_path``, ``binary_dir``, ``hinted_dir``,
    ``rollouts_dir``, ``judge_cache_root``), re-exported here;
  * filtered loaders for the two manifests and question_reliance.csv;
  * dataset_B through src.lib.selection (the only sanctioned selection path);
  * access to ``common.ssp_common``: the ONE SSP implementation (verified joins, sample-0 re-parse,
    binary-judge stitching), so the funnel's SSP flips are exactly the other figures';
  * file provenance helpers (size / mtime / sha256) for gather_meta.json.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common.paths import (  # noqa: F401  (re-exported for the gathers)
    binary_dir, hinted_dir, judge_cache_root, manifest_meta_path, manifest_path, reliance_path,
    resample_manifest_path, rollouts_dir,
)

MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
EXCLUDED_MODELS = ("qwen3.6-27b",)
STYLES = ["expert_opinion", "unethical_info", "tool_output", "consensus", "metadata",
          "answer_key_artifact", "grader_hacking", "post_hoc"]
SEEDS = (43, 44, 45, 46)
SMALL_N = 20


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path: Path, *, sha: bool = False) -> dict:
    path = Path(path)
    st = path.stat()
    out = {"path": str(path), "size": st.st_size,
           "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    if sha:
        out["sha256"] = sha256_file(path)
    return out


def shared_code_meta() -> dict:
    """sha256 of every shared module a yield gather depends on (recorded in gather_meta.json)."""
    sc = ssp_common()
    return {
        "git_sha": sc.repo_git_sha(),
        "yield_common.py": file_info(Path(__file__), sha=True),
        "ssp_common.py": file_info(sc.LIB_PATH, sha=True),
        **{rel: file_info(P.repo_path(rel), sha=True)
           for rel in ("src/lib/selection.py", "src/lib/resample.py", "src/lib/parsing.py", "src/lib/baseline.py")},
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# The shared SSP module
# ---------------------------------------------------------------------------

def ssp_common():
    """``common.ssp_common`` as a module. Call its load_rows() with cache_dir=None to build in memory without
    touching the shared cache."""
    from src.scripts.visualizations.common import ssp_common as sc
    return sc


# ---------------------------------------------------------------------------
# Loaders (filters as in the resample scripts)
# ---------------------------------------------------------------------------

def _filter(df: pd.DataFrame) -> pd.DataFrame:
    from src.lib.resample import DROPPED_HINTS, SMOKE_RUN_MARKER
    keep = ~df["run"].astype(str).str.contains(SMOKE_RUN_MARKER)
    keep &= ~df["hint_style"].astype(str).isin(DROPPED_HINTS)
    keep &= df["subject_model"].astype(str).isin(MODELS)
    out = df[keep].copy()
    extra = sorted(set(out["hint_style"].astype(str)) - set(STYLES))
    if extra:
        raise SystemExit(f"unexpected hint styles after filtering: {extra}")
    return out


def load_rollout_manifest(columns: list[str] | None = None, *, cueball: str | Path | None = None) -> pd.DataFrame:
    cols = None if columns is None else sorted(set(columns) | {"run", "hint_style", "subject_model"})
    return _filter(pd.read_parquet(P.manifest_path(cueball), columns=cols))


def load_resample_manifest(columns: list[str] | None = None, *, cueball: str | Path | None = None) -> pd.DataFrame:
    cols = None if columns is None else sorted(set(columns) | {"run", "hint_style", "subject_model"})
    return _filter(pd.read_parquet(P.resample_manifest_path(cueball), columns=cols))


def load_reliance(role: str | None = "used_candidate", *, cueball: str | Path | None = None) -> pd.DataFrame:
    rel = pd.read_csv(P.reliance_path(cueball))
    rel = _filter(rel)
    if role is not None:
        rel = rel[rel["role"] == role]
    if rel["rollout_id"].duplicated().any():
        raise SystemExit("question_reliance.csv: duplicate rollout_id within one role")
    return rel


def dataset_B(resample_manifest: pd.DataFrame) -> pd.DataFrame:
    """The RSP-unfaithful dataset: src.lib.selection's registered 'dataset_B' predicate, nothing re-derived."""
    from src.lib.selection import select_rows
    return select_rows("dataset_B", resample_manifest)


def rsp_unfaithful_pool(resample_manifest: pd.DataFrame) -> pd.DataFrame:
    """dataset_B's conditions with v2 −1 admitted and counted as UNFAITHFUL.

    = dataset_B (label 0/1, exclude_reason null)  ∪  the re-rolls meeting dataset_B's other conditions whose
    v2 label is −1. For those, exclude_reason is 'incoherent' (the reason −1 itself sets); an EARLIER reason such as
    truncated still excludes them. Re-rolls are never noise flips (checked here), so no later reason is hidden.
    Adds `v2_label` and `rsp_unfaithful` (label 0 or −1).
    """
    rm = resample_manifest
    reroll = rm["provenance"].astype(str) == "resample_k4"
    if (reroll & (rm["exclude_reason"].astype(object) == "noise_flip")).any():
        raise SystemExit("a re-roll carries exclude_reason == noise_flip: the −1 admission rule would need a re-test")
    B = dataset_B(rm)
    lab = pd.to_numeric(rm["judge_label_final"], errors="coerce")
    inc = (reroll & (rm["reliance_label"].astype(object) == "robust_used") & rm["to_hint"].fillna(False).astype(bool)
           & (lab == -1) & (rm["exclude_reason"].astype(object) == "incoherent"))
    pool = pd.concat([B, rm[inc]])
    if pool["rollout_id"].duplicated().any():
        raise SystemExit("dataset_B and the −1 admission overlap")
    pool = pool.copy()
    pool["v2_label"] = pd.to_numeric(pool["judge_label_final"], errors="coerce")
    pool["rsp_unfaithful"] = pool["v2_label"].isin([0, -1])
    return pool


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def wilson(k: int, n: int) -> tuple[float, float]:
    from src.lib.resample import wilson_interval
    return wilson_interval(int(k), int(n))


def cluster_bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, *, reps: int = 2000, seed: int = 0,
                               level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of sum(num)/sum(den) with CLUSTERS (questions) resampled with replacement.

    num[i], den[i] are cluster i's totals (e.g. unfaithful re-rolls, hinted rollouts of one question).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = len(num)
    if n == 0 or den.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for r0 in range(0, reps, 200):                  # chunks keep memory at 200 × n counts
        m = min(200, reps - r0)
        w = rng.multinomial(n, np.full(n, 1 / n), size=m)
        stats[r0:r0 + m] = (w @ num) / np.maximum(w @ den, 1e-12)
    a = (1 - level) / 2
    return (float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a)))


def json_dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))
