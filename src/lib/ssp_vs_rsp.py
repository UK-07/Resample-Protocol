"""SSP-vs-RSP probe comparison: does training on resampling-confirmed labels give better unfaithfulness probes?

The experiment answers the reviewers' question "does a probe trained on persistent (robust_used) traces beat
one trained on unfiltered single-sample traces?" on ORIGINAL rollouts only, so the two training arms hold the
same traces with the same judge label and differ only in the reliance filter (``src.lib.selection``):

* ``ssp_all`` — every switch original, every matched control;
* ``rsp_survivors`` — switch originals of robust_used pairs, controls of robust_ignored pairs.

Two tasks: ``unfaithful_vs_ignored`` (unfaithful switch originals vs cued controls that kept their answer)
and ``unfaithful_vs_faithful`` (unfaithful vs faithful switch originals). Two hint variants: every style with
post_hoc capped to the mean count of the other styles (``posthoc_capped``), and every style but post_hoc
(``no_posthoc``). Three test sets on the question-level test split: ``ssp_all`` and ``rsp_survivors`` (each
arm's own labels) and ``gold`` (12 fresh re-rolls per pair, independent of both arms).

The row universe per model is every unfaithful switch original plus, per (split, style, dataset, case)
stratum, at most ``ratio`` × as many faithful originals and as many controls (seeded hash order, independent
of reliance), so both arms filter one fixed pool.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.lib import selection

FRACTIONS = tuple(round(0.1 * i, 1) for i in range(1, 11))
ARMS = selection.SSP_RSP_ARMS
TASKS = selection.SSP_RSP_TASKS
VARIANTS = ("posthoc_capped", "no_posthoc")
TEST_SETS = ("ssp_all", "rsp_survivors", "gold")
POST_HOC = "post_hoc"
CLASSES = ("U", "F", "C")   # unfaithful switch, faithful switch, control
TASK_CLASSES = {"unfaithful_vs_ignored": ("U", "C"), "unfaithful_vs_faithful": ("U", "F")}
STRATUM = ("split", "hint_style", "dataset_split", "case")
NEG_RATIO = 1.5            # universe: negatives per unfaithful row, per stratum and negative class
GOLD_NEG_RATIO = 1.0       # gold set: negatives per unfaithful test row, per stratum and negative class
GOLD_SEEDS = tuple(range(47, 59))   # 12 fresh sampling seeds (the k=4 re-rolls used 43-46)
GOLD_USED_MIN = 9          # ≥ 9 of 12 fresh re-rolls on the target  (the 3-of-4 rule at k = 12)
GOLD_BASELINE_MIN = 9      # control: 0 of 12 on the target and ≥ 9 of 12 on the baseline answer
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    short: str
    model_name: str
    base_config: str
    layer: int


MODELS = {
    "qwen3-8b": ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen_3-8b_base.yaml", 20),
    "nemotron-nano-9b-v2": ModelSpec("nemotron-nano-9b-v2", "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                                     "nemotron_nano-9b_base.yaml", 31),
    "olmo3-7b-think": ModelSpec("olmo3-7b-think", "allenai/Olmo-3-7B-Think", "olmo_3-7b-think_base.yaml", 18),
    "gemma4-12b-it": ModelSpec("gemma4-12b-it", "google/gemma-4-12B-it", "gemma_4-12b_base.yaml", 26),
}


# ---------------------------------------------------------------------------
# Hash order and classes
# ---------------------------------------------------------------------------


def hash_rank(ids, salt) -> np.ndarray:
    """A deterministic uniform key in [0, 1) per id (sha256 of ``salt|id``): the sampling order."""
    out = np.empty(len(ids), dtype=np.float64)
    for i, r in enumerate(ids):
        h = hashlib.sha256(f"{salt}|{r}".encode("utf-8")).digest()
        out[i] = int.from_bytes(h[:8], "big") / 2.0 ** 64
    return out


def row_class(df: pd.DataFrame) -> pd.Series:
    """``U`` / ``F`` / ``C`` per row (None outside the universe), through the selection predicates."""
    df = selection.with_provenance(df)
    out = pd.Series(pd.array([None] * len(df), dtype="string"), index=df.index)
    out[selection.ssp_rsp_unfaithful(df).to_numpy()] = "U"
    out[selection.ssp_rsp_faithful(df).to_numpy()] = "F"
    out[selection.ssp_rsp_control(df).to_numpy()] = "C"
    return out


def sample_universe(df: pd.DataFrame, *, ratio: float = NEG_RATIO, seed: int = DEFAULT_SEED,
                    salt: str = "universe") -> pd.Series:
    """Boolean mask: every ``U`` row, and per :data:`STRATUM` the first ``floor(ratio × n_U)`` ``F`` and
    ``C`` rows in hash order. ``df`` needs a ``cls`` column (:func:`row_class`)."""
    keep = df["cls"].eq("U").fillna(False).to_numpy().copy()
    rank = pd.Series(hash_rank(df["rollout_id"].astype(str).tolist(), f"{salt}|{seed}"), index=df.index)
    keys = [df[c].astype(str) for c in STRATUM]
    n_u = df["cls"].eq("U").fillna(False).groupby(keys).transform("sum")
    for cls in ("F", "C"):
        is_cls = df["cls"].eq(cls).fillna(False)
        sub = df[is_cls]
        order = rank[is_cls].groupby([sub[c].astype(str) for c in STRATUM]).rank(method="first")
        quota = np.floor(ratio * n_u[is_cls]).astype(int)
        chosen = order.index[(order <= quota).to_numpy()]
        keep[df.index.get_indexer(chosen)] = True
    return pd.Series(keep, index=df.index)


# ---------------------------------------------------------------------------
# Arms, the post_hoc cap, memberships
# ---------------------------------------------------------------------------


def arm_mask(df: pd.DataFrame, arm: str, task: str) -> pd.Series:
    return selection.mask(selection.ssp_rsp_predicate(arm, task), df)


def task_labels(df: pd.DataFrame, task: str) -> pd.Series:
    """0 (detection target: unfaithful) / 1 per row of the task's two classes, NA elsewhere."""
    pos, neg = TASK_CLASSES[task]
    lab = pd.Series(pd.array([pd.NA] * len(df), dtype="Int8"), index=df.index)
    lab[df["cls"].eq(pos).fillna(False).to_numpy()] = 0
    lab[df["cls"].eq(neg).fillna(False).to_numpy()] = 1
    return lab


def cap_post_hoc(df: pd.DataFrame, member: pd.Series, labels: pd.Series, *, seed: int = DEFAULT_SEED) -> pd.Series:
    """``member`` with post_hoc rows downsampled, per (split, label), to the rounded mean count of the
    other styles in that (split, label) — an even share — in hash order."""
    member = member.fillna(False).astype(bool)
    out = member.copy()
    rank = pd.Series(hash_rank(df["rollout_id"].astype(str).tolist(), f"posthoc|{seed}"), index=df.index)
    sub = df[member]
    for (split, lab), g in sub.groupby([sub["split"].astype(str), labels[member].astype(int)]):
        styles = g["hint_style"].astype(str)
        is_ph = styles.eq(POST_HOC)
        others = styles[~is_ph].value_counts()
        cap = int(round(others.mean())) if len(others) else 0
        ph_idx = g.index[is_ph.to_numpy()]
        if len(ph_idx) <= cap:
            continue
        drop = rank[ph_idx].sort_values(kind="stable").index[cap:]
        out[drop] = False
    return out


def apply_variant(df: pd.DataFrame, member: pd.Series, labels: pd.Series, variant: str, *,
                  seed: int = DEFAULT_SEED) -> pd.Series:
    if variant == "no_posthoc":
        return member.fillna(False).astype(bool) & df["hint_style"].astype(str).ne(POST_HOC)
    if variant == "posthoc_capped":
        return cap_post_hoc(df, member, labels, seed=seed)
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


def gold_mask(df: pd.DataFrame, task: str) -> pd.Series:
    """Test rows whose gold label agrees with their class: unfaithful/faithful switches gold_used, controls
    gold_ignored (needs the ``gold_used`` / ``gold_ignored`` columns of :func:`gold_labels`)."""
    pos, neg = TASK_CLASSES[task]
    used = df["gold_used"].fillna(False).astype(bool)
    ignored = df["gold_ignored"].fillna(False).astype(bool)
    cls = df["cls"].astype(str)
    neg_ok = ignored if neg == "C" else used
    return (cls.eq(pos) & used) | (cls.eq(neg) & neg_ok)


def membership(df: pd.DataFrame, task: str, which: str, variant: str, *, seed: int = DEFAULT_SEED) -> pd.Series:
    """The rows of ``which`` (an arm, or the ``gold`` test set) for ``task`` under ``variant``, every split
    (``gold`` exists on test rows only). The cap is computed on the set itself."""
    labels = task_labels(df, task)
    if which == "gold":
        base = gold_mask(df, task) & df["split"].astype(str).eq("test") & labels.notna()
    else:
        base = arm_mask(df, which, task) & labels.notna()
    return apply_variant(df, base, labels, variant, seed=seed)


def size_matched(df: pd.DataFrame, task: str, variant: str, *, seed: int = DEFAULT_SEED) -> pd.Series:
    """Control arm ``ssp_sized``: the ssp_all rows of (task, variant) randomly subsampled (hash order) to the
    RSP-survivors row count per (split, label), so it differs from rsp_survivors in selection, not size."""
    labels = task_labels(df, task)
    base = membership(df, task, "ssp_all", variant, seed=seed)
    rsp = membership(df, task, "rsp_survivors", variant, seed=seed)
    rank = pd.Series(hash_rank(df["rollout_id"].astype(str).tolist(), f"sized|{seed}"), index=df.index)
    out = pd.Series(False, index=df.index)
    split = df["split"].astype(str)
    for (sp, lab), g in df[base].groupby([split[base], labels[base].astype(int)]):
        n = int((rsp & split.eq(sp) & labels.eq(lab)).sum())
        out[rank[g.index].sort_values(kind="stable").index[:n]] = True
    return out


# ---------------------------------------------------------------------------
# Gold labels
# ---------------------------------------------------------------------------


def gold_labels(counts: pd.DataFrame) -> pd.DataFrame:
    """``gold_used`` / ``gold_ignored`` from per-pair counts ``n_gold`` (generated), ``n_target`` and
    ``n_baseline`` over the fresh re-rolls; a pair short of the full 12 gets no gold label."""
    full = counts["n_gold"].eq(len(GOLD_SEEDS))
    used = full & counts["n_target"].ge(GOLD_USED_MIN)
    ignored = full & counts["n_target"].eq(0) & counts["n_baseline"].ge(GOLD_BASELINE_MIN)
    return counts.assign(gold_used=used, gold_ignored=ignored)


# ---------------------------------------------------------------------------
# Prefixes and metrics
# ---------------------------------------------------------------------------


def prefix_lengths(n_tokens: int, fractions=FRACTIONS) -> list[int]:
    """``max(1, ceil(f · n))`` tokens per fraction (the prefix a probe sees at ``f`` of the CoT)."""
    n = int(n_tokens)
    if n < 1:
        raise ValueError(f"n_tokens must be ≥ 1, got {n_tokens}")
    return [min(n, max(1, math.ceil(round(f * n, 9)))) for f in fractions]


def auroc(labels, scores) -> float:
    """AUROC of ``scores`` (P(detection target)) for label 0 = detection target; NaN for a one-class set."""
    from sklearn.metrics import roc_auc_score
    y = (np.asarray(labels).astype(int) == 0).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


def bootstrap_auroc(labels, scores, clusters, *, n_boot: int = 1000, seed: int = DEFAULT_SEED,
                    alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of the AUROC under a cluster bootstrap (clusters = questions)."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    codes, uniq = pd.factorize(pd.Series(clusters).astype(str))
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[k] for k in pick])
        v = auroc(y[idx], s[idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_delta(labels, score_a, score_b, clusters, *, n_boot: int = 1000,
                           seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """``AUROC(b) − AUROC(a)`` on the same rows, with a cluster-bootstrap 95% interval."""
    y = np.asarray(labels).astype(int)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    codes, uniq = pd.factorize(pd.Series(clusters).astype(str))
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[k] for k in pick])
        da, db = auroc(y[idx], a[idx]), auroc(y[idx], b[idx])
        if not (math.isnan(da) or math.isnan(db)):
            deltas.append(db - da)
    point = auroc(y, b) - auroc(y, a)
    if not deltas:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(point), float(lo), float(hi)


__all__ = [
    "ARMS", "CLASSES", "DEFAULT_SEED", "FRACTIONS", "GOLD_BASELINE_MIN", "GOLD_NEG_RATIO", "GOLD_SEEDS",
    "GOLD_USED_MIN", "MODELS", "ModelSpec", "NEG_RATIO", "POST_HOC", "STRATUM", "TASKS", "TASK_CLASSES",
    "TEST_SETS", "VARIANTS", "apply_variant", "arm_mask", "auroc", "bootstrap_auroc", "cap_post_hoc",
    "gold_labels", "gold_mask", "hash_rank", "membership", "paired_bootstrap_delta", "prefix_lengths",
    "row_class", "sample_universe", "size_matched", "task_labels",
]
