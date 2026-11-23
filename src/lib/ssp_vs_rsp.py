"""SSP-vs-RSP probe comparison: does training on resampling-confirmed labels give better unfaithfulness probes?

The task is unfaithful (judge label 0, the detection target) vs faithful (1) on ORIGINAL switch rollouts, so
the arms hold the same traces with the same judge label and differ only in the rows they keep:

* ``ssp_all`` — every switch original;
* ``rsp_survivors`` — the switch originals of robust_used pairs;
* ``ssp_sized`` — ssp_all subsampled to the rsp_survivors size (the control for training-set size).

Two hint variants: every style with post_hoc capped to the mean count of the other styles
(``posthoc_capped``), and every style but post_hoc (``no_posthoc``). Three test sets on the question-level
test split: ``ssp_all``, ``rsp_survivors`` and ``gold`` (switches confirmed by 12 fresh re-rolls).

The row universe per model is every unfaithful original plus, per (split, style, dataset, case) stratum, at
most ``NEG_RATIO`` × as many faithful ones, so every arm filters one fixed pool. Every subsample keeps the
first rows of a seeded hash order, independent of reliance.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.lib import selection
from src.lib.paths import resolve_data_path

ROOT = "${DATA_ROOT}/ssp_vs_rsp"
FRACTIONS = tuple(round(0.1 * i, 1) for i in range(1, 11))
ARMS = ("ssp_all", "rsp_survivors", "ssp_sized")
VARIANTS = ("posthoc_capped", "no_posthoc")
TEST_SETS = ("ssp_all", "rsp_survivors", "gold")
LABEL = "judge_label_final"
POST_HOC = "post_hoc"
STRATUM = ["split", "hint_style", "dataset_split", "case"]
NEG_RATIO = 1.5            # universe: faithful rows per unfaithful row, per stratum
GOLD_NEG_RATIO = 1.0       # gold pairs: faithful rows per unfaithful test row, per stratum
GOLD_SEEDS = tuple(range(47, 59))   # 12 fresh sampling seeds (the k=4 re-rolls used 43-46)
GOLD_USED_MIN = 9          # ≥ 9 of 12 fresh re-rolls on the target (the 3-of-4 rule at k = 12)
SEED = 42
META_COLUMNS = ["rollout_id", "question_id", "hint_style", "split", "to_hint", LABEL, "reliance_label",
                "exclude_reason", "provenance", "in_gold"]


class Model(NamedTuple):
    name: str     # Hugging Face id
    layer: int    # the residual-stream layer the activation probes read


MODELS = {
    "qwen3-8b": Model("Qwen/Qwen3-8B", 20),
    "nemotron-nano-9b-v2": Model("nvidia/NVIDIA-Nemotron-Nano-9B-v2", 31),
    "olmo3-7b-think": Model("allenai/Olmo-3-7B-Think", 18),
    "gemma4-12b-it": Model("google/gemma-4-12B-it", 26),
}


def path(rel: str) -> Path:
    return resolve_data_path(f"{ROOT}/{rel}")


def load_universe(model: str, *extra: str, gold: bool = False) -> pd.DataFrame:
    """The model's universe (``META_COLUMNS`` + ``extra``); ``gold`` adds the gold re-roll counts and a
    boolean ``gold_used`` (False outside the gold pairs)."""
    uni = pd.read_parquet(path(f"universe/{model}.parquet"), columns=[*META_COLUMNS, *extra])
    if gold:
        uni = uni.merge(pd.read_parquet(path(f"gold/{model}_gold_labels.parquet")), on="rollout_id", how="left")
        uni["gold_used"] = uni["gold_used"].eq(True)
    return uni


def hash_rank(ids: pd.Series, salt: str) -> pd.Series:
    """A deterministic uniform key in [0, 1) per id (sha256 of ``salt|id``): the sampling order."""
    return ids.map(lambda r: int.from_bytes(hashlib.sha256(f"{salt}|{r}".encode()).digest()[:8], "big") / 2.0 ** 64
                   ).astype(float)


def _head(df: pd.DataFrame, rows: pd.Series, keys: list[str], quota: pd.Series, salt: str) -> pd.Series:
    """``rows`` cut, per ``keys`` group, to its first ``quota`` rows in hash order."""
    order = hash_rank(df.loc[rows, "rollout_id"], f"{salt}|{SEED}").groupby([df.loc[rows, k] for k in keys])
    return (order.rank(method="first") <= quota[rows]).reindex(df.index, fill_value=False)


def sample_universe(df: pd.DataFrame, *, ratio: float = NEG_RATIO, salt: str = "universe") -> pd.Series:
    """Every unfaithful row plus, per :data:`STRATUM`, ``floor(ratio × n_unfaithful)`` faithful rows."""
    unfaithful = df[LABEL].eq(0)
    quota = np.floor(ratio * unfaithful.groupby([df[k] for k in STRATUM]).transform("sum"))
    return unfaithful | _head(df, ~unfaithful, STRATUM, quota, salt)


def cap_post_hoc(df: pd.DataFrame, member: pd.Series) -> pd.Series:
    """``member`` with its post_hoc rows cut, per (split, label), to the rounded mean count of the other styles."""
    post_hoc = df["hint_style"].eq(POST_HOC)
    others = member & ~post_hoc
    keys = [df["split"], df[LABEL]]
    n_styles = df["hint_style"].where(others).groupby(keys).transform("nunique")
    cap = (others.groupby(keys).transform("sum") / n_styles).round()
    return others | _head(df, member & post_hoc, ["split", LABEL], cap, "posthoc")


def membership(df: pd.DataFrame, which: str, variant: str) -> pd.Series:
    """The rows of ``which`` (an arm, or the ``gold`` test set of :func:`load_universe`) under ``variant``,
    every split. The post_hoc cap is computed on the set itself."""
    if which == "ssp_sized":
        ssp, rsp = (membership(df, arm, variant) for arm in ("ssp_all", "rsp_survivors"))
        return _head(df, ssp, ["split", LABEL], rsp.groupby([df["split"], df[LABEL]]).transform("sum"), "sized")
    base = df["gold_used"] if which == "gold" else selection.mask(which, df)
    return base & df["hint_style"].ne(POST_HOC) if variant == "no_posthoc" else cap_post_hoc(df, base)


def split_masks(uni: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """``{(variant, arm): (train mask, val mask)}`` and the test mask (every test row, whatever the variant)."""
    split = uni["split"]
    units = {(variant, arm): tuple((membership(uni, arm, variant) & split.eq(s)).to_numpy() for s in ("train", "val"))
             for variant in VARIANTS for arm in ARMS}
    return units, split.eq("test").to_numpy()


def prefix_lengths(n_tokens: int) -> list[int]:
    """``max(1, ceil(f · n))`` tokens per fraction (the prefix a probe sees at ``f`` of the CoT)."""
    return [max(1, math.ceil(round(f * n_tokens, 9))) for f in FRACTIONS]


def auroc(pos: np.ndarray, scores: np.ndarray) -> float:
    """AUROC of ``scores`` for the boolean detection-target mask ``pos`` (NaN for a one-class set)."""
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(scores)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def open_store(model: str):
    """The model's sequence store and its manifest indexed by ``rollout_id``."""
    from src.lib.activation_store import SequenceStore, make_backend, read_store_manifest

    backend = make_backend({"backend": "local", "local_dir": f"{ROOT}/activations"})
    manifest, meta = read_store_manifest(backend, f"{model}_ssp_rsp")
    return SequenceStore(backend, f"{model}_ssp_rsp", manifest, meta), manifest.set_index("rollout_id")


def write_predictions(model: str, probe: str, scores: dict, test_ids) -> None:
    """``predictions/<model>_<probe>.parquet`` (rollout_id, variant, arm, probe, fraction, score) from
    ``{(variant, arm): scores [n_test, len(FRACTIONS)]}``."""
    preds = [pd.DataFrame({"rollout_id": np.repeat(np.asarray(test_ids), len(FRACTIONS)), "variant": variant,
                           "arm": arm, "probe": probe, "fraction": np.tile(FRACTIONS, len(s)), "score": s.ravel()})
             for (variant, arm), s in scores.items()]
    path("predictions").mkdir(parents=True, exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(path(f"predictions/{model}_{probe}.parquet"), index=False)


__all__ = [
    "ARMS", "FRACTIONS", "GOLD_NEG_RATIO", "GOLD_SEEDS", "GOLD_USED_MIN", "LABEL", "META_COLUMNS", "MODELS", "Model",
    "NEG_RATIO", "POST_HOC", "ROOT", "SEED", "STRATUM", "TEST_SETS", "VARIANTS", "auroc", "cap_post_hoc",
    "hash_rank", "load_universe", "membership", "open_store", "path", "prefix_lengths", "sample_universe",
    "split_masks", "write_predictions",
]
