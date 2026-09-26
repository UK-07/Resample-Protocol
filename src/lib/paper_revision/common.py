"""Shared code for the revision analysis: loaders, the question-cluster bootstrap (10,000 replicates, seed 42,
questions resampled within dataset, every model / cue / original / re-roll of a question moving together) and
table helpers, ported from the verified 2026-09-26 RunPod analysis. Input paths are explicit and no data-bundle code is executed. The bootstrap weights are generated once from the
sorted canonical question list so all comparisons share the same draws."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ["nemotron-nano-9b-v2", "qwen3-8b", "qwen3.5-9b", "olmo3-7b-think", "gemma4-12b-it"]
MODEL_SHORT = {"nemotron-nano-9b-v2": "Nemotron", "qwen3-8b": "Qwen3-8B", "qwen3.5-9b": "Qwen3.5-9B",
               "olmo3-7b-think": "OLMo-3-7B", "gemma4-12b-it": "Gemma-4-12B"}
DATASETS = ["commonsense_qa", "medqa", "gpqa", "mmlu_pro"]
CUES = ["expert_opinion", "unethical_info", "tool_output", "consensus", "metadata", "answer_key_artifact",
        "grader_hacking", "post_hoc"]
CASES = ["positive", "negative"]
N_OPTIONS = {"commonsense_qa": 5, "medqa": 4, "gpqa": 4, "mmlu_pro": 10}
N_BOOT = 10000
SEED = 42
SMALL_N = 20


def load_pairs(results: Path) -> pd.DataFrame:
    df = pd.read_parquet(results / "pairs_master.parquet")
    for c in ("ssp_flip", "ssp_unfaithful", "ssp_faithful", "ssp_label_missing", "robust_used", "has_reliance",
              "ssp_flip_truncated", "ssp_eligible_truncated", "orig_role_covered", "persist1", "persist2", "persist3"):
        df[c] = df[c].fillna(False).astype(bool)
    df["truncated"] = df.truncated.fillna(False).astype(bool)
    df["clean"] = ~df.truncated & df.model_answer.notna()
    df["eligible"] = df.ssp_status == "eligible"
    df["eligible_clean"] = df.eligible & df.clean
    return df


def load_rerolls(results: Path) -> pd.DataFrame:
    return pd.read_parquet(results / "rerolls_long.parquet")


def wilson(k, n, z=1.96):
    k, n = float(k), float(n)
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


class Boot:
    """Holder for bootstrap replicates attached to a table (.attrs["boot"]); compares by identity so pandas
    merges/concats never compare the arrays."""
    def __init__(self, reps, levels):
        self.reps = reps; self.levels = levels
    def __eq__(self, other):
        return self is other
    def __hash__(self):
        return id(self)


class ClusterBootstrap:
    """Question-cluster bootstrap. Questions (canonical key dataset:original_index) are resampled with
    replacement WITHIN each dataset (multinomial counts), giving a (N_BOOT, Q) weight matrix shared by every
    statistic. A statistic is a ratio of weighted sums of per-question totals."""

    def __init__(self, qkeys: pd.Series, n_boot: int = N_BOOT, seed: int = SEED):
        keys = sorted(set(qkeys.astype(str)))
        self.qkeys = np.array(keys)
        self.qidx = {k: i for i, k in enumerate(keys)}
        self.Q = len(keys)
        self.dataset = np.array([k.split(":")[0] for k in keys])
        rng = np.random.default_rng(seed)
        W = np.zeros((n_boot, self.Q), dtype=np.float32)
        for d in sorted(set(self.dataset)):
            cols = np.where(self.dataset == d)[0]
            n = len(cols)
            W[:, cols] = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float32)
        self.W = W
        self.n_boot = n_boot
        self.seed = seed
        self.fingerprint = hashlib.sha256(W.tobytes()).hexdigest()[:16]

    def per_question(self, df: pd.DataFrame, value: pd.Series) -> np.ndarray:
        """Q-vector of per-question sums of ``value`` (rows of df; df.qkey gives the question)."""
        idx = df.qkey.astype(str).map(self.qidx).to_numpy()
        out = np.zeros(self.Q)
        np.add.at(out, idx, np.asarray(value, dtype=float))
        return out

    def per_question_groups(self, df: pd.DataFrame, value: pd.Series, groups: pd.Series) -> tuple[np.ndarray, list]:
        """(Q x G) matrix of per-question sums, one column per group value (sorted)."""
        g = groups.astype(str)
        levels = sorted(g.unique())
        gi = g.map({l: i for i, l in enumerate(levels)}).to_numpy()
        idx = df.qkey.astype(str).map(self.qidx).to_numpy()
        out = np.zeros((self.Q, len(levels)))
        np.add.at(out, (idx, gi), np.asarray(value, dtype=float))
        return out, levels

    def reps(self, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        """Bootstrap replicates of sum(num)/sum(den): (n_boot,) or (n_boot, G). NaN where the denominator is 0."""
        a = self.W @ num.astype(np.float32)
        b = self.W @ den.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(b > 0, a / b, np.nan)
        return r

    @staticmethod
    def summarize(reps: np.ndarray, point: float | np.ndarray, level: float = 0.95) -> dict | pd.DataFrame:
        a = (1 - level) / 2
        reps = np.asarray(reps, dtype=float)
        if reps.ndim == 1:
            ok = ~np.isnan(reps)
            n_ok = int(ok.sum())
            if n_ok < 0.5 * len(reps):
                return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan, "undefined_reps": int(len(reps) - n_ok)}
            return {"point": point, "ci_lo": float(np.nanpercentile(reps, 100 * a)),
                    "ci_hi": float(np.nanpercentile(reps, 100 * (1 - a))), "undefined_reps": int(len(reps) - n_ok)}
        ok = ~np.isnan(reps)
        n_ok = ok.sum(axis=0)
        lo = np.nanpercentile(reps, 100 * a, axis=0)
        hi = np.nanpercentile(reps, 100 * (1 - a), axis=0)
        bad = n_ok < 0.5 * reps.shape[0]
        lo = np.where(bad, np.nan, lo); hi = np.where(bad, np.nan, hi)
        return pd.DataFrame({"point": np.asarray(point, dtype=float), "ci_lo": lo, "ci_hi": hi,
                             "undefined_reps": reps.shape[0] - n_ok})

    @staticmethod
    def pvalue(reps_diff: np.ndarray) -> np.ndarray:
        """Two-sided bootstrap p-value for a difference: 2 * min(P(diff <= 0), P(diff >= 0)), NaN if undefined."""
        r = np.asarray(reps_diff, dtype=float)
        if r.ndim == 1:
            r = r[:, None]
        ok = ~np.isnan(r)
        n = ok.sum(axis=0).astype(float)
        le = (np.where(ok, r <= 0, False)).sum(axis=0) / np.maximum(n, 1)
        ge = (np.where(ok, r >= 0, False)).sum(axis=0) / np.maximum(n, 1)
        p = 2 * np.minimum(le, ge)
        p = np.minimum(p, 1.0)
        p = np.where(n < 0.5 * r.shape[0], np.nan, p)
        return p if p.shape[0] > 1 else p[0]


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.minimum(q, 1.0)
    res = np.empty(m); res[order] = q
    out[ok] = res
    return out


def ratio_table(bs: ClusterBootstrap, df: pd.DataFrame, num: pd.Series, den: pd.Series, keys: list[str],
                name: str = "rate") -> pd.DataFrame:
    """Per group (keys): numerator, denominator, unique questions, rate, cluster-bootstrap 95% CI (and the
    replicate matrix, attached as .attrs['reps'] with .attrs['levels'])."""
    if keys:
        g = df[keys].astype(str).agg("|".join, axis=1)
    else:
        g = pd.Series("all", index=df.index)
    NUM, levels = bs.per_question_groups(df, num, g)
    DEN, _ = bs.per_question_groups(df, den, g)
    point = np.where(DEN.sum(0) > 0, NUM.sum(0) / np.maximum(DEN.sum(0), 1e-12), np.nan)
    reps = bs.reps(NUM, DEN)
    s = bs.summarize(reps, point)
    nq = df.assign(_g=g, _d=np.asarray(den, dtype=float) > 0).query("_d").groupby("_g").qkey.nunique()
    t = pd.DataFrame({"group": levels, "num": NUM.sum(0), "den": DEN.sum(0)})
    t["n_questions"] = t.group.map(nq).fillna(0).astype(int)
    t[name] = s.point.to_numpy(); t[f"{name}_ci_lo"] = s.ci_lo.to_numpy(); t[f"{name}_ci_hi"] = s.ci_hi.to_numpy()
    t["undefined_reps"] = s.undefined_reps.to_numpy()
    if keys:
        parts = t.group.str.split("|", expand=True); parts.columns = keys
        t = pd.concat([parts, t.drop(columns="group")], axis=1)
    else:
        t = t.drop(columns="group")
    t.attrs["boot"] = Boot(reps, levels)
    return t


def plain(t):
    """A copy of a table without the bootstrap replicate arrays in .attrs (pandas concat compares attrs)."""
    o = t.copy(); o.attrs = {}
    return o


def cat(objs, **kw):
    return pd.concat([plain(o) for o in objs], **kw)


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
