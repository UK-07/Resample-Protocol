#!/usr/bin/env python3
"""The paper's judge-validation figures (GPU-free, no API key).

Four figures, each with variants, written as PDF + PNG plus ``figures_judge.md`` with a caption draft
per variant (the captions carry the counts they were drawn from):

* **J1** — agreement of the primary judge with the arbiter: Cohen's κ per cue style × subject model
  (and per role, one-vs-rest) on the stratified validation sample, with Walden & Wanner's 0.69 as the
  reference.
* **J2** — consistency of the binary verbalization judge with the role judge on every switched
  rollout of the paper grid: confusion of the binary label against the primary judge's role, per model.
* **J3** — the researcher's spot-check: precision of the primary judge's verdict per role on the
  reviewed rows (Wilson intervals), with the behaviourally confirmed false rejections singled out,
  and the human golden set as a second reference.
* **J4** — robustness of the headline to the judge: unfaithful rates, the share of rollouts that
  claim not to rely on the cue and the false-claim rate recomputed under arbiter (and reviewer)
  labels, side by side with the primary judge's (design-weighted, stratified bootstrap).

Inputs (defaults under ``${DATA_ROOT}``): the judge-validation run directory (``sample_judged.csv``,
``review_sheet.csv``), the shared manifest (the primary judge's roles for the sampled rows), the
prompt-backfill judge cache (the primary judge re-run on the corrected prompt, used for every sampled
run that has one), the shared ``question_reliance.csv`` (robust_used for the sampled rows), the label
overrides (the reviewer's labels), the cueball tree (``hinted_rollouts/rollout_manifest.parquet`` and
``binary_judge/*_binary_judged.csv``) and the manual golden set. Any input that is missing skips the
variants that need it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.lib.judge_validation import cohen_kappa, weighted_rate_bootstrap  # noqa: E402
from src.lib.llm_judge_verb import JUDGE_ROLES, load_judged_records  # noqa: E402
from src.lib.paths import resolve_data_path  # noqa: E402
from src.lib.resample import wilson_interval  # noqa: E402
from src.scripts.visualizations.paper_plots import (  # noqa: E402
    AXIS, CUE_NAMES, CUE_ORDER, DATASET_NAMES, DATASET_ORDER, FULL_W, GRID, HALF_W, INK, INK2, MODEL_COLOR, MODEL_NAMES, MODEL_ORDER, MUTED,
    ROLE_NAMES, ROLE_ORDER, SEQ_CMAP, SLOTS, SURFACE, Variant, apply_style, figure, heatmap, pct, pct_axis, tidy,
)

DEFAULT_VALIDATION_DIR = "${DATA_ROOT}/judge_validation/mmlu_pro_gpt5.6-terra"
DEFAULT_CUEBALL_DIR = "${DATA_ROOT}/cueball"
DEFAULT_SHARED_MANIFEST = "${DATA_ROOT}/hinted_rollouts/rollout_manifest.parquet"
DEFAULT_BACKFILL_CACHE_DIR = "${DATA_ROOT}/judge_cache_prompt_backfill"
DEFAULT_RELIANCE = "${DATA_ROOT}/resample/question_reliance.csv"
DEFAULT_OVERRIDES = "${DATA_ROOT}/hinted_rollouts/judge_label_overrides.csv"
DEFAULT_GOLDEN = "${DATA_ROOT}/manual_golden_dataset_judge/golden_dataset.csv"

REFERENCE_KAPPA = 0.69          # Walden & Wanner 2026: κ of their hint-use facet (50 items)
REFERENCE_KAPPA_LABEL = "Walden & Wanner κ = 0.69"
PAPER_MODELS = ["nemotron-nano-9b-v2", "olmo3-7b-think", "qwen3-8b", "qwen3.5-9b", "gemma4-12b-it"]
PAPER_CUES = list(CUE_ORDER)                                   # the paper's eight cue styles (grader_hacking included)
CLAIM_ROLES = ("rejected", "verification_only")                # "mentions the cue and claims not to rely on it"
LABEL_SETS = ["primary", "arbiter", "reviewer"]
LABEL_SET_NAMES = {"primary": "primary judge (GLM-5.3-Flash)", "arbiter": "arbiter (GPT-5.6-Terra)",
                   "reviewer": "reviewer-adjudicated"}
LABEL_SET_COLOR = {"primary": SLOTS[0], "arbiter": SLOTS[1], "reviewer": SLOTS[2]}
BINARY_NAMES = {1: "faithful", 0: "unfaithful"}
EMPH, DEEMPH = SLOTS[0], "#c3c2b7"
DIV_CMAP = LinearSegmentedColormap.from_list(
    "kappa_div", ["#e34948", "#ee8a89", "#f6c2c1", "#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6"])


# ------------------------------------------------------------------ data ---


@dataclass
class Inputs:
    sample: pd.DataFrame | None = None       # the validation sample with every label set
    sheet: pd.DataFrame | None = None        # the researcher's review sheet, primary role attached
    binary: pd.DataFrame | None = None       # binary-judge label × primary role, every switched rollout
    golden: pd.DataFrame | None = None       # the manual golden set
    notes: list[str] = field(default_factory=list)


def _int_label(values) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def source_stem(source_csv: str) -> str:
    return Path(str(source_csv)).stem.removesuffix("_judged")


def _cache_file(cache_dir: Path, judge_model: str, prompt_hash: str | None) -> Path | None:
    slug = str(judge_model).replace("/", "_").replace(":", "_")
    if prompt_hash:
        path = cache_dir / f"{slug}_binary_{prompt_hash}.jsonl"
        return path if path.exists() else None
    found = sorted(cache_dir.glob(f"{slug}_binary_*.jsonl"))
    return found[0] if len(found) == 1 else None


def load_sample(validation_dir: Path, shared_manifest: Path | None, backfill_dir: Path | None,
                reliance_csv: Path | None, overrides_csv: Path | None, notes: list[str],
                models: list[str] | None = None) -> pd.DataFrame | None:
    """The validation sample with ``primary_*``, ``arbiter_*`` and ``reviewer_label`` columns.

    The primary judge's label comes from the sample (the manifest's verdict); where the sampled run
    has a prompt-backfill cache the primary judge's re-run on the corrected prompt replaces it (label
    and role), because that is the verdict the reviewer saw and the arbiter answered. Roles of the
    other runs come from the shared manifest.
    """
    path = validation_dir / "sample_judged.csv"
    if not path.exists():
        notes.append(f"no validation sample at {path}: J1, J3a and J4 skipped")
        return None
    s = pd.read_csv(path)
    if models is not None:
        s = s[s["subject_model"].isin(models)].reset_index(drop=True)
        if not len(s):
            notes.append(f"no sampled rows for {models}: J1, J3a and J4 skipped")
            return None
    s["primary_label"] = _int_label(s["judge_label"]).astype(float)
    s["primary_role"] = None
    s["primary_source"] = "manifest"
    if shared_manifest is not None and shared_manifest.exists():
        man = pd.read_parquet(shared_manifest, columns=["rollout_id", "judge_role"])
        roles = man.set_index("rollout_id")["judge_role"]
        s["primary_role"] = s["rollout_id"].map(roles).where(lambda x: x.isin(JUDGE_ROLES), None)
    else:
        notes.append("shared manifest missing: the primary judge's roles are only known for runs with a backfill cache")
    if backfill_dir is not None and backfill_dir.exists():
        for stem, part in s.groupby(s["source_csv"].map(source_stem)):
            judge_model = str(part["judge_model"].iloc[0])
            prompt_hash = str(part["arbiter_prompt_hash"].iloc[0]) if "arbiter_prompt_hash" in part else None
            cache = _cache_file(backfill_dir / stem, judge_model, prompt_hash)
            if cache is None:
                continue
            recs = load_judged_records(cache)
            keys = list(zip(part["original_index"].astype(int), part["hint_style"].astype(str)))
            hit = [k in recs and recs[k].get("error") in (None, "None") for k in keys]
            idx = part.index[hit]
            s.loc[idx, "primary_label"] = [float(recs[k]["label"]) for k, h in zip(keys, hit) if h]
            s.loc[idx, "primary_role"] = [recs[k].get("hint_role") if recs[k].get("hint_role") in JUDGE_ROLES else None
                                          for k, h in zip(keys, hit) if h]
            s.loc[idx, "primary_source"] = "corrected_prompt"
            notes.append(f"{stem}: primary judge's corrected-prompt verdicts used for {len(idx)} of {len(part)} sampled rows")
    s["arbiter_label"] = _int_label(s["arbiter_label"])
    s["arbiter_role"] = s["arbiter_hint_role"].where(s["arbiter_hint_role"].isin(JUDGE_ROLES), None)
    s["reviewer_label"] = s["primary_label"]
    if overrides_csv is not None and overrides_csv.exists():
        ov = pd.read_csv(overrides_csv)
        final = ov.set_index("rollout_id")["judge_label_final"]
        reviewed = s["rollout_id"].isin(final.index)
        s.loc[reviewed, "reviewer_label"] = _int_label(s.loc[reviewed, "rollout_id"].map(final)).to_numpy()
        s["reviewed"] = reviewed
    else:
        s["reviewed"] = False
        notes.append("label overrides missing: reviewer labels equal the primary judge's")
    s["reliance_label"] = None
    if reliance_csv is not None and reliance_csv.exists():
        rel = pd.read_csv(reliance_csv, usecols=["rollout_id", "reliance_label"]).drop_duplicates("rollout_id")
        s["reliance_label"] = s["rollout_id"].map(rel.set_index("rollout_id")["reliance_label"])
    s["cell"] = s["stratum"].astype(str) + "|" + s["hint_style"].astype(str)
    s["both_binary"] = s["primary_label"].isin([0, 1]) & s["arbiter_label"].isin([0, 1])
    return s


def load_sheet(validation_dir: Path, sample: pd.DataFrame | None, notes: list[str]) -> pd.DataFrame | None:
    path = validation_dir / "review_sheet.csv"
    if not path.exists() or sample is None:
        notes.append("review sheet missing: J3a skipped")
        return None
    sheet = pd.read_csv(path)
    verdict = sheet["researcher_verdict"].fillna("").astype(str).str.strip().str.lower()
    sheet["verdict"] = verdict
    sheet["decided"] = verdict.isin(["flash", "arbiter", "both", "neither"])
    sheet["upheld"] = verdict.isin(["flash", "both"])
    cols = ["rollout_id", "primary_role", "primary_label", "reliance_label"]
    sheet = sheet.drop(columns=[c for c in cols[1:] if c in sheet.columns]).merge(sample[cols], on="rollout_id", how="left")
    return sheet


def _binary_cache_ok(meta_path: Path, sources: list[Path]) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    current = {p.name: [p.stat().st_size, int(p.stat().st_mtime)] for p in sources}
    return meta.get("sources") == current


def load_binary_labels(cueball_dir: Path, cache_dir: Path | None, notes: list[str]) -> pd.DataFrame | None:
    """One row per binary-judged rollout: ``source_csv`` (the role judge's file), index, cue, label."""
    files = sorted((cueball_dir / "binary_judge").glob("*_binary_judged.csv"))
    if not files:
        notes.append(f"no binary-judge files under {cueball_dir / 'binary_judge'}: J2 skipped")
        return None
    cache = (cache_dir / "binary_judge_labels.parquet") if cache_dir is not None else None
    meta = (cache_dir / "binary_judge_labels.json") if cache_dir is not None else None
    if cache is not None and cache.exists() and _binary_cache_ok(meta, files):
        return pd.read_parquet(cache)
    parts = []
    for f in files:
        part = pd.read_csv(f, usecols=["original_index", "hint_name", "judge_label"])
        part = part[part["judge_label"].notna()]
        part["source_csv"] = f.name.replace("_binary_judged.csv", "_judged.csv")
        parts.append(part)
        print(f"  read {f.name}: {len(part):,} binary verdicts", flush=True)
    out = pd.concat(parts, ignore_index=True).rename(columns={"hint_name": "hint_style", "judge_label": "binary_label"})
    out["original_index"] = out["original_index"].astype(int)
    out["binary_label"] = _int_label(out["binary_label"])
    if cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
        meta.write_text(json.dumps({"sources": {p.name: [p.stat().st_size, int(p.stat().st_mtime)] for p in files}}, indent=1))
    return out


def load_binary_vs_role(cueball_dir: Path, cache_dir: Path | None, models: list[str], cues: list[str],
                        notes: list[str]) -> pd.DataFrame | None:
    """Every switched rollout of the grid with a primary role and a binary-judge label."""
    manifest = cueball_dir / "hinted_rollouts" / "rollout_manifest.parquet"
    if not manifest.exists():
        notes.append(f"no manifest at {manifest}: J2 skipped")
        return None
    labels = load_binary_labels(cueball_dir, cache_dir, notes)
    if labels is None:
        return None
    m = pd.read_parquet(manifest, columns=["rollout_id", "subject_model", "dataset", "hint_style", "case", "original_index",
                                           "to_hint", "judge_label", "judge_role", "exclude_reason", "source_csv"])
    m = m[m["subject_model"].isin(models) & m["hint_style"].isin(cues) & m["to_hint"].fillna(False).astype(bool)]
    m["primary_label"] = _int_label(m["judge_label"])
    m = m[m["primary_label"].isin([0, 1]) & m["judge_role"].isin(JUDGE_ROLES)].rename(columns={"judge_role": "primary_role"})
    m["original_index"] = m["original_index"].astype(int)
    m["source_csv"] = m["source_csv"].map(lambda p: Path(str(p)).name)
    out = m.merge(labels, on=["source_csv", "original_index", "hint_style"], how="inner")
    out = out[out["binary_label"].isin([0, 1])]
    out["implied_label"] = (out["primary_role"] == "credited").astype(int)
    out["inconsistent"] = out["binary_label"].astype(int) != out["implied_label"]
    return out


def load_golden(path: Path | None, notes: list[str]) -> pd.DataFrame | None:
    if path is None or not path.exists():
        notes.append("golden set missing: J3b skipped")
        return None
    g = pd.read_csv(path)
    g["judge_label"] = _int_label(g["judge_label"])
    g["manual_label"] = _int_label(g["manual_label"])
    g["manual_label_blind"] = _int_label(g["manual_label_blind"]) if "manual_label_blind" in g else np.nan
    return g[g["judge_label"].isin([0, 1]) & g["manual_label"].isin([0, 1])]


# ----------------------------------------------------------------- stats ---


def kappa_grid(sample: pd.DataFrame, cues: list[str], models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """κ of the binary label (primary vs arbiter) per cue × model, plus an ``all cues`` row pooled over ``cues``; and the n per cell."""
    rows = sample[sample["both_binary"] & sample["hint_style"].isin(cues)]
    kappa = pd.DataFrame(index=cues + ["all"], columns=models, dtype=float)
    n = pd.DataFrame(0, index=cues + ["all"], columns=models, dtype=int)
    for m in models:
        for c in cues + ["all"]:
            part = rows[(rows["subject_model"] == m) & ((rows["hint_style"] == c) if c != "all" else True)]
            n.loc[c, m] = len(part)
            k = cohen_kappa(part["primary_label"].astype(int), part["arbiter_label"].astype(int)) if len(part) else None
            kappa.loc[c, m] = np.nan if k is None else k
    return kappa, n


def role_kappa_grid(sample: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-vs-rest κ per role between the primary judge's role and the arbiter's; n = rows either judge gave the role."""
    rows = sample[sample["primary_role"].isin(JUDGE_ROLES) & sample["arbiter_role"].isin(JUDGE_ROLES)]
    kappa = pd.DataFrame(index=ROLE_ORDER + ["all"], columns=models, dtype=float)
    n = pd.DataFrame(0, index=ROLE_ORDER + ["all"], columns=models, dtype=int)
    for m in models:
        part = rows[rows["subject_model"] == m]
        for r in ROLE_ORDER:
            a, b = (part["primary_role"] == r).astype(int), (part["arbiter_role"] == r).astype(int)
            n.loc[r, m] = int((a | b).sum())
            k = cohen_kappa(a, b) if len(part) and n.loc[r, m] else None
            kappa.loc[r, m] = np.nan if k is None else k
        k = cohen_kappa(part["primary_role"], part["arbiter_role"]) if len(part) else None
        kappa.loc["all", m] = np.nan if k is None else k
        n.loc["all", m] = len(part)
    return kappa, n


def role_precision(sheet: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Per primary role: precision of the primary verdict on the reviewed rows, and extrapolated to all judged rows.

    ``upheld`` = the reviewer sided with the primary judge (or found both judges right); ``decided`` rows
    carry any verdict. The extrapolation is (1 − d) · c + d · p, with d the design-weighted share of the
    sample's rows (of that role) on which the judges disagree, c the share of agreement controls the
    reviewer found both right, and p the share of reviewed *disagreements* of that role the reviewer
    decided for the primary judge.
    """
    dec = sheet[sheet["decided"]]
    controls = dec[dec["review_kind"] == "agreement_control"]
    c = float(controls["upheld"].mean()) if len(controls) else np.nan
    binary = sample[sample["both_binary"]]
    disagree = binary["primary_label"] != binary["arbiter_label"]
    rows = []
    for role in ROLE_ORDER + ["all"]:
        sel = dec if role == "all" else dec[dec["primary_role"] == role]
        pop = binary if role == "all" else binary[binary["primary_role"] == role]
        w = pop["design_weight"].to_numpy(dtype=float)
        d = float(w[disagree[pop.index]].sum() / w.sum()) if w.sum() else np.nan
        dis = sel[sel["review_kind"] == "disagreement"]
        p = float(dis["upheld"].mean()) if len(dis) else np.nan
        k, n = int(sel["upheld"].sum()), int(len(sel))
        lo, hi = wilson_interval(k, n)
        rows.append({"role": role, "k": k, "n": n, "precision": k / n if n else np.nan, "lo": lo, "hi": hi,
                     "disagreement_share": d, "n_disagreements": int(len(dis)),
                     "extrapolated": (1 - d) * c + d * p if n and not np.isnan(d) and not np.isnan(p) and not np.isnan(c) else np.nan})
    confirmed = dec[dec["primary_role"].isin(CLAIM_ROLES) & (dec["reliance_label"] == "robust_used")]
    k, n = int(confirmed["upheld"].sum()), int(len(confirmed))
    lo, hi = wilson_interval(k, n)
    rows.append({"role": "confirmed_false_rejection", "k": k, "n": n, "precision": k / n if n else np.nan, "lo": lo, "hi": hi,
                 "disagreement_share": np.nan, "n_disagreements": int((confirmed["review_kind"] == "disagreement").sum()),
                 "extrapolated": np.nan})
    out = pd.DataFrame(rows)
    out.attrs["controls_both_right"] = c
    out.attrs["n_controls"] = int(len(controls))
    return out


def weighted_metrics(sample: pd.DataFrame, models: list[str], *, seed: int, n_boot: int,
                     by_style: bool = False, cues: list[str] | None = None) -> pd.DataFrame:
    """Every J4 quantity per model (and cue) and label set, design-weighted with bootstrap intervals."""
    rows = sample[sample["both_binary"]]
    out = []
    for m in models:
        part_m = rows[rows["subject_model"] == m]
        groups = [(c, part_m[part_m["hint_style"] == c]) for c in (cues or [])] if by_style else [("all", part_m)]
        for cue, part in groups:
            if not len(part):
                continue
            has_rel = part["reliance_label"].notna().any()
            robust = (part["reliance_label"] == "robust_used").to_numpy()
            for ls in LABEL_SETS:
                label = part[f"{ls}_label"].to_numpy()
                role = part[f"{ls}_role"] if f"{ls}_role" in part else None
                claims = role.isin(CLAIM_ROLES).to_numpy() if role is not None and role.isin(JUDGE_ROLES).any() else None
                specs = {"unfaithful_ssp": (label == 0, np.ones(len(part), dtype=bool))}
                if claims is not None:
                    specs["claim_share"] = (claims, np.ones(len(part), dtype=bool))
                if has_rel:
                    specs["unfaithful_rsp"] = (robust & (label == 0), robust)
                    if claims is not None:
                        specs["false_claim_rate"] = (claims & robust, claims)
                for metric, (num, den) in specs.items():
                    est = weighted_rate_bootstrap(num, den, part["design_weight"], part["cell"], seed=seed, n_boot=n_boot)
                    out.append({"subject_model": m, "hint_style": cue, "label_set": ls, "metric": metric,
                                "rate": np.nan if est["rate"] is None else est["rate"],
                                "lo": est["ci95"][0] if est["ci95"] else np.nan, "hi": est["ci95"][1] if est["ci95"] else np.nan,
                                "n_num": est["n_numerator"], "n_den": est["n_denominator"]})
    return pd.DataFrame(out)


def population_rate(validation_dir: Path, model: str) -> float:
    """The primary judge's unfaithful rate over the whole frame of a model (label 0 among 0/1 rows)."""
    path = validation_dir / "frame.csv"
    if not path.exists():
        return np.nan
    f = pd.read_csv(path, usecols=["subject_model", "judge_label"])
    lab = _int_label(f.loc[f["subject_model"] == model, "judge_label"])
    lab = lab[lab.isin([0, 1])]
    return float((lab == 0).mean()) if len(lab) else np.nan


# --------------------------------------------------------------- drawing ---


def model_label(m: str) -> str:
    return MODEL_NAMES.get(m, m)


def kappa_heatmap(ax, fig, kappa: pd.DataFrame, n: pd.DataFrame, *, row_names: list[str], cbar: bool = True):
    norm = TwoSlopeNorm(vmin=0.0, vcenter=REFERENCE_KAPPA, vmax=1.0)
    shown = kappa.clip(lower=0.0)
    im = heatmap(ax, shown, cmap=DIV_CMAP, norm=norm, annotate=False, fig=fig if cbar else None,
                 cbar_label="Cohen's κ" if cbar else None, row_names=row_names, col_names=[model_label(m) for m in kappa.columns],
                 cbar_format=FuncFormatter(lambda v, _: f"{v:.2f}"))
    for i in range(kappa.shape[0]):
        for j in range(kappa.shape[1]):
            v, k = kappa.iloc[i, j], int(n.iloc[i, j])
            text = "—" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i - 0.12, text, ha="center", va="center", fontsize=7.5, color=INK)
            ax.text(j, i + 0.24, f"n = {k:,}", ha="center", va="center", fontsize=5.8, color=INK, alpha=0.75)
    if cbar and im.colorbar is not None:
        cb = im.colorbar
        cb.set_ticks([0, 0.35, REFERENCE_KAPPA, 1.0])
        cb.set_ticklabels(["0", "0.35", "0.69\nW&W", "1"])
        cb.ax.axhline(REFERENCE_KAPPA, color=INK, linewidth=0.8)
    return im


def dot_rows(ax, table: pd.DataFrame, *, y: np.ndarray, series: list[str], colors: dict, names: dict,
             offsets: dict, value_labels: bool = True, marker="o", size=24):
    """Dots with a CI whisker per (row, series); ``table`` has one row per (row index, series)."""
    for s in series:
        t = table[table["series"] == s].set_index("row")
        yy = np.array([y[i] + offsets[s] for i in t.index])
        ax.hlines(yy, t["lo"], t["hi"], color=INK, linewidth=0.7, zorder=2)
        ax.scatter(t["rate"], yy, s=size, marker=marker, color=colors[s], edgecolor=SURFACE, linewidth=0.9, zorder=3, label=names[s])
        if value_labels:
            for yi, v, hi in zip(yy, t["rate"], t["hi"]):
                if not pd.isna(v):
                    ax.text((hi if not pd.isna(hi) else v), yi, "  " + pct(v, 1), ha="left", va="center", fontsize=6, color=INK2)


def reference_line(ax, x: float, label: str):
    ax.axvline(x, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    right = x > 0.5 * sum(ax.get_xlim())
    ax.text(x, ax.get_ylim()[1], (label + " ") if right else (" " + label), ha="right" if right else "left",
            va="top", fontsize=6, color=INK2)


REGISTRY: dict[str, tuple] = {}


def register(figure_key: str, title: str):
    def deco(fn):
        REGISTRY[figure_key] = (title, fn)
        return fn
    return deco


# ---- J1 · κ primary vs arbiter -----------------------------------------------

@register("J1", "Agreement of the primary judge with the arbiter (Cohen's κ)")
def fig_j1(d: Inputs, cfg: dict) -> list[Variant]:
    out = []
    if d.sample is None:
        return out
    models = [m for m in MODEL_ORDER if m in set(d.sample["subject_model"])] or sorted(set(d.sample["subject_model"]))
    cues = [c for c in cfg["cues"] if c in set(d.sample["hint_style"])]
    kappa, n = kappa_grid(d.sample, cues, models)
    row_names = [CUE_NAMES.get(c, c) for c in cues] + ["all cues"]
    n_binary = int((d.sample["both_binary"] & d.sample["hint_style"].isin(cues)).sum())
    pooled = ", ".join(f"{model_label(m)} κ = {kappa.loc['all', m]:.2f} (n = {int(n.loc['all', m])})" for m in models)
    sources = "; ".join(s for s in d.notes if "corrected-prompt" in s)

    # a · heatmap, diverging around the 0.69 reference
    fig, ax = figure(HALF_W + 1.1, 0.34 * len(row_names) + 0.7)
    kappa_heatmap(ax, fig, kappa, n, row_names=row_names)
    ax.axhline(len(cues) - 0.5, color=INK2, linewidth=0.8)
    fig.tight_layout()
    out.append(Variant("J1", "a", "Cohen's κ between the primary judge (GLM-5.3-Flash) and the arbiter "
               "(GPT-5.6-Terra) on the binary verbalization label, per cue style and subject model, on the "
               f"stratified validation sample (MMLU-Pro; {n_binary} rows both judges labelled 0/1; n per cell "
               "printed). Colour diverges around κ = 0.69, the agreement Walden & Wanner report for their "
               "hint-use facet: blue cells agree better than that reference, red cells worse. The sample "
               "over-represents unfaithful verdicts by design, so κ is on a harder mix than the population. "
               f"Pooled: {pooled}." + (f" {sources}." if sources else ""), fig))

    # b · the same numbers as dots with the reference line
    fig, ax = figure(HALF_W + 1.3, 0.3 * len(row_names) + 0.9)
    y = np.arange(len(row_names))
    off = {m: o for m, o in zip(models, np.linspace(-0.18, 0.18, len(models)))}
    for m in models:
        ax.scatter(kappa[m].to_numpy(), y + off[m], s=26, color=MODEL_COLOR.get(m, SLOTS[0]), edgecolor=SURFACE,
                   linewidth=0.9, zorder=3, label=model_label(m))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}  (n {' / '.join(str(int(n.loc[c, m])) for m in models)})"
                        for name, c in zip(row_names, cues + ["all"])])
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Cohen's κ, primary judge vs arbiter")
    reference_line(ax, REFERENCE_KAPPA, REFERENCE_KAPPA_LABEL)
    ax.axhline(len(cues) - 0.5, color=AXIS, linewidth=0.6)
    tidy(ax, grid_axis="x")
    if len(models) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=len(models), columnspacing=1.2, handletextpad=0.3)
    else:
        ax.set_title(model_label(models[0]))
    fig.tight_layout()
    out.append(Variant("J1", "b", "The κ values of J1a as dots" + (", one colour per subject model" if len(models) > 1 else "")
               + ", with the 0.69 reference as a dashed line; n per cue (primary / arbiter rows both binary) in the row labels"
               + (", in the order of the legend." if len(models) > 1 else "."), fig))

    # c · per role, one-vs-rest
    rk, rn = role_kappa_grid(d.sample, models)
    if rn.to_numpy().sum():
        fig, ax = figure(HALF_W + 1.1, 0.34 * (len(ROLE_ORDER) + 1) + 0.7)
        kappa_heatmap(ax, fig, rk, rn, row_names=[ROLE_NAMES[r] for r in ROLE_ORDER] + ["all roles (5-class κ)"])
        ax.axhline(len(ROLE_ORDER) - 0.5, color=INK2, linewidth=0.8)
        fig.tight_layout()
        with_roles = int((d.sample["primary_role"].isin(JUDGE_ROLES) & d.sample["arbiter_role"].isin(JUDGE_ROLES)).sum())
        out.append(Variant("J1", "c", "Agreement on the role itself: one-vs-rest κ per role between the primary "
                   "judge's role and the arbiter's (n = rows on which either judge assigned the role), and the "
                   f"five-class κ over all roles, on the {with_roles} sampled rows where both judges gave a role. "
                   "The two mention-without-credit roles (verification only, rejected) are where the judges "
                   "part ways; no mention and credited are settled.", fig))
    return out


# ---- J2 · binary vs role -----------------------------------------------------

@register("J2", "Binary verbalization label vs role label, every switched rollout")
def fig_j2(d: Inputs, cfg: dict) -> list[Variant]:
    out = []
    if d.binary is None or not len(d.binary):
        return out
    b = d.binary
    models = [m for m in MODEL_ORDER if m in set(b["subject_model"])]
    single = len(models) == 1
    # facets: one panel / column per model, or per dataset when the figure covers a single model
    facet_col = "dataset" if single else "subject_model"
    facets = [x for x in (DATASET_ORDER if single else MODEL_ORDER) if x in set(b[facet_col])]
    facet_name = (lambda x: DATASET_NAMES.get(x, x)) if single else model_label
    facet_word = "dataset" if single else "model"
    scope = f"{model_label(models[0])}, " if single else ""
    total = len(b)
    per_facet_n = ", ".join(f"{facet_name(x)} {int((b[facet_col] == x).sum()):,}" for x in facets)
    inc_role = b.groupby("primary_role")["inconsistent"].mean().reindex(ROLE_ORDER)
    inc_text = ", ".join(f"{ROLE_NAMES[r]} {pct(inc_role[r], 1)}" for r in ROLE_ORDER if not pd.isna(inc_role[r]))

    # a · one confusion matrix per facet (row = role, column = binary label), colour = row share, text = count
    fig, axes = figure(FULL_W, 2.3, ncols=len(facets), sharey=True, gridspec_kw={"wspace": 0.12})
    axes = np.atleast_1d(axes)
    for ax, x in zip(axes, facets):
        part = b[b[facet_col] == x]
        counts = pd.crosstab(part["primary_role"], part["binary_label"]).reindex(index=ROLE_ORDER, columns=[1, 0], fill_value=0)
        share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        heatmap(ax, share, vmin=0, vmax=1, annotate=False, row_names=[ROLE_NAMES[r] for r in ROLE_ORDER],
                col_names=[BINARY_NAMES[c] for c in [1, 0]])
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                v, s = int(counts.iloc[i, j]), share.iloc[i, j]
                color = SURFACE if (not pd.isna(s) and s > 0.55) else INK
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=6.5, color=color)
        ax.set_title(f"{facet_name(x)}\nn = {len(part):,}", fontsize=7.5, loc="center")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    fig.supxlabel("binary judge's verdict", fontsize=7.5, color=INK2, y=-0.14)
    fig.tight_layout()
    out.append(Variant("J2", "a", "Every switched rollout" + (f" of {model_label(models[0])}" if single else " of the paper grid")
               + " carries two independent verdicts from the same judge model: the five-way role (rows; from the paper's "
               "judge prompt) and a plain binary faithful / unfaithful verdict from a separate minimal prompt (columns), "
               f"one panel per {facet_word}. Cells are counts, shaded by their share of the row. {total:,} rollouts with both "
               f"verdicts ({per_facet_n}); cells without role labels are absent. A credited role should read faithful and "
               "every other role unfaithful. The binary judge concurs on credited and no-mention rollouts and parts ways on "
               "the three mention-without-credit roles, above all neutral mentions, which it mostly reads as faithful. "
               f"Share of rollouts the binary judge contradicts, per role: {inc_text}.", fig))

    # b · inconsistency rate per role × facet
    inc = b.groupby(["primary_role", facet_col])["inconsistent"].mean().unstack().reindex(index=ROLE_ORDER, columns=facets)
    n = b.groupby(["primary_role", facet_col]).size().unstack().reindex(index=ROLE_ORDER, columns=facets).fillna(0).astype(int)
    fig, ax = figure(HALF_W + 1.6, 2.2)
    vmax = max(0.5, float(np.nanmax(inc.to_numpy())))
    heatmap(ax, inc, fig=fig, cbar_label="share the binary judge contradicts", vmin=0, vmax=vmax,
            annotate=False, row_names=[ROLE_NAMES[r] for r in ROLE_ORDER], col_names=[facet_name(x) for x in facets])
    for i in range(inc.shape[0]):
        for j in range(inc.shape[1]):
            v, k = inc.iloc[i, j], int(n.iloc[i, j])
            if pd.isna(v):
                continue
            color = SURFACE if v / vmax > 0.55 else INK
            ax.text(j, i - 0.12, pct(v, 1), ha="center", va="center", fontsize=7, color=color)
            ax.text(j, i + 0.25, f"n = {k:,}", ha="center", va="center", fontsize=5.6, color=color)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    if single:
        ax.set_title(model_label(models[0]))
    fig.tight_layout()
    out.append(Variant("J2", "b", f"Where the two verdicts disagree ({scope}per {facet_word}): the share of rollouts of each "
               f"role (rows) and {facet_word} (columns) whose binary verdict contradicts the role's implied label (credited → "
               "faithful, anything else → unfaithful), with the number of rollouts per cell. Disagreement is confined to the "
               "mention-without-credit roles — largest for neutral mentions, then verification only and rejected — and is "
               f"rare for credited and no mention, on every {facet_word}.", fig))

    # c · pooled confusion with row percentages
    counts = pd.crosstab(b["primary_role"], b["binary_label"]).reindex(index=ROLE_ORDER, columns=[1, 0], fill_value=0)
    share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = figure(HALF_W + 0.5, 2.2)
    heatmap(ax, share, vmin=0, vmax=1, annotate=False, row_names=[ROLE_NAMES[r] for r in ROLE_ORDER],
            col_names=[BINARY_NAMES[c] for c in [1, 0]])
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v, s = int(counts.iloc[i, j]), share.iloc[i, j]
            color = SURFACE if (not pd.isna(s) and s > 0.55) else INK
            ax.text(j, i - 0.12, f"{v:,}", ha="center", va="center", fontsize=7, color=color)
            ax.text(j, i + 0.25, pct(s, 1), ha="center", va="center", fontsize=5.8, color=color)
    ax.set_xlabel("binary judge's verdict")
    if single:
        ax.set_title(model_label(models[0]))
    fig.tight_layout()
    pooled_over = f"the four datasets of {model_label(models[0])}" if single else f"the {len(models)} models"
    out.append(Variant("J2", "c", f"The same confusion pooled over {pooled_over} ({total:,} rollouts): counts and "
               "row shares of the binary verdict for each role.", fig))
    return out


# ---- J3 · human spot-check ---------------------------------------------------

@register("J3", "Researcher spot-check: precision of the primary judge per role")
def fig_j3(d: Inputs, cfg: dict) -> list[Variant]:
    out = []
    if d.sheet is not None and d.sample is not None and d.sheet["decided"].any():
        prec = role_precision(d.sheet, d.sample)
        reviewed_roles = [r for r in ROLE_ORDER if int(prec.loc[prec["role"] == r, "n"].iloc[0]) > 0]
        order = reviewed_roles + ["all", "confirmed_false_rejection"]
        names = {**ROLE_NAMES, "all": "all roles", "confirmed_false_rejection": "confirmed false rejections\n(rejected or verif. only, robust_used)"}
        t = prec.set_index("role").reindex(order)
        y = np.arange(len(order))
        colors = [EMPH if r in CLAIM_ROLES or r == "confirmed_false_rejection" else DEEMPH for r in order]
        fig, ax = figure(FULL_W * 0.78, 0.33 * len(order) + 0.9)
        ax.hlines(y, t["lo"], t["hi"], color=INK, linewidth=0.8, zorder=2)
        ax.scatter(t["precision"], y, s=30, color=colors, edgecolor=SURFACE, linewidth=0.9, zorder=3)
        ok = t["extrapolated"].notna()
        ax.scatter(t.loc[ok, "extrapolated"], y[ok.to_numpy()], s=30, facecolor=SURFACE, edgecolor=[c for c, o in zip(colors, ok) if o],
                   linewidth=1.1, zorder=3)
        for yi, (k, n, hi) in enumerate(zip(t["k"], t["n"], t["hi"])):
            ax.text(1.07, yi, f"{int(k)}/{int(n)}", ha="left", va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([names[r] for r in order]); ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.04); pct_axis(ax, "x")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("share of reviewed verdicts the researcher upheld")
        ax.axhline(len(reviewed_roles) - 0.5, color=AXIS, linewidth=0.6)
        tidy(ax, grid_axis="x")
        handles = [Line2D([], [], marker="o", linestyle="none", color=EMPH, markeredgecolor=SURFACE, markersize=5.5,
                          label="on the reviewed rows (Wilson 95 %)"),
                   Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=EMPH, markersize=5.5,
                          label="extrapolated to all judged rows")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, columnspacing=1.0, handletextpad=0.3)
        fig.tight_layout()
        n_dec = int(prec.loc[prec["role"] == "all", "n"].iloc[0])
        unreviewed = [ROLE_NAMES[r] for r in ROLE_ORDER if r not in reviewed_roles]
        n_dis = int(d.sheet[d.sheet["decided"] & (d.sheet["review_kind"] == "disagreement")].shape[0])
        c = prec.attrs["controls_both_right"]
        cfr = prec[prec["role"] == "confirmed_false_rejection"].iloc[0]
        claim_rows = prec[prec["role"].isin(CLAIM_ROLES)]
        claim_text = ", ".join(f"{ROLE_NAMES[r.role]} {int(r.k)}/{int(r.n)}" for r in claim_rows.itertuples())
        out.append(Variant("J3", "a", "The researcher's spot-check, per role assigned by the primary judge: the share "
                   "of reviewed verdicts the researcher upheld (filled dots, Wilson 95 % intervals, k/n at right). The "
                   f"review sheet holds every primary-vs-arbiter disagreement ({n_dis} decided) plus {prec.attrs['n_controls']} "
                   f"random agreement controls ({pct(c, 0)} found right), so the filled dots are precision on the "
                   "contested rows; hollow dots extrapolate to all judged rows using each role's design-weighted "
                   f"disagreement share. Emphasised: the two roles behind the paper's false-claim rate ({claim_text} "
                   "upheld) and, at the bottom, the reviewed rollouts that are behaviourally confirmed false rejections "
                   f"(role rejected or verification only and robust_used on re-roll): {int(cfr.k)} of {int(cfr.n)} upheld. "
                   f"{n_dec} decided verdicts in all"
                   + (f"; no reviewed row carries the role {', '.join(unreviewed)}" if unreviewed else "") + ".", fig))

    if d.golden is not None and len(d.golden):
        g = d.golden
        rows = [("unfaithful verdicts (label 0)", g[g["judge_label"] == 0]), ("faithful verdicts (label 1)", g[g["judge_label"] == 1]), ("all", g)]
        y = np.arange(len(rows))
        fig, ax = figure(FULL_W * 0.7, 1.9)
        for yi, (name, part) in enumerate(rows):
            k, n = int((part["manual_label"] == part["judge_label"]).sum()), len(part)
            lo, hi = wilson_interval(k, n)
            ax.hlines(yi, lo, hi, color=INK, linewidth=0.8, zorder=2)
            ax.scatter(k / n if n else np.nan, yi, s=30, color=EMPH, edgecolor=SURFACE, linewidth=0.9, zorder=3)
            if part["manual_label_blind"].notna().any():
                kb = int((part["manual_label_blind"] == part["judge_label"]).sum())
                ax.scatter(kb / n if n else np.nan, yi, s=30, facecolor=SURFACE, edgecolor=EMPH, linewidth=1.1, zorder=3)
            ax.text(1.07, yi, f"{k}/{n}", ha="left", va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.04); pct_axis(ax, "x"); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("share of verdicts matching the human label")
        tidy(ax, grid_axis="x")
        handles = [Line2D([], [], marker="o", linestyle="none", color=EMPH, markeredgecolor=SURFACE, markersize=5.5, label="human label after review"),
                   Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=EMPH, markersize=5.5, label="blind first pass")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.62), ncol=2, columnspacing=1.0, handletextpad=0.3)
        fig.tight_layout()
        models = ", ".join(f"{model_label(m)} {int(c)}" for m, c in g["model"].value_counts().items())
        out.append(Variant("J3", "b", f"The human golden set: {len(g)} traces labelled by hand ({models}), drawn "
                   "balanced over the primary judge's verdict, and the share of the judge's verdicts that match the "
                   "human label (Wilson 95 %; hollow dots = the annotator's blind first pass, before seeing the judge's "
                   "reasoning). Unlike J3a this sample is not enriched for disagreements.", fig))
    return out


# ---- J4 · robustness to the judge -------------------------------------------

@register("J4", "Robustness of the headline to the judge")
def fig_j4(d: Inputs, cfg: dict) -> list[Variant]:
    out = []
    if d.sample is None:
        return out
    models = [m for m in MODEL_ORDER if m in set(d.sample["subject_model"])] or sorted(set(d.sample["subject_model"]))
    seed, n_boot = cfg["seed"], cfg["n_boot"]
    metrics = weighted_metrics(d.sample, models, seed=seed, n_boot=n_boot)
    n_rows = int(d.sample["both_binary"].sum())
    pop = {m: population_rate(cfg["validation_dir"], m) for m in models}
    offsets = dict(zip(LABEL_SETS, [-0.22, 0.0, 0.22]))

    def table(metric: str, ms: list[str] | None = None, cue: str = "all") -> pd.DataFrame:
        t = metrics[(metrics["metric"] == metric) & (metrics["hint_style"] == cue)]
        t = t[t["subject_model"].isin(ms or models)]
        return t.assign(row=t["subject_model"].map({m: i for i, m in enumerate(ms or models)}), series=t["label_set"])

    # a · the unfaithful rate per model under the three label sets
    t = table("unfaithful_ssp")
    fig, ax = figure(FULL_W * 0.8, max(2.0, 0.75 * len(models) + 0.9))
    y = np.arange(len(models))
    dot_rows(ax, t, y=y, series=LABEL_SETS, colors=LABEL_SET_COLOR, names=LABEL_SET_NAMES, offsets=offsets)
    for i, m in enumerate(models):
        if not np.isnan(pop[m]):
            ax.plot([pop[m], pop[m]], [i - 0.36, i + 0.36], color=INK2, linewidth=1.0, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([model_label(m) for m in models]); ax.invert_yaxis()
    pct_axis(ax, "x", top=min(1.0, float(t["hi"].max()) * 1.3 + 0.01))
    ax.set_xlabel("unfaithful rate among switched rollouts (MMLU-Pro)")
    tidy(ax, grid_axis="x")
    handles = [Line2D([], [], marker="o", linestyle="none", color=LABEL_SET_COLOR[s], markeredgecolor=SURFACE, markersize=5.5, label=LABEL_SET_NAMES[s]) for s in LABEL_SETS]
    handles.append(Line2D([], [], color=INK2, linewidth=1.0, label="primary judge, whole population"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.3 if len(models) > 1 else -0.5), ncol=2,
              columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout()
    rates = {(r.subject_model, r.label_set): (r.rate, r.lo, r.hi) for r in t.itertuples()}
    rate_text = " " + "; ".join(
        f"{model_label(m)}: " + ", ".join(f"{LABEL_SET_NAMES[s].split(' (')[0]} {pct(rates[(m, s)][0], 1)} "
                                          f"[{pct(rates[(m, s)][1], 1)}, {pct(rates[(m, s)][2], 1)}]" for s in LABEL_SETS)
        for m in models) + "."
    gap_text = ""
    if len(models) >= 2:
        a, b = models[0], models[1]
        gap_text = " Gap " + f"{model_label(a)} − {model_label(b)}: " + ", ".join(
            f"{LABEL_SET_NAMES[s].split(' (')[0]} {100 * (rates[(a, s)][0] - rates[(b, s)][0]):.1f} pp" for s in LABEL_SETS) + "."
    out.append(Variant("J4", "a", "The unfaithful rate (label 0 among switched, judged rollouts) recomputed on the "
               f"validation sample under three label sets: the primary judge, the arbiter and the reviewer-adjudicated "
               "labels (every reviewed disagreement takes the researcher's verdict, everything else the primary "
               f"judge's). Same {n_rows} rows for all three, re-weighted to the population by the sampling design, "
               "95 % intervals from a stratified bootstrap; the grey tick is the primary judge's rate over the whole "
               "population of judged switches (under the manifest's verdicts; for a run whose sampled rows were "
               f"re-judged on the corrected prompt the sample uses those verdicts instead).{rate_text}{gap_text} "
               "The primary judge's interval is zero-width by construction: the sampling strata are defined by its own "
               "label, so re-drawing within strata never changes its rate."
               + (" The ranking does not change with the judge." if len(models) >= 2 else ""), fig))

    # b · per cue style, primary vs arbiter, one panel per model
    cues = [c for c in cfg["cues"] if c in set(d.sample["hint_style"])]
    ms = weighted_metrics(d.sample, models, seed=seed, n_boot=n_boot, by_style=True, cues=cues)
    ms = ms[(ms["metric"] == "unfaithful_ssp") & (ms["label_set"] != "reviewer")]
    fig, axes = figure(FULL_W if len(models) > 1 else FULL_W * 0.6, 0.3 * len(cues) + 1.2, ncols=len(models), sharey=True,
                       gridspec_kw={"wspace": 0.25})
    axes = np.atleast_1d(axes)
    yy = np.arange(len(cues))
    two = {"primary": -0.17, "arbiter": 0.17}
    for ax, m in zip(axes, models):
        t = ms[ms["subject_model"] == m].assign(row=lambda x: x["hint_style"].map({c: i for i, c in enumerate(cues)}), series=lambda x: x["label_set"])
        dot_rows(ax, t, y=yy, series=["primary", "arbiter"], colors=LABEL_SET_COLOR, names=LABEL_SET_NAMES, offsets=two, value_labels=False, size=20)
        ax.set_title(model_label(m))
        top = float(t["hi"].max()) if len(t) and not np.isnan(t["hi"].max()) else 1.0
        pct_axis(ax, "x", top=min(1.0, top * 1.2 + 0.005))
        ax.set_xlabel("unfaithful rate", fontsize=7)
        tidy(ax, grid_axis="x")
    axes[0].set_yticks(yy); axes[0].set_yticklabels([CUE_NAMES.get(c, c) for c in cues]); axes[0].invert_yaxis()
    axes[-1].legend(handles=[Line2D([], [], marker="o", linestyle="none", color=LABEL_SET_COLOR[s], markeredgecolor=SURFACE, markersize=5, label=LABEL_SET_NAMES[s]) for s in ["primary", "arbiter"]],
                    loc="upper center", bbox_to_anchor=(-0.15 if len(models) > 1 else 0.4, -0.22), ncol=2 if len(models) > 1 else 1,
                    columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout()
    out.append(Variant("J4", "b", "The same recomputation per cue style, primary judge against arbiter, one panel "
               "per subject model with its own axis (design-weighted, stratified bootstrap 95 %). The cue "
               "fingerprint — which styles are concealed most — is the same under both judges; the arbiter reads "
               "the post-hoc cue somewhat more leniently.", fig))

    # c · the paper's re-roll-based quantities where reliance labels exist
    have_rel = [m for m in models if (d.sample[d.sample["subject_model"] == m]["reliance_label"].notna()).any()]
    specs = [("unfaithful_rsp", "unfaithful rate, robust_used switches (RSP)", have_rel, LABEL_SETS),
             ("claim_share", "claims not to rely on the cue\n(rejected or verification only), share of switches", models, ["primary", "arbiter"]),
             ("false_claim_rate", "false-claim rate:\nrobust_used among those claims", have_rel, ["primary", "arbiter"])]
    rows, labels = [], []
    for metric, name, ms_, sets in specs:
        for m in ms_:
            part = metrics[(metrics["metric"] == metric) & (metrics["subject_model"] == m) & (metrics["label_set"].isin(sets))]
            if not len(part):
                continue
            rows.append(part.assign(row=len(labels), series=part["label_set"]))
            labels.append(f"{name}\n{model_label(m)}")
    if rows:
        t = pd.concat(rows, ignore_index=True)
        left = t[t["metric"] != "false_claim_rate"]
        right = t[t["metric"] == "false_claim_rate"]
        panels = [(left, "share of switched rollouts"), (right, "share of those claims")]
        panels = [(tt, lab) for tt, lab in panels if len(tt)]
        widths = [3, 2][: len(panels)]
        fig, axes = figure(FULL_W, 0.55 * len(labels) + 1.2, ncols=len(panels), gridspec_kw={"width_ratios": widths, "wspace": 0.55})
        axes = np.atleast_1d(axes)
        for k, (ax, (tt, xlabel)) in enumerate(zip(axes, panels)):
            rows_here = sorted(tt["row"].unique())
            remap = {r: i for i, r in enumerate(rows_here)}
            tt = tt.assign(row=tt["row"].map(remap))
            y = np.arange(len(rows_here))
            dot_rows(ax, tt, y=y, series=LABEL_SETS, colors=LABEL_SET_COLOR, names=LABEL_SET_NAMES, offsets=offsets)
            ticks = [labels[r] for r in rows_here] if k == 0 else [labels[r].split("\n")[-1] for r in rows_here]
            if k > 0:
                ax.set_title("\n".join(labels[rows_here[0]].split("\n")[:-1]), fontsize=7)
            ax.set_yticks(y); ax.set_yticklabels(ticks, fontsize=6.6); ax.invert_yaxis()
            ax.set_ylim(len(rows_here) - 0.5, -0.5)
            pct_axis(ax, "x", top=min(1.0, float(tt["hi"].max()) * 1.35 + 0.02))
            ax.set_xlabel(xlabel, fontsize=7)
            tidy(ax, grid_axis="x")
        axes[0].legend(handles=handles[:3], loc="upper center", bbox_to_anchor=(0.9 if len(panels) > 1 else 0.4, -0.2), ncol=3,
                       columnspacing=1.0, handletextpad=0.3)
        fig.tight_layout()
        counts = "; ".join(f"{r.metric.replace('_', ' ')} · {model_label(r.subject_model)} · {r.label_set}: {r.n_num}/{r.n_den} rows"
                           for r in t.itertuples() if r.metric != "unfaithful_ssp")
        out.append(Variant("J4", "c", "The paper's re-roll-based quantities recomputed on the validation sample: the "
                   "RSP unfaithful rate (label 0 among switches whose pair is robust_used), the share of switches whose "
                   "reasoning claims not to rely on the cue (role rejected or verification only), and the false-claim "
                   "rate (robust_used among those claims), each under the primary judge's and the arbiter's labels or "
                   "roles (and the reviewer's labels where only the label matters). Reliance labels exist for "
                   f"{', '.join(model_label(m) for m in have_rel) or 'no model'} in the sample; roles are not reviewed, "
                   f"so the reviewer set has no role-based rows. Raw counts (numerator/denominator): {counts}.", fig))
    return out


# --------------------------------------------------------------------- CLI ---


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validation-dir", default=DEFAULT_VALIDATION_DIR, help="the judge_validation run directory")
    parser.add_argument("--cueball-dir", default=DEFAULT_CUEBALL_DIR, help="the paper tree (rollout manifest + binary_judge/)")
    parser.add_argument("--shared-manifest", default=DEFAULT_SHARED_MANIFEST, help="manifest holding the sampled runs' roles")
    parser.add_argument("--backfill-cache-dir", default=DEFAULT_BACKFILL_CACHE_DIR, help="prompt-backfill judge caches")
    parser.add_argument("--reliance", default=DEFAULT_RELIANCE, help="question_reliance.csv covering the sampled runs")
    parser.add_argument("--overrides", default=DEFAULT_OVERRIDES, help="judge_label_overrides.csv (the reviewer's labels)")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, help="the manual golden set (optional)")
    parser.add_argument("--out", default=None, help="output directory (default: <cueball-dir>/plots/judge_validation)")
    parser.add_argument("--only", default=None, help="comma-separated figures or variants, e.g. J1,J4a")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--cues", default="paper",
                        help="'paper' or 'all' (both: the eight cue styles of the paper), or comma-separated cue ids, e.g. "
                             "expert_opinion,post_hoc, which restricts the per-cue rows of J1a/J1b/J4b, their pooled row and "
                             "the J2 rollouts to those cues (drawn in the paper's cue order); J1c, J3 and J4a/J4c use every sampled row")
    parser.add_argument("--models", default="paper",
                        help="'paper' (the five models of the paper), 'all', or comma-separated model ids, e.g. nemotron-nano-9b-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--no-cache", action="store_true", help="re-read the binary-judge CSVs instead of the cached labels")
    args = parser.parse_args(argv)

    cueball = resolve_data_path(args.cueball_dir)
    validation_dir = resolve_data_path(args.validation_dir)
    out_dir = resolve_data_path(args.out) if args.out else cueball / "plots" / "judge_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    wanted = {w.strip() for w in args.only.split(",")} if args.only else None
    if args.models == "paper":
        models = PAPER_MODELS
    elif args.models == "all":
        models = MODEL_ORDER
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = sorted(set(models) - set(MODEL_ORDER))
        if unknown:
            parser.error(f"unknown model id(s) {unknown}; known: {MODEL_ORDER}")
    if args.cues in ("paper", "all"):
        cues = PAPER_CUES if args.cues == "paper" else CUE_ORDER
    else:
        given = {c.strip() for c in args.cues.split(",") if c.strip()}
        unknown = sorted(given - set(CUE_ORDER))
        if unknown or not given:
            parser.error(f"unknown cue id(s) {unknown}; known: {CUE_ORDER}" if unknown else "--cues names no cue")
        cues = [c for c in CUE_ORDER if c in given]          # paper order, de-duplicated
    cfg = {"cues": cues, "models": models,
           "seed": args.seed, "n_boot": args.n_boot, "validation_dir": validation_dir}

    apply_style()
    notes: list[str] = []
    print(f"loading the validation sample from {validation_dir} …", flush=True)
    sample = load_sample(validation_dir, resolve_data_path(args.shared_manifest), resolve_data_path(args.backfill_cache_dir),
                         resolve_data_path(args.reliance), resolve_data_path(args.overrides), notes, models=cfg["models"])
    sheet = load_sheet(validation_dir, sample, notes)
    if sheet is not None:
        sheet = sheet[sheet["subject_model"].isin(cfg["models"])].reset_index(drop=True)
    need_binary = wanted is None or any(w.startswith("J2") for w in wanted)
    binary = None
    if need_binary:
        print(f"loading binary-judge verdicts from {cueball / 'binary_judge'} …", flush=True)
        binary = load_binary_vs_role(cueball, None if args.no_cache else out_dir / "_cache", cfg["models"], cfg["cues"], notes)
    golden = load_golden(resolve_data_path(args.golden), notes)
    if golden is not None and "model" in golden.columns:
        golden = golden[golden["model"].isin(cfg["models"])].reset_index(drop=True)
    data = Inputs(sample=sample, sheet=sheet, binary=binary, golden=golden, notes=notes)
    for n in notes:
        print(f"  note: {n}", flush=True)

    index = ["# Judge-validation figures", "", f"Validation run: `{validation_dir}`; grid: `{cueball}`; models: "
             f"{', '.join(model_label(m) for m in cfg['models'])} — generated by `src.scripts.visualizations.judge_validation_plots`.", ""]
    if notes:
        index += ["Notes:", ""] + [f"- {n}" for n in notes] + [""]
    n = 0
    for key, (title, fn) in REGISTRY.items():
        if wanted and not any(w == key or (w.startswith(key) and len(w) > len(key)) for w in wanted):
            continue
        index += [f"## {key} — {title}", ""]
        for v in fn(data, cfg):
            if wanted and not (key in wanted or v.name in wanted):
                plt.close(v.fig); continue
            for ext in formats:
                v.fig.savefig(out_dir / f"{v.name}.{ext}", bbox_inches="tight", pad_inches=0.02)
            plt.close(v.fig)
            index += [f"**{v.name}** — {v.caption}", ""]
            n += 1
            print(f"  wrote {v.name}", flush=True)
    (out_dir / "figures_judge.md").write_text("\n".join(index) + "\n")
    print(f"{n} figure(s) → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
