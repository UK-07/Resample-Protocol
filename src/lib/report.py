"""Evaluation-side breakdown of probe predictions: every metric pooled / positive-only / negative-only.

A predictions table (``rollout_id``, ``p_unfaithful``, optional ``predicted_label``) is joined to
the manifest, labels come from the manifest's label column (never from the predictions file), and
the metrics are reported per case view, per group, with a KS case-shift diagnostic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.lib.probe_metrics import auroc, summarize
from src.lib.rollout_manifest import CASES
from src.lib.selection import encode_labels

CASE_VIEWS = ("pooled",) + CASES
PRIMARY_VIEW = "positive"
PREDICTION_COLUMNS = ["rollout_id", "p_unfaithful"]
DEFAULT_GROUP_KEYS = ("hint_style", "subject_model", "dataset")
DEFAULT_LABEL_COL = "judge_label_final"
# The metric columns of the rendered tables, in order.
TABLE_METRICS = [
    "n", "n_unfaithful", "n_faithful", "auroc", "accuracy", "balanced_accuracy",
    "precision_unfaithful", "recall_unfaithful", "recall_at_1pct_fpr", "recall_at_5pct_fpr", "ece",
]


def load_predictions(path) -> pd.DataFrame:
    """A predictions CSV (:data:`PREDICTION_COLUMNS`, optionally ``predicted_label``), validated."""
    return validate_predictions(pd.read_csv(path))


def validate_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PREDICTION_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"predictions are missing column(s) {missing}")
    out = pred.copy()
    out["rollout_id"] = out["rollout_id"].astype(str)
    dupes = out["rollout_id"][out["rollout_id"].duplicated()]
    if len(dupes):
        raise ValueError(f"predictions hold duplicate rollout_id(s), e.g. {dupes.iloc[0]!r}")
    score = pd.to_numeric(out["p_unfaithful"], errors="coerce")
    if score.isna().any() or ((score < 0) | (score > 1)).any():
        raise ValueError("p_unfaithful must be a probability in [0, 1] on every row")
    out["p_unfaithful"] = score.astype(float)
    if "predicted_label" in out.columns:
        lab = pd.to_numeric(out["predicted_label"], errors="coerce")
        if lab.isna().any() or not lab.isin([0, 1]).all():
            raise ValueError("predicted_label must be 0 or 1 on every row")
        out["predicted_label"] = lab.astype(int)
    return out


def join_predictions(pred: pd.DataFrame, manifest: pd.DataFrame, *, label_col: str = DEFAULT_LABEL_COL,
                     group_keys=DEFAULT_GROUP_KEYS) -> pd.DataFrame:
    """The predictions joined to their manifest rows, with ``label`` encoded from ``label_col``.

    Every prediction must name a manifest row, and every joined row must carry a valid case.
    """
    pred = validate_predictions(pred)
    keep = ["rollout_id", "case", label_col] + [k for k in group_keys if k in manifest.columns]
    keep += [c for c in ("split",) if c in manifest.columns]
    missing = [c for c in ("rollout_id", "case", label_col) if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing column(s) {missing}")
    rows = manifest[list(dict.fromkeys(keep))].copy()
    rows["rollout_id"] = rows["rollout_id"].astype(str)
    if rows["rollout_id"].duplicated().any():
        raise ValueError("manifest holds duplicate rollout_id(s)")
    joined = pred.merge(rows, on="rollout_id", how="left", indicator=True)
    unknown = joined.loc[joined["_merge"] != "both", "rollout_id"]
    if len(unknown):
        raise ValueError(f"{len(unknown)} prediction(s) name rollout_id(s) the manifest lacks, e.g. {unknown.iloc[0]!r}")
    joined = joined.drop(columns="_merge")
    bad_case = joined["case"].isna() | ~joined["case"].astype(object).isin(CASES)
    if bad_case.any():
        raise ValueError(f"{int(bad_case.sum())} joined row(s) carry no valid case")
    joined["label"] = encode_labels(joined, label_col).to_numpy()
    return joined


def case_view(joined: pd.DataFrame, view: str) -> pd.DataFrame:
    """The rows of one case view: ``pooled`` (all), ``positive`` or ``negative``."""
    if view not in CASE_VIEWS:
        raise ValueError(f"unknown view {view!r}; choose from {CASE_VIEWS}")
    if view == "pooled":
        return joined
    return joined[joined["case"].astype(object) == view]


def _metrics(rows: pd.DataFrame) -> dict:
    out = summarize(rows["label"].to_numpy(), rows["p_unfaithful"].to_numpy())
    if "predicted_label" in rows.columns and len(rows):
        # The caller's own decision rule (threshold, calibration) next to the 0.5 one.
        pred_unf = rows["predicted_label"].to_numpy() == 0
        is_unf = rows["label"].to_numpy() == 0
        out["accuracy_predicted_label"] = float((pred_unf == is_unf).mean())
    return out


def three_way(joined: pd.DataFrame) -> dict[str, dict]:
    """:func:`summarize` for each of :data:`CASE_VIEWS`."""
    return {view: _metrics(case_view(joined, view)) for view in CASE_VIEWS}


def three_way_by_group(joined: pd.DataFrame, key: str) -> dict[str, dict[str, dict]]:
    """Per value of ``key`` (sorted), the three views."""
    out: dict[str, dict[str, dict]] = {}
    for value, rows in joined.groupby(joined[key].astype(str), sort=True):
        out[str(value)] = three_way(rows)
    return out


def ks_two_sample(a, b) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov statistic and asymptotic two-sided p-value (NaN when a side is empty)."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return math.nan, math.nan
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / n1
    cdf_b = np.searchsorted(b, grid, side="right") / n2
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_eff = n1 * n2 / (n1 + n2)
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
    if lam == 0:
        return d, 1.0
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def ks_case_shift(joined: pd.DataFrame) -> dict[str, dict]:
    """Per class (``unfaithful`` = label 0, ``faithful`` = label 1): KS statistic and p-value
    between the scores of positive-case and negative-case rows, with both counts."""
    out = {}
    for name, label in (("unfaithful", 0), ("faithful", 1)):
        rows = joined[joined["label"] == label]
        pos = rows.loc[rows["case"].astype(object) == "positive", "p_unfaithful"].to_numpy()
        neg = rows.loc[rows["case"].astype(object) == "negative", "p_unfaithful"].to_numpy()
        stat, p = ks_two_sample(pos, neg)
        out[name] = {
            "ks_statistic": stat, "p_value": p, "n_positive": int(len(pos)), "n_negative": int(len(neg)),
            "mean_score_positive": float(pos.mean()) if len(pos) else math.nan,
            "mean_score_negative": float(neg.mean()) if len(neg) else math.nan,
        }
    return out


def delta_auroc(joined_a: pd.DataFrame, joined_b: pd.DataFrame, *, view: str = PRIMARY_VIEW) -> dict:
    """AUROC of two prediction sets on the same ``view`` and their difference (``a − b``).

    Both sets must cover the same rollouts.
    """
    a, b = case_view(joined_a, view), case_view(joined_b, view)
    ids_a, ids_b = set(a["rollout_id"]), set(b["rollout_id"])
    if ids_a != ids_b:
        raise ValueError(f"the two prediction sets cover different rollouts on view {view!r} "
                         f"({len(ids_a ^ ids_b)} differ)")
    auc_a = auroc(a["label"].to_numpy(), a["p_unfaithful"].to_numpy())
    auc_b = auroc(b["label"].to_numpy(), b["p_unfaithful"].to_numpy())
    return {"view": view, "n": int(len(a)), "auroc_a": auc_a, "auroc_b": auc_b, "delta_auroc": auc_a - auc_b}


def build_report(pred: pd.DataFrame, manifest: pd.DataFrame, *, label_col: str = DEFAULT_LABEL_COL,
                 group_keys=DEFAULT_GROUP_KEYS, name: str | None = None) -> dict:
    """Everything: the three-way overall metrics, per-group three-way blocks, the KS diagnostic."""
    joined = join_predictions(pred, manifest, label_col=label_col, group_keys=group_keys)
    keys = [k for k in group_keys if k in joined.columns]
    return {
        "name": name,
        "label_col": label_col,
        "primary_view": PRIMARY_VIEW,
        "views": list(CASE_VIEWS),
        "n_predictions": int(len(joined)),
        "n_by_case": {c: int((joined["case"].astype(object) == c).sum()) for c in CASES},
        "splits": sorted(joined["split"].dropna().astype(str).unique().tolist()) if "split" in joined else [],
        "overall": three_way(joined),
        "by_group": {k: three_way_by_group(joined, k) for k in keys},
        "ks_case_shift": ks_case_shift(joined),
    }


def three_way_table(blocks: dict[str, dict], *, metrics=TABLE_METRICS) -> pd.DataFrame:
    """A ``metric × view`` table from one three-way block (missing metrics are NaN)."""
    return pd.DataFrame({view: [blocks.get(view, {}).get(m, math.nan) for m in metrics] for view in CASE_VIEWS},
                        index=list(metrics))


def group_table(by_group: dict[str, dict[str, dict]], *, metric: str = "auroc",
                counts: bool = True) -> pd.DataFrame:
    """One row per group value: ``<metric>`` per view (plus ``n`` per view when ``counts``)."""
    rows = []
    for value, blocks in by_group.items():
        row = {"group": value}
        for view in CASE_VIEWS:
            row[f"{metric}_{view}"] = blocks[view].get(metric, math.nan)
            if counts:
                row[f"n_{view}"] = blocks[view].get("n", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_table(df: pd.DataFrame, **kwargs) -> str:
    try:
        return df.to_markdown(floatfmt=".3f", **kwargs)
    except ImportError:  # tabulate is optional
        return df.to_string(float_format="{:.3f}".format, **kwargs)


def render_markdown(report: dict) -> str:
    title = report.get("name") or "probe evaluation"
    lines = [
        f"# {title}: three-way case breakdown",
        "",
        f"Label column `{report['label_col']}`; {report['n_predictions']} predictions "
        f"({', '.join(f'{c}: {n}' for c, n in report['n_by_case'].items())})"
        + (f"; split(s) {', '.join(report['splits'])}" if report.get("splits") else "") + ".",
        "",
        f"**Primary (pre-registered): `{report['primary_view']}`.** `pooled` is secondary, `negative` exploratory "
        "(a negative-case flip can be genuine re-derivation, so its used/verbalised labels are muddier).",
        "",
        "## Overall",
        "",
        _fmt_table(three_way_table(report["overall"])),
        "",
    ]
    for key, blocks in report.get("by_group", {}).items():
        lines += [f"## AUROC by {key}", "", _fmt_table(group_table(blocks), index=False), ""]
    ks = report.get("ks_case_shift", {})
    if ks:
        table = pd.DataFrame(ks).T
        lines += [
            "## Case-shift diagnostic (KS, positive vs negative scores within a class)",
            "",
            "A large statistic means the probe scores the class differently by case — it partially encodes "
            "case (correctness-related features) rather than reliance / verbalisation.",
            "",
            _fmt_table(table),
            "",
        ]
    return "\n".join(lines)


__all__ = [
    "CASE_VIEWS", "DEFAULT_GROUP_KEYS", "DEFAULT_LABEL_COL", "PREDICTION_COLUMNS", "PRIMARY_VIEW", "TABLE_METRICS",
    "build_report", "case_view", "delta_auroc", "group_table", "join_predictions", "ks_case_shift", "ks_two_sample",
    "load_predictions", "render_markdown", "three_way", "three_way_by_group", "three_way_table",
    "validate_predictions",
]
