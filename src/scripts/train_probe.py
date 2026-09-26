#!/usr/bin/env python3
"""Train the TF-IDF / linear / attention probes with leave-one-hint-out evaluation.

One YAML config names a probe dataset parquet, the probe type and its hyper-parameter grid
(the loop lives in :mod:`src.lib.probe_training`). Per held-out hint style one probe per
*unit* — ``(layer, "na")`` for attention, ``(layer, point)`` for linear, ``("na", "na")``
for TF-IDF — is trained over every grid point, the best trial per unit is kept by validation
AUROC, and both test sets (``test_id`` / ``test_ood``) are scored per unit and as their
mean (the ensemble). Run directory::

    <output.dir>/<stamp>_<run_name>/
        config.yaml            the resolved config      run.log
        folds.json             the fold plan            search_results.csv
        fold_<held_out_style>/
            predictions/<test_set>__layer_<L>__<point>.csv   rollout_id, p_unfaithful
            predictions/<test_set>__ensemble.csv
            report/<test_set>__layer_<L>__<point>_report.{json,md}   (+ the ensemble's)
            probe__layer_<L>__<point>.pt                     vectorizer.joblib (tfidf)
        results.json  results_summary.csv  aggregate.json  ablation.json (--compare)
        plots/auroc_by_unit.png  plots/auroc_by_fold.png  plots/training_curves_<fold>.png

Every predictions file carries :data:`src.lib.report.PREDICTION_COLUMNS`, so
``probe_eval_report.py`` scores it unchanged.

Usage::

    uv run python -m src.scripts.train_probe --config configs/probes/nemotron_tfidf.yaml --dry-run
    uv run python -m src.scripts.train_probe --config configs/probes/nemotron_linear.yaml --folds metadata,consensus
    uv run python -m src.scripts.train_probe --config configs/probes/nemotron_attention.yaml --only-trial lr=0.001,aggregation=softmax
    uv run python -m src.scripts.train_probe --config <pooled run's config> --compare <positive-only run dir>

The activation probes read the store folder ``model.activations.folder`` (null = the
dataset's ``name``); the store's ``dataset_spec.fingerprint`` need not equal the training
dataset's, but every fold id must be in the store manifest. The attention probe's
``model.layers`` must lie within the store's ``seq_layers`` (and, on a local mirror, its
``mirrored_layers``); the linear probe reads the points of every layer in ``layers``.
``--dry-run`` reads the dataset and the store manifest / text-cache status, prints the fold
plan, the grid and the size estimates, writes ``config.yaml`` + ``dry_run.json`` and reads
no tensor.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.lib.activation_store import (  # noqa: E402
    GB,
    MIRROR_SOURCE_KEY,
    POINTS,
    STORE_FILE_KINDS,
    LayerSelectionError,
    PointStore,
    SequenceStore,
    cgroup_memory_limit_bytes,
    is_mirror,
    make_backend,
    read_store_manifest,
    resolve_kind_layers,
    resolve_store_layers,
    stored_layers,
)
from src.lib.config import load_config  # noqa: E402
from src.lib.probe_datasets import dataset_fingerprint, read_dataset  # noqa: E402
from src.lib.probe_training import (  # noqa: E402
    ENSEMBLE_KEY,
    NA,
    TEST_SETS,
    Fold,
    FoldResult,
    PointSource,
    SequenceSource,
    TfidfSource,
    TrainConfig,
    Trial,
    evaluate,
    expand_grid,
    folds_to_json,
    parse_train_config,
    plan_folds,
    train_fold,
)
from src.lib.report import (  # noqa: E402
    CASE_VIEWS,
    PRIMARY_VIEW,
    TABLE_METRICS,
    build_report,
    delta_auroc,
    join_predictions,
    load_predictions,
    render_markdown,
)
from src.lib.text_features import (  # noqa: E402
    TextPipelineConfig,
    _read_cache as read_text_cache,
    cache_path,
    dataset_name_for,
    normalised_corpus,
    reasoning_sha,
    save_vectorizer,
)

RUN_LOG = "run.log"
PLOTS_DIR = "plots"
FOLD_ROLES = (("train", "train_ids"), ("val", "val_ids"), ("test_id", "test_id_ids"), ("test_ood", "test_ood_ids"))
#: The metrics ``aggregate.json`` summarises over folds.
AGGREGATE_METRICS = ("auroc", "recall_at_1pct_fpr", "recall_at_5pct_fpr", "balanced_accuracy")
#: The view the plots show.
PLOT_VIEW = "pooled"
#: The unit slot of the per-fold ensemble in the metrics tables.
ENSEMBLE_UNIT = (ENSEMBLE_KEY, NA)
BYTES_PER_ELEMENT = 2   # the stores hold bf16
LOGGER_NAME = "train_probe"


def unit_name(unit) -> str:
    """``layer_<L>__<point>`` (``layer_na__na`` for TF-IDF), ``ensemble`` for the per-fold ensemble."""
    if unit == ENSEMBLE_UNIT or unit == ENSEMBLE_KEY:
        return ENSEMBLE_KEY
    layer, point = unit
    return f"layer_{layer}__{point}"


def unit_fields(unit) -> tuple[str, str]:
    """The ``(layer, point)`` strings of a unit for the CSV / JSON tables."""
    if unit == ENSEMBLE_UNIT or unit == ENSEMBLE_KEY:
        return ENSEMBLE_UNIT
    return str(unit[0]), str(unit[1])


def _jsonable(obj):
    """Numpy scalars → Python, NaN / ±inf → null, paths → strings, tuples → lists."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    return path


def _fmt_gb(n_bytes: float) -> str:
    return f"{n_bytes / GB:.2f} GB"


def setup_logging(run_dir: Path) -> logging.Logger:
    """``run.log`` (DEBUG) + stdout (INFO) on the ``train_probe`` logger; earlier handlers closed."""
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    close_logging(log)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(run_dir / RUN_LOG, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def close_logging(log: logging.Logger) -> None:
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


def make_run_dir(cfg: TrainConfig, *, now: Optional[datetime] = None) -> Path:
    """``<output.dir>/<stamp>_<run_name>/``; an existing directory gets a ``_2``, ``_3``, … suffix."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{cfg.output.run_name}" if cfg.output.run_name else stamp
    run_dir = cfg.output.dir / base
    k = 1
    while run_dir.exists():
        k += 1
        run_dir = cfg.output.dir / f"{base}_{k}"
    run_dir.mkdir(parents=True)
    return run_dir


def resolved_config(cfg: TrainConfig, folds: Optional[Sequence[str]], trials: Optional[Sequence[Trial]] = None) -> dict:
    """``cfg.to_dict()`` with the effective fold list and, under ``--only-trial``, the ``search:``
    block narrowed to that trial's overrides — what ``config.yaml`` holds (re-parseable)."""
    out = cfg.to_dict()
    if folds is not None:
        out["dataset"]["folds"] = list(folds)
    if trials is not None and len(trials) == 1 and any(isinstance(v, list) for v in cfg.search.values()):
        out["search"] = json.loads(json.dumps(dict(trials[0].overrides)))
    return out


def fold_plan_records(plan: Sequence[Fold], df: pd.DataFrame) -> list[dict]:
    """One record per (fold, role): rows, questions, label-0 and label-1 counts."""
    by_id = df.set_index(df["rollout_id"].astype(str))
    labels = by_id["label"].astype(int)
    questions = by_id["question_id"].astype(str)
    out = []
    for fold in plan:
        for role, attr in FOLD_ROLES:
            ids = list(getattr(fold, attr))
            lab = labels.loc[ids] if ids else labels.iloc[0:0]
            out.append({
                "held_out_style": fold.held_out_style, "role": role, "n_rows": len(ids),
                "n_questions": int(questions.loc[ids].nunique()) if ids else 0,
                "n_label_0": int((lab == 0).sum()), "n_label_1": int((lab == 1).sum()),
            })
    return out


def fold_plan_text(records: list[dict]) -> str:
    return pd.DataFrame(records).to_string(index=False)


def sequence_bytes(n_tokens: dict, ids: Sequence[str], *, n_layers: int, hidden: int) -> int:
    """Σ n_cot_tokens × layers × hidden × 2 over ``ids``."""
    return int(sum(int(n_tokens[str(i)]) for i in ids)) * n_layers * hidden * BYTES_PER_ELEMENT


def point_bytes(n_rows: int, *, n_points: int, n_layers: int, hidden: int) -> int:
    """rows × points × layers × hidden × 2 (what ``PointSource`` materialises per fold)."""
    return int(n_rows) * n_points * n_layers * hidden * BYTES_PER_ELEMENT


def dataloader_ram_bytes(cfg: TrainConfig, *, n_layers: int, hidden: int) -> int:
    """Host RAM of ``(num_workers × prefetch_factor + 1)`` padded batches of ``max_batch_tokens`` tokens."""
    slots = cfg.dataloader.num_workers * cfg.dataloader.prefetch_factor + 1
    return slots * cfg.training.max_batch_tokens * n_layers * hidden * BYTES_PER_ELEMENT


def size_estimates(cfg: TrainConfig, plan: Sequence[Fold], *, n_layers: int, hidden: int,
                   n_tokens: Optional[dict], n_trials: int = 1) -> dict:
    """Per fold the bytes one epoch streams (sequences) or materialises (points), plus the
    dataloader's host-RAM estimate and the container limit."""
    kind = "sequence" if cfg.probe_type == "attention" else "point"
    per_fold = {}
    norm = int(cfg.probe.get("norm_tokens") or 0) if cfg.probe.get("standardize") else 0
    for fold in plan:
        if kind == "sequence":
            train_val = sequence_bytes(n_tokens, fold.train_ids + fold.val_ids, n_layers=n_layers, hidden=hidden)
            tests = sequence_bytes(n_tokens, fold.test_id_ids + fold.test_ood_ids, n_layers=n_layers, hidden=hidden)
            norm_bytes = min(norm, sum(int(n_tokens[str(i)]) for i in fold.train_ids)) * n_layers * hidden * BYTES_PER_ELEMENT
            per_fold[fold.held_out_style] = {
                "per_epoch_train_val_bytes": train_val, "test_sets_bytes": tests, "norm_pass_bytes": norm_bytes,
                "per_run_bytes": (train_val * cfg.training.epochs + norm_bytes) * n_trials + tests}
        else:
            n_rows = sum(len(getattr(fold, attr)) for _, attr in FOLD_ROLES)
            per_fold[fold.held_out_style] = {"materialised_bytes": point_bytes(
                n_rows, n_points=len(cfg.probe["points"]), n_layers=n_layers, hidden=hidden)}
    out = {"kind": kind, "n_layers": n_layers, "hidden": hidden, "n_trials": int(n_trials), "per_fold": per_fold,
           "cgroup_memory_limit_bytes": cgroup_memory_limit_bytes()}
    if kind == "sequence":
        out["dataloader_ram_bytes"] = dataloader_ram_bytes(cfg, n_layers=n_layers, hidden=hidden)
        out["max_per_epoch_train_val_bytes"] = max((v["per_epoch_train_val_bytes"] for v in per_fold.values()), default=0)
    else:
        out["max_materialised_bytes"] = max((v["materialised_bytes"] for v in per_fold.values()), default=0)
    return out


def estimates_text(cfg: TrainConfig, est: dict) -> list[str]:
    limit = est.get("cgroup_memory_limit_bytes")
    limit_text = _fmt_gb(limit) if limit else "unreadable/unlimited"
    lines = []
    if est["kind"] == "sequence":
        lines.append(f"Bytes per epoch (train + val sequences, Σ n_cot_tokens × {est['n_layers']} layers × "
                     f"{est['hidden']} × 2 B): max over folds {_fmt_gb(est['max_per_epoch_train_val_bytes'])}")
        for style, v in est["per_fold"].items():
            lines.append(f"  fold {style}: {_fmt_gb(v['per_epoch_train_val_bytes'])} per epoch, test sets "
                         f"{_fmt_gb(v['test_sets_bytes'])}, ≈ {_fmt_gb(v['per_run_bytes'])} over {cfg.training.epochs} "
                         f"epochs × {est['n_trials']} trial(s) (+ the standardisation pass; at most, early stopping cuts it)")
        d = cfg.dataloader
        lines.append(f"Host RAM for the dataloader: ({d.num_workers} workers × {d.prefetch_factor} prefetch + 1) × "
                     f"{cfg.training.max_batch_tokens:,} tokens × {est['n_layers']} layers × {est['hidden']} × 2 B ≈ "
                     f"{_fmt_gb(est['dataloader_ram_bytes'])} (container limit {limit_text})")
        if limit and est["dataloader_ram_bytes"] > limit:
            lines.append("  [warn] the dataloader's prefetched batches alone exceed the container limit — lower "
                         "dataloader.num_workers / prefetch_factor, training.max_batch_tokens or model.layers")
    else:
        lines.append(f"Bytes materialised per fold (rows × {len(cfg.probe['points'])} points × {est['n_layers']} layers × "
                     f"{est['hidden']} × 2 B): max over folds {_fmt_gb(est['max_materialised_bytes'])} "
                     f"(container limit {limit_text})")
        if limit and est["max_materialised_bytes"] > limit:
            lines.append("  [warn] a fold's point cache exceeds the container limit — fewer layers or points per run")
    return lines


class StoreHandle:
    """An opened store folder: backend, manifest, meta, the resolved layers and its identity block."""

    def __init__(self, backend, folder: str, manifest: pd.DataFrame, meta: dict, layers: list[int],
                 *, training_fingerprint: Optional[str] = None):
        self.backend, self.folder, self.manifest, self.meta, self.layers = backend, folder, manifest, meta, layers
        self.n_tokens = dict(zip(manifest["rollout_id"].astype(str), manifest["n_cot_tokens"].astype(int)))
        self.hidden = int(meta["hidden_size"])
        self.training_fingerprint = training_fingerprint

    def identity(self) -> dict:
        m = self.meta
        spec = dict(m.get("dataset_spec") or {})
        out = {"backend": self.backend.describe(), "folder": self.folder, "n_rollouts": int(len(self.manifest)),
               "stored_layers": list(m["layers"]), "stored_seq_layers": stored_layers(m, "seq"),
               "layers": list(self.layers), "hidden_size": self.hidden,
               "dtype": m.get("dtype"), "model_name": m.get("model_name"), "short_name": m.get("short_name"),
               "dataset_spec": {"name": spec.get("name"), "fingerprint": spec.get("fingerprint"),
                                "n_rows": spec.get("n_rows"), "built_utc": spec.get("built_utc")},
               "training_dataset_fingerprint": self.training_fingerprint,
               "same_dataset": bool(spec.get("fingerprint")) and spec.get("fingerprint") == self.training_fingerprint,
               "collected_utc": m.get("collected_utc")}
        if is_mirror(m):
            out["mirrored_layers"] = list(m.get("mirrored_layers") or [])
            out["mirrored_points_layers"] = list(m.get("mirrored_points_layers") or [])
            out["mirror_source"] = dict(m.get(MIRROR_SOURCE_KEY) or {})
        return out


def store_kinds(probe_type: str) -> tuple[str, ...]:
    """The store file kind a probe type reads: ``seq`` for the attention probe, ``points`` for the linear one."""
    return ("seq",) if probe_type == "attention" else ("points",)


def check_store(df: pd.DataFrame, ds_meta: dict, manifest: pd.DataFrame, meta: dict, plan: Sequence[Fold], *,
                subject_model: Optional[str], layers, points: Sequence[str], kinds: Sequence[str] = STORE_FILE_KINDS,
                ) -> list[int]:
    """Dataset-vs-store checks (raise on the first failure): the meta carries a
    ``dataset_spec.fingerprint`` (it need not equal the training dataset's), the ``short_name``
    matches ``subject_model`` when both are set, every fold id is in the store manifest, the
    requested layers are stored (and mirrored) for ``kinds``, the points are in ``POINTS``.
    Returns the resolved layer list."""
    stored_fp = (meta.get("dataset_spec") or {}).get("fingerprint")
    if not stored_fp:
        raise ValueError(f"the store meta carries no dataset_spec.fingerprint — not a store written by "
                         f"collect_probe_activations.py (dataset {ds_meta.get('name')!r})")
    short = meta.get("short_name")
    if subject_model is not None and short is not None and str(short) != str(subject_model):
        raise ValueError(f"the store holds activations of {short!r}, the config's model.subject_model is "
                         f"{subject_model!r}")
    stored_ids = set(manifest["rollout_id"].astype(str))
    needed = sorted({i for fold in plan for _, attr in FOLD_ROLES for i in getattr(fold, attr)})
    missing = [i for i in needed if i not in stored_ids]
    if missing:
        raise ValueError(f"{len(missing)} of {len(needed)} rollout(s) the folds use are not in the store manifest "
                         f"({len(stored_ids)} rows), e.g. {missing[:10]} — the dataset and the store disagree "
                         "(a rollout the collector skipped must be absent from the dataset)")
    try:
        resolved = resolve_store_layers(None if layers is None else list(layers), meta["layers"])
        for kind in kinds:
            resolved = resolve_kind_layers(None if layers is None else list(layers), meta, kind)
    except LayerSelectionError as e:
        raise ValueError(f"model.layers: {str(e).replace('data.layers', 'model.layers')}") from e
    bad = [p for p in points if p not in POINTS]
    if bad:
        raise ValueError(f"probe.points {bad} are not stored points {list(POINTS)}")
    return resolved


def store_folder(cfg: TrainConfig, ds_meta: dict) -> str:
    """The store folder a run reads: ``model.activations.folder``, else the dataset's ``name``."""
    return cfg.model.folder or str(ds_meta["name"])


def open_store(cfg: TrainConfig, df: pd.DataFrame, ds_meta: dict, plan: Sequence[Fold], *, api=None) -> StoreHandle:
    """``make_backend`` + ``read_store_manifest`` for the layers and file kind the probe reads + :func:`check_store`."""
    backend = make_backend(cfg.model.activations, api=api)
    folder = store_folder(cfg, ds_meta)
    kinds = store_kinds(cfg.probe_type)
    try:
        manifest, meta = read_store_manifest(backend, folder, layers=None if cfg.model.layers is None
                                             else list(cfg.model.layers), kinds=kinds)
    except LayerSelectionError as e:
        raise ValueError(f"model.layers: {str(e).replace('data.layers', 'model.layers')}") from e
    points = list(cfg.probe["points"]) if cfg.probe_type == "linear" else list(POINTS)
    layers = check_store(df, ds_meta, manifest, meta, plan, subject_model=cfg.model.subject_model,
                         layers=cfg.model.layers, points=points, kinds=kinds)
    return StoreHandle(backend, folder, manifest, meta, layers, training_fingerprint=dataset_fingerprint(df))


def text_cache_status(ds_meta: dict, df: pd.DataFrame, text_cfg: TextPipelineConfig, *, nlp=None) -> dict:
    """Which normalised-text cache file a run would use and how many dataset ids it already serves (read-only)."""
    name = dataset_name_for(ds_meta)
    fingerprint = text_cfg.fingerprint(nlp=nlp)
    path = cache_path(name, fingerprint, text_cfg)
    out = {"path": str(path), "fingerprint": fingerprint, "exists": path.exists(), "n_cached_rows": 0,
           "n_ids_cached": 0, "n_ids_to_normalise": int(len(df))}
    if path.exists():
        cached = read_text_cache(path)
        have = dict(zip(cached["rollout_id"].astype(str), cached["reasoning_sha"].astype(str)))
        hits = sum(have.get(str(i)) == reasoning_sha(t) for i, t in zip(df["rollout_id"], df["reasoning"]))
        out.update(n_cached_rows=int(len(cached)), n_ids_cached=int(hits), n_ids_to_normalise=int(len(df) - hits))
    return out


def build_source(cfg: TrainConfig, df: pd.DataFrame, ds_meta: dict, *, store: Optional[StoreHandle], log):
    """The feature adapter of the probe type: ``TfidfSource`` over ``normalised_corpus``, or a
    ``SequenceSource`` / ``PointSource`` over the opened store."""
    labels = dict(zip(df["rollout_id"].astype(str), df["label"].astype(int)))
    if cfg.probe_type == "tfidf":
        text_cfg = TextPipelineConfig.from_dict(cfg.probe["text"])
        corpus = normalised_corpus(ds_meta, df, text_cfg, log=log.info)
        return TfidfSource(corpus, cfg.probe["vectorizer"], labels=labels, text_cfg=text_cfg,
                           corpus_for=lambda text: normalised_corpus(ds_meta, df, TextPipelineConfig.from_dict(text),
                                                                     log=log.info))
    assert store is not None
    d = cfg.dataloader
    if cfg.probe_type == "attention":
        seq = SequenceStore(store.backend, store.folder, store.manifest, store.meta, layers=store.layers,
                            fetch_threads=d.fetch_threads)
        return SequenceSource(seq, store.layers, labels=labels, n_tokens=store.n_tokens, num_workers=d.num_workers,
                              fetch_threads=d.fetch_threads, prefetch_factor=d.prefetch_factor)
    pts = PointStore(store.backend, store.folder, store.manifest, store.meta, layers=store.layers,
                     fetch_threads=d.fetch_threads)
    return PointSource(pts, store.layers, cfg.probe["points"], labels=labels)


def _overrides_of(trials: Sequence[Trial], trial_id: Optional[str]) -> dict:
    for t in trials:
        if t.id == trial_id:
            return dict(t.overrides)
    return {}


def write_fold_outputs(run_dir: Path, result: FoldResult, source, df: pd.DataFrame, *, label_col: str,
                       cfg: TrainConfig, dataset_info: dict, store_info: Optional[dict], device: str, log) -> dict:
    """Checkpoints (+ the tfidf vectorizer), then predictions + reports for both test sets and every
    unit (+ the ensemble). Returns the fold's record: ``metrics[test_set][layer][point][view]``,
    per-unit training summaries and the best-trial curves."""
    style = result.held_out_style
    fold_dir = run_dir / f"fold_{style}"
    pred_dir, rep_dir = fold_dir / "predictions", fold_dir / "report"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    trial_of = {u: r.best_trial for u, r in result.units.items()}
    # Checkpoints are saved before any prediction is scored, so a failing report never loses them.
    units_record, curves = {}, {}
    for unit, r in result.units.items():
        name = unit_name(unit)
        layer, point = unit_fields(unit)
        overrides = _overrides_of(result.trials, r.best_trial)
        checkpoint = {**r.state, "best_epoch": r.best_epoch, "best_score": r.best_score, "val_metrics": r.val_metrics,
                      "history": r.history, "trial": r.best_trial, "trial_overrides": overrides, "trials": r.trials,
                      "unit": {"layer": layer, "point": point}, "held_out_style": style, "config": cfg.to_dict(),
                      "dataset": dataset_info, "store": store_info}
        torch.save(checkpoint, fold_dir / f"probe__{name}.pt")
        units_record[name] = {"layer": layer, "point": point, "best_trial": r.best_trial, "best_epoch": r.best_epoch,
                              "best_score": r.best_score, "val_metrics": r.val_metrics, "trials": r.trials,
                              "trial_overrides": overrides}
        hist = r.history.get(r.best_trial, [])
        curves[name] = {"layer": layer, "point": point, "trial": r.best_trial, "epoch": [h["epoch"] for h in hist],
                        "val_auroc": [h["val_auroc"] for h in hist], "train_auroc": [h["train_auroc"] for h in hist]}
        val_auc = (r.val_metrics or {}).get("auroc", math.nan)
        log.info(f"fold {style!r} {name}: best trial {r.best_trial!r} epoch {r.best_epoch} val AUROC {val_auc:.3f}")
    if source.input_kind == "sparse":
        save_vectorizer(source.vectorizer, fold_dir / "vectorizer.joblib")
    metrics: dict = {}
    for test_set in TEST_SETS:
        ids = result.fold.ids(test_set)
        if not ids:
            log.warning(f"fold {style!r}: {test_set} holds no row")
        per_unit = evaluate(result, source, ids, test_set, device=device)
        metrics[test_set] = {}
        for unit, frame in per_unit.items():
            name = unit_name(unit)
            layer, point = unit_fields(unit)
            frame.to_csv(pred_dir / f"{test_set}__{name}.csv", index=False)
            rep = build_report(frame, df, label_col=label_col, name=f"{run_dir.name}: fold {style} / {test_set} / {name}")
            rep.update({"fold": style, "test_set": test_set, "layer": layer, "point": point,
                        "trial": trial_of.get(unit), "predictions": f"predictions/{test_set}__{name}.csv"})
            write_json(rep_dir / f"{test_set}__{name}_report.json", rep)
            (rep_dir / f"{test_set}__{name}_report.md").write_text(render_markdown(rep))
            metrics[test_set].setdefault(layer, {})[point] = rep["overall"]
            auc = rep["overall"][PLOT_VIEW].get("auroc", math.nan)
            auc_p = rep["overall"][PRIMARY_VIEW].get("auroc", math.nan)
            log.info(f"fold {style!r} {test_set} {name}: AUROC pooled {auc:.3f} / positive {auc_p:.3f} "
                     f"(n = {rep['n_predictions']}, trial {trial_of.get(unit)!r})")
    return {"held_out_style": style, "metrics": metrics, "units": units_record, "curves": curves}


def _render_override(value) -> Any:
    return value if isinstance(value, (int, float, str, bool)) or value is None else json.dumps(value)


def search_results_rows(fold_records: Sequence[dict]) -> list[dict]:
    """One row per (fold, trial, layer, point): the trial's overrides, best epoch / val score / val AUROC, whether selected."""
    rows = []
    for rec in fold_records:
        for name, u in rec["units"].items():
            if name == ENSEMBLE_KEY:
                continue
            for trial_id, t in u["trials"].items():
                val = t.get("val_metrics") or {}
                row = {"fold": rec["held_out_style"], "trial": trial_id, "layer": u["layer"], "point": u["point"],
                       "best_epoch": t.get("best_epoch"), "best_val_score": t.get("best_score"),
                       "val_auroc": val.get("auroc", math.nan), "val_loss": val.get("loss", math.nan),
                       "selected": trial_id == u["best_trial"]}
                rows.append(row)
    return rows


def search_results_frame(fold_records: Sequence[dict], trials: Sequence[Trial]) -> pd.DataFrame:
    rows = search_results_rows(fold_records)
    by_id = {t.id: t.overrides for t in trials}
    keys = sorted({k for t in trials for k in t.overrides})
    for row in rows:
        for k in keys:
            row[k] = _render_override(by_id.get(row["trial"], {}).get(k))
    columns = ["fold", "trial", "layer", "point", *keys, "best_epoch", "best_val_score", "val_auroc", "val_loss", "selected"]
    return pd.DataFrame(rows, columns=columns)


def summary_rows(fold_records: Sequence[dict]) -> list[dict]:
    """One row per (fold, test_set, layer, point, view): ``TABLE_METRICS`` + the unit's trial."""
    rows = []
    for rec in fold_records:
        trial_by_unit = {(u["layer"], u["point"]): u["best_trial"] for u in rec["units"].values()}
        for test_set, by_layer in rec["metrics"].items():
            for layer, by_point in by_layer.items():
                for point, views in by_point.items():
                    for view in CASE_VIEWS:
                        m = views.get(view, {})
                        rows.append({"fold": rec["held_out_style"], "test_set": test_set, "layer": layer, "point": point,
                                     "view": view, "trial": trial_by_unit.get((layer, point)),
                                     **{k: m.get(k, math.nan) for k in TABLE_METRICS}})
    return rows


def _stats(values: Sequence[float]) -> dict:
    arr = np.asarray([math.nan if v is None else v for v in values], dtype=float)
    valid = arr[~np.isnan(arr)]
    n = int(len(valid))
    return {"mean": float(valid.mean()) if n else math.nan,
            "std": float(valid.std(ddof=1)) if n > 1 else math.nan,
            "min": float(valid.min()) if n else math.nan, "max": float(valid.max()) if n else math.nan, "n": n}


def aggregate(fold_records: Sequence[dict]) -> dict:
    """Mean / sample std / min / max / n over folds per (test_set, layer, point, view) of
    :data:`AGGREGATE_METRICS`, plus ``ood_minus_id_auroc`` per (layer, point, view) — the per-fold
    gap ``AUROC(test_ood) − AUROC(test_id)`` with the same statistics."""
    folds = [rec["held_out_style"] for rec in fold_records]
    cells: dict[tuple, dict[str, list]] = {}
    gaps: dict[tuple, dict[str, float]] = {}
    for rec in fold_records:
        for test_set, by_layer in rec["metrics"].items():
            for layer, by_point in by_layer.items():
                for point, views in by_point.items():
                    for view in CASE_VIEWS:
                        cell = cells.setdefault((test_set, layer, point, view), {m: [] for m in AGGREGATE_METRICS})
                        for m in AGGREGATE_METRICS:
                            cell[m].append(views.get(view, {}).get(m, math.nan))
        for layer, by_point in rec["metrics"].get("test_id", {}).items():
            for point, views in by_point.items():
                for view in CASE_VIEWS:
                    a_id = views.get(view, {}).get("auroc", math.nan)
                    a_ood = rec["metrics"].get("test_ood", {}).get(layer, {}).get(point, {}).get(view, {}).get("auroc", math.nan)
                    gaps.setdefault((layer, point, view), {})[rec["held_out_style"]] = a_ood - a_id
    by_unit = [{"test_set": ts, "layer": layer, "point": point, "view": view,
                **{m: _stats(vals) for m, vals in cell.items()}}
               for (ts, layer, point, view), cell in cells.items()]
    gap = [{"layer": layer, "point": point, "view": view, **_stats(list(per_fold.values())), "per_fold": per_fold}
           for (layer, point, view), per_fold in gaps.items()]
    return {"folds": folds, "n_folds": len(folds), "metrics": list(AGGREGATE_METRICS), "views": list(CASE_VIEWS),
            "by_unit": by_unit, "ood_minus_id_auroc": gap}


def write_run_tables(run_dir: Path, fold_records: Sequence[dict], trials: Sequence[Trial], *, results_base: dict,
                     timing: dict, t_start: float, complete: bool) -> tuple[pd.DataFrame, dict]:
    """``search_results.csv``, ``results_summary.csv``, ``aggregate.json`` and ``results.json`` over
    the folds finished so far. Returns the summary frame and the aggregate."""
    search_results_frame(fold_records, trials).to_csv(run_dir / "search_results.csv", index=False)
    summary = pd.DataFrame(summary_rows(fold_records), columns=["fold", "test_set", "layer", "point", "view", "trial",
                                                                *TABLE_METRICS])
    summary.to_csv(run_dir / "results_summary.csv", index=False)
    agg = aggregate(fold_records)
    write_json(run_dir / "aggregate.json", agg)
    results = {
        **results_base, "complete": bool(complete),
        "folds": [rec["held_out_style"] for rec in fold_records],
        "training": {rec["held_out_style"]: rec["units"] for rec in fold_records},
        "metrics": {rec["held_out_style"]: rec["metrics"] for rec in fold_records},
        "aggregate": agg, "timing_seconds": dict(timing), "total_seconds": time.perf_counter() - t_start,
    }
    write_json(run_dir / "results.json", results)
    return summary, agg


def _unit_aurocs(fold_records: Sequence[dict], view: str) -> dict:
    """``{test_set: {(layer, point): {fold: auroc}}}`` (the ensemble excluded)."""
    out: dict = {ts: {} for ts in TEST_SETS}
    for rec in fold_records:
        for test_set in TEST_SETS:
            for layer, by_point in rec["metrics"].get(test_set, {}).items():
                if layer == ENSEMBLE_KEY:
                    continue
                for point, views in by_point.items():
                    out[test_set].setdefault((layer, point), {})[rec["held_out_style"]] = views.get(view, {}).get("auroc", math.nan)
    return out


def plot_auroc_by_unit(fold_records: Sequence[dict], out_path: Path, *, view: str = PLOT_VIEW) -> Path:
    """Two panels (test_id / test_ood): x = layer, one line per point (a single line for the attention
    probe), thin per-fold lines + the bold mean; a TF-IDF run (no layer) gets one bar per test set."""
    data = _unit_aurocs(fold_records, view)
    numeric = all(str(layer).isdigit() for ts in data.values() for layer, _ in ts)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, test_set in zip(axes, TEST_SETS):
        units = data[test_set]
        if numeric and units:
            points = sorted({p for _, p in units})
            for k, point in enumerate(points):
                layers = sorted({int(l) for l, p in units if p == point})
                folds = sorted({f for u in units.values() for f in u})
                for fold in folds:
                    ys = [units.get((str(l), point), {}).get(fold, math.nan) for l in layers]
                    ax.plot(layers, ys, color=cmap(k), alpha=0.25, linewidth=0.8)
                means = [float(np.nanmean([v for v in units.get((str(l), point), {}).values()] or [math.nan]))
                         for l in layers]
                ax.plot(layers, means, color=cmap(k), linewidth=2.2, marker="o",
                        label="CoT sequence (attention)" if point == NA else point)
            ax.set_xlabel("layer")
            ax.set_xticks(sorted({int(l) for l, _ in units}))
            ax.legend(fontsize=8, title="point")
        else:
            labels = ["tfidf" if u == (NA, NA) else unit_name(u) for u in units]
            means = [float(np.nanmean(list(v.values()) or [math.nan])) for v in units.values()]
            ax.bar(labels, means, color="steelblue", edgecolor="black")
            for i, (u, per_fold) in enumerate(units.items()):
                ax.scatter([i] * len(per_fold), list(per_fold.values()), color="black", s=12, zorder=3)
        ax.axhline(0.5, color="gray", linewidth=0.8)
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"{test_set} — AUROC ({view}; thin = folds, bold = mean)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def best_unit_by_fold(fold_records: Sequence[dict]) -> dict[str, str]:
    """The unit with the highest validation score in each fold (selection on val, never on test)."""
    out = {}
    for rec in fold_records:
        cands = [(u["best_score"] if u["best_score"] == u["best_score"] else -math.inf, name)
                 for name, u in rec["units"].items()]
        out[rec["held_out_style"]] = max(cands)[1] if cands else ""
    return out


def plot_auroc_by_fold(fold_records: Sequence[dict], out_path: Path, *, view: str = PLOT_VIEW) -> Path:
    """Bars per held-out style: the fold's best unit (by validation score) on test_id and test_ood."""
    best = best_unit_by_fold(fold_records)
    styles = [rec["held_out_style"] for rec in fold_records]
    values = {ts: [] for ts in TEST_SETS}
    for rec in fold_records:
        u = rec["units"].get(best[rec["held_out_style"]], {})
        for ts in TEST_SETS:
            values[ts].append(rec["metrics"].get(ts, {}).get(u.get("layer"), {}).get(u.get("point"), {})
                              .get(view, {}).get("auroc", math.nan))
    x = np.arange(len(styles))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(styles) + 3), 4.5))
    for k, ts in enumerate(TEST_SETS):
        bars = ax.bar(x + (k - 0.5) * width, values[ts], width, label=ts, edgecolor="black")
        for bar, v in zip(bars, values[ts]):
            if v == v:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({best[s]})" for s in styles], fontsize=8)
    ax.axhline(0.5, color="gray", linewidth=0.8)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel(f"AUROC ({view})")
    ax.set_title("Best unit per fold (chosen on validation): in-distribution vs held-out style")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_training_curves(fold_record: dict, out_path: Path) -> Path:
    """Validation AUROC per epoch of every unit's best trial — one panel per point, one line per layer."""
    curves = fold_record["curves"]
    points = sorted({c["point"] for c in curves.values()})
    fig, axes = plt.subplots(1, max(1, len(points)), figsize=(4.5 * max(1, len(points)), 3.8), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, point in zip(axes[0], points):
        items = [(name, c) for name, c in curves.items() if c["point"] == point]
        for k, (name, c) in enumerate(sorted(items)):
            color = cmap(k / max(1, len(items) - 1)) if len(items) > 1 else "steelblue"
            ax.plot(c["epoch"], c["val_auroc"], marker="o", markersize=3, color=color,
                    label=f"layer {c['layer']} ({c['trial']})")
        ax.set_title(f"fold {fold_record['held_out_style']} — val AUROC, point {point}")
        ax.set_xlabel("epoch")
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("val AUROC (best trial)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def compare_runs(run_dir: Path, other_dir: Path, df: pd.DataFrame, *, label_col: str, view: str = PRIMARY_VIEW) -> dict:
    """``report.delta_auroc`` (this run − the other) on every ``(fold, test_set, unit)`` predictions
    file both runs hold, on ``view``; a pair covering different rollouts is recorded as an error."""
    other_dir = Path(other_dir)
    entries = []
    for pred_path in sorted(run_dir.glob("fold_*/predictions/*.csv")):
        rel = pred_path.relative_to(run_dir)
        other_path = other_dir / rel
        if not other_path.exists():
            continue
        fold = pred_path.parent.parent.name[len("fold_"):]
        test_set, name = pred_path.stem.split("__", 1)
        entry = {"fold": fold, "test_set": test_set, "unit": name, "predictions": str(rel)}
        try:
            a = join_predictions(load_predictions(pred_path), df, label_col=label_col)
            b = join_predictions(load_predictions(other_path), df, label_col=label_col)
            entry.update(delta_auroc(a, b, view=view))
        except ValueError as e:
            entry["error"] = str(e)
        entries.append(entry)
    per_unit: dict[tuple, list] = {}
    for e in entries:
        if "delta_auroc" in e:
            per_unit.setdefault((e["test_set"], e["unit"]), []).append(e["delta_auroc"])
    summary = [{"test_set": ts, "unit": u, **_stats(vals)} for (ts, u), vals in sorted(per_unit.items())]
    return {"a": str(run_dir), "b": str(other_dir), "view": view, "label_col": label_col,
            "n_pairs": len(entries), "entries": entries, "summary": summary}


def _restrict_trials(trials: list[Trial], only_trial: Optional[str]) -> list[Trial]:
    if only_trial is None:
        return trials
    picked = [t for t in trials if t.id == only_trial]
    if not picked:
        raise ValueError(f"--only-trial {only_trial!r} is not a trial of this grid; trials: {[t.id for t in trials]}")
    return picked


def main(config_path, *, dry_run: bool = False, folds: Optional[Sequence[str]] = None,
         only_trial: Optional[str] = None, compare=None, device: Optional[str] = None, api=None) -> Path:
    """Run (or dry-run) one training config; returns the run directory.

    ``folds`` restricts the held-out styles, ``only_trial`` the grid to one trial id, ``compare``
    names another run directory for ``ablation.json``, ``device`` overrides the cuda-if-available
    default; ``api`` injects the ``HfApi``."""
    cfg_path = Path(config_path).resolve()
    cfg = parse_train_config(load_config(str(cfg_path)))
    folds = None if folds is None else [str(f) for f in folds]
    if folds is not None and not folds:
        raise ValueError("--folds names no style (an empty list); omit it to train every fold")
    if compare is not None and not any(Path(compare).glob("fold_*/predictions/*.csv")):
        raise ValueError(f"--compare {compare!r} is not a finished train_probe run directory "
                         "(no fold_*/predictions/*.csv)")
    effective_folds = folds if folds is not None else (list(cfg.dataset.folds) if cfg.dataset.folds else None)
    run_dir = make_run_dir(cfg)
    log = setup_logging(run_dir)
    try:
        return _run(cfg, cfg_path, run_dir, log, dry_run=dry_run, folds=effective_folds, only_trial=only_trial,
                    compare=compare, device=device, api=api)
    finally:
        close_logging(log)


def _run(cfg: TrainConfig, cfg_path: Path, run_dir: Path, log: logging.Logger, *, dry_run: bool, folds, only_trial,
         compare, device, api) -> Path:
    t_start = time.perf_counter()
    log.info(f"Run directory : {run_dir}")
    log.info(f"Config        : {cfg_path}")
    trials = _restrict_trials(expand_grid(cfg), only_trial)
    resolved = resolved_config(cfg, folds, trials)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_jsonable(resolved), sort_keys=False))
    log.info("Resolved config:\n" + json.dumps(_jsonable(resolved), indent=2))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    df, ds_meta = read_dataset(cfg.dataset.path)
    label_col = ds_meta.get("label_col")
    if not label_col:
        raise ValueError(f"{cfg.dataset.path}: the dataset sidecar carries no 'label_col' — the reports need it")
    dataset_info = {"path": str(cfg.dataset.path), "name": ds_meta.get("name"), "fingerprint": ds_meta.get("fingerprint"),
                    "n_rows": int(len(df)), "label_col": label_col, "predicate": ds_meta.get("predicate"),
                    "built_utc": ds_meta.get("built_utc")}
    log.info(f"Dataset {ds_meta.get('name')!r}: {len(df)} rows, label column {label_col!r}, "
             f"fingerprint {ds_meta.get('fingerprint')}, built {ds_meta.get('built_utc')}")

    # Folds and grid
    plan = plan_folds(df, folds, cases=cfg.dataset.cases)
    write_json(run_dir / "folds.json", folds_to_json(plan))
    records = fold_plan_records(plan, df)
    log.info(f"Fold plan ({len(plan)} fold(s), cases={cfg.dataset.cases}):\n" + fold_plan_text(records))
    log.info(f"Grid: {len(trials)} trial(s): {[t.id for t in trials]}")

    # Features: the store or the text cache
    store = store_info = None
    text_status = None
    if cfg.probe_type == "tfidf":
        text_cfg = TextPipelineConfig.from_dict(cfg.probe["text"])
        units = [(NA, NA)]
        if dry_run:
            text_status = text_cache_status(ds_meta, df, text_cfg)
            log.info(f"Text cache {text_status['path']} (fingerprint {text_status['fingerprint']}): "
                     f"{'exists' if text_status['exists'] else 'absent'}, {text_status['n_cached_rows']} cached rows, "
                     f"{text_status['n_ids_cached']} of {len(df)} dataset ids served, "
                     f"{text_status['n_ids_to_normalise']} to normalise (not written by a dry run)")
        estimates = None
    else:
        store = open_store(cfg, df, ds_meta, plan, api=api)
        store_info = store.identity()
        spec = store_info["dataset_spec"]
        log.info(f"Store {store_info['backend']}/{store.folder} (folder from "
                 f"{'model.activations.folder' if cfg.model.folder else 'the dataset name'}): "
                 f"{store_info['n_rollouts']} rollouts, stored layers {store_info['stored_layers']} (seq files for "
                 f"{store_info['stored_seq_layers']}), requested {store.layers}, hidden {store.hidden}, "
                 f"dtype {store_info['dtype']}, model {store_info['model_name']!r}")
        if "mirrored_layers" in store_info:
            src = store_info["mirror_source"]
            log.info(f"Store is a local mirror of {src.get('source', '?')} (mirrored {src.get('mirrored_utc', '?')}): "
                     f"seq layers {store_info['mirrored_layers']}, points layers {store_info['mirrored_points_layers']} "
                     f"— the run reads {store_kinds(cfg.probe_type)[0]} files of layers {store.layers}")
        log.info(f"Store collected from dataset {spec['name']!r} ({spec['n_rows']} rows, built {spec['built_utc']}, "
                 f"fingerprint {spec['fingerprint']}); training dataset {ds_meta.get('name')!r} fingerprint "
                 f"{store_info['training_dataset_fingerprint']} — "
                 f"{'the same dataset' if store_info['same_dataset'] else 'a different dataset (shared store; every fold id is in the store)'}")
        units = [(l, NA) for l in store.layers] if cfg.probe_type == "attention" else \
            [(l, p) for l in store.layers for p in cfg.probe["points"]]
        estimates = size_estimates(cfg, plan, n_layers=len(store.layers), hidden=store.hidden, n_tokens=store.n_tokens,
                                   n_trials=len(trials))
        for line in estimates_text(cfg, estimates):
            (log.warning if line.strip().startswith("[warn]") else log.info)(line)
    log.info(f"Units: {len(units)} — {units}")

    if dry_run:
        write_json(run_dir / "dry_run.json", {
            "config": resolved, "dataset": dataset_info, "store": store_info, "text_cache": text_status,
            "fold_plan": records, "trials": [{"id": t.id, "overrides": t.overrides} for t in trials],
            "units": [list(unit_fields(u)) for u in units], "estimates": estimates, "device": device})
        log.info(f"Dry run — no tensor read, nothing trained; wrote {run_dir / 'config.yaml'} and {run_dir / 'dry_run.json'}")
        return run_dir

    # Training
    source = build_source(cfg, df, ds_meta, store=store, log=log)
    log.info(f"Device {device}; source {type(source).__name__} ({source.input_kind})")
    fold_records: list[dict] = []
    timing: dict[str, float] = {}
    results_base = {
        "run_dir": str(run_dir), "config": resolved, "dataset": dataset_info, "store": store_info, "device": device,
        "trials": [{"id": t.id, "overrides": t.overrides} for t in trials], "units": [list(unit_fields(u)) for u in units],
        "planned_folds": [f.held_out_style for f in plan],
    }
    if not plan:
        raise ValueError("no fold to train (the dataset holds no hint style)")
    for k, fold in enumerate(plan):
        t0 = time.perf_counter()
        result = train_fold(fold, source, cfg.probe, cfg.training, trials, device=device, log=log, fold_index=k)
        rec = write_fold_outputs(run_dir, result, source, df, label_col=label_col, cfg=cfg, dataset_info=dataset_info,
                                 store_info=store_info, device=device, log=log)
        plot_training_curves(rec, run_dir / PLOTS_DIR / f"training_curves_{fold.held_out_style}.png")
        fold_records.append(rec)
        timing[fold.held_out_style] = time.perf_counter() - t0
        log.info(f"fold {fold.held_out_style!r} done in {timing[fold.held_out_style]:.0f}s")
        # Rewritten after every fold, so an interrupted run keeps the finished folds' tables.
        summary, agg = write_run_tables(run_dir, fold_records, trials, results_base=results_base, timing=timing,
                                        t_start=t_start, complete=(k + 1 == len(plan)))
        clear = getattr(source, "clear", None)
        if clear is not None:
            clear()
        del result

    plot_auroc_by_unit(fold_records, run_dir / PLOTS_DIR / "auroc_by_unit.png")
    plot_auroc_by_fold(fold_records, run_dir / PLOTS_DIR / "auroc_by_fold.png")
    pooled = summary[summary["view"] == PLOT_VIEW].pivot_table(index=["layer", "point"], columns=["fold", "test_set"],
                                                              values="auroc", aggfunc="first")
    log.info(f"AUROC ({PLOT_VIEW}) per unit × (fold, test set):\n" + pooled.to_string(float_format=lambda x: f"{x:.3f}"))
    for entry in agg["ood_minus_id_auroc"]:
        if entry["view"] == PRIMARY_VIEW:
            log.info(f"LOHO gap (test_ood − test_id AUROC, {PRIMARY_VIEW}) layer {entry['layer']} point {entry['point']}: "
                     f"mean {entry['mean']:.3f} ± {entry['std']:.3f} over {entry['n']} fold(s)")
    if compare is not None:
        ablation = compare_runs(run_dir, Path(compare), df, label_col=label_col)
        write_json(run_dir / "ablation.json", ablation)
        log.info(f"Ablation vs {compare}: {ablation['n_pairs']} matching predictions file(s); "
                 + "; ".join(f"{s['test_set']} {s['unit']}: Δ {s['mean']:+.3f}" for s in ablation["summary"]))
    log.info(f"Outputs in {run_dir} ({time.perf_counter() - t_start:.0f}s)")
    return run_dir


def cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to the YAML config (extends: resolved).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read the dataset and the store manifest / text-cache status, print the fold plan, the "
                             "grid and the size estimates; write config.yaml + dry_run.json; read no tensor.")
    parser.add_argument("--folds", default=None, metavar="STYLE,STYLE",
                        help="Only these held-out styles (comma-separated).")
    parser.add_argument("--only-trial", default=None, metavar="ID", help="Only this trial id of the grid.")
    parser.add_argument("--compare", default=None, metavar="RUN_DIR",
                        help="Another run directory: ΔAUROC (this − other) per matching predictions file → ablation.json.")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu).")
    args = parser.parse_args(argv)
    folds = [f.strip() for f in args.folds.split(",") if f.strip()] if args.folds else None
    main(args.config, dry_run=args.dry_run, folds=folds, only_trial=args.only_trial, compare=args.compare,
         device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
