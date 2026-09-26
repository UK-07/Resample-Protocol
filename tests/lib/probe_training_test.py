"""Tests for src/lib/probe_training.py: CPU-only, over a fake local activation store and an in-memory corpus."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from src.lib import activation_store as st
from src.lib import probe_training as pt
from src.lib.probe_datasets import fold_roles
from src.lib.probe_metrics import auroc
from src.lib.probes import load_probe, p_detection_target
from src.lib.report import PREDICTION_COLUMNS


class TestTokenBudgetBatchSampler(unittest.TestCase):
    lengths = [10, 200, 30, 400, 50, 60, 70, 800, 90, 100, 110, 120]

    def _check(self, batches, max_size, budget):
        seen = sorted(i for b in batches for i in b)
        self.assertEqual(seen, list(range(len(self.lengths))))
        for b in batches:
            self.assertLessEqual(len(b), max_size)
            if len(b) > 1:
                self.assertLessEqual(len(b) * max(self.lengths[i] for i in b), budget)

    def test_every_index_once_and_budgets_hold(self):
        s = pt.TokenBudgetBatchSampler(self.lengths, 500, 4, shuffle=True, seed=0)
        self._check(list(iter(s)), 4, 500)
        self._check(s.batches(), 4, 500)

    def test_long_sample_alone(self):
        s = pt.TokenBudgetBatchSampler(self.lengths, 500, 4, shuffle=False)
        for b in s.batches():
            if 7 in b:
                self.assertEqual(b, [7])

    def test_shuffle_changes_epochs_and_plan_is_deterministic(self):
        s = pt.TokenBudgetBatchSampler(self.lengths, 500, 4, shuffle=True, seed=3)
        first, second = list(iter(s)), list(iter(s))
        self.assertNotEqual(first, second)
        self.assertEqual(s.batches(), s.batches())

    def test_len_matches_iteration(self):
        s = pt.TokenBudgetBatchSampler(self.lengths, 500, 4, shuffle=False)
        self.assertEqual(len(s), len(list(iter(s))))


class TestCollateAndHelpers(unittest.TestCase):
    def test_collate_keeps_dtype_and_pads(self):
        items = [
            ({0: torch.ones(3, 4, dtype=torch.bfloat16), 5: torch.ones(3, 4, dtype=torch.bfloat16)},
             torch.tensor(1), torch.tensor(1.0)),
            ({0: torch.ones(5, 4, dtype=torch.bfloat16), 5: torch.ones(5, 4, dtype=torch.bfloat16)},
             torch.tensor(0), torch.tensor(1.0)),
        ]
        H, y, lengths = pt.collate_multi_layer(items)
        self.assertEqual(H[0].dtype, torch.bfloat16)
        self.assertEqual(H[0].shape, (2, 5, 4))
        self.assertTrue(torch.equal(H[5][0, 3:], torch.zeros(2, 4, dtype=torch.bfloat16)))
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(lengths.tolist(), [3, 5])

    def test_prefix_lengths_within_bounds(self):
        rng = np.random.default_rng(0)
        lengths = torch.tensor([1, 5, 40, 400])
        for _ in range(20):
            out = pt.sample_prefix_lengths(lengths, 0.25, rng)
            self.assertEqual(out.dtype, lengths.dtype)
            for L, p in zip(lengths.tolist(), out.tolist()):
                self.assertGreaterEqual(p, max(1, int(L * 0.25)))
                self.assertLessEqual(p, L)

    def test_class_weights_balanced(self):
        w = pt.class_weights([0, 1, 1, 1], "balanced")
        self.assertAlmostEqual(float(w.mean()), 1.0)
        self.assertGreater(float(w[0]), float(w[1]))
        self.assertIsNone(pt.class_weights([0, 1], "none"))
        self.assertIsNone(pt.class_weights([1, 1], "balanced"))
        with self.assertRaises(ValueError):
            pt.class_weights([0, 1], "sqrt")


# ---------------------------------------------------------------------------
# Fixtures: a 60-row synthetic dataset, a fake local store, a corpus
# ---------------------------------------------------------------------------

HIDDEN = 4
LAYERS = (0, 1)
STYLES = ("consensus", "metadata", "sycophancy")
FOLDER = "spec"
N_PER_STYLE = 20
_SPLIT_OF = {0: "train", 1: "train", 2: "train", 3: "val", 4: "test"}
_TEXT = {0: "hint says answer key mention option", 1: "compute derive reason step verify"}


def make_rows() -> pd.DataFrame:
    """60 rows: 3 styles × 20; per style 12 train / 4 val / 4 test, both labels in every role."""
    records = []
    for style in STYLES:
        for i in range(N_PER_STYLE):
            label = (i // 2) % 2
            records.append({
                "rollout_id": f"nemotron:run:{style}:{i}", "question_id": f"q:{style}:{i}",
                "hint_style": style, "case": "positive" if i % 2 == 0 else "negative", "split": _SPLIT_OF[i % 5],
                "label": label, "subject_model": "nemotron", "dataset": "medqa",
                "n_tokens": 2 + i % 5, "reasoning": f"{_TEXT[label]} filler{i % 3}",
            })
    return pd.DataFrame(records)


def signal(label: int, layer: int, rng: np.random.Generator, shape, scale: float = 1.0) -> torch.Tensor:
    """Label 0 rows sit around +scale, label 1 rows around -scale (per layer a different factor) plus noise."""
    base = (scale if label == 0 else -scale) * (1.0 + 0.5 * layer)
    return torch.tensor(base + 0.5 * rng.standard_normal(shape), dtype=torch.float32)


def write_store(root: Path, rows: pd.DataFrame, *, seed: int = 0, scale: float = 1.0, layers=LAYERS,
                seq_layers=None) -> st.LocalBackend:
    """A local store over ``rows`` in several parts; ``seq_layers`` (None = every layer) get seq files."""
    rng = np.random.default_rng(seed)
    layers = [int(l) for l in layers]
    seq_layers = layers if seq_layers is None else [int(l) for l in seq_layers]
    labels = dict(zip(rows["rollout_id"], rows["label"]))
    items = [SimpleNamespace(rollout_id=r, n_tokens=int(n)) for r, n in zip(rows["rollout_id"], rows["n_tokens"])]
    parts = st.plan_parts(items, hidden=HIDDEN, n_layers=len(layers), max_bytes=25 * len(st.POINTS) * HIDDEN * 2)
    backend = st.LocalBackend(root)
    meta = {"points": list(st.POINTS), "layers": layers, "hidden_size": HIDDEN, "dtype": st.STORE_DTYPE_NAME}
    if seq_layers != layers:
        meta["seq_layers"] = seq_layers
    writer = st.StoreWriter(backend, FOLDER, meta=meta, upload_workers=1)
    with contextlib.redirect_stdout(io.StringIO()):
        for part in parts:
            for layer in layers:
                for kind, fname in (("seq", part.seq_file(layer)), ("points", part.points_file(layer))):
                    if kind == "seq" and layer not in seq_layers:
                        continue
                    pw = writer.new_part_writer(fname, part.entries(kind, hidden=HIDDEN), {"layer": str(layer)})
                    for it in part.items:
                        shape = (it.n_tokens, HIDDEN) if kind == "seq" else (len(st.POINTS), HIDDEN)
                        pw.add(it.rollout_id, signal(labels[it.rollout_id], layer, rng, shape, scale).to(torch.bfloat16))
                    writer.submit_file(f"part{part.index:02d}", fname, pw.finish())
            manifest_rows = [{**{c: "" for c in st.STORE_MANIFEST_COLUMNS}, "rollout_id": it.rollout_id,
                              "n_cot_tokens": it.n_tokens, "seq_file": st.part_template("seq", part.index),
                              "points_file": st.part_template("points", part.index)} for it in part.items]
            writer.finalize_group(f"part{part.index:02d}", rows=manifest_rows)
        writer.close()
    return backend


def open_stores(backend: st.LocalBackend, layers=None):
    manifest, meta = st.read_store_manifest(backend, FOLDER)
    seq = st.SequenceStore(backend, FOLDER, manifest, meta, layers=layers)
    points = st.PointStore(backend, FOLDER, manifest, meta, layers=layers)
    n_tokens = dict(zip(manifest["rollout_id"], manifest["n_cot_tokens"].astype(int)))
    return seq, points, n_tokens


def corpus_of(rows: pd.DataFrame) -> pd.Series:
    return pd.Series(rows["reasoning"].tolist(), index=rows["rollout_id"].tolist())


def labels_of(rows: pd.DataFrame) -> dict:
    return dict(zip(rows["rollout_id"], rows["label"].astype(int)))


MODEL_BLOCK = {"subject_model": "nemotron-nano-9b-v2",
               "activations": {"backend": "local", "local_dir": "${DATA_ROOT}/probe_activations/nemotron"},
               "layers": list(LAYERS)}


def config(probe: dict, *, model=True, **top) -> dict:
    cfg = {"dataset": {"path": "${DATA_ROOT}/probe_datasets/ds.parquet"}, "probe": probe,
           "output": {"dir": "${DATA_ROOT}/probe_runs", "run_name": "t"}}
    if model:
        cfg["model"] = json.loads(json.dumps(MODEL_BLOCK))
    cfg.update(top)
    return cfg


class StoreFixture(unittest.TestCase):
    """A shared fake store per test class (built once) with the synthetic rows."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.rows = make_rows()
        cls.backend = write_store(Path(cls._tmp.name) / "store", cls.rows)
        cls.labels = labels_of(cls.rows)

    @classmethod
    def tearDownClass(cls):
        st.reset_read_caches()
        cls._tmp.cleanup()

    def sequence_source(self, layers=None, **kw) -> pt.SequenceSource:
        seq, _, n_tokens = open_stores(self.backend, layers)
        return pt.SequenceSource(seq, layers, labels=self.labels, n_tokens=n_tokens, num_workers=0, **kw)

    def point_source(self, points, layers=None) -> pt.PointSource:
        _, pts, _ = open_stores(self.backend, layers)
        return pt.PointSource(pts, layers, points, labels=self.labels)

    def tfidf_source(self, vectorizer=None) -> pt.TfidfSource:
        return pt.TfidfSource(corpus_of(self.rows), vectorizer or {"min_df": 1, "ngram_range": [1, 1]},
                              labels=self.labels)

    def fold(self, style="metadata", cases="both") -> pt.Fold:
        return next(f for f in pt.plan_folds(self.rows, [style], cases=cases))


# ---------------------------------------------------------------------------
# Config schema and grid
# ---------------------------------------------------------------------------


class TestParseTrainConfig(unittest.TestCase):
    """parse_train_config / expand_grid / apply_trial."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"DATA_ROOT": "/data/root"})
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def test_parse_train_config_rejects_unknown_keys_and_model_block_for_tfidf(self):
        cases = [
            (config({"type": "attention"}, extra=1), "unknown key"),
            (config({"type": "attention"}, training={"lr": 1e-3, "momentum": 0.9}), "training: unknown key"),
            (config({"type": "attention", "dropout": 0.1}), "probe: unknown key"),
            (config({"type": "attention"}, dataloader={"workers": 2}), "dataloader: unknown key"),
            (config({"type": "attention"}, output={"dir": "${DATA_ROOT}/x", "stamp": 1}), "output: unknown key"),
            (config({"type": "tfidf"}), "must be absent for probe type 'tfidf'"),
            (config({"type": "linear"}, model=False), "model: is required"),
            (config({"type": "linear", "points": ["pre_cot", "middle"]}), "probe.points"),
            (config({"type": "linear"}, training={"min_prefix_frac": 0.5}), "only valid for probe type 'attention'"),
            (config({"type": "attention"}, dataset={"path": "${DATA_ROOT}/d.parquet", "cases": "all"}), "dataset.cases"),
            (config({"type": "attention"}, dataset={"path": "${DATA_ROOT}/d.parquet", "folds": ["a", "a"]}), "duplicates"),
            (config({"type": "attention"}, training={"class_weighting": "sqrt"}), "class_weighting"),
            (config({"type": "attention"}, training={"epochs": 0}), "training.epochs"),
            (config({"type": "attention"}, dataset={"path": "${DATA_ROOT}/d.parquet", "extra": 1}), "dataset: unknown"),
            (config({"type": "tfidf", "text": {"stem": True}}, model=False), "probe.text: unknown keys"),
            (config({"type": "tfidf", "vectorizer": {"min_df": 0}}, model=False), "vectorizer.min_df"),
            (config({"type": "attention", "aggregation": "mean"}), "invalid block for trial 'default'"),
            (config({"type": "attention"}, search={"heads": [1, 0]}), "invalid block for trial 'heads=0'"),
            (config({"type": "linear", "num_classes": 3}), "probe.num_classes must be 2"),
        ]
        for cfg, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    pt.parse_train_config(cfg)
        bad_model = config({"type": "attention"})
        bad_model["model"]["activations"]["bucket"] = "x"
        with self.assertRaisesRegex(ValueError, "model.activations: unknown key"):
            pt.parse_train_config(bad_model)
        bad_model = config({"type": "attention"})
        bad_model["model"]["activations"]["backend"] = "s3"
        with self.assertRaisesRegex(ValueError, "model.activations.backend"):
            pt.parse_train_config(bad_model)
        bad_model = config({"type": "attention"})
        bad_model["model"]["layers"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "model.layers has duplicates"):
            pt.parse_train_config(bad_model)
        for bad in ("", "  ", "/", ".", "..", 3, "a/b", ["x"]):
            bad_model = config({"type": "attention"})
            bad_model["model"]["activations"]["folder"] = bad
            with self.assertRaisesRegex(ValueError, "model.activations.folder", msg=repr(bad)):
                pt.parse_train_config(bad_model)

    def test_parse_train_config_defaults_and_paths(self):
        cfg = pt.parse_train_config(config({"type": "attention", "heads": 2}))
        t = cfg.training
        self.assertEqual((t.epochs, t.lr, t.weight_decay, t.batch_size, t.max_batch_tokens),
                         (20, 1e-3, 1e-2, 16, 65536))
        self.assertEqual((t.min_prefix_frac, t.grad_clip, t.patience, t.class_weighting, t.seed),
                         (0.1, 1.0, 5, "balanced", 42))
        d = cfg.dataloader
        self.assertEqual((d.num_workers, d.fetch_threads, d.prefetch_factor), (8, 4, 2))
        self.assertEqual(cfg.probe, {"type": "attention", "heads": 2, "standardize": True, "norm_tokens": 500_000})
        self.assertEqual(cfg.dataset.path, Path("/data/root/probe_datasets/ds.parquet"))
        self.assertTrue(cfg.dataset.path.is_absolute())
        self.assertIsNone(cfg.dataset.folds)
        self.assertEqual(cfg.dataset.cases, "both")
        self.assertEqual(cfg.output.dir, Path("/data/root/probe_runs"))
        self.assertEqual(cfg.output.run_name, "t")
        self.assertEqual(cfg.model.layers, tuple(LAYERS))
        self.assertEqual(cfg.model.activations["backend"], "local")
        self.assertIsNone(cfg.model.folder)
        self.assertNotIn("folder", cfg.model.activations)
        self.assertIsNone(cfg.to_dict()["model"]["activations"]["folder"])
        with_folder = config({"type": "attention"})
        with_folder["model"]["activations"]["folder"] = " shared "
        parsed = pt.parse_train_config(with_folder)
        self.assertEqual(parsed.model.folder, "shared")
        self.assertEqual(parsed.model.activations, cfg.model.activations)
        self.assertEqual(parsed.to_dict()["model"]["activations"]["folder"], "shared")
        self.assertEqual(pt.parse_train_config(parsed.to_dict()).model.folder, "shared")
        self.assertEqual(cfg.probe_type, "attention")
        self.assertEqual(cfg.search, {})
        self.assertEqual([t.id for t in pt.expand_grid(cfg)], [pt.DEFAULT_TRIAL_ID])

        linear = pt.parse_train_config(config({"type": "linear"}, training={"class_weighting": False}))
        self.assertEqual(linear.training.min_prefix_frac, 1.0)
        self.assertEqual(linear.training.class_weighting, "none")
        self.assertEqual(linear.probe, {"type": "linear", "points": list(st.POINTS), "standardize": True})
        self.assertEqual(pt.parse_train_config(config({"type": "linear"}, training={"class_weighting": True}))
                         .training.class_weighting, "balanced")

        tfidf = pt.parse_train_config(config({"type": "tfidf"}, model=False,
                                             dataset={"path": "${DATA_ROOT}/d.parquet", "folds": ["metadata"],
                                                      "cases": "positive"}))
        self.assertIsNone(tfidf.model)
        self.assertEqual(tfidf.probe["text"]["spacy_model"], "en_core_web_sm")
        self.assertEqual(tfidf.probe["vectorizer"]["ngram_range"], [1, 2])
        self.assertEqual(tfidf.dataset.folds, ("metadata",))
        self.assertEqual(tfidf.dataset.cases, "positive")
        self.assertEqual(tfidf.training.min_prefix_frac, 1.0)
        json.dumps(tfidf.to_dict())
        as_dict = json.loads(json.dumps(cfg.to_dict()))
        self.assertEqual(as_dict["dataset"]["path"], "/data/root/probe_datasets/ds.parquet")
        self.assertEqual(as_dict["training"]["min_prefix_frac"], 0.1)

    def test_grid_expansion_and_trial_ids(self):
        cfg = pt.parse_train_config(config({"type": "attention"},
                                           search={"lr": [1e-3, 1e-2], "heads": [1, 2], "weight_decay": 0.0}))
        self.assertEqual(cfg.search, {"training.lr": [1e-3, 1e-2], "probe.heads": [1, 2], "training.weight_decay": 0.0})
        trials = pt.expand_grid(cfg)
        self.assertEqual([t.id for t in trials],
                         ["heads=1,lr=0.001", "heads=2,lr=0.001", "heads=1,lr=0.01", "heads=2,lr=0.01"])
        for t in trials:
            self.assertEqual(t.overrides["training.weight_decay"], 0.0)
        training, probe = pt.apply_trial(cfg.training, cfg.probe, trials[3])
        self.assertEqual((training.lr, training.weight_decay, probe["heads"]), (1e-2, 0.0, 2))
        self.assertEqual(cfg.training.lr, 1e-3)          # the base blocks are untouched
        self.assertNotIn("heads", cfg.probe)
        # nested vectorizer keys and an explicit prefix
        tfidf = pt.parse_train_config(config({"type": "tfidf"}, model=False,
                                             search={"vectorizer.min_df": [1, 2], "training.lr": 0.5,
                                                     "probe.vectorizer.binary": [True, False]}))
        trials = pt.expand_grid(tfidf)
        self.assertEqual([t.id for t in trials],
                         ["vectorizer.binary=true,vectorizer.min_df=1", "vectorizer.binary=false,vectorizer.min_df=1",
                          "vectorizer.binary=true,vectorizer.min_df=2", "vectorizer.binary=false,vectorizer.min_df=2"])
        training, probe = pt.apply_trial(tfidf.training, tfidf.probe, trials[2])
        self.assertEqual((training.lr, probe["vectorizer"]["min_df"], probe["vectorizer"]["binary"]), (0.5, 2, True))
        self.assertEqual(probe["vectorizer"]["ngram_range"], [1, 2])   # the rest of the block is kept
        # no lists → one trial carrying the scalar override
        one = pt.parse_train_config(config({"type": "attention"}, search={"lr": 0.5}))
        self.assertEqual(pt.expand_grid(one), [pt.Trial(id="default", overrides={"training.lr": 0.5})])
        self.assertEqual(pt.apply_trial(one.training, one.probe, pt.expand_grid(one)[0])[0].lr, 0.5)

    def test_search_key_must_exist(self):
        bad = [
            ({"type": "attention"}, {"momentum": [0.9]}, "not a sweepable"),
            ({"type": "attention"}, {"type": ["linear"]}, "not a sweepable"),
            ({"type": "linear"}, {"points": [["pre_cot"]]}, "not a sweepable"),
            ({"type": "attention"}, {"vectorizer.min_df": [1]}, "not a sweepable"),
            ({"type": "attention"}, {"probe.lr": [1e-3]}, "not a sweepable"),
            ({"type": "attention"}, {"lr": []}, "empty list"),
            ({"type": "attention"}, {"lr": [1e-3], "training.lr": [1e-2]}, "both name training.lr"),
            ({"type": "linear"}, {"min_prefix_frac": [0.5, 1.0]}, "only valid for probe type 'attention'"),
            ({"type": "attention"}, {"lr": [1e-3, 1e-3]}, "duplicate trial ids"),
        ]
        for probe, search, message in bad:
            with self.subTest(search=search):
                with self.assertRaisesRegex(ValueError, message):
                    pt.parse_train_config(config(probe, model=probe["type"] != "tfidf", search=search))
        with self.assertRaisesRegex(ValueError, "available: .*training.lr"):
            pt.parse_train_config(config({"type": "attention"}, search={"nope": 1}))
        ok = pt.parse_train_config(config({"type": "tfidf"}, model=False,
                                          search={"text.lowercase": [True, False], "probe.num_classes": 2}))
        self.assertEqual(set(ok.search), {"probe.text.lowercase", "probe.num_classes"})


# ---------------------------------------------------------------------------
# Fold plan
# ---------------------------------------------------------------------------


class TestFoldPlan(unittest.TestCase):
    """plan_folds / folds_to_json / folds_from_json."""

    def test_plan_folds_cases_filter_only_touches_train_val(self):
        rows = make_rows()
        both = pt.plan_folds(rows, None)
        self.assertEqual([f.held_out_style for f in both], sorted(STYLES))
        for fold in both:
            roles = fold_roles(rows, fold.held_out_style)
            for role, attr in (("train", "train_ids"), ("val", "val_ids"), ("test_id", "test_id_ids"),
                               ("test_ood", "test_ood_ids")):
                expected = sorted(rows.loc[(roles == role).fillna(False).to_numpy(), "rollout_id"])
                self.assertEqual(list(getattr(fold, attr)), expected)
            self.assertEqual(fold.ids("test_ood"), fold.test_ood_ids)
            self.assertTrue(all(i.split(":")[2] == fold.held_out_style for i in fold.test_ood_ids))
            self.assertTrue(all(i.split(":")[2] != fold.held_out_style for i in fold.train_ids + fold.test_id_ids))
        positive = pt.plan_folds(rows, ["metadata"], cases="positive")
        self.assertEqual(len(positive), 1)
        case_of = dict(zip(rows["rollout_id"], rows["case"]))
        p, b = positive[0], next(f for f in both if f.held_out_style == "metadata")
        self.assertTrue(all(case_of[i] == "positive" for i in p.train_ids + p.val_ids))
        self.assertEqual(set(p.train_ids), {i for i in b.train_ids if case_of[i] == "positive"})
        self.assertEqual(set(p.val_ids), {i for i in b.val_ids if case_of[i] == "positive"})
        self.assertEqual(p.test_id_ids, b.test_id_ids)
        self.assertEqual(p.test_ood_ids, b.test_ood_ids)
        self.assertLess(len(p.train_ids), len(b.train_ids))
        self.assertEqual(len(b.train_ids), 24)
        self.assertEqual(len(b.test_ood_ids), 4)
        with self.assertRaisesRegex(ValueError, "pushback.*styles present"):
            pt.plan_folds(rows, ["pushback"])
        with self.assertRaisesRegex(ValueError, "cases must be one of"):
            pt.plan_folds(rows, None, cases="all")
        with self.assertRaisesRegex(ValueError, "split"):
            pt.plan_folds(rows.assign(split=None), None)

    def test_folds_json_roundtrip(self):
        plan = pt.plan_folds(make_rows(), None, cases="negative")
        data = json.loads(json.dumps(pt.folds_to_json(plan)))
        self.assertEqual(pt.folds_from_json(data), plan)
        self.assertEqual(sorted(data[0]), sorted(["held_out_style", "train_ids", "val_ids", "test_id_ids", "test_ood_ids"]))
        with self.assertRaisesRegex(ValueError, "unknown key"):
            pt.folds_from_json([{**data[0], "extra": 1}])


# ---------------------------------------------------------------------------
# Feature adapters
# ---------------------------------------------------------------------------


class TestSources(StoreFixture):
    """The three feature adapters over the fake store / corpus."""

    def test_sequence_source_batches_by_token_budget(self):
        src = self.sequence_source()
        self.assertEqual(src.units, [(0, "na"), (1, "na")])
        self.assertEqual(src.input_kind, "sequence")
        self.assertEqual(src.feature_dim((0, "na")), HIDDEN)
        ids = self.rows["rollout_id"].tolist()[:12]
        n_tokens = dict(zip(self.rows["rollout_id"], self.rows["n_tokens"]))
        seq, _, _ = open_stores(self.backend)
        seen = []
        for batch in src.batches(ids, shuffle=False, seed=0, max_batch_tokens=10, batch_size=4):
            self.assertIsInstance(batch, pt.Batch)
            B = len(batch.ids)
            self.assertLessEqual(B, 4)
            T = int(batch.lengths.max())
            if B > 1:
                self.assertLessEqual(B * T, 10)
            for unit in src.units:
                H = batch.features[unit]
                self.assertEqual(H.dtype, torch.bfloat16)
                self.assertEqual(tuple(H.shape), (B, T, HIDDEN))
            self.assertEqual(batch.lengths.tolist(), [n_tokens[i] for i in batch.ids])
            self.assertEqual(batch.labels.tolist(), [self.labels[i] for i in batch.ids])
            self.assertEqual(batch.weights.tolist(), [1.0] * B)
            for k, rid in enumerate(batch.ids):
                stored = seq.get(rid)
                for layer in LAYERS:
                    self.assertTrue(torch.equal(batch.features[(layer, "na")][k, :n_tokens[rid]], stored[layer]))
            seen.extend(batch.ids)
        self.assertEqual(sorted(seen), sorted(ids))
        plan = lambda seed: [b.ids for b in src.batches(ids, shuffle=True, seed=seed, max_batch_tokens=10, batch_size=4)]
        self.assertEqual(plan(1), plan(1))
        self.assertNotEqual(plan(1), plan(2))
        self.assertEqual(sorted(i for b in plan(1) for i in b), sorted(ids))
        sub = self.sequence_source(layers=[1])
        self.assertEqual(sub.units, [(1, "na")])
        self.assertEqual(list(next(iter(sub.batches(ids[:2], shuffle=False, seed=0, max_batch_tokens=100,
                                                     batch_size=8))).features), [(1, "na")])
        # a subset of the store's layers fetches only that subset per sample
        seq_all, _, _ = open_stores(self.backend)
        subset = pt.SequenceSource(seq_all, [1], labels=self.labels, n_tokens=n_tokens, num_workers=0)
        with mock.patch.object(seq_all, "get", wraps=seq_all.get) as get:
            list(subset.batches(ids[:3], shuffle=False, seed=0, max_batch_tokens=100, batch_size=8))
        self.assertEqual(get.call_count, 3)
        self.assertTrue(all(call.args[1] == [1] for call in get.call_args_list))
        self.assertEqual(list(src.batches([], shuffle=False, seed=0, max_batch_tokens=10, batch_size=4)), [])
        with self.assertRaisesRegex(ValueError, "not stored in the store"):
            self.sequence_source(layers=[7])
        seq_all, _, _ = open_stores(self.backend)
        with self.assertRaisesRegex(ValueError, "not read by the store"):
            pt.SequenceSource(seq_all, [7], labels=self.labels, n_tokens=n_tokens)
        with self.assertRaisesRegex(KeyError, "has no label"):
            list(src.batches(["nemotron:run:metadata:99"], shuffle=False, seed=0, max_batch_tokens=10, batch_size=4))

    def test_point_source_materialises_once_per_fold(self):
        src = self.point_source(["pre_cot", "last_cot"])
        self.assertEqual(src.units, [(0, "pre_cot"), (0, "last_cot"), (1, "pre_cot"), (1, "last_cot")])
        self.assertEqual(src.input_kind, "vector")
        ids = self.rows["rollout_id"].tolist()[:10]
        with mock.patch.object(src.store, "get_many", wraps=src.store.get_many) as get_many:
            for _ in range(3):
                batches = list(src.batches(ids, shuffle=False, seed=0, max_batch_tokens=1, batch_size=4))
            self.assertEqual(get_many.call_count, 2)          # once per point, for every layer at once
            self.assertEqual(src.n_fetches, 2)
            self.assertEqual([len(b.ids) for b in batches], [4, 4, 2])
            self.assertIsNone(batches[0].lengths)
            for b in batches:
                for unit, X in b.features.items():
                    self.assertEqual(X.dtype, torch.bfloat16)
                    self.assertEqual(tuple(X.shape), (len(b.ids), HIDDEN))
                for k, rid in enumerate(b.ids):
                    for layer, point in src.units:
                        self.assertTrue(torch.equal(b.features[(layer, point)][k], src.store.get(rid, point)[layer]))
                self.assertEqual(b.labels.tolist(), [self.labels[i] for i in b.ids])
            list(src.batches(ids[:3], shuffle=False, seed=0, max_batch_tokens=1, batch_size=4))
            self.assertEqual(get_many.call_count, 4)          # a different id tuple is a new fold set
            src.clear()
            list(src.batches(ids, shuffle=False, seed=0, max_batch_tokens=1, batch_size=4))
            self.assertEqual(get_many.call_count, 6)
        shuffled = [b.ids for b in src.batches(ids, shuffle=True, seed=3, max_batch_tokens=1, batch_size=4)]
        self.assertNotEqual([i for b in shuffled for i in b], ids)
        self.assertEqual(sorted(i for b in shuffled for i in b), sorted(ids))
        with self.assertRaisesRegex(ValueError, "points must be"):
            self.point_source(["pre_cot", "pre_cot"])

    def test_tfidf_source_fits_on_train_only(self):
        rows = self.rows
        corpus = corpus_of(rows)
        val_only = "nemotron:run:metadata:3"          # a val row: give it a token nobody else has
        corpus[val_only] = corpus[val_only] + " unicorn"
        src = pt.TfidfSource(corpus, {"min_df": 1, "ngram_range": [1, 1]}, labels=self.labels)
        self.assertEqual(src.units, [("na", "na")])
        self.assertEqual(src.input_kind, "sparse")
        with self.assertRaisesRegex(RuntimeError, "fit"):
            src.feature_dim(("na", "na"))
        fold = self.fold("metadata")
        src.fit(fold.train_ids)
        self.assertNotIn("unicorn", src.vectorizer.vocabulary_)
        self.assertIn("hint", src.vectorizer.vocabulary_)
        self.assertEqual(src.fitted_on, fold.train_ids)
        self.assertEqual(src.fit_calls, 1)
        self.assertEqual(src.feature_dim(("na", "na")), len(src.vectorizer.vocabulary_))
        src.fit(fold.train_ids)
        src.fit(fold.train_ids, probe_cfg={"vectorizer": {"min_df": 1, "ngram_range": [1, 1]}})
        self.assertEqual(src.fit_calls, 1)                    # same settings, same ids: free
        src.fit(fold.train_ids, probe_cfg={"vectorizer": {"min_df": 1, "ngram_range": [1, 2]}})
        self.assertEqual(src.fit_calls, 2)
        self.assertIn("hint says", src.vectorizer.vocabulary_)
        src.fit(fold.train_ids, probe_cfg={"vectorizer": {"min_df": 1, "ngram_range": [1, 1]}})
        self.assertEqual(src.fit_calls, 2)                    # an earlier fit is kept, not redone
        self.assertNotIn("hint says", src.vectorizer.vocabulary_)
        src.fit(fold.train_ids, probe_cfg={"vectorizer": {"min_df": 1, "ngram_range": [1, 2]}})
        F_ = src.feature_dim(("na", "na"))
        with mock.patch.object(pt, "transform", wraps=pt.transform) as tf:
            for _ in range(3):
                list(src.batches(list(fold.val_ids), shuffle=False, seed=0, max_batch_tokens=1, batch_size=3))
        self.assertEqual(tf.call_count, 1)                    # transformed once per (fit, ids)
        src.clear()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            src.matrix(list(fold.val_ids))
        src.fit(fold.train_ids, probe_cfg={"vectorizer": {"min_df": 1, "ngram_range": [1, 2]}})
        self.assertEqual(src.fit_calls, 3)
        # corpus_for is only consulted for a text setting other than the base pipeline's
        calls = []
        base_text = {"lowercase": True}
        with_text = pt.TfidfSource(corpus, {"min_df": 1, "ngram_range": [1, 1]}, labels=self.labels,
                                   text_cfg=base_text, corpus_for=lambda t: calls.append(t) or corpus)
        with_text.fit(fold.train_ids, probe_cfg={"text": base_text})
        self.assertEqual(calls, [])
        with_text.fit(fold.train_ids, probe_cfg={"text": {"lowercase": False}})
        self.assertEqual(calls, [{"lowercase": False}])
        batches = list(src.batches(list(fold.val_ids), shuffle=False, seed=0, max_batch_tokens=1, batch_size=3))
        self.assertEqual([len(b.ids) for b in batches], [3, 3, 2])
        for b in batches:
            X = b.features[("na", "na")]
            self.assertEqual(X.layout, torch.sparse_csr)
            self.assertEqual(tuple(X.shape), (len(b.ids), F_))
            self.assertEqual(X.dtype, torch.float32)
            self.assertIsNone(b.lengths)
            self.assertEqual(b.labels.tolist(), [self.labels[i] for i in b.ids])
        dense = torch.cat([b.features[("na", "na")].to_dense() for b in batches])
        self.assertTrue(torch.allclose(dense, torch.from_numpy(src.matrix(list(fold.val_ids)).toarray())))
        with self.assertRaisesRegex(KeyError, "no text in the corpus"):
            list(src.batches(["nemotron:run:metadata:99"], shuffle=False, seed=0, max_batch_tokens=1, batch_size=3))
        with self.assertRaisesRegex(ValueError, "corpus_for needs text_cfg"):
            pt.TfidfSource(corpus, labels=self.labels, corpus_for=lambda text: corpus)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            pt.TfidfSource(pd.concat([corpus, corpus]), labels=self.labels)


# ---------------------------------------------------------------------------
# The loop and evaluation
# ---------------------------------------------------------------------------


def training(**kw) -> pt.TrainingConfig:
    base = {"epochs": 3, "lr": 0.05, "weight_decay": 0.0, "batch_size": 8, "max_batch_tokens": 64,
            "min_prefix_frac": 1.0, "grad_clip": 1.0, "patience": 5, "class_weighting": "balanced", "seed": 7}
    base.update(kw)
    return pt.TrainingConfig(**base)


class TestTrainFold(StoreFixture):
    """train_fold / FoldResult / evaluate over the fake store."""

    def test_train_fold_selects_best_trial_by_val_auroc(self):
        # A weaker signal than the class store's, so the untrained probe is not already perfect.
        with tempfile.TemporaryDirectory() as tmp:
            backend = write_store(Path(tmp) / "noisy", self.rows, seed=1, scale=0.15)
            _, pts, _ = open_stores(backend, [1])
            src = pt.PointSource(pts, [1], ["pre_cot"], labels=self.labels)
            self._check_best_trial(src)
            st.reset_read_caches()

    def _check_best_trial(self, src):
        fold = self.fold("metadata")
        probe = {"type": "linear", "points": ["pre_cot"], "standardize": True}
        trials = [pt.Trial("lr=1e-09", {"training.lr": 1e-9}), pt.Trial("lr=0.05", {"training.lr": 0.05})]
        result = pt.train_fold(fold, src, probe, training(epochs=6), trials, device="cpu", log=None, fold_index=0)
        self.assertEqual(result.held_out_style, "metadata")
        self.assertEqual(list(result.units), [(1, "pre_cot")])
        r = result.units[(1, "pre_cot")]
        self.assertEqual(set(r.trials), {"lr=1e-09", "lr=0.05"})
        self.assertEqual(r.best_trial, "lr=0.05")
        self.assertEqual(r.best_score, max(t["best_score"] for t in r.trials.values()))
        self.assertGreater(r.trials["lr=0.05"]["best_score"], r.trials["lr=1e-09"]["best_score"])
        self.assertEqual(r.best_epoch, r.trials["lr=0.05"]["best_epoch"])
        self.assertEqual(r.val_metrics, r.trials["lr=0.05"]["val_metrics"])
        self.assertIn("auroc", r.val_metrics)
        self.assertEqual(r.val_metrics["n"], len(fold.val_ids))
        for trial_id, hist in r.history.items():
            self.assertEqual(len(hist), 6)
            self.assertEqual(set(hist[0]), {"epoch", "train_loss", "train_auroc", "val_loss", "val_auroc"})
            self.assertEqual([h["epoch"] for h in hist], [1, 2, 3, 4, 5, 6])
        self.assertEqual(r.state["probe_config"]["type"], "linear")
        self.assertEqual(r.state["probe_config"]["hidden_dim"], HIDDEN)
        self.assertIn("linear.weight", r.state["model_state_dict"])
        probe_obj = load_probe(r.state)
        self.assertFalse(torch.equal(probe_obj.std, torch.ones(HIDDEN)))     # standardised
        # ties go to the earlier trial
        tie = pt.train_fold(fold, src, probe, training(epochs=2), [pt.Trial("a", {}), pt.Trial("b", {})])
        u = tie.units[(1, "pre_cot")]
        self.assertEqual(u.trials["a"]["best_score"], u.trials["b"]["best_score"])
        self.assertEqual(u.best_trial, "a")
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            pt.train_fold(fold, src, probe, training(epochs=1), [pt.Trial("a", {}), pt.Trial("a", {})])
        # a seed sweep takes effect: different initialisation / shuffles per trial
        seeds = pt.train_fold(fold, src, {**probe, "standardize": False}, training(epochs=2),
                              [pt.Trial("seed=1", {"training.seed": 1}), pt.Trial("seed=2", {"training.seed": 2})])
        h = seeds.units[(1, "pre_cot")].history
        self.assertNotEqual(h["seed=1"], h["seed=2"])
        same = pt.train_fold(fold, src, {**probe, "standardize": False}, training(epochs=2, seed=2), None)
        self.assertEqual(same.units[(1, "pre_cot")].history[pt.DEFAULT_TRIAL_ID], h["seed=2"])

    def test_early_stopping_per_unit(self):
        src = self.point_source(["pre_cot", "mean_cot"], layers=[0])
        fold = self.fold("consensus")
        probe = {"type": "linear", "points": ["pre_cot", "mean_cot"], "standardize": False}
        logs = []
        result = pt.train_fold(fold, src, probe, training(epochs=10, lr=1e-12, patience=2), None, log=logs.append)
        for unit, r in result.units.items():
            hist = r.history[pt.DEFAULT_TRIAL_ID]
            self.assertEqual(len(hist), 3, unit)                 # epoch 1 sets the best, 2 bad epochs, stop
            self.assertEqual(r.best_epoch, 1)
            self.assertEqual(r.best_trial, pt.DEFAULT_TRIAL_ID)
        self.assertTrue(any("early stop at epoch 3" in m for m in logs))
        full = pt.train_fold(fold, src, probe, training(epochs=3, lr=1e-12, patience=0))
        for r in full.units.values():
            self.assertEqual(len(r.history[pt.DEFAULT_TRIAL_ID]), 3)
        with self.assertRaisesRegex(ValueError, "no training rows"):
            pt.train_fold(pt.Fold("x", (), fold.val_ids, (), ()), src, probe, training())

    def test_predictions_keyed_by_rollout_id_full_prefix(self):
        src = self.sequence_source()
        fold = self.fold("sycophancy")
        probe = {"type": "attention", "heads": 1, "standardize": True, "norm_tokens": 40}
        result = pt.train_fold(fold, src, probe, training(epochs=2, min_prefix_frac=0.5), None, fold_index=2)
        ids = list(fold.test_id_ids)
        rng = np.random.default_rng(0)
        permuted = [ids[i] for i in rng.permutation(len(ids))]
        preds = result.predict(permuted)
        self.assertEqual(set(preds), set(src.units))
        seq, _, n_tokens = open_stores(self.backend)
        for unit, frame in preds.items():
            self.assertEqual(list(frame.columns), PREDICTION_COLUMNS)
            self.assertEqual(frame["rollout_id"].tolist(), permuted)
            probe_obj = load_probe(result.units[unit].state).eval()
            for rid, p in zip(frame["rollout_id"], frame["p_unfaithful"]):
                H = seq.get(rid)[unit[0]].unsqueeze(0).float()
                with torch.no_grad():
                    expected = float(p_detection_target(probe_obj(H, torch.tensor([n_tokens[rid]])))[0])
                self.assertAlmostEqual(float(p), expected, places=5)
        again = result.predict(ids)
        for unit in preds:
            a = preds[unit].set_index("rollout_id")["p_unfaithful"]
            b = again[unit].set_index("rollout_id")["p_unfaithful"]
            self.assertTrue(np.allclose(a.reindex(ids).to_numpy(), b.to_numpy()))
        self.assertEqual(len(result.predict([])[(0, "na")]), 0)

    def test_ensemble_is_mean_over_units(self):
        src = self.point_source(["pre_cot", "last_cot"])
        fold = self.fold("metadata")
        result = pt.train_fold(fold, src, {"type": "linear", "points": ["pre_cot", "last_cot"]}, training(epochs=2))
        out = pt.evaluate(result, src, fold.test_ood_ids, "test_ood")
        self.assertEqual(set(out), set(src.units) | {pt.ENSEMBLE_KEY})
        ens = out[pt.ENSEMBLE_KEY]
        self.assertEqual(list(ens.columns), PREDICTION_COLUMNS)
        self.assertEqual(ens["rollout_id"].tolist(), list(fold.test_ood_ids))
        stacked = np.stack([out[u]["p_unfaithful"].to_numpy() for u in src.units])
        self.assertTrue(np.allclose(ens["p_unfaithful"].to_numpy(), stacked.mean(axis=0)))
        self.assertIs(result.predictions["test_ood"], out)
        # frames aligned on rollout_id, not position
        shuffled = {u: f.iloc[::-1].reset_index(drop=True) if k else f for k, (u, f) in enumerate(out.items())
                    if u != pt.ENSEMBLE_KEY}
        self.assertTrue(np.allclose(pt.ensemble_predictions(shuffled)["p_unfaithful"].to_numpy(),
                                    stacked.mean(axis=0)))

    def test_run_is_deterministic_on_cpu(self):
        rows = self.rows
        plan_a = pt.folds_to_json(pt.plan_folds(rows, None, cases="positive"))
        plan_b = pt.folds_to_json(pt.plan_folds(rows.sample(frac=1.0, random_state=1), None, cases="positive"))
        self.assertEqual(plan_a, plan_b)
        fold = self.fold("consensus", cases="positive")
        probe = {"type": "attention", "heads": 2, "position_bias": True, "standardize": True, "norm_tokens": 30}
        trials = [pt.Trial("h=1", {"probe.heads": 1}), pt.Trial("h=2", {"probe.heads": 2})]
        runs = []
        for _ in range(2):
            src = self.sequence_source()
            result = pt.train_fold(fold, src, probe, training(epochs=2, min_prefix_frac=0.3, batch_size=4,
                                                               max_batch_tokens=12), trials, fold_index=1)
            preds = pt.evaluate(result, src, fold.test_id_ids, "test_id")
            runs.append((result, preds))
        (ra, pa), (rb, pb) = runs
        for unit in ra.units:
            self.assertEqual(ra.units[unit].history, rb.units[unit].history)
            self.assertEqual((ra.units[unit].best_trial, ra.units[unit].best_epoch),
                             (rb.units[unit].best_trial, rb.units[unit].best_epoch))
            for k, v in ra.units[unit].state["model_state_dict"].items():
                self.assertTrue(torch.equal(v, rb.units[unit].state["model_state_dict"][k]))
        for key in pa:
            pd.testing.assert_frame_equal(pa[key], pb[key])
        self.assertEqual(len(ra.units[(0, "na")].history["h=1"]), 2)

    def test_end_to_end_three_probe_types(self):
        rows, styles = self.rows, ["consensus", "sycophancy"]
        seq, pts, n_tokens = open_stores(self.backend)
        with mock.patch.dict(os.environ, {"DATA_ROOT": self._tmp.name}):
            dataset = {"path": "${DATA_ROOT}/ds.parquet", "folds": styles, "cases": "both"}
            train_block = {"epochs": 3, "lr": 0.05, "batch_size": 8, "max_batch_tokens": 64, "patience": 2, "seed": 3}
            configs = {
                "attention": pt.parse_train_config(config(
                    {"type": "attention", "heads": 1, "norm_tokens": 50}, dataset=dataset,
                    training={**train_block, "min_prefix_frac": 0.5}, search={"lr": [0.05, 0.01]})),
                "linear": pt.parse_train_config(config(
                    {"type": "linear", "points": ["pre_cot", "mean_cot"]}, dataset=dataset, training=train_block,
                    search={"weight_decay": [0.0, 0.1]})),
                "tfidf": pt.parse_train_config(config(
                    {"type": "tfidf", "vectorizer": {"min_df": 1}}, model=False, dataset=dataset,
                    training=train_block, search={"vectorizer.ngram_range": [[1, 1], [1, 2]]})),
            }
        expected_units = {"attention": {(0, "na"), (1, "na")},
                          "linear": {(l, p) for l in LAYERS for p in ("pre_cot", "mean_cot")},
                          "tfidf": {("na", "na")}}
        results = {}
        for kind, cfg in configs.items():
            trials = pt.expand_grid(cfg)
            self.assertEqual(len(trials), 2)
            plan = pt.plan_folds(rows, cfg.dataset.folds, cases=cfg.dataset.cases)
            self.assertEqual([f.held_out_style for f in plan], styles)
            if kind == "attention":
                source = pt.SequenceSource(seq, cfg.model.layers, labels=self.labels, n_tokens=n_tokens,
                                           num_workers=0, fetch_threads=cfg.dataloader.fetch_threads,
                                           prefetch_factor=cfg.dataloader.prefetch_factor)
            elif kind == "linear":
                source = pt.PointSource(pts, cfg.model.layers, cfg.probe["points"], labels=self.labels)
            else:
                source = pt.TfidfSource(corpus_of(rows), cfg.probe["vectorizer"], labels=self.labels,
                                        text_cfg=cfg.probe["text"])
            per_fold = []
            for k, fold in enumerate(plan):
                result = pt.train_fold(fold, source, cfg.probe, cfg.training, trials, device="cpu", fold_index=k)
                self.assertEqual(set(result.units), expected_units[kind])
                for test_set in pt.TEST_SETS:
                    out = pt.evaluate(result, source, fold.ids(test_set), test_set)
                    self.assertEqual(set(out), expected_units[kind] | {pt.ENSEMBLE_KEY})
                    for frame in out.values():
                        self.assertEqual(list(frame.columns), PREDICTION_COLUMNS)
                        self.assertEqual(frame["rollout_id"].tolist(), list(fold.ids(test_set)))
                        self.assertTrue(frame["p_unfaithful"].between(0, 1).all())
                self.assertEqual(set(result.predictions), set(pt.TEST_SETS))
                for unit, r in result.units.items():
                    self.assertIn(r.best_trial, {t.id for t in trials})
                    self.assertGreaterEqual(r.best_epoch, 1)
                    self.assertEqual(set(r.history), {t.id for t in trials})
                    self.assertTrue(all(1 <= len(h) <= cfg.training.epochs for h in r.history.values()))
                    self.assertEqual(set(r.trials), {t.id for t in trials})
                    self.assertIn("auroc", r.val_metrics)
                    self.assertEqual(r.state["probe_config"]["type"], kind)
                    load_probe(r.state)
                if kind == "tfidf":
                    self.assertEqual(source.fitted_on, fold.train_ids)
                    self.assertIsNotNone(source.vectorizer)
                    self.assertEqual(source.fit_calls, 2 * (k + 1))     # one fit per trial, none for predict
                    self.assertEqual(len(source.vectorizer.vocabulary_),
                                     result.units[("na", "na")].state["probe_config"]["n_features"])
                per_fold.append(result)
            results[kind] = per_fold
        # the synthetic signal is linearly separable at every point: the ensemble finds it
        for fold_result in results["linear"]:
            for test_set in pt.TEST_SETS:
                frame = fold_result.predictions[test_set][pt.ENSEMBLE_KEY]
                y = [self.labels[i] for i in frame["rollout_id"]]
                self.assertGreater(auroc(y, frame["p_unfaithful"].to_numpy()), 0.9, (fold_result.held_out_style, test_set))
        for fold_result in results["tfidf"]:
            frame = fold_result.predictions["test_ood"][pt.ENSEMBLE_KEY]
            y = [self.labels[i] for i in frame["rollout_id"]]
            self.assertGreater(auroc(y, frame["p_unfaithful"].to_numpy()), 0.9)


if __name__ == "__main__":
    unittest.main()
