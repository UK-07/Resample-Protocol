"""Tests for src/scripts/train_probe.py over a fake local activation store, a fake spaCy pipeline and a
60-row synthetic dataset parquet (CPU, no network)."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
import yaml

from src.lib import activation_store as st
from src.lib import probe_training as pt
from src.lib import text_features as tf
from src.lib.config import load_config
from src.lib.probe_datasets import DATASET_COLUMNS, dataset_fingerprint
from src.lib.probes import load_probe
from src.lib.report import CASE_VIEWS, PREDICTION_COLUMNS, TABLE_METRICS, delta_auroc, join_predictions, load_predictions
from src.scripts import probe_eval_report
from src.scripts import train_probe as tp
from tests.lib.probe_training_test import FOLDER, LAYERS, make_rows, write_store
from tests.lib.text_features_test import FakeNLP, fake_spacy

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_COL = "judge_label_final"


def dataset_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """The synthetic rows as a full ``DATASET_COLUMNS`` frame (the label column mirrors ``label``)."""
    out = pd.DataFrame({c: [None] * len(rows) for c in DATASET_COLUMNS})
    for c in ("rollout_id", "question_id", "hint_style", "case", "split", "label", "subject_model", "dataset", "reasoning"):
        out[c] = rows[c].to_numpy()
    n = len(rows)
    out["dataset_split"] = "test"
    out["original_index"] = range(n)
    out["subject_model_id"] = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    out["run"] = "run"
    out["target_option"] = "B"
    out["groundtruth"] = "A"
    out[LABEL_COL] = rows["label"].astype(int).to_numpy()
    out["source_csv"] = "x.csv"
    out["provenance"] = "resample_k4"
    out["is_resample"] = True
    out["sample_seed"] = 43
    out["balance_kept"] = True
    out["prompt"] = [f"Q{i}?" for i in range(n)]
    out["hinted_prompt"] = [f"Q{i}? [hint B]" for i in range(n)]
    out["rollout"] = [f"{r}</think>\n<answer>B</answer>" for r in rows["reasoning"]]
    out["choices"] = json.dumps(["a", "b", "c", "d"])
    out["option_letters"] = json.dumps(list("ABCD"))
    return out[list(DATASET_COLUMNS)]


def write_dataset(folder: Path, df: pd.DataFrame, *, name: str = FOLDER, fingerprint: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    parquet = folder / f"{name}.parquet"
    df.to_parquet(parquet, index=False)
    meta = {"name": name, "label_col": LABEL_COL, "predicate": "verbalised_vs_unverbalised",
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": fingerprint or dataset_fingerprint(df), "n_rows": int(len(df)),
            "hint_styles": sorted(df["hint_style"].unique())}
    (folder / f"{name}.meta.json").write_text(json.dumps(meta))
    return parquet


def write_fake_store(root: Path, rows: pd.DataFrame, *, fingerprint: str, short_name: str | None = "nemotron-nano-9b-v2",
                     **kw) -> Path:
    """tests/lib's ``write_store`` plus the ``dataset_spec`` / ``short_name`` the trainer checks."""
    write_store(root, rows, **kw)
    meta_path = root / FOLDER / st.STORE_MANIFEST_META_NAME
    meta = json.loads(meta_path.read_text())
    meta["dataset_spec"] = {"name": FOLDER, "fingerprint": fingerprint}
    if short_name is not None:
        meta["short_name"] = short_name
    meta_path.write_text(json.dumps(meta))
    return root


def patch_spacy():
    return mock.patch.object(tf, "_import_spacy", return_value=fake_spacy(load=lambda name, disable: FakeNLP()))


TRAIN = {"epochs": 2, "lr": 0.05, "batch_size": 8, "max_batch_tokens": 64, "patience": 2, "seed": 3}


class TrainProbeScriptTest(unittest.TestCase):
    """src/scripts/train_probe.py over the fake store / corpus."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.rows = make_rows()
        cls.df = dataset_frame(cls.rows)
        cls.fingerprint = dataset_fingerprint(cls.df)
        cls.parquet = write_dataset(cls.root / "probe_datasets", cls.df)
        cls.store = write_fake_store(cls.root / "store", cls.rows, fingerprint=cls.fingerprint)

    @classmethod
    def tearDownClass(cls):
        st.reset_read_caches()
        cls._tmp.cleanup()

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(dir=self.root))
        self._env = mock.patch.dict(os.environ, {"DATA_ROOT": str(self.root)})
        self._env.start()
        self.addCleanup(self._env.stop)
        self._spacy = patch_spacy()
        self._spacy.start()
        self.addCleanup(self._spacy.stop)

    # -- config helpers -----------------------------------------------------------------

    def config(self, kind: str, *, run_name: str | None = None, folds=("consensus", "sycophancy"), cases="both",
               store: Path | None = None, layers=list(LAYERS), search=None, **top) -> Path:
        cfg = {
            "dataset": {"path": str(self.parquet), "folds": list(folds) if folds else None, "cases": cases},
            "training": dict(TRAIN),
            "dataloader": {"num_workers": 0, "fetch_threads": 1, "prefetch_factor": 2},
            "output": {"dir": str(self.out), "run_name": run_name or kind},
        }
        if kind == "tfidf":
            cfg["probe"] = {"type": "tfidf", "vectorizer": {"min_df": 1, "ngram_range": [1, 1]},
                            "text": {"n_process": 1, "cache_dir": str(self.out / "text_cache")}}
            cfg["search"] = search if search is not None else {"lr": [0.05, 0.01]}
        else:
            cfg["model"] = {"subject_model": "nemotron-nano-9b-v2", "layers": layers,
                            "activations": {"backend": "local", "local_dir": str(store or self.store)}}
            if kind == "linear":
                cfg["probe"] = {"type": "linear", "points": ["pre_cot", "mean_cot"]}
                cfg["search"] = search if search is not None else {"weight_decay": [0.0, 0.1]}
            else:
                cfg["probe"] = {"type": "attention", "heads": 1, "norm_tokens": 50}
                cfg["training"]["min_prefix_frac"] = 0.5
                cfg["search"] = search if search is not None else {"lr": [0.05, 0.01]}
        cfg.update(top)
        path = self.out / f"{run_name or kind}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def run_main(self, config_path: Path, **kw) -> tuple[Path, str]:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            run_dir = tp.main(config_path, device="cpu", **kw)
        return run_dir, buf.getvalue()

    # -- tests ---------------------------------------------------------------------------

    def test_dry_run_reads_no_tensors_and_prints_plan(self):
        for kind in ("attention", "linear", "tfidf"):
            with self.subTest(kind=kind):
                cfg = self.config(kind, run_name=f"dry_{kind}")
                with mock.patch.object(st, "local_handle", side_effect=AssertionError("a tensor was read")), \
                        mock.patch.object(tf, "normalise_texts", side_effect=AssertionError("text normalised")):
                    run_dir, out = self.run_main(cfg, dry_run=True)
                self.assertEqual(sorted(p.name for p in run_dir.iterdir()), ["config.yaml", "dry_run.json", "folds.json", "run.log"])
                self.assertIn("Fold plan (2 fold(s)", out)
                self.assertIn("Grid: 2 trial(s)", out)
                self.assertIn("Dry run", out)
                plan = json.loads((run_dir / "dry_run.json").read_text())
                self.assertEqual([f["held_out_style"] for f in plan["fold_plan"][::4]], ["consensus", "sycophancy"])
                self.assertEqual({f["role"] for f in plan["fold_plan"]}, {"train", "val", "test_id", "test_ood"})
                train = next(f for f in plan["fold_plan"] if f["role"] == "train")
                self.assertEqual((train["n_rows"], train["n_label_0"] + train["n_label_1"]), (24, 24))
                self.assertEqual(len(plan["trials"]), 2)
                self.assertEqual(plan["dataset"]["fingerprint"], self.fingerprint)
                if kind == "tfidf":
                    self.assertIsNone(plan["store"])
                    self.assertFalse(plan["text_cache"]["exists"])
                    self.assertEqual(plan["text_cache"]["n_ids_to_normalise"], 60)
                    self.assertIn("Text cache", out)
                    self.assertFalse((self.out / "text_cache").exists())
                else:
                    self.assertEqual(plan["store"]["n_rollouts"], 60)
                    self.assertEqual(plan["store"]["layers"], list(LAYERS))
                    self.assertEqual(plan["store"]["folder"], FOLDER)
                    self.assertEqual(plan["store"]["dataset_spec"]["fingerprint"], self.fingerprint)
                    self.assertTrue(plan["store"]["same_dataset"])
                    self.assertIn("Store collected from dataset", out)
                    est = plan["estimates"]
                    self.assertIn("Bytes", out)
                    if kind == "attention":
                        n_tok = self.rows.set_index("rollout_id")["n_tokens"]
                        fold = pt.plan_folds(self.df, ["consensus"])[0]
                        expected = int(n_tok.loc[list(fold.train_ids + fold.val_ids)].sum()) * 2 * 4 * 2
                        self.assertEqual(est["per_fold"]["consensus"]["per_epoch_train_val_bytes"], expected)
                        self.assertEqual(est["dataloader_ram_bytes"], (0 * 2 + 1) * 64 * 2 * 4 * 2)
                    else:
                        self.assertEqual(est["per_fold"]["consensus"]["materialised_bytes"], 44 * 2 * 2 * 4 * 2)
                cfg_back = yaml.safe_load((run_dir / "config.yaml").read_text())
                pt.parse_train_config(cfg_back)

    def test_outputs_written_for_each_probe_type(self):
        expected_units = {"attention": [f"layer_{l}__na" for l in LAYERS],
                          "linear": [f"layer_{l}__{p}" for l in LAYERS for p in ("pre_cot", "mean_cot")],
                          "tfidf": ["layer_na__na"]}
        for kind in ("tfidf", "linear", "attention"):
            with self.subTest(kind=kind):
                run_dir, out = self.run_main(self.config(kind))
                units = expected_units[kind]
                for name in ("config.yaml", "run.log", "folds.json", "search_results.csv", "results.json",
                             "results_summary.csv", "aggregate.json", "plots/auroc_by_unit.png", "plots/auroc_by_fold.png"):
                    self.assertTrue((run_dir / name).exists(), name)
                self.assertFalse((run_dir / "ablation.json").exists())
                folds = pt.folds_from_json(json.loads((run_dir / "folds.json").read_text()))
                self.assertEqual([f.held_out_style for f in folds], ["consensus", "sycophancy"])
                for fold in folds:
                    fdir = run_dir / f"fold_{fold.held_out_style}"
                    self.assertTrue((run_dir / "plots" / f"training_curves_{fold.held_out_style}.png").exists())
                    self.assertEqual((fdir / "vectorizer.joblib").exists(), kind == "tfidf")
                    if kind == "tfidf":
                        tf.load_vectorizer(fdir / "vectorizer.joblib")
                    for test_set in pt.TEST_SETS:
                        for name in units + ["ensemble"]:
                            pred = pd.read_csv(fdir / "predictions" / f"{test_set}__{name}.csv")
                            self.assertEqual(list(pred.columns), PREDICTION_COLUMNS)
                            self.assertEqual(pred["rollout_id"].tolist(), list(fold.ids(test_set)))
                            self.assertTrue(pred["p_unfaithful"].between(0, 1).all())
                            rep = json.loads((fdir / "report" / f"{test_set}__{name}_report.json").read_text())
                            self.assertEqual(set(rep["overall"]), set(CASE_VIEWS))
                            self.assertEqual((rep["fold"], rep["test_set"]), (fold.held_out_style, test_set))
                            self.assertEqual(rep["label_col"], LABEL_COL)
                            self.assertIn("## Overall", (fdir / "report" / f"{test_set}__{name}_report.md").read_text())
                    for name in units:
                        ckpt = torch.load(fdir / f"probe__{name}.pt", weights_only=False)
                        for key in ("probe_config", "model_state_dict", "best_epoch", "val_metrics", "history", "trial",
                                    "trial_overrides", "unit", "config", "dataset"):
                            self.assertIn(key, ckpt, key)
                        self.assertEqual(ckpt["probe_config"]["type"], kind)
                        self.assertEqual(set(ckpt["history"]), {t.id for t in pt.expand_grid(pt.parse_train_config(
                            yaml.safe_load((run_dir / "config.yaml").read_text())))})
                        load_probe(ckpt)
                    self.assertFalse((fdir / "probe__ensemble.pt").exists())
                results = json.loads((run_dir / "results.json").read_text())
                for style in ("consensus", "sycophancy"):
                    for test_set in pt.TEST_SETS:
                        block = results["metrics"][style][test_set]
                        layers = {"attention": [str(l) for l in LAYERS], "linear": [str(l) for l in LAYERS], "tfidf": ["na"]}[kind]
                        self.assertEqual(sorted(block), sorted(layers + ["ensemble"]))
                        for layer in layers:
                            for point, views in block[layer].items():
                                self.assertEqual(set(views), set(CASE_VIEWS))
                                self.assertIn("auroc", views["pooled"])
                        self.assertEqual(list(block["ensemble"]), ["na"])
                    self.assertEqual(set(results["training"][style]), set(units))
                    for u in results["training"][style].values():
                        self.assertIn(u["best_trial"], {t["id"] for t in results["trials"]})
                summary = pd.read_csv(run_dir / "results_summary.csv")
                self.assertEqual(list(summary.columns), ["fold", "test_set", "layer", "point", "view", "trial", *TABLE_METRICS])
                self.assertEqual(len(summary), 2 * 2 * (len(units) + 1) * 3)
                self.assertTrue(summary.loc[summary["layer"] != "ensemble", "trial"].notna().all())
                search = pd.read_csv(run_dir / "search_results.csv")
                self.assertEqual(len(search), 2 * 2 * len(units))
                self.assertEqual(int(search["selected"].sum()), 2 * len(units))
                self.assertIn("best_val_score", search.columns)
                agg = json.loads((run_dir / "aggregate.json").read_text())
                self.assertEqual(agg["n_folds"], 2)
                self.assertEqual(len(agg["by_unit"]), 2 * (len(units) + 1) * 3)
                self.assertIn("val AUROC", out)
                self.assertIn("Outputs in", (run_dir / "run.log").read_text())
                if kind != "tfidf":
                    self.assertEqual(results["store"]["dataset_spec"]["fingerprint"], self.fingerprint)

    def test_store_from_other_dataset_accepted_when_ids_covered(self):
        other_fp = "0" * 64
        store = write_fake_store(self.out / "other_store", self.rows, fingerprint=other_fp)
        meta_path = store / FOLDER / st.STORE_MANIFEST_META_NAME
        meta = json.loads(meta_path.read_text())
        meta["dataset_spec"].update(name="superset_spec", n_rows=999)
        meta_path.write_text(json.dumps(meta))
        run_dir, out = self.run_main(self.config("linear", store=store, folds=("consensus",)))
        results = json.loads((run_dir / "results.json").read_text())
        self.assertTrue(results["complete"])
        self.assertEqual(results["store"]["dataset_spec"],
                         {"name": "superset_spec", "fingerprint": other_fp, "n_rows": 999, "built_utc": None})
        self.assertEqual(results["store"]["training_dataset_fingerprint"], self.fingerprint)
        self.assertFalse(results["store"]["same_dataset"])
        self.assertEqual(results["dataset"]["fingerprint"], self.fingerprint)
        self.assertIn("collected from dataset 'superset_spec' (999 rows", out)
        self.assertIn("a different dataset (shared store", out)
        # The same-dataset store is flagged as such.
        run_dir, out = self.run_main(self.config("linear", folds=("consensus",)), dry_run=True)
        plan = json.loads((run_dir / "dry_run.json").read_text())
        self.assertTrue(plan["store"]["same_dataset"])
        self.assertIn("the same dataset", out)
        # A store from another dataset that lacks fold ids is still refused.
        partial = write_fake_store(self.out / "other_partial", self.rows.iloc[5:], fingerprint=other_fp)
        with self.assertRaisesRegex(ValueError, r"5 of 60 rollout\(s\).*not in the store manifest"):
            self.run_main(self.config("linear", store=partial))

    def test_missing_store_ids_are_an_error(self):
        partial = self.rows.iloc[5:]
        store = write_fake_store(self.out / "partial_store", partial, fingerprint=self.fingerprint)
        with self.assertRaisesRegex(ValueError, r"5 of 60 rollout\(s\).*nemotron:run:consensus:0") as ctx:
            self.run_main(self.config("attention", store=store))
        self.assertIn("consensus:4", str(ctx.exception))

    def test_requested_layers_and_points_validated(self):
        with self.assertRaisesRegex(ValueError, r"model\.layers.*\[7\].*not stored in the store"):
            self.run_main(self.config("attention", layers=[0, 7]))
        with self.assertRaisesRegex(ValueError, "probe.points"):
            self.run_main(self.config("linear", probe={"type": "linear", "points": ["pre_cot", "middle"]}))
        store = write_fake_store(self.out / "foreign_store", self.rows, fingerprint=self.fingerprint, short_name="qwen3.5-9b")
        with self.assertRaisesRegex(ValueError, "activations of 'qwen3.5-9b'"):
            self.run_main(self.config("linear", store=store))
        with self.assertRaisesRegex(ValueError, "no dataset_spec.fingerprint"):
            write_fake_store(self.out / "nofp_store", self.rows, fingerprint=None)
            self.run_main(self.config("linear", store=self.out / "nofp_store"))

    def test_store_folder_key_reads_named_folder(self):
        store = write_fake_store(self.out / "shared_root", self.rows, fingerprint="1" * 64)
        (store / FOLDER).rename(store / "shared")
        model = {"subject_model": "nemotron-nano-9b-v2", "layers": list(LAYERS),
                 "activations": {"backend": "local", "local_dir": str(store), "folder": "shared"}}
        run_dir, out = self.run_main(self.config("linear", folds=("consensus",), model=model), dry_run=True)
        plan = json.loads((run_dir / "dry_run.json").read_text())
        self.assertEqual(plan["store"]["folder"], "shared")
        self.assertEqual(plan["store"]["dataset_spec"]["fingerprint"], "1" * 64)
        self.assertIn("/shared (folder from model.activations.folder)", out)
        cfg_back = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.assertEqual(cfg_back["model"]["activations"]["folder"], "shared")
        self.assertEqual(pt.parse_train_config(cfg_back).model.folder, "shared")
        # Without the key the dataset's name is read — which this root lacks.
        model_default = {**model, "activations": {**model["activations"], "folder": None}}
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_main(self.config("linear", folds=("consensus",), model=model_default), dry_run=True)
        run_dir, out = self.run_main(self.config("linear", folds=("consensus",)), dry_run=True)
        self.assertIn(f"/{FOLDER} (folder from the dataset name)", out)

    def test_aggregate_over_folds(self):
        run_dir, _ = self.run_main(self.config("linear"))
        agg = json.loads((run_dir / "aggregate.json").read_text())
        summary = pd.read_csv(run_dir / "results_summary.csv", dtype={"layer": str, "point": str})
        self.assertEqual(agg["metrics"], list(tp.AGGREGATE_METRICS))
        for cell in agg["by_unit"]:
            rows = summary[(summary["test_set"] == cell["test_set"]) & (summary["layer"] == cell["layer"])
                           & (summary["point"] == cell["point"]) & (summary["view"] == cell["view"])]
            self.assertEqual(len(rows), 2)
            for metric in tp.AGGREGATE_METRICS:
                values = rows[metric].dropna()
                stats = cell[metric]
                self.assertEqual(stats["n"], len(values))
                if len(values):
                    self.assertAlmostEqual(stats["mean"], float(values.mean()), places=9)
                    self.assertAlmostEqual(stats["min"], float(values.min()), places=9)
                    self.assertAlmostEqual(stats["max"], float(values.max()), places=9)
                    if len(values) > 1:
                        self.assertAlmostEqual(stats["std"], float(values.std(ddof=1)), places=9)
        gaps = agg["ood_minus_id_auroc"]
        self.assertEqual(len(gaps), (2 * 2 + 1) * 3)
        for g in gaps:
            for fold, delta in g["per_fold"].items():
                sel = summary[(summary["fold"] == fold) & (summary["layer"] == g["layer"]) & (summary["point"] == g["point"])
                              & (summary["view"] == g["view"])].set_index("test_set")["auroc"]
                expected = sel["test_ood"] - sel["test_id"]
                if expected == expected:
                    self.assertAlmostEqual(delta, expected, places=9)
                else:
                    self.assertIsNone(delta)

    def test_compare_writes_ablation(self):
        positive_dir, _ = self.run_main(self.config("linear", run_name="pos", cases="positive"))
        pooled_dir, out = self.run_main(self.config("linear", run_name="pooled"), compare=positive_dir)
        ablation = json.loads((pooled_dir / "ablation.json").read_text())
        self.assertEqual((ablation["view"], ablation["b"]), ("positive", str(positive_dir)))
        self.assertEqual(ablation["n_pairs"], 2 * 2 * (2 * 2 + 1))
        self.assertIn("Ablation vs", out)
        for entry in ablation["entries"]:
            self.assertNotIn("error", entry)
            a = join_predictions(load_predictions(pooled_dir / entry["predictions"]), self.df, label_col=LABEL_COL)
            b = join_predictions(load_predictions(positive_dir / entry["predictions"]), self.df, label_col=LABEL_COL)
            expected = delta_auroc(a, b)
            self.assertEqual(entry["n"], expected["n"])
            for key in ("auroc_a", "auroc_b", "delta_auroc"):
                if expected[key] == expected[key]:
                    self.assertAlmostEqual(entry[key], expected[key], places=9)
        self.assertTrue(ablation["summary"])
        self.assertIn("ensemble", {s["unit"] for s in ablation["summary"]})

    def test_predictions_file_is_probe_eval_report_compatible(self):
        run_dir, _ = self.run_main(self.config("tfidf"))
        pred = run_dir / "fold_consensus" / "predictions" / "test_id__layer_na__na.csv"
        out_dir = self.out / "eval_report"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = probe_eval_report.main(["--predictions", str(pred), "--manifest", str(self.parquet),
                                           "--label-col", LABEL_COL, "--output-dir", str(out_dir)])
        self.assertEqual(code, 0)
        self.assertIn("pooled", buf.getvalue())
        external = json.loads((out_dir / "test_id__layer_na__na_report.json").read_text())
        own = json.loads((run_dir / "fold_consensus" / "report" / "test_id__layer_na__na_report.json").read_text())
        self.assertEqual(json.loads(json.dumps(tp._jsonable(external["overall"]))), own["overall"])
        self.assertEqual(external["by_group"].keys(), own["by_group"].keys())

    def test_folds_flag_restricts_run(self):
        run_dir, _ = self.run_main(self.config("linear", folds=None), folds=["metadata"], only_trial="weight_decay=0.1")
        self.assertEqual([p.name for p in run_dir.glob("fold_*")], ["fold_metadata"])
        self.assertEqual([f["held_out_style"] for f in json.loads((run_dir / "folds.json").read_text())], ["metadata"])
        self.assertEqual(yaml.safe_load((run_dir / "config.yaml").read_text())["dataset"]["folds"], ["metadata"])
        search = pd.read_csv(run_dir / "search_results.csv")
        self.assertEqual(set(search["trial"]), {"weight_decay=0.1"})
        self.assertEqual(len(search), 4)
        # config.yaml reproduces the run: the grid is narrowed to the picked trial's hyper-parameters.
        back = pt.parse_train_config(yaml.safe_load((run_dir / "config.yaml").read_text()))
        self.assertEqual(back.search, {"training.weight_decay": 0.1})
        self.assertEqual([t.overrides for t in pt.expand_grid(back)], [{"training.weight_decay": 0.1}])
        with self.assertRaisesRegex(ValueError, "pushback.*styles present"):
            self.run_main(self.config("linear", run_name="bad_fold"), folds=["pushback"])
        with self.assertRaisesRegex(ValueError, "--only-trial 'lr=1'"):
            self.run_main(self.config("linear", run_name="bad_trial"), only_trial="lr=1")
        with self.assertRaisesRegex(ValueError, "--folds names no style"):
            tp.cli(["--config", str(self.config("linear", run_name="empty_folds")), "--folds", ",", "--device", "cpu"])
        with self.assertRaisesRegex(ValueError, "--compare .* not a finished train_probe run"):
            self.run_main(self.config("linear", run_name="bad_compare"), compare=self.out / "nowhere")
        self.assertFalse(list(self.out.glob("*_empty_folds")) + list(self.out.glob("*_bad_compare")))

    def test_run_tables_written_after_every_fold(self):
        real = tp.write_fold_outputs
        calls = []

        def second_fold_dies(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("simulated crash in fold 2")
            return real(*args, **kwargs)

        with mock.patch.object(tp, "write_fold_outputs", side_effect=second_fold_dies):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                self.run_main(self.config("linear", run_name="interrupted"))
        run_dir = next(self.out.glob("*_interrupted"))
        results = json.loads((run_dir / "results.json").read_text())
        self.assertEqual((results["folds"], results["planned_folds"], results["complete"]),
                         (["consensus"], ["consensus", "sycophancy"], False))
        self.assertEqual(set(results["metrics"]), {"consensus"})
        self.assertEqual(json.loads((run_dir / "aggregate.json").read_text())["n_folds"], 1)
        self.assertEqual(set(pd.read_csv(run_dir / "results_summary.csv")["fold"]), {"consensus"})
        self.assertEqual(set(pd.read_csv(run_dir / "search_results.csv")["fold"]), {"consensus"})
        self.assertFalse((run_dir / "plots" / "auroc_by_unit.png").exists())
        complete_dir, _ = self.run_main(self.config("linear", run_name="complete"))
        self.assertTrue(json.loads((complete_dir / "results.json").read_text())["complete"])

    def test_sources_cleared_between_folds(self):
        with mock.patch.object(pt.PointSource, "clear", autospec=True, side_effect=pt.PointSource.clear) as clear:
            self.run_main(self.config("linear"))
        self.assertEqual(clear.call_count, 2)
        with mock.patch.object(pt.TfidfSource, "clear", autospec=True, side_effect=pt.TfidfSource.clear) as clear:
            self.run_main(self.config("tfidf"))
        self.assertEqual(clear.call_count, 2)

    def test_local_mirror_with_layer_subset_trains(self):
        mirror = self.out / "mirror"
        source = st.LocalBackend(self.store)
        with contextlib.redirect_stdout(io.StringIO()):
            st.mirror_store(source, FOLDER, mirror, layers=None)   # points only
        self.assertFalse(list((mirror / FOLDER).glob("seq_layer_*")))
        run_dir, out = self.run_main(self.config("linear", store=mirror, folds=("consensus",)))
        results = json.loads((run_dir / "results.json").read_text())
        self.assertTrue(results["complete"])
        self.assertEqual((results["store"]["layers"], results["store"]["mirrored_layers"],
                          results["store"]["mirrored_points_layers"]), (list(LAYERS), [], list(LAYERS)))
        self.assertEqual(results["store"]["mirror_source"]["backend"], "local")
        self.assertIn("Store is a local mirror of", out)
        self.assertIn("points files of layers [0, 1]", out)
        with self.assertRaisesRegex(ValueError, r"model\.layers: layers \[0, 1\] are not in this mirror.*seq.*--layers 0,1"):
            self.run_main(self.config("attention", store=mirror, run_name="att_nomirror"), dry_run=True)
        with self.assertRaisesRegex(ValueError, r"model\.layers: this mirror of the store holds no seq files.*--layers"):
            self.run_main(self.config("attention", store=mirror, layers=None, run_name="att_null0"), dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            st.mirror_store(source, FOLDER, mirror, layers=[1])
        st.reset_read_caches()
        run_dir, out = self.run_main(self.config("attention", store=mirror, layers=[1], folds=("consensus",),
                                                 run_name="att_l1"))
        results = json.loads((run_dir / "results.json").read_text())
        self.assertTrue(results["complete"])
        self.assertEqual((results["store"]["layers"], results["store"]["mirrored_layers"]), ([1], [1]))
        self.assertEqual(sorted(results["metrics"]["consensus"]["test_id"]), ["1", "ensemble"])
        self.assertIn("seq files of layers [1]", out)
        run_dir, _ = self.run_main(self.config("attention", store=mirror, layers=None, folds=("consensus",),
                                               run_name="att_null"), dry_run=True)
        self.assertEqual(json.loads((run_dir / "dry_run.json").read_text())["store"]["layers"], [1])
        with self.assertRaisesRegex(ValueError, r"model\.layers: layers \[0\] are not in this mirror"):
            self.run_main(self.config("attention", store=mirror, layers=[0, 1], run_name="att_l01"), dry_run=True)
        # The linear probe still reads every layer's points from the same mirror.
        run_dir, _ = self.run_main(self.config("linear", store=mirror, folds=("consensus",), run_name="lin2"), dry_run=True)
        self.assertEqual(json.loads((run_dir / "dry_run.json").read_text())["store"]["layers"], list(LAYERS))

    def test_store_with_seq_layers_subset(self):
        store = write_fake_store(self.out / "seq_store", self.rows, fingerprint=self.fingerprint,
                                 layers=(0, 1, 2), seq_layers=(1,))
        stems = {p.name.split("_part")[0] for p in (store / FOLDER).glob("*.safetensors")}
        self.assertEqual(stems, {"seq_layer_1", "points_layer_0", "points_layer_1", "points_layer_2"})
        run_dir, out = self.run_main(self.config("linear", store=store, layers=[0, 1, 2], folds=("consensus",),
                                                 run_name="lin_seq"))
        results = json.loads((run_dir / "results.json").read_text())
        self.assertTrue(results["complete"])
        self.assertEqual((results["store"]["stored_layers"], results["store"]["stored_seq_layers"], results["store"]["layers"]),
                         ([0, 1, 2], [1], [0, 1, 2]))
        self.assertEqual(sorted(results["metrics"]["consensus"]["test_id"]), ["0", "1", "2", "ensemble"])
        self.assertIn("(seq files for [1])", out)
        run_dir, _ = self.run_main(self.config("attention", store=store, layers=[1], folds=("consensus",), run_name="att_seq"))
        results = json.loads((run_dir / "results.json").read_text())
        self.assertTrue(results["complete"])
        self.assertEqual((results["store"]["layers"], results["store"]["stored_seq_layers"]), ([1], [1]))
        run_dir, _ = self.run_main(self.config("attention", store=store, layers=None, run_name="att_seq_null"), dry_run=True)
        self.assertEqual(json.loads((run_dir / "dry_run.json").read_text())["store"]["layers"], [1])
        with self.assertRaisesRegex(ValueError, r"model\.layers: layers \[0\] have no seq files in this store.*seq_layers: \[1\]"):
            self.run_main(self.config("attention", store=store, layers=[0], run_name="att_seq_0"), dry_run=True)
        with self.assertRaisesRegex(ValueError, r"model\.layers: layers \[0, 2\] have no seq files"):
            self.run_main(self.config("attention", store=store, layers=[0, 1, 2], run_name="att_seq_012"), dry_run=True)
        with self.assertRaisesRegex(ValueError, r"model\.layers.*\[7\].*not stored in the store"):
            self.run_main(self.config("attention", store=store, layers=[7], run_name="att_seq_7"), dry_run=True)

    def test_configs_parse(self):
        expected = {"tfidf": 6, "linear": 4, "attention": 4}
        models = {   # config prefix → (short name, collector config stem)
            "nemotron": ("nemotron-nano-9b-v2", "nemotron"),
            "olmo3-7b-think": ("olmo3-7b-think", "olmo"),
            "qwen3-8b": ("qwen3-8b", "qwen3-8b"),
            "gemma4-12b-it": ("gemma4-12b-it", "gemma4-12b"),
        }
        for prefix, (short, collect_stem) in models.items():
            collect = load_config(str(REPO_ROOT / "configs" / f"collect_probe_activations_{collect_stem}.yaml"))
            points_layers = tuple(collect["layers"])
            seq_layers = tuple(collect["seq_layers"] if collect.get("seq_layers") is not None else collect["layers"])
            for kind, n_trials in expected.items():
                with self.subTest(model=prefix, kind=kind):
                    cfg = pt.parse_train_config(load_config(str(REPO_ROOT / "configs" / "probes" / f"{prefix}_{kind}.yaml")))
                    self.assertEqual(cfg.probe_type, kind)
                    self.assertEqual(len(pt.expand_grid(cfg)), n_trials)
                    self.assertEqual(cfg.output.dir, self.root / "trained_probes" / "probes")
                    self.assertEqual(cfg.output.run_name, f"{prefix}_{kind}")
                    self.assertEqual(cfg.dataset.path.name, f"{short}_verbalised_vs_unverbalised.parquet")
                    self.assertIsNone(cfg.dataset.folds)
                    self.assertEqual(cfg.dataset.cases, "both")
                    if kind == "tfidf":
                        self.assertIsNone(cfg.model)
                        self.assertEqual(cfg.probe["text"]["cache_dir"], "${DATA_ROOT}/probe_datasets/text_cache")
                    else:
                        if kind == "linear":
                            self.assertEqual(cfg.model.layers, points_layers)   # every points layer of the store
                        else:
                            self.assertEqual(len(cfg.model.layers), 1)
                            self.assertIn(cfg.model.layers[0], seq_layers)      # a layer the store has seq files for
                            self.assertEqual(cfg.model.layers[0], points_layers[3])   # the middle one
                        self.assertEqual(cfg.model.activations["backend"], "local")
                        self.assertEqual(cfg.model.activations["local_dir"], "/root/probe_activations_mirror")
                        self.assertIsNone(cfg.model.activations["hf_repo_id"])
                        self.assertNotIn("folder", cfg.model.activations)   # split off into ModelConfig.folder
                        self.assertEqual(cfg.model.folder, f"{short}_shared")
                        self.assertEqual(cfg.model.folder, collect["store_folder"])
                        self.assertEqual(cfg.model.subject_model, short)
                        self.assertEqual((cfg.dataloader.num_workers, cfg.dataloader.fetch_threads, cfg.dataloader.prefetch_factor),
                                         (8, 4, 2))
                    if kind == "linear":
                        self.assertEqual(cfg.probe["points"], list(st.POINTS))
                    if kind == "attention":
                        self.assertEqual((cfg.probe["heads"], cfg.probe["position_bias"], cfg.training.min_prefix_frac), (4, True, 0.1))
            local = load_config(str(REPO_ROOT / "configs" / "probes" / f"{prefix}_probe_local.yaml"))
            self.assertEqual(set(local), {"model", "dataloader", "output"})
            self.assertEqual(local["model"]["layers"], list(points_layers))
            for extended in ("linear", "attention") + (("tfidf",) if prefix != "nemotron" else ()):
                raw = yaml.safe_load((REPO_ROOT / "configs" / "probes" / f"{prefix}_{extended}.yaml").read_text())
                self.assertEqual(raw["extends"], f"{prefix}_probe_local.yaml")
        base = load_config(str(REPO_ROOT / "configs" / "probes" / "nemotron_probe_base.yaml"))
        self.assertEqual(set(base), {"model", "dataloader", "output"})
        self.assertEqual((base["model"]["activations"]["backend"], base["model"]["activations"]["folder"]),
                         ("hf", "nemotron-nano-9b-v2_shared"))
        local = load_config(str(REPO_ROOT / "configs" / "probes" / "nemotron_probe_local.yaml"))
        self.assertEqual(set(local), {"model", "dataloader", "output"})
        self.assertEqual(local["model"]["activations"]["folder"], base["model"]["activations"]["folder"])
        self.assertEqual(local["model"]["subject_model"], base["model"]["subject_model"])
        self.assertEqual(local["dataloader"], base["dataloader"])
        self.assertEqual(local["output"], base["output"])


if __name__ == "__main__":
    unittest.main()
