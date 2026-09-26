"""Tests for src/scripts/select_resample_set.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.lib.resample import ITEM_CHOICES_COLUMN, ITEM_COLUMNS, SELECTION_COLUMNS
from src.scripts.select_resample_set import main
from src.lib.rollout_manifest import read_manifest, read_manifest_meta, write_manifest
from tests.scripts.resample_fixtures import MODEL, MODEL_ID, RUN, STEM, make_data_root


class SelectResampleSetScriptTest(unittest.TestCase):
    def test_writes_selection_report_meta_and_items(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_data_root(Path(td))
            with patch.dict(os.environ, {"DATA_ROOT": str(root)}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--seed", "3"]), 0)
            out = root / "resample"
            selection = pd.read_csv(out / "resample_selection.csv")
            self.assertEqual(list(selection.columns), SELECTION_COLUMNS)
            self.assertEqual(set(selection["run"]), {RUN})  # the smoke run is out
            self.assertEqual((selection["role"] == "used_candidate").sum(), 8)
            # 8/8 metadata cell: 4 used, 6 kept -> 4 controls; 6/8: 2 used, 2 kept; consensus: 2 used, 2 kept.
            self.assertEqual((selection["role"] == "control").sum(), 8)
            report = pd.read_csv(out / "resample_selection_report.csv")
            self.assertEqual(int(report["control_shortfall"].sum()), 0)
            meta = json.loads((out / "resample_selection.meta.json").read_text())
            self.assertEqual(meta["k"], 4)
            self.assertEqual(meta["sample_seeds"], [43, 44, 45, 46])
            self.assertEqual(meta["plan"][0]["n_rollouts"], 16 * 4)
            self.assertEqual(meta["plan"][0]["model_name"], MODEL_ID)
            items = pd.read_csv(out / "items" / f"{STEM}_resample_items.csv")
            # The fixture's manifest meta names a baseline, so the items carry the option texts.
            self.assertEqual(list(items.columns), ITEM_COLUMNS + [ITEM_CHOICES_COLUMN])
            self.assertEqual(len(items), 16)
            self.assertTrue(items[ITEM_CHOICES_COLUMN].map(lambda c: len(json.loads(c)) == 4).all())
            self.assertTrue(items["hinted_prompt"].str.startswith("hinted Q").all())
            items_meta = json.loads((out / "items" / f"{STEM}_resample_items.meta.json").read_text())
            self.assertEqual(items_meta["model_name"], MODEL_ID)
            self.assertEqual(items_meta["source_recipe"]["seed"], 42)
            self.assertEqual(items_meta["n_options"], 4)

    def test_seed_count_must_match_k(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_data_root(Path(td))
            with patch.dict(os.environ, {"DATA_ROOT": str(root)}), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--k", "3"])

    def test_no_items_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_data_root(Path(td))
            with patch.dict(os.environ, {"DATA_ROOT": str(root)}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--no-items"]), 0)
            self.assertFalse((root / "resample" / "items").exists())

    def test_extend_keeps_stored_runs_and_draws_only_new_ones(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_data_root(Path(td))
            out = root / "resample"
            with patch.dict(os.environ, {"DATA_ROOT": str(root)}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--seed", "3"]), 0)
                first = (out / "resample_selection.csv").read_text()
                first_items = (out / "items" / f"{STEM}_resample_items.csv").read_text()

                # Nothing new in the manifest: the stored selection is left exactly as it is.
                self.assertEqual(main(["--seed", "99", "--extend"]), 0)
                self.assertEqual((out / "resample_selection.csv").read_text(), first)

                # A second run lands in the manifest (same rows under a new run tag and source).
                path = root / "hinted_rollouts" / "rollout_manifest.parquet"
                manifest, meta = read_manifest(path), read_manifest_meta(path)
                late = manifest[manifest["run"] == RUN].copy()
                late["run"] = "medqa-test_late"
                late["rollout_id"] = late["rollout_id"].str.replace(RUN, "medqa-test_late")
                late_stem = f"{MODEL}_medqa-test_late_baseline_hinted_rollouts"
                late["source_csv"] = f"{late_stem}_judged.csv"
                source = next(s for s in meta["sources"] if s["run"] == RUN)
                write_manifest(pd.concat([manifest, late], ignore_index=True), path,
                               {"sources": meta["sources"] + [{**source, "run": "medqa-test_late",
                                                               "source_csv": f"{late_stem}_judged.csv"}]})
                rollouts = root / "hinted_rollouts"
                (rollouts / f"{late_stem}_judged.csv").write_text((rollouts / f"{STEM}_judged.csv").read_text())

                self.assertEqual(main(["--seed", "99", "--extend"]), 0)
            merged = pd.read_csv(out / "resample_selection.csv")
            self.assertEqual(list(merged.columns), SELECTION_COLUMNS)
            self.assertEqual(set(merged["run"]), {RUN, "medqa-test_late"})
            # The first run's rows (drawn with seed 3) are byte-for-byte the stored ones.
            self.assertTrue((out / "resample_selection.csv").read_text().startswith(first))
            self.assertEqual((merged["run"] == "medqa-test_late").sum(), 16)
            self.assertEqual((out / "items" / f"{STEM}_resample_items.csv").read_text(), first_items)
            self.assertTrue((out / "items" / f"{late_stem}_resample_items.csv").exists())
            meta = json.loads((out / "resample_selection.meta.json").read_text())
            self.assertEqual(meta["n_selected"], 32)
            self.assertEqual([p["run"] for p in meta["plan"]], [RUN, "medqa-test_late"])
            self.assertEqual(meta["extends"]["kept_runs"], [f"{MODEL}/{RUN}"])
            report = pd.read_csv(out / "resample_selection_report.csv")
            self.assertEqual(set(report["run"]), {RUN, "medqa-test_late"})

    def test_extend_refuses_a_different_k(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_data_root(Path(td))
            with patch.dict(os.environ, {"DATA_ROOT": str(root)}), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([]), 0)
                with self.assertRaises(SystemExit):
                    main(["--extend", "--k", "2", "--sample-seeds", "43,44"])


if __name__ == "__main__":
    unittest.main()
