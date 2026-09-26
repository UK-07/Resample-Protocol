"""Tests for src/scripts/assign_splits.py — the split stage CLI."""

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

from src.lib.rollout_manifest import read_manifest, read_manifest_meta
from src.lib.selection import HINTED_ONCE, RESAMPLE_K4
from src.lib.splits import assert_assigned
from src.scripts.assign_splits import main
from tests.scripts.resample_fixtures import make_data_root


class AssignSplitsScriptTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = make_data_root(Path(self.td.name))
        self.manifest = self.root / "hinted_rollouts" / "rollout_manifest.parquet"
        self.resample = self.root / "resample" / "resample_manifest.parquet"

    def run_main(self, argv=()):
        out = io.StringIO()
        with patch.dict(os.environ, {"DATA_ROOT": str(self.root)}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(list(argv))
        return code, out.getvalue()

    def write_resample_manifest(self) -> pd.DataFrame:
        base = read_manifest(self.manifest)
        orig = base.iloc[:4].copy()
        orig["provenance"] = HINTED_ONCE
        rerolls = orig.copy()
        rerolls["rollout_id"] = rerolls["rollout_id"] + "#rs43"
        rerolls["provenance"] = RESAMPLE_K4
        df = pd.concat([orig, rerolls], ignore_index=True)
        self.resample.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.resample, index=False)
        self.resample.with_suffix(".meta.json").write_text(json.dumps({"n_rows": len(df)}))
        return df

    def test_dry_run_writes_nothing(self):
        code, out = self.run_main(["--dry-run", "--seed", "3"])
        self.assertEqual(code, 0, out)
        self.assertIn("dry run", out)
        self.assertTrue(read_manifest(self.manifest)["split"].isna().all())

    def test_assigns_writes_meta_and_propagates(self):
        self.write_resample_manifest()
        code, out = self.run_main(["--seed", "3"])
        self.assertEqual(code, 0, out)
        manifest = read_manifest(self.manifest)
        assert_assigned(manifest)
        meta = read_manifest_meta(self.manifest)
        self.assertIn("sources", meta)  # the build's provenance survives the rewrite
        self.assertEqual(meta["split"]["seed"], 3)
        self.assertEqual(meta["split"]["fractions"], {"train": 0.70, "val": 0.15, "test": 0.15})
        self.assertTrue(meta["split"]["report"])
        resample = pd.read_parquet(self.resample)
        assert_assigned(resample)
        by_q = manifest.drop_duplicates("question_id").set_index("question_id")["split"]
        self.assertEqual(list(resample["split"]), list(by_q.loc[resample["question_id"]]))
        sidecar = json.loads(self.resample.with_suffix(".meta.json").read_text())
        self.assertEqual(sidecar["split"]["propagated_from"], str(self.manifest))

    def test_refuses_to_reassign_without_force(self):
        self.assertEqual(self.run_main(["--seed", "3"])[0], 0)
        code, out = self.run_main(["--seed", "4"])
        self.assertEqual(code, 1)
        self.assertIn("already assigned", out)
        self.assertEqual(self.run_main(["--seed", "4", "--force"])[0], 0)

    def test_extend_assigns_only_new_questions_and_a_wiped_split_is_refused(self):
        self.assertEqual(self.run_main(["--seed", "3"])[0], 0)
        before = read_manifest(self.manifest)
        # A rebuild that added a question (and carried the rest over).
        extra = before.iloc[:1].copy()
        extra["rollout_id"] = "x:medqa-test_0911:metadata:999"
        extra["question_id"] = "medqa:test:999"
        extra["original_index"] = 999
        extra["split"] = None
        grown = pd.concat([before, extra], ignore_index=True)
        grown.to_parquet(self.manifest, index=False)
        code, out = self.run_main(["--seed", "3"])
        self.assertEqual(code, 1)
        self.assertIn("--extend", out)
        code, out = self.run_main(["--seed", "4", "--extend"])
        self.assertEqual(code, 1)
        self.assertIn("seed 3", out)
        code, out = self.run_main(["--seed", "3", "--extend"])
        self.assertEqual(code, 0, out)
        after = read_manifest(self.manifest)
        assert_assigned(after)
        self.assertEqual(list(after["split"].iloc[:len(before)]), list(before["split"]))
        self.assertEqual(read_manifest_meta(self.manifest)["split"]["extended"]["n_rows_added"], 1)
        # The column wiped but the sidecar still recording an assignment: refuse without --force.
        wiped = after.copy()
        wiped["split"] = None
        wiped.to_parquet(self.manifest, index=False)
        code, out = self.run_main(["--seed", "3"])
        self.assertEqual(code, 1)
        self.assertIn("rebuilt without carrying it over", out)

    def test_custom_fractions(self):
        code, out = self.run_main(["--fractions", "0.5,0.25,0.25"])
        self.assertEqual(code, 0, out)
        self.assertEqual(read_manifest_meta(self.manifest)["split"]["fractions"], {"train": 0.5, "val": 0.25, "test": 0.25})


if __name__ == "__main__":
    unittest.main()
