"""Tests for src/scripts/resample_relabel.py."""

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

from src.lib.rollout_manifest import MANIFEST_COLUMNS
from src.scripts.resample_relabel import EXTRA_COLUMNS, main, resampled_csv
from src.scripts.select_resample_set import main as select_main
from tests.scripts.resample_fixtures import STEM, make_data_root, write_rerolls


class RelabelScriptTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = make_data_root(Path(self.td.name))
        with patch.dict(os.environ, {"DATA_ROOT": str(self.root)}), contextlib.redirect_stdout(io.StringIO()):
            select_main([])

    def run_main(self, argv=()):
        out = io.StringIO()
        with patch.dict(os.environ, {"DATA_ROOT": str(self.root)}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(["--no-token-len", *argv])
        return code, out.getvalue()

    def test_resampled_csv_prefers_judged(self):
        d = self.root / "resample" / "rollouts"
        d.mkdir(parents=True)
        self.assertIsNone(resampled_csv(d, f"{STEM}_judged.csv", 43))
        (d / f"{STEM}_rs43.csv").write_text("x")
        self.assertEqual(resampled_csv(d, f"{STEM}_judged.csv", 43).name, f"{STEM}_rs43.csv")
        (d / f"{STEM}_rs43_judged.csv").write_text("x")
        self.assertEqual(resampled_csv(d, f"{STEM}_judged.csv", 43).name, f"{STEM}_rs43_judged.csv")

    def test_relabels_and_reports(self):
        # A no-hint vote for the target on every question: originals become noise_flip, re-rolls never.
        baseline_csv = next((self.root / "baselines").glob("*_baseline.csv"))
        base = pd.read_csv(baseline_csv)
        base["sample_answers"] = '["A","A","A","A","A","A","A","B"]'
        base.to_csv(baseline_csv, index=False)
        write_rerolls(self.root)
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        res = self.root / "resample"
        rollouts = pd.read_parquet(res / "resample_manifest.parquet")
        rerolls = rollouts[rollouts["is_resample"]]
        self.assertTrue((rerolls["baseline_hint_votes"] >= 1).all())
        self.assertEqual(int((rerolls["exclude_reason"] == "noise_flip").sum()), 0)
        self.assertEqual(set(rollouts.loc[~rollouts["is_resample"], "provenance"]), {"hinted_once"})
        self.assertEqual(set(rerolls["provenance"]), {"resample_k4"})
        self.assertEqual(list(rollouts.columns), list(MANIFEST_COLUMNS) + EXTRA_COLUMNS)
        # Label-configuration columns: re-rolls only, null on every original row.
        originals = rollouts[~rollouts["is_resample"]]
        self.assertTrue(originals["label_verbalised_rest"].isna().all())
        self.assertTrue(originals["label_used_ignored"].isna().all())
        self.assertEqual(rerolls["label_verbalised_rest"].value_counts().to_dict(), {"verbalised": 16, "rest": 40})
        self.assertEqual(rerolls["label_used_ignored"].value_counts().to_dict(), {"used": 24, "ignored": 32})
        self.assertEqual(len(rollouts), 16 * 5)
        self.assertEqual(int(rollouts["is_resample"].sum()), 64)
        self.assertEqual(set(rollouts.loc[~rollouts["is_resample"], "sample_seed"].astype(int)), {42})
        self.assertTrue(rollouts.loc[rollouts["is_resample"], "rollout_id"].str.contains("#rs4").all())
        questions = pd.read_csv(res / "question_reliance.csv")
        self.assertEqual(len(questions), 16)
        counts = questions["reliance_label"].value_counts().to_dict()
        self.assertEqual(counts, {"robust_used": 8, "robust_ignored": 8})
        self.assertTrue((questions.loc[questions["role"] == "used_candidate", "k_to_hint_count"] == 3).all())
        report = json.loads((res / "resample_relabel_report.json").read_text())
        self.assertEqual(report["n_rerolls"], 64)
        self.assertEqual(report["rerolls_noise_flip"], "disabled")
        self.assertEqual(report["reliance_labels"]["robust_used"], 8)
        self.assertEqual(report["contrast_a"]["train/used"], 32)
        self.assertEqual(report["contrast_a"]["train/ignored"], 40)
        self.assertEqual(report["contrast_a"]["challenge/ignored"], 8)
        self.assertEqual(report["paired_a_questions"], 8)
        self.assertEqual(report["contrast_b_rollouts"], 32)
        self.assertEqual(report["paired_b_questions"], 8)
        survival = pd.read_csv(res / "survival_by_hint.csv").set_index("hint_style")
        self.assertAlmostEqual(survival.loc["metadata", "survival_rate"], 1.0)
        self.assertEqual(survival.loc["metadata", "n_single_used"], 6)
        text = (res / "survival_table.md").read_text()
        self.assertIn("## Per hint style (pooled", text)
        self.assertIn("k=4 times with its stored hinted prompt (seeds 43–46, temperature", text)
        self.assertIn("## Noise model vs. observed", text)
        self.assertIn("robust_used", text)
        self.assertEqual(report["missing"], [])
        meta = json.loads((res / "resample_manifest.meta.json").read_text())
        self.assertEqual(meta["n_rows"], 80)
        self.assertEqual(len(meta["sources"]), 4)

    def test_partial_rerolls_leave_labels_null_and_report_missing(self):
        write_rerolls(self.root, seeds=(43, 44), judged=False)
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        res = self.root / "resample"
        questions = pd.read_csv(res / "question_reliance.csv")
        self.assertTrue(questions["reliance_label"].isna().all())
        self.assertTrue((questions["k_n"] == 2).all())
        report = json.loads((res / "resample_relabel_report.json").read_text())
        self.assertEqual(len(report["missing"]), 2)
        self.assertFalse(report["sources"][0]["judged"])
        self.assertGreater(report["sources"][0]["n_unjudged_switched"], 0)
        self.assertIn("missing:", out)

    def test_no_rerolls_fails(self):
        code, out = self.run_main()
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()


class NoiseVsObservedTest(unittest.TestCase):
    def test_noise_model_is_scoped_to_the_selected_pairs(self):
        from src.scripts.resample_relabel import noise_vs_observed
        from tests.lib.resample_test import manifest_rows
        # m1 has no off-target flips, m2 does; a selection covering m1 only must not see m2's noise.
        manifest = pd.concat([
            manifest_rows(4, 0, 4, model="m1"),
            manifest_rows(4, 4, 4, model="m2", start=100),
        ], ignore_index=True)
        questions = pd.DataFrame({
            "rollout_id": [f"m1:medqa-test_0911:metadata:{i}" for i in range(4)],
            "subject_model": "m1", "run": "medqa-test_0911", "hint_style": "metadata", "case": "positive",
            "role": "used_candidate", "reliance_label": ["robust_used"] * 3 + ["weak_used"],
            "k_to_hint_count": [3, 3, 4, 1],
        })
        out = noise_vs_observed(questions, manifest).set_index(["hint_style", "case"])
        self.assertEqual(out.loc[("metadata", "positive"), "noise_share_of_to_hint"], 0.0)
        pooled = noise_vs_observed(questions.assign(subject_model=["m1", "m1", "m2", "m2"]), manifest)
        self.assertGreater(pooled["noise_share_of_to_hint"].iloc[0], 0.0)
