"""Tests for src/scripts/probe_eval_report.py — the three-way report CLI on a toy predictions file."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.scripts.probe_eval_report import main
from tests.lib.report_test import toy


class ProbeEvalReportScriptTest(unittest.TestCase):
    def test_renders_three_way_tables_from_a_toy_predictions_file(self):
        pred, manifest = toy()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest.to_parquet(tmp / "manifest.parquet", index=False)
            pred.to_csv(tmp / "toy_predictions.csv", index=False)
            other = pred.assign(p_unfaithful=1 - pred["p_unfaithful"])
            other.to_csv(tmp / "inverted_predictions.csv", index=False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--predictions", str(tmp / "toy_predictions.csv"), "--manifest", str(tmp / "manifest.parquet"),
                             "--compare", str(tmp / "inverted_predictions.csv")])
            self.assertEqual(code, 0)
            text = out.getvalue()
            for view in ("pooled", "positive", "negative"):
                self.assertIn(view, text)
            report = json.loads((tmp / "toy_predictions_report.json").read_text())
            self.assertEqual(report["name"], "toy_predictions")
            self.assertEqual(set(report["overall"]), {"pooled", "positive", "negative"})
            self.assertGreater(report["ablation"]["delta_auroc"], 0.5)
            md = (tmp / "toy_predictions_report.md").read_text()
            self.assertIn("## Overall", md)
            self.assertIn("## Ablation", md)
            # Plain relative paths work (only ${DATA_ROOT}-style ones go through the resolver).
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--predictions", "toy_predictions.csv", "--manifest", "manifest.parquet",
                                 "--output-dir", "out"])
            finally:
                os.chdir(cwd)
            self.assertEqual(code, 0)
            self.assertTrue((tmp / "out" / "toy_predictions_report.md").exists())


if __name__ == "__main__":
    unittest.main()
