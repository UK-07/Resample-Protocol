"""Tests for src/scripts/resample_noise_model.py."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.scripts.resample_noise_model import check_option_widths, main
from tests.scripts.resample_fixtures import MODEL, RUN, make_data_root


class NoiseModelScriptTest(unittest.TestCase):
    def test_writes_tables_without_smoke_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_data_root(Path(td))
            with patch.dict(os.environ, {"DATA_ROOT": str(root)}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([]), 0)
            out = root / "resample"
            cells = pd.read_csv(out / "noise_model_cells.csv")
            self.assertEqual(set(cells["run"]), {RUN})
            self.assertEqual(set(cells["hint_style"]), {"metadata", "consensus"})
            meta_row = cells[cells["hint_style"] == "metadata"].iloc[0]
            # 6 to target, 2 off target over 4 options -> 1 noise-shaped switch.
            self.assertEqual(meta_row["n_to_hint"], 6)
            self.assertEqual(meta_row["n_off_target"], 2)
            self.assertAlmostEqual(meta_row["noise_flip_estimate"], 1.0)
            strat = pd.read_csv(out / "noise_model_by_stability.csv")
            self.assertEqual(set(strat["stability_bin"].astype(str)), {"8", "6"})
            self.assertTrue((out / "noise_model_stability_summary.csv").exists())
            text = (out / "noise_model.md").read_text()
            self.assertIn("## Per cell", text)
            self.assertIn("answer", text.lower())

    def test_width_check_catches_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_data_root(Path(td))
            baseline = root / "baselines" / f"{MODEL}_{RUN}_baseline.csv"
            base = pd.read_csv(baseline)
            base["choices"] = '["a","b","c","d","e"]'
            base.to_csv(baseline, index=False)
            meta = {"sources": [{"subject_model": MODEL, "run": RUN, "dataset": "medqa", "baseline_csv": str(baseline)}]}
            problems = check_option_widths(meta, {(MODEL, RUN)})
            self.assertEqual(len(problems), 1)
            self.assertIn("!= registered 4", problems[0])
            with patch.dict(os.environ, {"DATA_ROOT": str(root)}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([]), 1)
                self.assertEqual(main(["--skip-width-check"]), 0)

    def test_only_filter_with_no_match_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_data_root(Path(td))
            with patch.dict(os.environ, {"DATA_ROOT": str(root)}), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--only", "no-such-run"]), 1)


if __name__ == "__main__":
    unittest.main()
