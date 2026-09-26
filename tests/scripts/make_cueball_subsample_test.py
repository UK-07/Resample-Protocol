"""Tests for src/scripts/make_cueball_subsample.py (the row filter and sidecar block)."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.scripts.make_cueball_subsample import N, SEED, filter_csv, subsample_block


class TestHelpers(unittest.TestCase):
    def test_filter_keeps_rows_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.csv"
            pd.DataFrame({"original_index": ["3", "1", "2"], "v": ["a,b", "007", ""]}).to_csv(src, index=False)
            out, n = filter_csv(src, Path(tmp) / "sub" / "out.csv", {"1", "3"})
            self.assertEqual(n, 3)
            self.assertEqual(out["original_index"].tolist(), ["3", "1"])
            self.assertEqual(out["v"].tolist(), ["a,b", "007"])  # strings, no re-formatting
            back = pd.read_csv(Path(tmp) / "sub" / "out.csv", dtype=str, keep_default_na=False)
            self.assertEqual(back["v"].tolist(), ["a,b", "007"])

    def test_subsample_block_names_the_rule_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.csv"
            src.write_text("a\n1\n")
            block = subsample_block(Path(tmp) / "tree", src, 10, 4, "now")
            self.assertEqual(block["question_list"], str(Path(tmp) / "tree" / "mmlu_pro_1000_questions.csv"))
            self.assertIn(f"n={N}, random_state={SEED}", block["rule"])
            self.assertEqual((block["n_source_rows"], block["n_rows"], block["created_utc"]), (10, 4, "now"))
            self.assertEqual(block["source"]["path"], str(src))


if __name__ == "__main__":
    unittest.main()
