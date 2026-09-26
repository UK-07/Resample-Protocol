"""Regression gates for all revised-paper tables and the exact-sign correction."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.lib.paper_revision import workflow


class PaperRevisionVerificationTest(unittest.TestCase):
    ALPHA = "s1_persistence/alpha_vs_measured_noise.csv"
    COUNTS = "plot_tables/fig_reliance_composition_ssp_flips.csv"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference"
        self.generated = self.root / "generated"
        ordinary = pd.DataFrame({
            "cell": ["a", "b", "c"],
            "count": [65772, 53281, 20000],
            "nullable_count": [65772.0, np.nan, 20000.0],
            "rate": [.859, .135, .752],
            "flag": [True, False, True],
        })
        alpha = pd.DataFrame({
            "grouping": "subject_model+dataset+hint_style+case",
            "subject_model": [f"fixture-{i}" for i in range(8)],
            "rate": np.arange(8) / 13,
            "p_strict": .5,
            "q_bh_strict": [.051] + [.6] * 7,
        })
        for name in workflow.EXPECTED_RECOMPUTED_TABLES:
            table = alpha.copy() if name == self.ALPHA else ordinary.copy()
            self.write(self.reference, name, table)
            if name == self.ALPHA:
                table.loc[:5, "p_strict"] += .001
                table.loc[:3, "q_bh_strict"] += .001
            self.write(self.generated, name, table)

    @staticmethod
    def write(root, name, table):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    def change(self, name, column, row, value):
        table = pd.read_csv(self.generated / name)
        table.loc[row, column] = value
        self.write(self.generated, name, table)

    def verify(self):
        return workflow.verify_recomputed(self.reference, self.generated)

    def test_complete_fixture_passes_and_records_only_known_corrections(self):
        report = self.verify()
        self.assertEqual(len(workflow.EXPECTED_RECOMPUTED_TABLES), 24)
        self.assertEqual(set(report), workflow.EXPECTED_RECOMPUTED_TABLES)
        corrections = report[self.ALPHA]["alpha_sign_corrections"]
        self.assertEqual(len(corrections["p_strict"]), 6)
        self.assertEqual(len(corrections["q_bh_strict"]), 4)
        self.assertEqual(corrections["p_strict"][0]["keys"]["subject_model"], "fixture-0")
        self.assertIn("nullable_count", report[self.COUNTS]["integer_columns_compared_exactly"])

    def test_missing_table_is_rejected(self):
        (self.generated / self.COUNTS).unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete recomputed table set"):
            self.verify()

    def test_unexpected_table_is_rejected(self):
        self.write(self.generated, "extra.csv", pd.DataFrame({"x": [1]}))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.verify()

    def test_column_reordering_is_rejected(self):
        frame = pd.read_csv(self.generated / self.COUNTS)
        self.write(self.generated, self.COUNTS, frame.iloc[:, ::-1])
        with self.assertRaisesRegex(ValueError, "rows or columns"):
            self.verify()

    def test_large_nullable_count_cannot_use_relative_tolerance(self):
        # A relative tolerance of 2e-6 would incorrectly accept this error.
        self.change(self.COUNTS, "nullable_count", 0, 65772.1)
        with self.assertRaisesRegex(AssertionError, "Integer-valued column differs"):
            self.verify()

    def test_float_absolute_tolerance_has_no_relative_allowance(self):
        self.change(self.COUNTS, "rate", 0, .859 + 3e-7)
        with self.assertRaisesRegex(AssertionError, "Floating column differs"):
            self.verify()

    def test_float_rounding_within_absolute_tolerance_is_accepted(self):
        self.change(self.COUNTS, "rate", 0, .859 + 1e-7)
        report = self.verify()
        self.assertAlmostEqual(report[self.COUNTS]["max_absolute_numeric_difference"], 1e-7)

    def test_changed_missing_mask_is_rejected(self):
        self.change(self.COUNTS, "nullable_count", 1, 0)
        with self.assertRaisesRegex(ValueError, "missing-value mask"):
            self.verify()

    def test_alpha_infinity_is_rejected(self):
        self.change(self.ALPHA, "p_strict", 0, np.inf)
        with self.assertRaisesRegex(ValueError, "Non-finite alpha"):
            self.verify()

    def test_alpha_out_of_range_is_rejected(self):
        self.change(self.ALPHA, "q_bh_strict", 0, 1.001)
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            self.verify()

    def test_changed_alpha_missing_mask_is_rejected(self):
        self.change(self.ALPHA, "p_strict", 0, np.nan)
        with self.assertRaisesRegex(ValueError, "undefined p/q"):
            self.verify()

    def test_alpha_correction_cannot_exceed_documented_bounds(self):
        for column, value in (("p_strict", .507), ("q_bh_strict", .058)):
            with self.subTest(column=column):
                path = self.generated / self.ALPHA
                original = path.read_bytes()
                self.change(self.ALPHA, column, 0, value)
                with self.assertRaisesRegex(ValueError, "exceeds its documented bound"):
                    self.verify()
                path.write_bytes(original)

    def test_six_p_and_four_q_corrections_are_required(self):
        for column, value, expected in (("p_strict", .5, 6), ("q_bh_strict", .051, 4)):
            with self.subTest(column=column):
                path = self.generated / self.ALPHA
                original = path.read_bytes()
                self.change(self.ALPHA, column, 0, value)
                with self.assertRaisesRegex(ValueError, f"Expected {expected} corrected"):
                    self.verify()
                path.write_bytes(original)

    def test_even_small_correction_cannot_change_bh_decision(self):
        self.change(self.ALPHA, "q_bh_strict", 0, .049)
        with self.assertRaisesRegex(ValueError, "changed a reported BH decision"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
