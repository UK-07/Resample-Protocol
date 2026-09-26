"""Tests for src/scripts/unfaithfulness_metrics_summary.py."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.scripts.unfaithfulness_metrics_summary import (
    COUNT_COLUMNS,
    EXCLUDED_FLAG,
    EXCLUDED_FOOTNOTE,
    GROUP_KEYS,
    READ_COLUMNS,
    excluded_cell,
    main,
    n_excluded,
    pair_identity,
    render_report,
    summarize_csv,
    token_budget,
)
from tests.lib.rollout_manifest_test import make_chunk


class TestSummarizeCsv(unittest.TestCase):
    def test_counts_and_judges(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "m_d_hinted_rollouts_judged.csv"
            make_chunk().to_csv(csv, index=False)
            counts, judges = summarize_csv(csv, chunk_rows=3)
        self.assertEqual(list(counts.columns), COUNT_COLUMNS)
        self.assertEqual(counts.index.names, GROUP_KEYS)
        self.assertEqual(int(counts["n_rollouts"].sum()), 8)
        self.assertTrue(judges)
        pos = counts.loc[("positive", "expert_opinion")]
        # idx 5 never closed its reasoning; idx 5 and 6 have no answer.
        self.assertEqual((int(pos["n_truncated"]), int(pos["n_unanswered"])), (1, 2))
        # unanswered 2 + incoherent 1 + unjudged 1 = 4 of 7 rollouts.
        self.assertEqual(n_excluded(pos), 4)
        self.assertEqual(excluded_cell(pos), f"57.1% {EXCLUDED_FLAG}")

    def test_header_only_csv_yields_empty_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "m_d_hinted_rollouts_judged.csv"
            csv.write_text(",".join(READ_COLUMNS) + "\n")
            counts, judges = summarize_csv(csv, chunk_rows=3)
        self.assertEqual(len(counts), 0)
        self.assertEqual(list(counts.columns), COUNT_COLUMNS)
        self.assertEqual(counts.index.names, GROUP_KEYS)
        self.assertEqual(judges, set())
        # The report's totals row sums to zero rather than failing on it.
        self.assertEqual(int(counts.sum()["n_rollouts"]), 0)


class TestExcludedFlag(unittest.TestCase):
    def _counts(self, n_unanswered: int) -> pd.DataFrame:
        index = pd.MultiIndex.from_tuples([("positive", "tool_output")], names=GROUP_KEYS)
        return pd.DataFrame({
            "n_rollouts": [100], "n_changed": [40], "n_switched": [30],
            "n_unfaithful": [5], "n_faithful": [20], "n_incoherent": [3],
            "n_truncated": [n_unanswered], "n_unanswered": [n_unanswered],
        }, index=index)

    def _report(self, n_unanswered: int) -> str:
        return render_report([{
            "path": Path("/x/m_d_hinted_rollouts_judged.csv"), "model": "m", "dataset": "d",
            "sidecar": {}, "counts": self._counts(n_unanswered),
            "judge_model": "j", "judge_prompt": "p",
        }])

    def test_cell_above_threshold_is_flagged_and_footnoted(self):
        # unanswered 9 + incoherent 3 + unjudged 2 = 14 %: flagged.
        self.assertEqual(excluded_cell(self._counts(9).iloc[0]), f"14.0% {EXCLUDED_FLAG}")
        report = self._report(9)
        self.assertIn(f"| 9 | 14.0% {EXCLUDED_FLAG} |", report)
        # Once under the overview, once under the pair table.
        self.assertEqual(report.count(EXCLUDED_FOOTNOTE), 2)

    def test_cell_at_or_below_threshold_is_not_flagged(self):
        # unanswered 5 + incoherent 3 + unjudged 2 = 10 %: not above the threshold.
        self.assertEqual(excluded_cell(self._counts(5).iloc[0]), "10.0%")
        report = self._report(5)
        self.assertIn("| 5 | 10.0% |", report)
        self.assertNotIn(EXCLUDED_FOOTNOTE, report)


class TestSmokeSkipping(unittest.TestCase):
    def _write(self, tmp: Path, stem: str) -> None:
        make_chunk().to_csv(tmp / f"{stem}_judged.csv", index=False)

    def test_smoke_runs_skipped_unless_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "qwen3-8b_medqa-test_0911_baseline_hinted_rollouts")
            self._write(root, "qwen3-8b_medqa-test_smoke_baseline_hinted_rollouts")
            out = root / "summary.md"
            buf = io.StringIO()
            with patch.object(sys, "argv", ["x", "--dir", tmp, "--output", str(out)]), \
                    contextlib.redirect_stdout(buf):
                main()
            report = out.read_text()
            self.assertIn("## qwen3-8b — medqa-test_0911", report)
            self.assertNotIn("medqa-test_smoke", report)
            self.assertIn("skipping qwen3-8b_medqa-test_smoke", buf.getvalue())
            with patch.object(sys, "argv", ["x", "--dir", tmp, "--output", str(out), "--include-smoke"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertIn("## qwen3-8b — medqa-test_smoke", out.read_text())


class TestPairIdentity(unittest.TestCase):
    def test_from_sidecar_then_file_name(self):
        judged = Path("/x/qwen3.5-9b_medqa-test_0911_baseline_hinted_rollouts_judged.csv")
        self.assertEqual(pair_identity(judged, {}), ("qwen3.5-9b", "medqa-test_0911"))
        self.assertEqual(
            pair_identity(judged, {"baseline_csv": "/b/gemma3-4b-it_mmlu-pro_baseline.csv"}),
            ("gemma3-4b-it", "mmlu-pro"),
        )
        merged = Path("/x/qwen3.5-9b_mmlu-pro_hinted-rollouts-0828_merged_judged.csv")
        self.assertEqual(pair_identity(merged, {}), ("qwen3.5-9b", "mmlu-pro"))


class TestTokenBudget(unittest.TestCase):
    def test_recorded_and_missing(self):
        self.assertEqual(
            token_budget({"max_tokens": 16384, "max_model_len": 24576}), ("16384", "24576")
        )
        # A missing key and a null value read the same way.
        self.assertEqual(token_budget({}), ("unknown", "unknown"))
        self.assertEqual(token_budget({"max_tokens": None}), ("unknown", "unknown"))


if __name__ == "__main__":
    unittest.main()
