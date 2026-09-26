"""Tests for src/lib/report.py."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib.probe_metrics import auroc
from src.lib.report import (
    CASE_VIEWS,
    build_report,
    case_view,
    delta_auroc,
    group_table,
    join_predictions,
    ks_case_shift,
    ks_two_sample,
    load_predictions,
    render_markdown,
    three_way,
    three_way_table,
    validate_predictions,
)
from tests.lib.resample_test import manifest_rows


def toy(seed: int = 0, n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``n`` positive + ``n`` negative judged rows and predictions that separate classes on positive rows only."""
    pos = manifest_rows(n, 0, 0, case="positive", hint="metadata")
    neg = manifest_rows(n, 0, 0, case="negative", hint="consensus", start=1000)
    manifest = pd.concat([pos, neg], ignore_index=True)
    rng = np.random.default_rng(seed)
    label = manifest["judge_label_final"].astype(int).to_numpy()
    strength = np.where(manifest["case"] == "positive", 0.4, 0.05)
    p = np.clip(0.5 + np.where(label == 0, strength, -strength) + rng.normal(0, 0.1, len(manifest)), 0.001, 0.999)
    pred = pd.DataFrame({"rollout_id": manifest["rollout_id"], "p_unfaithful": p,
                         "predicted_label": (p <= 0.5).astype(int)})
    return pred, manifest


class ValidationTest(unittest.TestCase):
    def test_predictions_are_validated(self):
        with self.assertRaises(ValueError):
            validate_predictions(pd.DataFrame({"rollout_id": ["a"]}))
        with self.assertRaises(ValueError):
            validate_predictions(pd.DataFrame({"rollout_id": ["a", "a"], "p_unfaithful": [0.1, 0.2]}))
        with self.assertRaises(ValueError):
            validate_predictions(pd.DataFrame({"rollout_id": ["a"], "p_unfaithful": [1.5]}))
        with self.assertRaises(ValueError):
            validate_predictions(pd.DataFrame({"rollout_id": ["a"], "p_unfaithful": [0.5], "predicted_label": [2]}))

    def test_join_requires_known_ids_and_takes_labels_from_the_manifest(self):
        pred, manifest = toy()
        joined = join_predictions(pred, manifest)
        self.assertEqual(len(joined), len(pred))
        self.assertEqual(list(joined["label"]), list(manifest["judge_label_final"].astype(int)))
        self.assertIn("hint_style", joined.columns)
        with self.assertRaises(ValueError):
            join_predictions(pd.concat([pred, pd.DataFrame({"rollout_id": ["ghost"], "p_unfaithful": [0.5]})]), manifest)

    def test_join_with_contrast_a_labels(self):
        pred, manifest = toy(n=4)
        manifest["contrast_a"] = ["used", "ignored"] * 4
        joined = join_predictions(pred, manifest, label_col="contrast_a")
        self.assertEqual(list(joined["label"]), [0, 1] * 4)


class ViewsTest(unittest.TestCase):
    def test_three_views_filter_on_case_only_here(self):
        pred, manifest = toy()
        joined = join_predictions(pred, manifest)
        self.assertEqual(len(case_view(joined, "pooled")), 80)
        self.assertEqual(set(case_view(joined, "positive")["case"]), {"positive"})
        self.assertEqual(set(case_view(joined, "negative")["case"]), {"negative"})
        with self.assertRaises(ValueError):
            case_view(joined, "all")

    def test_three_way_metrics_match_direct_computation(self):
        pred, manifest = toy()
        joined = join_predictions(pred, manifest)
        blocks = three_way(joined)
        self.assertEqual(set(blocks), set(CASE_VIEWS))
        pos = joined[joined["case"] == "positive"]
        self.assertAlmostEqual(blocks["positive"]["auroc"], auroc(pos["label"], pos["p_unfaithful"]))
        self.assertAlmostEqual(blocks["pooled"]["auroc"], auroc(joined["label"], joined["p_unfaithful"]))
        self.assertGreater(blocks["positive"]["auroc"], blocks["negative"]["auroc"])
        self.assertEqual(blocks["pooled"]["n"], 80)
        self.assertIn("accuracy_predicted_label", blocks["pooled"])

    def test_tables(self):
        pred, manifest = toy()
        report = build_report(pred, manifest, name="toy")
        table = three_way_table(report["overall"])
        self.assertEqual(list(table.columns), list(CASE_VIEWS))
        self.assertIn("auroc", table.index)
        by_hint = group_table(report["by_group"]["hint_style"])
        self.assertEqual(set(by_hint["group"]), {"metadata", "consensus"})
        # metadata rows are all positive: its negative view is empty.
        row = by_hint.set_index("group").loc["metadata"]
        self.assertEqual(row["n_negative"], 0)
        self.assertTrue(math.isnan(row["auroc_negative"]))


class KSTest(unittest.TestCase):
    def test_ks_statistic_and_p_value(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
        d, p = ks_two_sample(a, b)
        self.assertLess(d, 0.1)
        self.assertGreater(p, 0.05)
        d2, p2 = ks_two_sample(a, rng.normal(1.5, 1, 500))
        self.assertGreater(d2, 0.5)
        self.assertLess(p2, 1e-6)
        self.assertEqual(ks_two_sample([1.0, 2.0], [1.0, 2.0]), (0.0, 1.0))
        self.assertTrue(all(math.isnan(v) for v in ks_two_sample([], [1.0])))

    def test_case_shift_diagnostic_flags_a_case_encoding_probe(self):
        pred, manifest = toy()
        shift = ks_case_shift(join_predictions(pred, manifest))
        self.assertEqual(set(shift), {"unfaithful", "faithful"})
        # The toy probe scores unfaithful rows ~0.9 on positive and ~0.55 on negative cases.
        self.assertGreater(shift["unfaithful"]["ks_statistic"], 0.5)
        self.assertLess(shift["unfaithful"]["p_value"], 0.01)
        self.assertEqual(shift["unfaithful"]["n_positive"], 20)
        # A probe blind to case: same score distribution on both.
        flat = pred.copy()
        flat["p_unfaithful"] = np.where(manifest["judge_label_final"].astype(int) == 0, 0.8, 0.2)
        shift = ks_case_shift(join_predictions(flat, manifest))
        self.assertEqual(shift["unfaithful"]["ks_statistic"], 0.0)


class AblationTest(unittest.TestCase):
    def test_delta_auroc_on_the_primary_view(self):
        pred, manifest = toy()
        other = pred.copy()
        other["p_unfaithful"] = 1 - other["p_unfaithful"]  # an inverted probe
        a = join_predictions(pred, manifest)
        b = join_predictions(other, manifest)
        out = delta_auroc(a, b)
        self.assertEqual(out["view"], "positive")
        self.assertEqual(out["n"], 40)
        self.assertAlmostEqual(out["delta_auroc"], out["auroc_a"] - out["auroc_b"])
        self.assertGreater(out["delta_auroc"], 0.5)
        with self.assertRaises(ValueError):
            delta_auroc(a, b.iloc[:10])


class RenderTest(unittest.TestCase):
    def test_toy_predictions_file_renders_three_way_tables(self):
        pred, manifest = toy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_predictions.csv"
            pred.to_csv(path, index=False)
            loaded = load_predictions(path)
        report = build_report(loaded, manifest, name="toy probe")
        self.assertEqual(report["primary_view"], "positive")
        self.assertEqual(report["n_by_case"], {"positive": 40, "negative": 40})
        text = render_markdown(report)
        for heading in ("## Overall", "## AUROC by hint_style", "## AUROC by subject_model",
                        "## Case-shift diagnostic"):
            self.assertIn(heading, text)
        for view in CASE_VIEWS:
            self.assertIn(view, text)
        self.assertIn("Primary (pre-registered): `positive`", text)


if __name__ == "__main__":
    unittest.main()
