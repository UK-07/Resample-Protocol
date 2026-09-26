"""Statistical regression checks for the repository-owned paper analysis."""
from pathlib import Path
import tempfile
import hashlib
import unittest

import numpy as np
import pandas as pd

from src.lib.paper_figures import common as rc
from src.lib.paper_figures.recompute import (
    alpha_contrast_pvalues, build, persistence_tables, role_tables,
)


class QuestionClusterTest(unittest.TestCase):
    def test_all_cues_and_rerolls_of_a_question_share_one_draw(self):
        rows = pd.DataFrame({
            "qkey": ["d:1", "d:1", "d:1", "d:2", "d:2", "d:3"],
            "value": [1, 0, 1, 0, 0, 1],
        })
        bootstrap = rc.ClusterBootstrap(rows.qkey, n_boot=200, seed=42)
        numerator = bootstrap.per_question(rows, rows.value)
        denominator = bootstrap.per_question(rows, np.ones(len(rows)))
        np.testing.assert_array_equal(numerator, [2, 0, 1])
        np.testing.assert_array_equal(denominator, [3, 2, 1])
        actual = bootstrap.reps(numerator, denominator)
        # These are question multiplicities, not six independent rollout draws.
        expected = (
            2 * bootstrap.W[:, 0] + bootstrap.W[:, 2]
        ) / (
            3 * bootstrap.W[:, 0] + 2 * bootstrap.W[:, 1] + bootstrap.W[:, 2]
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(bootstrap.Q, 3)

    def test_dataset_strata_and_question_order_are_stable(self):
        questions = pd.Series(["b:9", "a:2", "a:1", "b:8", "a:2"])
        a = rc.ClusterBootstrap(questions, n_boot=200, seed=42)
        b = rc.ClusterBootstrap(questions.iloc[::-1], n_boot=200, seed=42)
        np.testing.assert_array_equal(a.W, b.W)
        for dataset in ("a", "b"):
            np.testing.assert_array_equal(a.W[:, a.dataset == dataset].sum(1), 2)


class AlphaSignTest(unittest.TestCase):
    def test_exact_zero_is_included_in_both_sign_tails(self):
        rows = pd.DataFrame({
            "qkey": ["d:1", "d:2", "d:3"],
            "n_options": [5, 5, 5],
            "ssp_flip": [True, True, False],
        })
        q_counts = np.array([1, 4, 1])
        measured = np.array([1, 1, 0])
        # Six third-option switches / (5 - 2) equal two measured switches.
        # Direct float64 ratio subtraction rounds this exact tie below zero.
        naive = np.sum(q_counts / 3) / 2 - np.sum(measured) / 2
        self.assertLess(naive, 0)
        self.assertEqual(rc.ClusterBootstrap.pvalue(np.repeat(naive, 10)), 0)
        bootstrap = rc.ClusterBootstrap(rows.qkey, n_boot=10, seed=42)
        # One draw of each question is a valid cluster-bootstrap realization;
        # repeat it to isolate exact-zero tail handling from Monte Carlo noise.
        bootstrap.W[:] = 1
        exact = alpha_contrast_pvalues(bootstrap, rows, q_counts, measured, [])
        self.assertEqual(exact, 1)
        rows["ssp_flip"] = False
        self.assertTrue(np.isnan(alpha_contrast_pvalues(bootstrap, rows, q_counts, measured, [])))


class FiguresTableTest(unittest.TestCase):
    @staticmethod
    def pairs():
        # All six persistent-count outcomes are independently inspectable. The
        # incoherent row has an emitted rejected role, which must not place it
        # in the coherent-rejected numerator.
        k = np.array([0, 1, 2, 3, 4, 4])
        return pd.DataFrame({
            "rollout_id": [f"r{i}" for i in range(6)],
            "qkey": ["gpqa:0", "gpqa:0", "gpqa:1", "gpqa:1", "gpqa:2", "gpqa:2"],
            "subject_model": "fixture-model",
            "dataset": "gpqa",
            "hint_style": "expert_opinion",
            "case": "positive",
            "k_n": 4,
            "k_hit": k,
            "k_missing": [0, 1, 0, 0, 0, 0],
            "k_eff": [4, 3, 4, 4, 4, 4],
            "ssp_flip": True,
            "ssp_unfaithful": True,
            "persist1": k >= 1,
            "persist2": k >= 2,
            "persist3": k >= 3,
            "eligible_clean": True,
            "model_answer": "B",
            "target_option": "B",
            "b0": "A",
            "n_options": 4,
            "role_label": [-1, 0, 1, 1, 0, 1],
            "role_orig": ["rejected", "rejected", "verification_only", "credited", "neutral", "none"],
        })

    def test_persistence_counts_and_threshold_denominators(self):
        rows = self.pairs()
        bootstrap = rc.ClusterBootstrap(rows.qkey, n_boot=200, seed=42)
        with tempfile.TemporaryDirectory() as path:
            persistence_tables(rows, bootstrap, Path(path))
            result = pd.read_csv(Path(path) / "s1_persistence/persistence_counts_and_survival.csv")
            pooled = result[(result.population == "ssp_flips") & (result.grouping == "pooled")].iloc[0]
            self.assertEqual(pooled.n, 6)
            self.assertEqual(pooled.n_questions, 3)
            np.testing.assert_array_equal(pooled[["k0", "k1", "k2", "k3", "k4"]].to_numpy(), [1, 1, 1, 1, 2])
            self.assertAlmostEqual(pooled.surv_ge1, 5 / 6)
            self.assertAlmostEqual(pooled.surv_ge2, 4 / 6)
            self.assertAlmostEqual(pooled.surv_ge3, 3 / 6)
            self.assertEqual(pooled.n_missing_any, 1)
            # The CI is calculated from totals on the same three question
            # clusters, including both cue rows in each question draw.
            reps = bootstrap.reps(np.array([0, 1, 2]), np.array([2, 2, 2]))
            lo, hi = np.percentile(reps, [2.5, 97.5])
            self.assertAlmostEqual(pooled.surv_ge3_ci_lo, lo)
            self.assertAlmostEqual(pooled.surv_ge3_ci_hi, hi)

    def test_role_categories_are_exclusive_and_reference_all_flips(self):
        rows = self.pairs()
        bootstrap = rc.ClusterBootstrap(rows.qkey, n_boot=200, seed=42)
        with tempfile.TemporaryDirectory() as path:
            role_tables(rows, bootstrap, Path(path))
            result = pd.read_csv(Path(path) / "s2_roles/role_category_persistence.csv")
            pooled = result[result.grouping == "pooled"].set_index("category")
            self.assertEqual(pooled.loc["incoherent", "n"], 1)
            self.assertEqual(pooled.loc["coherent_rejected", "n"], 1)
            self.assertEqual(pooled.loc["covered_all", "n"], 6)
            self.assertEqual(pooled.loc["coherent_rejected", "ref_n"], 6)
            self.assertAlmostEqual(pooled.loc["coherent_rejected", "diff_vs_all_flips"], -0.5)

    def test_build_recomputes_complete_fixture_without_modifying_inputs(self):
        pairs = self.pairs()
        pairs["baseline_stability"] = 8
        pairs["binary_judge_label"] = [0, 0, 1, 0, 1, 1]
        pairs["truncated"] = False
        pairs["ssp_status"] = "eligible"
        pairs["robust_used"] = pairs.k_hit >= 3
        for column in ("ssp_faithful", "ssp_label_missing", "ssp_flip_truncated",
                       "ssp_eligible_truncated"):
            pairs[column] = False
        pairs["has_reliance"] = True
        pairs["orig_role_covered"] = True
        rerolls = []
        for pair in pairs.itertuples():
            for draw in range(4):
                rerolls.append({
                    "source_rollout_id": pair.rollout_id,
                    "dataset": pair.dataset,
                    "original_index": int(pair.qkey.split(":")[1]),
                    "subject_model": pair.subject_model,
                    "hint_style": pair.hint_style,
                    "case": pair.case,
                    "lab": [0, 1, -1, 1][draw],
                    "hit_rsp_pool": pair.k_hit >= 3 and draw < pair.k_hit,
                })
        with tempfile.TemporaryDirectory() as path:
            source, output = Path(path) / "inputs", Path(path) / "derived"
            source.mkdir()
            pairs.to_parquet(source / "pairs_master.parquet", index=False)
            pd.DataFrame(rerolls).to_parquet(source / "rerolls_long.parquet", index=False)
            def hashes():
                return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.iterdir()}
            before = hashes()
            report = build(source, output, n_boot=100, seed=42)
            self.assertEqual(before, hashes())
            self.assertEqual(report["n_questions"], 3)
            self.assertEqual(report["n_pairs"], 6)
            self.assertEqual(report["n_rerolls"], 24)
            benchmark = pd.read_csv(output / "plot_tables_followup/benchmark_cells_model_dataset_cue_case_clusterCI.csv")
            self.assertEqual(len(benchmark), 1)
            cell = benchmark.iloc[0]
            self.assertEqual(cell.ssp_unf, 3)
            self.assertEqual(cell.ssp_labeled, 6)
            self.assertEqual(cell.rsp_unf_incl_minus1, 6)
            self.assertEqual(cell.rsp_judged, 11)
            self.assertAlmostEqual(cell.ssp_rate, 1 / 2)
            self.assertAlmostEqual(cell.rsp_rate, 6 / 11)
            self.assertAlmostEqual(cell.gap_rsp_minus_ssp, 6 / 11 - 1 / 2)
            self.assertIn("plot_tables/fig_reliance_composition_ssp_flips.csv", report["tables"])

    def test_cannot_write_into_source_results(self):
        with tempfile.TemporaryDirectory() as path:
            source = Path(path)
            for destination in (source, source / "nested"):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(ValueError, "read-only"):
                        build(source, destination)
            self.assertFalse((source / "nested").exists())


if __name__ == "__main__":
    unittest.main()
