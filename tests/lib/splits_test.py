"""Tests for src/lib/splits.py — the question-level split assignment."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.lib.splits import (
    SPLIT_FRACTIONS,
    assert_assigned,
    assign_question_splits,
    assign_splits,
    carry_over_splits,
    propagate_splits,
    question_hash,
    question_signatures,
    split_meta,
    split_report,
)
from tests.lib.resample_test import manifest_rows


def multi_cell_manifest(n_questions: int = 400) -> pd.DataFrame:
    """Every question under three hints; questions < n/4 negative under model m2, positive under m1."""
    parts = []
    for hint in ("metadata", "consensus", "post_hoc"):
        parts.append(manifest_rows(n_questions, 0, 0, hint=hint, model="m1"))
        neg = manifest_rows(n_questions // 4, 0, 0, hint=hint, model="m2", case="negative")
        parts.append(neg)
    return pd.concat(parts, ignore_index=True)


class HashTest(unittest.TestCase):
    def test_hash_is_deterministic_seeded_and_uniform(self):
        self.assertEqual(question_hash(1, "medqa:test:1"), question_hash(1, "medqa:test:1"))
        self.assertNotEqual(question_hash(1, "medqa:test:1"), question_hash(2, "medqa:test:1"))
        u = np.array([question_hash(0, f"q:{i}") for i in range(5000)])
        self.assertTrue(((u >= 0) & (u < 1)).all())
        self.assertLess(abs(u.mean() - 0.5), 0.02)


class SignatureTest(unittest.TestCase):
    def test_signature_is_the_set_of_cells(self):
        df = multi_cell_manifest(8)
        sig = question_signatures(df)
        self.assertEqual(sig["medqa:test:0"], (("consensus", "negative"), ("consensus", "positive"),
                                              ("metadata", "negative"), ("metadata", "positive"),
                                              ("post_hoc", "negative"), ("post_hoc", "positive")))
        self.assertEqual(sig["medqa:test:7"], (("consensus", "positive"), ("metadata", "positive"),
                                              ("post_hoc", "positive")))


class AssignTest(unittest.TestCase):
    def test_one_split_per_question_and_stratified_per_cell(self):
        df = multi_cell_manifest(400)
        out = assign_splits(df, seed=7)
        assert_assigned(out)
        self.assertEqual(list(out.columns), list(df.columns))
        self.assertTrue(out["split"].notna().all())
        per_q = out.groupby("question_id")["split"].nunique()
        self.assertTrue((per_q == 1).all())
        report = split_report(out).set_index(["hint_style", "case"])
        for (hint, case), row in report.iterrows():
            for name, frac in SPLIT_FRACTIONS.items():
                # Every cell is a union of signature groups, each cut to within one question.
                self.assertAlmostEqual(row[f"frac_q_{name}"], frac, delta=2.0 / row["n_questions"] + 1e-9,
                                       msg=f"{hint}/{case}/{name}")

    def test_deterministic_and_order_independent(self):
        df = multi_cell_manifest(60)
        a = assign_splits(df, seed=3).set_index("rollout_id")["split"]
        b = assign_splits(df.sample(frac=1, random_state=9), seed=3).set_index("rollout_id")["split"]
        self.assertTrue(a.sort_index().equals(b.sort_index()))
        c = assign_splits(df, seed=4).set_index("rollout_id")["split"]
        self.assertFalse(a.equals(c))

    def test_tiny_signature_groups_are_unbiased(self):
        # 300 singleton signature groups: the per-signature offset spreads them over the splits.
        parts = [manifest_rows(1, 0, 0, hint=f"h{i}", start=i) for i in range(300)]
        out = assign_question_splits(pd.concat(parts, ignore_index=True), seed=11)
        counts = out.value_counts(normalize=True)
        self.assertAlmostEqual(counts["train"], 0.70, delta=0.08)
        self.assertAlmostEqual(counts["test"], 0.15, delta=0.06)

    def test_refuses_to_reassign_unless_forced(self):
        df = multi_cell_manifest(20)
        out = assign_splits(df, seed=1)
        with self.assertRaises(ValueError):
            assign_splits(out, seed=2)
        forced = assign_splits(out, seed=1, force=True)
        self.assertTrue(forced["split"].equals(out["split"]))

    def test_extend_keeps_the_stored_assignment_and_fills_new_questions(self):
        df = multi_cell_manifest(40)
        # First assignment: questions 0-29 under the first hint only.
        early = (df["original_index"] < 30) & (df["hint_style"] == "metadata")
        first = assign_splits(df[early], seed=1)
        grown = pd.concat([first, df[~early]], ignore_index=True)
        out = assign_splits(grown, seed=1, extend=True)
        assert_assigned(out)
        old_ids = first["rollout_id"]
        self.assertEqual(list(out.set_index("rollout_id").loc[old_ids, "split"]), list(first["split"]))
        # New rows of an old question inherit its split; new questions were drawn afresh.
        by_q = first.drop_duplicates("question_id").set_index("question_id")["split"]
        old_q = out["question_id"].isin(by_q.index)
        self.assertEqual(list(out.loc[old_q, "split"]), list(by_q.loc[out.loc[old_q, "question_id"]]))
        self.assertEqual(out.loc[~old_q, "question_id"].nunique(), 10)
        # A rerun with nothing to extend is a no-op; a fresh assignment of the grown frame differs.
        self.assertTrue(assign_splits(out, seed=1, extend=True)["split"].equals(out["split"]))
        self.assertFalse(assign_splits(grown, seed=1, force=True)["split"].equals(out["split"]))

    def test_carry_over_on_rebuild(self):
        previous = assign_splits(multi_cell_manifest(20), seed=1)
        rebuilt = multi_cell_manifest(24)  # four new questions
        out, report = carry_over_splits(rebuilt, previous)
        self.assertEqual(report["n_questions_carried"], 20)
        self.assertEqual(report["n_unassigned"], int(rebuilt["question_id"].isin(
            [f"medqa:test:{i}" for i in range(20, 24)]).sum()))
        by_q = previous.drop_duplicates("question_id").set_index("question_id")["split"]
        carried = out[out["split"].notna()]
        self.assertEqual(list(carried["split"]), list(by_q.loc[carried["question_id"]]))
        untouched, report = carry_over_splits(rebuilt, multi_cell_manifest(20))
        self.assertFalse(report["previous_had_split"])
        self.assertTrue(untouched["split"].isna().all())

    def test_bad_fractions(self):
        df = multi_cell_manifest(4)
        with self.assertRaises(ValueError):
            assign_splits(df, seed=1, fractions={"train": 0.5, "val": 0.5, "test": 0.5})
        with self.assertRaises(ValueError):
            assign_splits(df, seed=1, fractions={"train": 1.0, "dev": 0.0, "test": 0.0})

    def test_missing_columns(self):
        with self.assertRaises(ValueError):
            assign_splits(pd.DataFrame({"question_id": ["a"]}), seed=1)


class PropagateTest(unittest.TestCase):
    def test_rerolls_inherit_the_question_split(self):
        base = assign_splits(multi_cell_manifest(30), seed=5)
        rerolls = base.iloc[:12].copy()
        rerolls["rollout_id"] = rerolls["rollout_id"] + "#rs43"
        rerolls["split"] = None
        out = propagate_splits(rerolls, base)
        assert_assigned(out)
        want = base[["question_id", "split"]].drop_duplicates().set_index("question_id")["split"]
        self.assertEqual(list(out["split"]), list(want.loc[out["question_id"]]))

    def test_unknown_question_is_an_error_unless_allowed(self):
        base = assign_splits(multi_cell_manifest(10), seed=5)
        target = pd.DataFrame({"question_id": ["medqa:test:0", "other:test:99"]})
        with self.assertRaises(ValueError):
            propagate_splits(target, base)
        out = propagate_splits(target, base, allow_missing=True)
        self.assertTrue(pd.isna(out["split"].iloc[1]))

    def test_source_must_be_fully_assigned_and_consistent(self):
        base = multi_cell_manifest(10)
        with self.assertRaises(ValueError):
            propagate_splits(base, base)  # source split null
        assigned = assign_splits(base, seed=1)
        assigned.loc[0, "split"] = "test" if assigned.loc[0, "split"] != "test" else "train"
        with self.assertRaises(ValueError):
            propagate_splits(base, assigned)  # question straddles splits


class AssertAndReportTest(unittest.TestCase):
    def test_assert_assigned_errors(self):
        df = multi_cell_manifest(5)
        with self.assertRaises(ValueError):
            assert_assigned(df)
        with self.assertRaises(ValueError):
            assert_assigned(df.drop(columns="split"))
        out = assign_splits(df, seed=1)
        out.loc[0, "split"] = "dev"
        with self.assertRaises(ValueError):
            assert_assigned(out)

    def test_report_and_meta(self):
        out = assign_splits(multi_cell_manifest(40), seed=2)
        report = split_report(out)
        self.assertEqual(report.iloc[0]["hint_style"], "all")
        self.assertEqual(int(report.iloc[0]["n_questions"]), 40)
        self.assertEqual(set(report["case"]), {"all", "positive", "negative"})
        meta = split_meta(out, seed=2)
        self.assertEqual(meta["n_questions"], 40)
        self.assertEqual(sum(meta["questions_per_split"].values()), 40)
        self.assertEqual(meta["fractions"], SPLIT_FRACTIONS)
        unassigned = split_report(multi_cell_manifest(3))
        self.assertIn("q_unassigned", unassigned.columns)


if __name__ == "__main__":
    unittest.main()
