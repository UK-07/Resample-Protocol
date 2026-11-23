"""Tests for src.lib.ssp_vs_rsp and the SSP-vs-RSP predicates of src.lib.selection."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib import probe_datasets as pdx
from src.lib import selection
from src.lib import ssp_vs_rsp as X


def frame(rows):
    base = {"provenance": "hinted_once", "exclude_reason": None, "split": "train", "hint_style": "metadata",
            "dataset_split": "mmlu_pro-test", "case": "positive", "question_id": "q"}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["rollout_id"] = [f"r{i}" for i in range(len(df))]
    df["to_hint"] = df["to_hint"].astype("boolean")
    df["judge_label_final"] = pd.array(df["judge_label_final"], dtype="Int64")
    return df


U = {"to_hint": True, "judge_label_final": 0, "role": "used_candidate"}
F = {"to_hint": True, "judge_label_final": 1, "role": "used_candidate"}
C = {"to_hint": False, "judge_label_final": None, "role": "control"}


class PredicateTest(unittest.TestCase):
    def setUp(self):
        self.df = frame([
            {**U, "reliance_label": "robust_used"}, {**U, "reliance_label": "weak_used"},
            {**U, "reliance_label": "mixed"}, {**F, "reliance_label": "robust_used"},
            {**F, "reliance_label": "weak_used"}, {**C, "reliance_label": "robust_ignored"},
            {**C, "reliance_label": "mixed"},
            {**U, "reliance_label": "robust_used", "exclude_reason": "noise_flip"},
            {**U, "reliance_label": "robust_used", "exclude_reason": "incoherent"},
            {**U, "reliance_label": "robust_used", "provenance": "resample_k4"},
        ])

    def ids(self, name):
        return selection.select(name, self.df)

    def test_ssp_all_keeps_every_reliance_label_and_noise_flips(self):
        self.assertEqual(self.ids("ssp_all_unfaithful_vs_ignored"), ["r0", "r1", "r2", "r5", "r6", "r7"])
        self.assertEqual(self.ids("ssp_all_unfaithful_vs_faithful"), ["r0", "r1", "r2", "r3", "r4", "r7"])

    def test_rsp_survivors_is_the_reliance_filtered_subset(self):
        self.assertEqual(self.ids("rsp_survivors_unfaithful_vs_ignored"), ["r0", "r5", "r7"])
        self.assertEqual(self.ids("rsp_survivors_unfaithful_vs_faithful"), ["r0", "r3", "r7"])
        for task in selection.SSP_RSP_TASKS:
            rsp = set(self.ids(selection.ssp_rsp_predicate("rsp_survivors", task)))
            self.assertTrue(rsp <= set(self.ids(selection.ssp_rsp_predicate("ssp_all", task))))

    def test_labels(self):
        lab = selection.labels_for("ssp_all_unfaithful_vs_ignored", self.df).set_index("rollout_id")["label"]
        self.assertEqual(lab.to_dict(), {"r0": 0, "r1": 0, "r2": 0, "r5": 1, "r6": 1, "r7": 0})
        lab = selection.labels_for("ssp_all_unfaithful_vs_faithful", self.df).set_index("rollout_id")["label"]
        self.assertEqual(lab.loc["r3"], 1)
        self.assertEqual(lab.loc["r0"], 0)

    def test_universe_and_classes(self):
        self.assertEqual(self.ids("ssp_rsp_universe"), ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"])
        cls = X.row_class(self.df)
        self.assertEqual(cls.tolist()[:8], ["U", "U", "U", "F", "F", "C", "C", "U"])
        self.assertTrue(pd.isna(cls.iloc[8]) and pd.isna(cls.iloc[9]))

    def test_bad_arm_or_task(self):
        with self.assertRaises(ValueError):
            selection.ssp_rsp_predicate("ssp", "unfaithful_vs_ignored")
        with self.assertRaises(ValueError):
            selection.ssp_rsp_predicate("ssp_all", "used_vs_ignored")


class UniverseAndCapTest(unittest.TestCase):
    def test_sample_universe_quota_per_stratum(self):
        rows = [{**U, "reliance_label": "robust_used"}] * 4 + [{**F, "reliance_label": "robust_used"}] * 20 \
            + [{**C, "reliance_label": "mixed"}] * 3 + [{**F, "reliance_label": "robust_used", "hint_style": "consensus"}] * 5
        df = frame(rows)
        df["cls"] = X.row_class(df)
        keep = X.sample_universe(df, ratio=1.5)
        kept = df[keep.to_numpy()]
        self.assertEqual(int(kept["cls"].eq("U").sum()), 4)
        self.assertEqual(int((kept["cls"].eq("F") & kept["hint_style"].eq("metadata")).sum()), 6)
        self.assertEqual(int(kept["cls"].eq("C").sum()), 3)       # fewer than the quota: all kept
        self.assertEqual(int(kept["hint_style"].eq("consensus").sum()), 0)   # no U in that stratum
        self.assertTrue(keep.equals(X.sample_universe(df, ratio=1.5)))      # deterministic

    def test_post_hoc_cap_is_an_even_share(self):
        rows = []
        for style, n in (("metadata", 4), ("consensus", 6), ("post_hoc", 30)):
            rows += [{**U, "reliance_label": "robust_used", "hint_style": style}] * n
        df = frame(rows)
        df["cls"] = X.row_class(df)
        member = X.membership(df, "unfaithful_vs_faithful", "ssp_all", "posthoc_capped")
        kept = df[member.to_numpy()]
        self.assertEqual(int(kept["hint_style"].eq("post_hoc").sum()), 5)   # mean of 4 and 6
        self.assertEqual(int(kept["hint_style"].ne("post_hoc").sum()), 10)
        none = X.membership(df, "unfaithful_vs_faithful", "ssp_all", "no_posthoc")
        self.assertEqual(int(none.sum()), 10)

    def test_gold_membership(self):
        df = frame([{**U, "split": "test"}, {**U, "split": "test"}, {**C, "split": "test"}, {**C, "split": "test"},
                    {**F, "split": "test"}])
        df["reliance_label"] = "mixed"
        df["cls"] = X.row_class(df)
        counts = pd.DataFrame({"rollout_id": df["rollout_id"], "n_gold": [12, 12, 12, 12, 11],
                               "n_target": [9, 8, 0, 1, 12], "n_baseline": [0, 0, 9, 11, 0]})
        g = X.gold_labels(counts)
        self.assertEqual(g["gold_used"].tolist(), [True, False, False, False, False])
        self.assertEqual(g["gold_ignored"].tolist(), [False, False, True, False, False])
        df = df.merge(g[["rollout_id", "gold_used", "gold_ignored"]], on="rollout_id")
        self.assertEqual(df.loc[X.membership(df, "unfaithful_vs_ignored", "gold", "no_posthoc").to_numpy(),
                                "rollout_id"].tolist(), ["r0", "r2"])


class PrefixAndMetricTest(unittest.TestCase):
    def test_prefix_lengths(self):
        self.assertEqual(X.prefix_lengths(10), list(range(1, 11)))
        self.assertEqual(X.prefix_lengths(1), [1] * 10)
        self.assertEqual(X.prefix_lengths(7)[-1], 7)
        self.assertTrue(all(a <= b for a, b in zip(X.prefix_lengths(333), X.prefix_lengths(333)[1:])))
        with self.assertRaises(ValueError):
            X.prefix_lengths(0)

    def test_auroc_detection_target_is_label_zero(self):
        self.assertEqual(X.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertTrue(np.isnan(X.auroc([1, 1], [0.1, 0.2])))
        lo, hi = X.bootstrap_auroc([0, 0, 1, 1] * 5, [0.9, 0.4, 0.5, 0.1] * 5, list("abcd") * 5, n_boot=200)
        self.assertLessEqual(lo, hi)
        d, lo, hi = X.paired_bootstrap_delta([0, 0, 1, 1] * 5, [0.5] * 20, [0.9, 0.8, 0.2, 0.1] * 5,
                                             [str(i) for i in range(20)], n_boot=200)
        self.assertAlmostEqual(d, 0.5)


class DatasetProvenanceTest(unittest.TestCase):
    def test_sidecar_provenances(self):
        self.assertEqual(pdx.dataset_provenances({}), (selection.RESAMPLE_K4,))
        self.assertEqual(pdx.dataset_provenances({"provenances": ["hinted_once"]}), ("hinted_once",))
        with self.assertRaises(ValueError):
            pdx.dataset_provenances({"provenances": ["resample_k12"]})
        with self.assertRaises(ValueError):
            pdx.dataset_provenances({"provenances": []})


class NormalisePrefixesTest(unittest.TestCase):
    def test_cutoffs_select_tokens_ending_before_them(self):
        from src.lib.text_features import TextPipelineConfig, normalise_prefixes

        class Tok:
            def __init__(self, text, idx):
                self.text, self.idx, self.lemma_ = text, idx, text
                self.is_space = self.is_punct = self.is_stop = self.like_num = False

        class NLP:
            meta = {}
            max_length = 10 ** 6

            def pipe(self, texts, batch_size, n_process):
                for t in texts:
                    pos, toks = 0, []
                    for w in t.split(" "):
                        toks.append(Tok(w, pos))
                        pos += len(w) + 1
                    yield toks

        cfg = TextPipelineConfig(lowercase=True, remove_stopwords=False, lemmatize=False, n_process=1)
        out = normalise_prefixes(["Alpha beta gamma"], [[0, 5, 10, 16]], cfg, nlp=NLP())
        self.assertEqual(out, [["", "alpha", "alpha beta", "alpha beta gamma"]])
        with self.assertRaises(ValueError):
            normalise_prefixes(["a"], [], cfg, nlp=NLP())


if __name__ == "__main__":
    unittest.main()
