"""Tests for src.lib.ssp_vs_rsp and the SSP-vs-RSP predicates of src.lib.selection."""

import unittest

import numpy as np
import pandas as pd

from src.lib import probe_datasets as pdx
from src.lib import selection
from src.lib import ssp_vs_rsp as X


def frame(rows):
    base = {"provenance": "hinted_once", "exclude_reason": None, "split": "train", "hint_style": "metadata",
            "dataset_split": "mmlu_pro-test", "case": "positive", "to_hint": True, "reliance_label": "robust_used"}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["rollout_id"] = [f"r{i}" for i in range(len(df))]
    df["to_hint"] = df["to_hint"].astype("boolean")
    df["judge_label_final"] = pd.array(df["judge_label_final"], dtype="Int64")
    return df


U = {"judge_label_final": 0}
F = {"judge_label_final": 1}


def ids(df, mask):
    return df.loc[mask.to_numpy(), "rollout_id"].tolist()


class PredicateTest(unittest.TestCase):
    def setUp(self):
        self.df = frame([
            U, {**U, "reliance_label": "weak_used"}, {**U, "reliance_label": "mixed"}, F,
            {**F, "reliance_label": "weak_used"}, {**U, "exclude_reason": "noise_flip"},
            {**U, "exclude_reason": "incoherent"}, {**U, "provenance": "resample_k4"},
            {"judge_label_final": None, "to_hint": False, "reliance_label": "robust_ignored"},
        ])

    def test_ssp_all_keeps_every_reliance_label_and_noise_flips(self):
        self.assertEqual(selection.select("ssp_all", self.df), ["r0", "r1", "r2", "r3", "r4", "r5"])

    def test_rsp_survivors_is_the_robust_used_subset(self):
        self.assertEqual(selection.select("rsp_survivors", self.df), ["r0", "r3", "r5"])

    def test_label_zero_is_unfaithful(self):
        lab = selection.labels_for("ssp_all", self.df).set_index("rollout_id")["label"]
        self.assertEqual(lab.to_dict(), {"r0": 0, "r1": 0, "r2": 0, "r3": 1, "r4": 1, "r5": 0})


class SamplingTest(unittest.TestCase):
    def test_sample_universe_quota_per_stratum(self):
        df = frame([U] * 4 + [F] * 20 + [{**F, "hint_style": "consensus"}] * 5)
        keep = X.sample_universe(df, ratio=1.5)
        kept = df[keep.to_numpy()]
        self.assertEqual(int(kept["judge_label_final"].eq(0).sum()), 4)
        self.assertEqual(int(kept["judge_label_final"].eq(1).sum()), 6)
        self.assertEqual(int(kept["hint_style"].eq("consensus").sum()), 0)   # no unfaithful row in that stratum
        self.assertTrue(keep.equals(X.sample_universe(df, ratio=1.5)))      # deterministic

    def test_post_hoc_cap_is_an_even_share(self):
        df = frame([{**U, "hint_style": style} for style, n in (("metadata", 4), ("consensus", 6), ("post_hoc", 30))
                    for _ in range(n)])
        kept = df[X.membership(df, "ssp_all", "posthoc_capped").to_numpy()]
        self.assertEqual(int(kept["hint_style"].eq("post_hoc").sum()), 5)   # mean of 4 and 6
        self.assertEqual(int(kept["hint_style"].ne("post_hoc").sum()), 10)
        self.assertEqual(int(X.membership(df, "ssp_all", "no_posthoc").sum()), 10)

    def test_ssp_sized_matches_the_rsp_survivors_counts(self):
        df = frame([U] * 3 + [{**U, "reliance_label": "weak_used"}] * 5 + [F] * 2 + [{**F, "reliance_label": "mixed"}] * 4)
        sized = X.membership(df, "ssp_sized", "no_posthoc")
        self.assertEqual(df[sized.to_numpy()]["judge_label_final"].value_counts().to_dict(), {0: 3, 1: 2})
        self.assertTrue((sized <= X.membership(df, "ssp_all", "no_posthoc")).all())

    def test_gold_membership_and_split_masks(self):
        df = frame([{**U, "split": "test"}, {**F, "split": "test"}, {**F, "split": "val"}, U])
        df["gold_used"] = [True, False, False, False]
        self.assertEqual(ids(df, X.membership(df, "gold", "no_posthoc")), ["r0"])
        masks, test = X.split_masks(df)
        self.assertEqual(set(masks), {(v, arm) for v in X.VARIANTS for arm in X.ARMS})
        train, val = masks["no_posthoc", "ssp_all"]
        self.assertEqual((train.tolist(), val.tolist(), test.tolist()),
                         ([False, False, False, True], [False, False, True, False], [True, True, False, False]))


class PrefixAndMetricTest(unittest.TestCase):
    def test_prefix_lengths(self):
        self.assertEqual(X.prefix_lengths(10), list(range(1, 11)))
        self.assertEqual(X.prefix_lengths(1), [1] * 10)
        self.assertEqual(X.prefix_lengths(7)[-1], 7)
        self.assertTrue(all(a <= b for a, b in zip(X.prefix_lengths(333), X.prefix_lengths(333)[1:])))

    def test_auroc(self):
        pos = np.array([True, True, False, False])
        self.assertEqual(X.auroc(pos, np.array([0.9, 0.8, 0.2, 0.1])), 1.0)
        self.assertEqual(X.auroc(pos, np.array([0.9, 0.1, 0.8, 0.2])), 0.5)
        self.assertEqual(X.auroc(pos, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        self.assertTrue(np.isnan(X.auroc(np.array([False, False]), np.array([0.1, 0.2]))))


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


if __name__ == "__main__":
    unittest.main()
