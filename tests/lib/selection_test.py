"""Tests for src/lib/selection.py."""

from __future__ import annotations

import unittest

import pandas as pd

from src.lib import selection
from src.lib.selection import (
    CASE_FILTERS,
    HINTED_ONCE,
    LABEL_CONFIGS,
    LABEL_ENCODINGS,
    RESAMPLE_K4,
    TRAINING_DATASETS,
    encode_labels,
    labels_for,
    mask,
    resample_provenance,
    select,
    select_rows,
    with_provenance,
)
from tests.lib.resample_test import label_config_frame, manifest_rows


def resample_frame() -> pd.DataFrame:
    """A resample-manifest slice: three questions × (original + 2 re-rolls), with the re-sampling columns."""
    rows = []

    def add(qid, prov, case, rl, to_hint, label, contrast_a, a_set, excl=None, seed=None):
        rid = f"m:run:metadata:{qid}" + (f"#rs{seed}" if seed else "")
        rows.append({
            "rollout_id": rid, "question_id": f"medqa:test:{qid}", "dataset": "medqa", "hint_style": "metadata",
            "case": case, "subject_model": "m", "to_hint": to_hint, "judge_label_final": label,
            "exclude_reason": excl, "provenance": prov, "reliance_label": rl,
            "contrast_a": contrast_a, "contrast_a_set": a_set, "split": None,
        })

    add(0, HINTED_ONCE, "positive", "robust_used", True, 0, "used", "train")
    add(0, RESAMPLE_K4, "positive", "robust_used", True, 0, "used", "train", seed=43)
    add(0, RESAMPLE_K4, "positive", "robust_used", True, 1, "used", "train", seed=44)
    add(1, HINTED_ONCE, "negative", "robust_ignored", False, None, "ignored", "train")
    add(1, RESAMPLE_K4, "negative", "robust_ignored", False, None, "ignored", "train", seed=43)
    add(1, RESAMPLE_K4, "negative", "robust_ignored", False, None, "ignored", "train", seed=44)
    add(2, HINTED_ONCE, "positive", "weak_used", True, 1, "used", "challenge")
    add(2, RESAMPLE_K4, "positive", "weak_used", True, 1, "used", "challenge", seed=43)
    add(2, RESAMPLE_K4, "positive", "weak_used", True, None, "used", "challenge", excl="truncated", seed=44)
    return pd.DataFrame(rows)


class ProvenanceTest(unittest.TestCase):
    def test_base_manifest_is_hinted_once(self):
        df = manifest_rows(2, 0, 0)
        out = with_provenance(df)
        self.assertTrue((out["provenance"] == HINTED_ONCE).all())
        self.assertNotIn("provenance", df.columns)  # the input is untouched

    def test_is_resample_is_mapped(self):
        df = manifest_rows(2, 0, 0).assign(is_resample=[False, True])
        self.assertEqual(list(with_provenance(df)["provenance"]), [HINTED_ONCE, RESAMPLE_K4])

    def test_explicit_column_is_kept_and_validated(self):
        df = manifest_rows(1, 0, 0).assign(provenance="resample_k4", is_resample=False)
        self.assertEqual(with_provenance(df)["provenance"].iloc[0], RESAMPLE_K4)
        with self.assertRaises(ValueError):
            with_provenance(df.assign(provenance="bogus"))

    def test_resample_provenance_value(self):
        self.assertEqual(resample_provenance(4), RESAMPLE_K4)


class DatasetCTest(unittest.TestCase):
    def test_selects_clean_judged_switched_rows_of_the_base_manifest(self):
        df = manifest_rows(4, 2, 3)  # 4 to_hint (labels 0,1,0,1; the first noise_flip-flagged by votes), 2 off, 3 kept
        df.loc[0, "exclude_reason"] = "noise_flip"
        df.loc[2, "judge_label_final"] = -1
        df.loc[2, "exclude_reason"] = "incoherent"
        ids = select("dataset_C", df)
        self.assertEqual(ids, ["m:medqa-test_0911:metadata:1", "m:medqa-test_0911:metadata:3"])
        rows = select_rows("dataset_C", df)
        self.assertEqual(list(rows["provenance"]), [HINTED_ONCE, HINTED_ONCE])

    def test_every_exclude_reason_is_dropped(self):
        from src.lib.rollout_manifest import EXCLUDE_REASONS

        df = manifest_rows(len(EXCLUDE_REASONS) + 1, 0, 0)
        df["judge_label_final"] = 1
        df["exclude_reason"] = list(EXCLUDE_REASONS) + [None]
        kept = select("dataset_C", df)
        self.assertEqual(kept, [f"m:medqa-test_0911:metadata:{len(EXCLUDE_REASONS)}"])
        self.assertIn("baseline_relabelled", EXCLUDE_REASONS)
        self.assertIn("judge_input_degraded", EXCLUDE_REASONS)

    def test_unjudged_switched_rows_are_never_selected(self):
        df = manifest_rows(2, 0, 0)
        df["judge_label_final"] = None
        df["judge_label_final_model"] = None
        self.assertEqual(select("dataset_C", df), [])

    def test_rerolls_are_not_dataset_C(self):
        self.assertEqual(select("dataset_C", resample_frame()), ["m:run:metadata:0", "m:run:metadata:2"])


class DatasetCIncoherentTest(unittest.TestCase):
    def frame(self):
        df = manifest_rows(5, 1, 1)  # 5 to_hint, 1 off-target, 1 kept
        df["judge_label_final"] = [0, -1, -1, 1, -1, None, None]
        df["exclude_reason"] = [None, "incoherent", "truncated", None, "incoherent", None, None]
        return df

    def test_selects_only_incoherent_switched_rows(self):
        df = self.frame()
        ids = select("dataset_C_incoherent", df)
        self.assertEqual(ids, ["m:medqa-test_0911:metadata:1", "m:medqa-test_0911:metadata:4"])
        rows = select_rows("dataset_C_incoherent", df)
        self.assertTrue((rows["judge_label_final"] == -1).all())
        self.assertTrue((rows["exclude_reason"] == "incoherent").all())
        # A -1 row whose earlier reason (truncated) won is not in the set.
        self.assertNotIn("m:medqa-test_0911:metadata:2", ids)

    def test_disjoint_from_dataset_C_and_not_a_training_set(self):
        df = self.frame()
        self.assertFalse(set(select("dataset_C", df)) & set(select("dataset_C_incoherent", df)))
        self.assertNotIn("dataset_C_incoherent", TRAINING_DATASETS)
        self.assertIsNone(selection.get("dataset_C_incoherent").label_col)
        with self.assertRaises(ValueError):
            labels_for("dataset_C_incoherent", df)

    def test_rerolls_are_not_in_the_set(self):
        df = resample_frame()
        df.loc[df["provenance"] == RESAMPLE_K4, "judge_label_final"] = -1
        df.loc[df["provenance"] == RESAMPLE_K4, "exclude_reason"] = "incoherent"
        self.assertEqual(select("dataset_C_incoherent", df), [])


class ResampleDatasetsTest(unittest.TestCase):
    def test_dataset_B_is_the_robust_used_rerolls_with_a_binary_verdict(self):
        self.assertEqual(select("dataset_B", resample_frame()),
                         ["m:run:metadata:0#rs43", "m:run:metadata:0#rs44"])

    def test_dataset_A_train_and_challenge(self):
        df = resample_frame()
        self.assertEqual(select("dataset_A", df), [
            "m:run:metadata:0#rs43", "m:run:metadata:0#rs44", "m:run:metadata:1#rs43", "m:run:metadata:1#rs44",
        ])
        # The truncated re-roll of q2 is excluded; the judged one stays a challenge row.
        self.assertEqual(select("dataset_A_challenge", df), ["m:run:metadata:2#rs43"])

    def test_missing_columns_are_an_error_not_an_empty_selection(self):
        with self.assertRaises(ValueError) as ctx:
            select("dataset_B", manifest_rows(2, 0, 0))
        self.assertIn("reliance_label", str(ctx.exception))

    def test_unknown_predicate(self):
        with self.assertRaises(KeyError):
            select("dataset_Z", resample_frame())


class CaseHandlingTest(unittest.TestCase):
    def test_training_datasets_never_filter_on_case(self):
        for name in TRAINING_DATASETS:
            self.assertFalse(selection.get(name).filters_case)
            self.assertNotIn("case", selection.get(name).requires)

    def test_positive_only_entries_are_the_only_case_filters(self):
        filtering = sorted(n for n, p in selection.REGISTRY.items() if p.filters_case)
        self.assertEqual(filtering, sorted(f"{n}_positive_only" for n in TRAINING_DATASETS))
        for name in filtering:
            self.assertEqual(selection.get(name).base, name.removesuffix("_positive_only"))
        for name, pred in selection.REGISTRY.items():
            if name not in filtering:
                self.assertNotIn("case", pred.requires, name)
        import inspect
        for fn in (mask, select, select_rows, labels_for):
            self.assertEqual(inspect.signature(fn).parameters["cases"].default, "both", fn.__name__)

    def test_positive_only_restricts_the_base(self):
        df = resample_frame()
        self.assertEqual(select("dataset_A_positive_only", df), ["m:run:metadata:0#rs43", "m:run:metadata:0#rs44"])
        self.assertEqual(select("dataset_B_positive_only", df), select("dataset_B", df))


class LabelConfigsTest(unittest.TestCase):
    def setUp(self):
        self.df, self.ids = label_config_frame()

    def rs(self, key, *seeds):
        return [f"{self.ids[key]}#rs{s}" for s in seeds]

    def test_verbalised_vs_unverbalised_equals_dataset_B(self):
        pred = selection.get("verbalised_vs_unverbalised")
        self.assertEqual(pred.base, "dataset_B")
        self.assertIs(pred.fn, selection.get("dataset_B").fn)
        self.assertEqual(pred.requires, selection.get("dataset_B").requires)
        self.assertEqual(pred.label_col, "judge_label_final")
        for frame in (self.df, resample_frame()):
            self.assertEqual(select("verbalised_vs_unverbalised", frame), select("dataset_B", frame))
            pd.testing.assert_frame_equal(labels_for("verbalised_vs_unverbalised", frame), labels_for("dataset_B", frame))
        self.assertEqual(select("verbalised_vs_unverbalised", self.df), sorted(
            self.rs("0", 43, 44) + self.rs("1", 44, 45, 46) + self.rs("10", 43, 44, 45, 46)))

    def test_verbalised_vs_rest_rows_and_labels(self):
        got = labels_for("verbalised_vs_rest", self.df).set_index("rollout_id")["label"].to_dict()
        want = {
            **{r: 1 for r in self.rs("0", 43) + self.rs("1", 44, 46) + self.rs("10", 45, 46)},   # used & 1
            **{r: 0 for r in self.rs("0", 44) + self.rs("1", 45) + self.rs("10", 43, 44)},       # used & 0
            **{r: 0 for r in self.rs("3", 43, 44, 45) + self.rs("11", 43, 44, 45, 46)},         # ignored
        }
        self.assertEqual(got, want)
        # -1 / unjudged used rows, off-pattern re-rolls, weak/mixed questions and originals are out.
        self.assertNotIn(self.rs("0", 45)[0], got)
        self.assertNotIn(self.rs("1", 43)[0], got)
        self.assertNotIn(self.rs("0", 46)[0], got)
        self.assertNotIn(self.rs("3", 46)[0], got)
        self.assertFalse(any(sid in got for sid in self.ids.values()))
        df = self.df.copy()
        df.loc[df["rollout_id"] == self.rs("0", 43)[0], "exclude_reason"] = "truncated"
        self.assertNotIn(self.rs("0", 43)[0], select("verbalised_vs_rest", df))

    def test_used_vs_ignored_rows_and_labels(self):
        got = labels_for("used_vs_ignored", self.df).set_index("rollout_id")["label"].to_dict()
        want = {
            **{r: 0 for r in self.rs("0", 43, 44) + self.rs("1", 43, 44, 45, 46) + self.rs("10", 43, 44, 45, 46)},
            **{r: 1 for r in self.rs("3", 43, 44, 45) + self.rs("11", 43, 44, 45, 46)},
        }
        self.assertEqual(got, want)
        # The label ignores the verdict: the unjudged used row (1#rs43) is in, label 0 ...
        self.assertIn(self.rs("1", 43)[0], got)
        # ... and the -1 used row (0#rs45) is "used" in the column but out through exclude_reason == incoherent.
        minus_one = self.df.set_index("rollout_id").loc[self.rs("0", 45)[0]]
        self.assertEqual(minus_one["label_used_ignored"], "used")
        self.assertEqual(minus_one["exclude_reason"], "incoherent")
        self.assertNotIn(self.rs("0", 45)[0], select("used_vs_ignored", self.df))
        df = self.df.copy()
        df.loc[df["rollout_id"] == self.rs("3", 43)[0], "exclude_reason"] = "unanswered"
        self.assertNotIn(self.rs("3", 43)[0], select("used_vs_ignored", df))

    def test_label_configs_are_registered_and_labelled(self):
        self.assertEqual(LABEL_CONFIGS, ("verbalised_vs_unverbalised", "verbalised_vs_rest", "used_vs_ignored"))
        self.assertEqual(TRAINING_DATASETS, ("dataset_A", "dataset_B", "dataset_C"))
        self.assertEqual(LABEL_ENCODINGS["label_verbalised_rest"], {"rest": 0, "verbalised": 1})
        self.assertEqual(LABEL_ENCODINGS["label_used_ignored"], {"used": 0, "ignored": 1})
        # The encodings cover exactly the classes assign_contrasts writes.
        from src.lib.resample import USED_IGNORED_CLASSES, VERBALISED_REST_CLASSES
        self.assertEqual(set(LABEL_ENCODINGS["label_verbalised_rest"]), set(VERBALISED_REST_CLASSES))
        self.assertEqual(set(LABEL_ENCODINGS["label_used_ignored"]), set(USED_IGNORED_CLASSES))
        for name in LABEL_CONFIGS:
            pred = selection.get(name)
            self.assertIn(pred.label_col, LABEL_ENCODINGS)
            self.assertFalse(pred.filters_case)
            self.assertNotIn("case", pred.requires)
            self.assertIn("exclude_reason", pred.requires)
            self.assertIn("provenance", pred.requires)
            labels = labels_for(name, self.df)
            self.assertGreater(len(labels), 0)
            self.assertEqual(set(labels["label"]), {0, 1})
            rows = select_rows(name, self.df)
            self.assertTrue((rows["provenance"] == RESAMPLE_K4).all())
            self.assertTrue(rows["exclude_reason"].isna().all())

    def test_cases_argument_filters_and_validates(self):
        self.assertEqual(CASE_FILTERS, ("both", "positive", "negative"))
        for name in LABEL_CONFIGS + TRAINING_DATASETS:
            both = select(name, self.df, cases="both")
            self.assertEqual(both, select(name, self.df))
            pos = select(name, self.df, cases="positive")
            neg = select(name, self.df, cases="negative")
            self.assertEqual(sorted(pos + neg), both)
            self.assertTrue(all(":10" not in r and ":11" not in r for r in pos))
            self.assertTrue(all(":10" in r or ":11" in r for r in neg))
        self.assertEqual(select("used_vs_ignored", self.df, cases="negative"),
                         sorted(self.rs("10", 43, 44, 45, 46) + self.rs("11", 43, 44, 45, 46)))
        neg_labels = labels_for("used_vs_ignored", self.df, cases="negative").set_index("rollout_id")["label"].to_dict()
        self.assertEqual(neg_labels, {**{r: 0 for r in self.rs("10", 43, 44, 45, 46)}, **{r: 1 for r in self.rs("11", 43, 44, 45, 46)}})
        m = mask("used_vs_ignored", self.df, cases="positive")
        self.assertEqual(m.name, "used_vs_ignored")
        self.assertEqual(int(m.sum()), 9)
        self.assertEqual(len(select_rows("used_vs_ignored", self.df, cases="positive")), 9)
        for fn in (mask, select, select_rows, labels_for):
            with self.assertRaises(ValueError) as ctx:
                fn("used_vs_ignored", self.df, cases="all")
            self.assertIn("('both', 'positive', 'negative')", str(ctx.exception))
        with self.assertRaises(ValueError):
            select("used_vs_ignored", self.df.drop(columns=["case"]), cases="positive")
        self.assertEqual(select("used_vs_ignored", self.df.drop(columns=["case"])), select("used_vs_ignored", self.df))

    def test_missing_label_column_is_an_error(self):
        for name, col in (("verbalised_vs_rest", "label_verbalised_rest"), ("used_vs_ignored", "label_used_ignored")):
            with self.assertRaises(ValueError) as ctx:
                select(name, resample_frame())
            self.assertIn(col, str(ctx.exception))
            with self.assertRaises(ValueError):
                select(name, manifest_rows(2, 0, 0))


class SplitRestrictionTest(unittest.TestCase):
    def test_split_argument_requires_an_assigned_split(self):
        df = resample_frame()
        with self.assertRaises(ValueError):
            select("dataset_B", df, split="train")
        df["split"] = ["train", "train", "train", "test", "test", "test", "val", "val", "val"]
        self.assertEqual(select("dataset_B", df, split="train"), ["m:run:metadata:0#rs43", "m:run:metadata:0#rs44"])
        self.assertEqual(select("dataset_B", df, split="test"), [])
        with self.assertRaises(ValueError):
            select("dataset_B", df, split="dev")

    def test_mask_is_index_aligned_and_named(self):
        df = resample_frame().set_index(pd.Index(range(100, 109)))
        m = mask("dataset_B", df)
        self.assertEqual(list(m.index), list(range(100, 109)))
        self.assertEqual(m.name, "dataset_B")
        self.assertEqual(int(m.sum()), 2)


class LabelsTest(unittest.TestCase):
    def test_encode_judge_labels_and_contrast_a(self):
        df = resample_frame()
        b = labels_for("dataset_B", df)
        self.assertEqual(list(b["label"]), [0, 1])
        a = labels_for("dataset_A", df)
        self.assertEqual(list(a["label"]), [0, 0, 1, 1])  # used → 0, ignored → 1

    def test_unencodable_label_is_an_error(self):
        with self.assertRaises(ValueError):
            encode_labels(pd.DataFrame({"contrast_a": ["used", None]}), "contrast_a")
        with self.assertRaises(ValueError):
            encode_labels(pd.DataFrame({"judge_label_final": [0, -1]}), "judge_label_final")
        with self.assertRaises(ValueError):
            encode_labels(pd.DataFrame({"foo": [0]}), "foo")


if __name__ == "__main__":
    unittest.main()
