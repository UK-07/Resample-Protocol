"""Tests for src/lib/probe_datasets.py — dataset specs, balance, fold roles and the fingerprint."""

from __future__ import annotations

import unittest

import pandas as pd

from src.lib import selection
from src.lib.probe_datasets import (
    DEFAULT_BALANCE_GROUPS,
    FOLD_ROLES,
    BalanceSpec,
    DatasetSpec,
    apply_balance,
    dataset_fingerprint,
    fold_report,
    fold_roles,
    parse_spec,
    question_checks,
    select_dataset_rows,
    stable_seed,
)
from tests.scripts.resample_fixtures import PROBE_STYLES, probe_manifest

MANIFEST = probe_manifest()


def spec(**overrides) -> DatasetSpec:
    cfg = {"name": "cfg_a", "manifest": "${DATA_ROOT}/resample/resample_manifest.parquet",
           "predicate": "verbalised_vs_unverbalised"}
    cfg.update(overrides)
    return parse_spec(cfg)


def balance_frame() -> pd.DataFrame:
    """Two groups on one (model, run, style) — positive: 9 label-1 / 3 label-0 with every majority row
    in split test; negative: 4 / 4 (already balanced) — plus a third group with label 1 only."""
    rows = []

    def add(case, label, n, split, start):
        for i in range(n):
            rows.append({"rollout_id": f"m:r:metadata:{start + i}#rs43", "question_id": f"d:test:{start + i}",
                         "subject_model": "m", "run": "r", "hint_style": "metadata", "case": case,
                         "label": label, "split": split})

    add("positive", 1, 9, "test", 0)
    add("positive", 0, 3, "train", 100)
    add("negative", 1, 4, "train", 200)
    add("negative", 0, 4, "train", 300)
    rows.extend({"rollout_id": f"m:r:consensus:{400 + i}#rs43", "question_id": f"d:test:{400 + i}",
                 "subject_model": "m", "run": "r", "hint_style": "consensus", "case": "positive",
                 "label": 1, "split": "train"} for i in range(5))
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("Int8")
    df["split"] = df["split"].astype("string")
    return df


class SpecTest(unittest.TestCase):
    """parse_spec / DatasetSpec / BalanceSpec validation."""

    def test_parse_spec_rejects_unknown_keys_and_bad_predicate(self):
        with self.assertRaisesRegex(ValueError, "unknown dataset spec key"):
            spec(layers=[1, 2])
        with self.assertRaisesRegex(ValueError, "unknown predicate"):
            spec(predicate="no_such_predicate")
        with self.assertRaisesRegex(ValueError, "missing required key"):
            parse_spec({"name": "x"})
        with self.assertRaisesRegex(ValueError, "name must match"):
            spec(name="-bad name")
        with self.assertRaisesRegex(ValueError, "cases must be one of"):
            spec(cases="pos")
        with self.assertRaisesRegex(ValueError, "seed must be an int"):
            spec(seed="42")
        with self.assertRaisesRegex(ValueError, "subject_models must be null or a non-empty list"):
            spec(subject_models="m1")
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            parse_spec(["not", "a", "mapping"])

    def test_predicate_without_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "carries no label column"):
            spec(predicate="dataset_C_incoherent")
        # Any registered predicate with a label column parses (re-rolls-only is checked at selection time).
        self.assertEqual(spec(predicate="dataset_C").label_col, "judge_label_final")
        for name in selection.LABEL_CONFIGS:
            self.assertEqual(spec(predicate=name).predicate, name)

    def test_parse_spec_defaults_and_balance_forms(self):
        s = spec()
        self.assertEqual((s.cases, s.seed, s.output_dir), ("both", 42, "${DATA_ROOT}/probe_datasets"))
        self.assertIsNone(s.subject_models)
        self.assertEqual(s.balance, BalanceSpec())
        self.assertEqual(s.manifest, "${DATA_ROOT}/resample/resample_manifest.parquet")  # raw, unresolved
        self.assertEqual(spec(balance="none").balance.method, "none")
        self.assertEqual(spec(balance={"method": "none"}).balance.r, None)
        b = spec(balance={"method": "ratio", "r": 2, "groups": ["hint_style"]}).balance
        self.assertEqual((b.method, b.r, b.groups), ("ratio", 2.0, ("hint_style",)))
        self.assertEqual(spec(balance={"method": "ratio", "r": 1.5}).balance.groups, DEFAULT_BALANCE_GROUPS)
        self.assertEqual(spec(subject_models=["m1"], runs=["r"], hint_styles=["metadata"]).subject_models, ("m1",))
        null = spec(seed=None, cases=None, output_dir=None, subject_models=None, balance=None)  # null = default
        self.assertEqual((null.seed, null.cases, null.output_dir, null.balance), (42, "both", s.output_dir, BalanceSpec()))
        self.assertEqual(s.to_dict()["balance"], {"method": "none", "r": None, "groups": list(DEFAULT_BALANCE_GROUPS)})

    def test_balance_spec_validation(self):
        with self.assertRaisesRegex(ValueError, "r is required"):
            spec(balance={"method": "ratio"})
        with self.assertRaisesRegex(ValueError, "must be ≥ 1.0"):
            spec(balance={"method": "ratio", "r": 0.5})
        with self.assertRaisesRegex(ValueError, "must be null"):
            spec(balance={"method": "none", "r": 1.0})
        with self.assertRaisesRegex(ValueError, "unknown balance key"):
            spec(balance={"method": "ratio", "r": 1.0, "ratio": 2})
        with self.assertRaisesRegex(ValueError, "balance.method must be one of"):
            spec(balance="undersample")
        with self.assertRaisesRegex(ValueError, "groups must be"):
            BalanceSpec(method="ratio", r=1.0, groups=())

    def test_stable_seed_matches_activation_store(self):
        from src.lib.activation_store import stable_seed as reference
        for parts in ((42, "m1", "run", "metadata", "positive"), (0,), (7, "a|b", ""), ("x", 1.5)):
            self.assertEqual(stable_seed(*parts), reference(*parts))


class SelectRowsTest(unittest.TestCase):
    """select_dataset_rows: predicate + restrictions + label, the split and re-roll assertions."""

    def test_select_rows_requires_assigned_split(self):
        with self.assertRaisesRegex(ValueError, "assign_splits"):
            select_dataset_rows(probe_manifest(split_seed=None), spec())
        partial = MANIFEST.copy()
        partial.loc[partial.index[:40], "split"] = None
        with self.assertRaisesRegex(ValueError, "unassigned"):
            select_dataset_rows(partial, spec())

    def test_select_rows_rejects_originals(self):
        with self.assertRaisesRegex(ValueError, "not k=4 re-rolls"):
            select_dataset_rows(MANIFEST, spec(predicate="dataset_C"))

    def test_unknown_run_or_model_lists_available(self):
        with self.assertRaisesRegex(ValueError, r"subject_models .*\['m9'\].*available: \['m1', 'm2'\]"):
            select_dataset_rows(MANIFEST, spec(subject_models=["m1", "m9"]))
        with self.assertRaisesRegex(ValueError, r"runs .*\['nope'\].*available: \['gpqa-diamond_0912', 'medqa-test_0911'\]"):
            select_dataset_rows(MANIFEST, spec(runs=["nope"]))
        with self.assertRaisesRegex(ValueError, r"hint_styles .*\['pushback'\].*available: \['consensus'"):
            select_dataset_rows(MANIFEST, spec(hint_styles=["pushback"]))
        # A run that exists under the predicate, but not for the requested model, must not silently drop.
        partial = MANIFEST[~((MANIFEST["subject_model"] == "m1") & (MANIFEST["run"] == "medqa-test_0911"))]
        select_dataset_rows(partial, spec(subject_models=["m2"], runs=["medqa-test_0911"]))
        with self.assertRaisesRegex(ValueError, r"runs .*once \['subject_models'\] restrict.*\['medqa-test_0911'\]"):
            select_dataset_rows(partial, spec(subject_models=["m1"], runs=["medqa-test_0911"]))

    def test_cases_filter_applied(self):
        both = select_dataset_rows(MANIFEST, spec())
        pos = select_dataset_rows(MANIFEST, spec(cases="positive"))
        neg = select_dataset_rows(MANIFEST, spec(cases="negative"))
        self.assertEqual(set(pos["case"]), {"positive"})
        self.assertEqual(set(neg["case"]), {"negative"})
        self.assertEqual(len(pos) + len(neg), len(both))
        self.assertEqual(set(pos["rollout_id"]) | set(neg["rollout_id"]), set(both["rollout_id"]))

    def test_restrictions_and_label_encoding(self):
        rows = select_dataset_rows(MANIFEST, spec(subject_models=["m2"], runs=["medqa-test_0911"],
                                                  hint_styles=["metadata", "post_hoc"]))
        self.assertEqual(set(rows["subject_model"]), {"m2"})
        self.assertEqual(set(rows["run"]), {"medqa-test_0911"})
        self.assertEqual(set(rows["hint_style"]), {"metadata", "post_hoc"})
        self.assertEqual(set(rows["provenance"]), {"resample_k4"})
        self.assertEqual(str(rows["label"].dtype), "Int8")
        self.assertTrue(rows["balance_kept"].all())
        self.assertEqual(rows["label"].tolist(), rows["judge_label_final"].astype(int).tolist())
        # used_vs_ignored: used → 0 (the detection target), ignored → 1.
        rows = select_dataset_rows(MANIFEST, spec(predicate="used_vs_ignored"))
        self.assertEqual(rows.loc[rows["label"] == 0, "label_used_ignored"].unique().tolist(), ["used"])
        self.assertEqual(rows.loc[rows["label"] == 1, "label_used_ignored"].unique().tolist(), ["ignored"])
        with self.assertRaisesRegex(ValueError, "selects no row"):
            select_dataset_rows(MANIFEST.iloc[:0], spec())


class BalanceTest(unittest.TestCase):
    """apply_balance."""

    def setUp(self):
        self.frame = balance_frame()

    def counts(self, df):
        """Per case of the metadata groups: label counts (the consensus group is the missing-class one)."""
        meta = df[df["hint_style"] == "metadata"]
        return {k: {lab: int((g["label"] == lab).sum()) for lab in (0, 1)} for k, g in meta.groupby("case")}

    def test_balance_none_is_identity(self):
        out, report = apply_balance(self.frame, spec(balance="none"))
        self.assertEqual(out["rollout_id"].tolist(), self.frame["rollout_id"].tolist())
        self.assertTrue(out["balance_kept"].all())
        self.assertEqual(report["method"], "none")
        self.assertEqual(report["n_dropped"], 0)

    def test_ratio_balance_bounds_majority_per_group(self):
        out, report = apply_balance(self.frame, spec(balance={"method": "ratio", "r": 2.0}))
        pos = self.counts(out)["positive"]
        self.assertEqual(pos, {0: 3, 1: 6})  # keep = min(9, floor(2 × 3))
        self.assertEqual(self.counts(out)["negative"], {0: 4, 1: 4})  # 4 ≤ 2 × 4: untouched
        entry = report["per_group"]["m|r|metadata|positive"]
        self.assertEqual((entry["before"], entry["after"]), ({"0": 3, "1": 9}, {"0": 3, "1": 6}))
        self.assertEqual((entry["majority_label"], entry["n_dropped"], entry["missing_class"]), (1, 3, False))
        self.assertEqual((report["r"], report["groups"]), (2.0, list(DEFAULT_BALANCE_GROUPS)))
        self.assertEqual((report["n_before"], report["n_after"], report["n_dropped"]), (25, 22, 3))
        self.assertTrue(out["balance_kept"].all())
        # A fractional r rounds down; a product that is exact in decimal is not rounded off (1.15 × 20 = 23).
        r = pd.DataFrame({"rollout_id": [f"x:{i}" for i in range(50)], "subject_model": "m", "run": "r",
                          "hint_style": "h", "case": "positive", "label": [0] * 20 + [1] * 30, "split": "train"})
        out, _ = apply_balance(r, spec(balance={"method": "ratio", "r": 1.15}))
        self.assertEqual(int((out["label"] == 1).sum()), 23)

    def test_ratio_one_is_full_balance(self):
        out, _ = apply_balance(self.frame, spec(balance={"method": "ratio", "r": 1.0}))
        self.assertEqual(self.counts(out)["positive"], {0: 3, 1: 3})
        self.assertEqual(self.counts(out)["negative"], {0: 4, 1: 4})

    def test_ratio_applies_to_test_rows_too(self):
        # Every majority row of the positive group sits in split == test: the balance still drops them.
        out, _ = apply_balance(self.frame, spec(balance={"method": "ratio", "r": 1.0}))
        pos = out[(out["case"] == "positive") & (out["hint_style"] == "metadata")]
        self.assertEqual(int((pos["split"] == "test").sum()), 3)
        self.assertEqual(int((pos["split"] == "train").sum()), 3)

    def test_missing_class_group_untouched_and_reported(self):
        out, report = apply_balance(self.frame, spec(balance={"method": "ratio", "r": 1.0}))
        only_one = out[out["hint_style"] == "consensus"]
        self.assertEqual(len(only_one), 5)
        entry = report["per_group"]["m|r|consensus|positive"]
        self.assertTrue(entry["missing_class"])
        self.assertEqual(entry["after"], {"0": 0, "1": 5})
        self.assertEqual(report["n_groups_missing_class"], 1)

    def test_balance_is_seeded_and_nested(self):
        loose, _ = apply_balance(self.frame, spec(balance={"method": "ratio", "r": 2.0}))
        tight, _ = apply_balance(self.frame, spec(balance={"method": "ratio", "r": 1.0}))
        self.assertTrue(set(tight["rollout_id"]) < set(loose["rollout_id"]))
        again, _ = apply_balance(self.frame, spec(balance={"method": "ratio", "r": 2.0}))
        self.assertEqual(loose["rollout_id"].tolist(), again["rollout_id"].tolist())
        shuffled = self.frame.sample(frac=1.0, random_state=3)
        reordered, _ = apply_balance(shuffled, spec(balance={"method": "ratio", "r": 2.0}))
        self.assertEqual(set(reordered["rollout_id"]), set(loose["rollout_id"]))
        other_seed, _ = apply_balance(self.frame, spec(balance={"method": "ratio", "r": 2.0}, seed=1))
        self.assertNotEqual(set(other_seed["rollout_id"]), set(loose["rollout_id"]))
        self.assertEqual(len(other_seed), len(loose))

    def test_minority_never_dropped(self):
        for r in (1.0, 1.5, 2.0, 10.0):
            out, _ = apply_balance(self.frame, spec(balance={"method": "ratio", "r": r}))
            minority = self.frame[(self.frame["case"] == "positive") & (self.frame["label"] == 0)]
            self.assertTrue(set(minority["rollout_id"]) <= set(out["rollout_id"]))
        # The majority is chosen per group: label 0 is the majority where it outnumbers label 1.
        flipped = self.frame.assign(label=(1 - self.frame["label"]).astype("Int8"))
        out, report = apply_balance(flipped, spec(balance={"method": "ratio", "r": 1.0}))
        self.assertEqual(report["per_group"]["m|r|metadata|positive"]["majority_label"], 0)
        self.assertEqual(self.counts(out)["positive"], {0: 3, 1: 3})

    def test_ratio_on_the_fixture(self):
        rows = select_dataset_rows(MANIFEST, spec(balance={"method": "ratio", "r": 1.0}))
        out, report = apply_balance(rows, spec(balance={"method": "ratio", "r": 1.0}))
        per = out.groupby(list(DEFAULT_BALANCE_GROUPS))["label"].agg(lambda s: (int((s == 0).sum()), int((s == 1).sum())))
        self.assertTrue(all(a == b for a, b in per))
        self.assertEqual(report["n_groups"], 32)
        self.assertEqual(report["labels_after"], {"0": 256, "1": 256})


class FoldRolesTest(unittest.TestCase):
    """fold_roles and fold_report."""

    def setUp(self):
        self.rows = select_dataset_rows(MANIFEST, spec())

    def test_fold_roles_held_out_style_test_only(self):
        roles = fold_roles(self.rows, "metadata")
        self.assertEqual(str(roles.dtype), "string")
        self.assertTrue(roles.index.equals(self.rows.index))
        held = self.rows["hint_style"] == "metadata"
        self.assertTrue(roles[held & (self.rows["split"] != "test")].isna().all())
        self.assertEqual(set(roles[held & (self.rows["split"] == "test")]), {"test_ood"})
        self.assertGreater(int((roles == "test_ood").sum()), 0)

    def test_fold_roles_other_styles_follow_split(self):
        roles = fold_roles(self.rows, "metadata")
        other = self.rows["hint_style"] != "metadata"
        expected = self.rows.loc[other, "split"].astype(str).map({"train": "train", "val": "val", "test": "test_id"})
        self.assertEqual(roles[other].astype(str).tolist(), expected.tolist())
        self.assertTrue(roles[other].notna().all())
        self.assertEqual(set(roles.dropna()), set(FOLD_ROLES))

    def test_fold_roles_unknown_style_raises(self):
        with self.assertRaisesRegex(ValueError, "pushback.*styles present"):
            fold_roles(self.rows, "pushback")
        with self.assertRaisesRegex(ValueError, "split"):
            fold_roles(self.rows.assign(split=None), "metadata")

    def test_fold_report_counts_and_question_consistency(self):
        report = fold_report(self.rows, PROBE_STYLES)
        self.assertEqual(set(report), set(PROBE_STYLES))
        for style, entry in report.items():
            roles = fold_roles(self.rows, style)
            self.assertEqual(set(entry["roles"]), set(FOLD_ROLES))
            for role, counts in entry["roles"].items():
                g = self.rows[(roles == role).fillna(False).to_numpy()]
                self.assertEqual(counts["n_rows"], len(g))
                self.assertEqual(counts["n_questions"], g["question_id"].nunique())
                self.assertEqual(counts["n_label_0"] + counts["n_label_1"], len(g))
                self.assertEqual(sum(counts["per_case"].values()), len(g))
                self.assertEqual(sum(counts["per_subject_model"].values()), len(g))
            self.assertEqual(entry["n_dropped"], int(roles.isna().sum()))
            self.assertTrue(entry["train_val_vs_test_disjoint"])
            self.assertTrue(entry["ood_questions_in_test_split"])
            self.assertTrue(entry["question_consistency"])
            self.assertTrue(entry["ood_has_both_labels"])
        # In the fixture a question lives under one style, so test_ood questions are never test_id ones —
        # informational only (the resample set is drawn per cell), never a consistency failure.
        self.assertFalse(report["metadata"]["ood_questions_subset_of_id"])
        self.assertEqual(report["metadata"]["n_ood_only_questions"], report["metadata"]["roles"]["test_ood"]["n_questions"])
        # A test_ood question hand-moved into a train row: the split stage's one-split-per-question rule
        # refuses the frame outright (fold_roles asserts it) ...
        roles = fold_roles(self.rows, "metadata")
        ood_row = self.rows[(roles == "test_ood").fillna(False).to_numpy()].iloc[[0]]
        moved = ood_row.assign(rollout_id=ood_row["rollout_id"] + "#moved", hint_style="consensus", split="train")
        with self.assertRaisesRegex(ValueError, "straddle"):
            fold_report(pd.concat([self.rows, moved], ignore_index=True), ["metadata"])
        # ... and on the checks themselves (hand-built roles) the overlap breaks question_consistency.
        broken = pd.concat([self.rows, moved], ignore_index=True)
        broken_roles = pd.Series(pd.array(list(roles.astype(object).where(roles.notna(), None)) + ["train"], dtype="string"),
                                 index=broken.index)
        checks = question_checks(broken, broken_roles)
        self.assertFalse(checks["train_val_vs_test_disjoint"])
        self.assertFalse(checks["question_consistency"])
        self.assertTrue(checks["ood_questions_in_test_split"])
        # A test_ood row outside split test breaks (b) on its own.
        bad_split = self.rows.copy()
        bad_split.loc[ood_row.index, "split"] = "val"
        self.assertFalse(question_checks(bad_split, roles)["ood_questions_in_test_split"])
        self.assertFalse(question_checks(bad_split, roles)["question_consistency"])
        # The checks align roles by index, so a reordered frame gives the same answer; a foreign roles
        # series is refused rather than read positionally.
        self.assertEqual(question_checks(self.rows.sample(frac=1.0, random_state=9), roles),
                         question_checks(self.rows, roles))
        with self.assertRaisesRegex(ValueError, "not aligned"):
            question_checks(self.rows, roles.iloc[1:])
        # A fold whose held-out test rows carry one label only is flagged.
        one_label = self.rows[~((self.rows["hint_style"] == "metadata") & (self.rows["label"] == 0))]
        self.assertFalse(fold_report(one_label, ["metadata"])["metadata"]["ood_has_both_labels"])


class FingerprintTest(unittest.TestCase):
    """dataset_fingerprint."""

    def test_fingerprint_is_order_independent(self):
        rows = select_dataset_rows(MANIFEST, spec())
        fp = dataset_fingerprint(rows)
        self.assertEqual(len(fp), 64)
        self.assertEqual(dataset_fingerprint(rows.sample(frac=1.0, random_state=5)), fp)
        self.assertEqual(dataset_fingerprint(rows.iloc[::-1]), fp)
        self.assertEqual(dataset_fingerprint(rows.drop(columns=["case", "run"])), fp)  # other columns are not hashed
        self.assertNotEqual(dataset_fingerprint(rows.iloc[1:]), fp)

    def test_fingerprint_tracks_label_and_split(self):
        rows = select_dataset_rows(MANIFEST, spec())
        fp = dataset_fingerprint(rows)
        relabelled = rows.copy()
        relabelled.loc[relabelled.index[0], "label"] = 1 - int(relabelled["label"].iloc[0])
        self.assertNotEqual(dataset_fingerprint(relabelled), fp)
        resplit = rows.copy()
        resplit.loc[resplit.index[0], "split"] = "val" if rows["split"].iloc[0] != "val" else "test"
        self.assertNotEqual(dataset_fingerprint(resplit), fp)


if __name__ == "__main__":
    unittest.main()
