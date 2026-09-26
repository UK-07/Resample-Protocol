"""Tests for src/lib/resample.py — the re-sampling design's GPU-free logic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib.resample import (
    ITEM_CHOICES_COLUMN,
    ITEM_COLUMNS,
    MATCH_KEYS,
    SELECTION_COLUMNS,
    USED_IGNORED_CLASSES,
    VERBALISED_REST_CLASSES,
    assign_contrasts,
    extract_items,
    filter_manifest,
    noise_model,
    noise_model_by_stability,
    reliance_label,
    reliance_labels,
    select_resample_set,
    stability_bin,
    stability_summary,
    survival_table,
    wilson_interval,
)


def manifest_rows(n_to_hint: int, n_off: int, n_kept: int, *, hint="metadata", case="positive",
                  dataset="medqa", stability=8, run="medqa-test_0911", model="m", start=0) -> pd.DataFrame:
    """A manifest slice: ``n_to_hint`` to-target (judged 0/1 alternating), ``n_off`` third-option, ``n_kept`` unchanged."""
    rows = []
    idx = start
    for kind, n in (("to_hint", n_to_hint), ("off", n_off), ("kept", n_kept)):
        for i in range(n):
            answer = {"to_hint": "B", "off": "C", "kept": "A"}[kind]
            rows.append({
                "rollout_id": f"{model}:{run}:{hint}:{idx}", "question_id": f"{dataset}:test:{idx}",
                "dataset": dataset, "dataset_split": "test", "original_index": idx,
                "subject_model": model, "subject_model_id": "org/m", "run": run,
                "hint_style": hint, "case": case, "target_option": "B", "groundtruth": "A",
                "baseline_modal_answer": "A", "baseline_stability": stability, "baseline_n_samples": 8,
                "baseline_hint_votes": 1 if (kind == "to_hint" and i == 0) else 0,
                "model_answer": answer, "changed": kind != "kept", "to_hint": kind == "to_hint",
                "judge_model": "j" if kind == "to_hint" else None, "judge_prompt": None,
                "judge_label": (i % 2) if kind == "to_hint" else None,
                "judge_label_final": (i % 2) if kind == "to_hint" else None,
                "judge_label_final_model": "j" if kind == "to_hint" else None,
                "judge_confidence": None, "judge_role": None, "hint_quote": None,
                "trace_token_len": 100, "truncated": False, "max_tokens_used": 16384,
                "exclude_reason": None, "split": None, "source_csv": f"{model}_{run}_judged.csv",
            })
            idx += 1
    return pd.DataFrame(rows)


class FilterAndBinTest(unittest.TestCase):
    def test_filter_drops_smoke_runs_and_dropped_hints(self):
        df = pd.concat([
            manifest_rows(1, 0, 0, run="medqa-test_0911"),
            manifest_rows(1, 0, 0, run="medqa-test_smoke", start=10),
            manifest_rows(1, 0, 0, hint="pushback", start=20),
            manifest_rows(1, 0, 0, hint="authority", start=30),
        ], ignore_index=True)
        kept = filter_manifest(df)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept["run"].iloc[0], "medqa-test_0911")

    def test_stability_bins(self):
        bins = stability_bin(pd.Series([8, 7, 6, 5, 4, None], dtype="Int8"))
        self.assertEqual(list(bins.astype(object).where(bins.notna(), None)), ["8", "7", "6", "<=5", "<=5", None])
        self.assertTrue(bins.cat.ordered)


class NoiseModelTest(unittest.TestCase):
    def test_estimate_uses_off_target_flips_over_n_minus_2(self):
        # 4 options: 60 to target, 32 off target -> 32/2 = 16 noise-shaped switches.
        df = manifest_rows(60, 32, 100)
        nm = noise_model(df)
        self.assertEqual(len(nm), 1)
        row = nm.iloc[0]
        self.assertEqual(row["n_rollouts"], 192)
        self.assertEqual(row["n_changed"], 92)
        self.assertEqual(row["n_to_hint"], 60)
        self.assertEqual(row["n_off_target"], 32)
        self.assertEqual(row["n_options"], 4)
        self.assertAlmostEqual(row["noise_flip_estimate"], 16.0)
        self.assertAlmostEqual(row["noise_share_of_to_hint"], 16 / 60)
        self.assertAlmostEqual(row["to_hint_rate"], 60 / 192)
        chance = 16 / 192
        self.assertAlmostEqual(row["to_hint_rate_excess"], (60 / 192 - chance) / (1 - chance))
        self.assertAlmostEqual(row["prior_target_rate"], 1 / (192 * 8))
        self.assertEqual(row["n_unfaithful"], 30)
        self.assertEqual(row["n_faithful"], 30)
        self.assertEqual(row["n_noise_flagged"], 1)

    def test_ten_option_dataset_divides_by_eight(self):
        df = manifest_rows(60, 32, 0, dataset="mmlu_pro")
        self.assertAlmostEqual(noise_model(df).iloc[0]["noise_flip_estimate"], 4.0)

    def test_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            noise_model(manifest_rows(1, 0, 0, dataset="riddles"))

    def test_stratified_and_pooled_tables(self):
        df = pd.concat([
            manifest_rows(10, 2, 10, stability=8),
            manifest_rows(10, 6, 4, stability=6, start=100),
            manifest_rows(10, 6, 4, stability=6, dataset="mmlu_pro", run="pro", start=200),
        ], ignore_index=True)
        strat = noise_model_by_stability(df)
        self.assertEqual(set(strat.index.get_level_values("stability_bin")), {"8", "6"})
        pooled = stability_summary(df)
        six = pooled.xs("6", level="stability_bin").iloc[0]
        # Pooled over a 4- and a 10-option run: 6/2 + 6/8 noise-shaped switches.
        self.assertAlmostEqual(six["noise_flip_estimate"], 3.0 + 0.75)
        self.assertTrue(pd.isna(six["n_options"]))
        self.assertEqual(six["n_to_hint"], 20)


class SelectionTest(unittest.TestCase):
    def test_used_candidates_and_matched_controls(self):
        df = pd.concat([
            manifest_rows(5, 1, 20, stability=8),
            manifest_rows(3, 0, 1, stability=6, start=100),
        ], ignore_index=True)
        selection, report = select_resample_set(df, seed=1)
        self.assertEqual(list(selection.columns), SELECTION_COLUMNS)
        used = selection[selection["role"] == "used_candidate"]
        ctrl = selection[selection["role"] == "control"]
        self.assertEqual(len(used), 8)
        self.assertTrue(used["rollout_id"].isin(df[df["to_hint"]]["rollout_id"]).all())
        # 8/8 cell: 5 controls of the 20 kept; 6/8 cell: only 1 kept -> shortfall 2.
        self.assertEqual(len(ctrl), 6)
        self.assertTrue((df.set_index("rollout_id").loc[ctrl["rollout_id"], "model_answer"] == "A").all())
        rep = report.set_index("stability_bin")
        self.assertEqual(rep.loc["8", "n_control"], 5)
        self.assertEqual(rep.loc["6", "control_shortfall"], 2)
        self.assertEqual(list(report.columns[:5]), MATCH_KEYS)

    def test_draw_is_seeded(self):
        df = manifest_rows(3, 0, 30)
        a, _ = select_resample_set(df, seed=7)
        b, _ = select_resample_set(df, seed=7)
        c, _ = select_resample_set(df, seed=8)
        self.assertEqual(list(a["rollout_id"]), list(b["rollout_id"]))
        self.assertNotEqual(list(a["rollout_id"]), list(c["rollout_id"]))

    def test_truncated_and_unanswered_rows_are_never_controls(self):
        df = manifest_rows(2, 0, 4)
        df.loc[df["model_answer"] == "A", "truncated"] = [True, False, False, False]
        df.loc[df.index[-1], "model_answer"] = None
        selection, _ = select_resample_set(df, seed=1)
        ctrl = selection[selection["role"] == "control"]
        self.assertEqual(len(ctrl), 2)


class ItemsTest(unittest.TestCase):
    def test_extract_items_keeps_selected_pairs_only(self):
        df = manifest_rows(2, 0, 2)
        selection, _ = select_resample_set(df, seed=1)
        source = pd.DataFrame({
            "original_index": [0, 1, 2, 3, 9], "sample_type": "positive",
            "hint_name": ["metadata"] * 4 + ["consensus"], "prompt": "q", "hinted_prompt": "hq",
            "hinted_answer": "B", "baseline_answer": "A", "groundtruth": "A",
            "rollout": "<think>x</think><answer>B</answer>", "reasoning": "x", "final_answer": "B",
            "additional_fields": "{}", "judge_label": "",
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m_medqa-test_0911_judged.csv"
            source.to_csv(path, index=False)
            items = extract_items(path, selection, chunk_rows=2)
            self.assertEqual(list(items.columns), ITEM_COLUMNS)
            self.assertEqual(sorted(items["original_index"]), [0, 1, 2, 3])
            self.assertEqual(set(items["role"]), {"used_candidate", "control"})
            self.assertEqual(set(items["n_options"]), {4})
            with self.assertRaises(ValueError):
                extract_items(path, selection.assign(original_index=[50, 51, 52, 53]))
            # With the source's baseline the items also carry the option texts
            # (the parser's labelled-tag contradiction check on the re-rolls).
            baseline = Path(tmp) / "m_medqa-test_0911_baseline.csv"
            pd.DataFrame({
                "original_index": [0, 1, 2, 3], "choices": ['["a","b","c","d"]'] * 4, "prompt": "q",
            }).to_csv(baseline, index=False)
            items = extract_items(path, selection, chunk_rows=2, baseline_csv=baseline)
            self.assertEqual(list(items.columns), ITEM_COLUMNS + [ITEM_CHOICES_COLUMN])
            self.assertEqual(set(items[ITEM_CHOICES_COLUMN]), {'["a","b","c","d"]'})


def rerolls(source_ids: list[str], answers: dict[str, list[str | None]], labels: dict[str, list] | None = None,
            *, seeds=(43, 44, 45, 46)) -> pd.DataFrame:
    """Manifest-shaped re-roll rows: per source id, one answer per seed."""
    rows = []
    for sid in source_ids:
        for j, seed in enumerate(seeds[: len(answers[sid])]):
            ans = answers[sid][j]
            lab = (labels or {}).get(sid, [None] * 4)[j]
            rows.append({
                "rollout_id": f"{sid}#rs{seed}", "source_rollout_id": sid, "sample_seed": seed,
                "model_answer": ans, "baseline_modal_answer": "A", "target_option": "B",
                "to_hint": ans == "B", "changed": ans is not None and ans != "A",
                "truncated": ans is None, "judge_label": lab, "judge_label_final": lab,
            })
    return pd.DataFrame(rows)


class RelianceTest(unittest.TestCase):
    def test_label_rules(self):
        self.assertEqual(reliance_label("used_candidate", 4, 4, 4, 0), "robust_used")
        self.assertEqual(reliance_label("used_candidate", 4, 4, 3, 1), "robust_used")
        self.assertEqual(reliance_label("used_candidate", 4, 4, 2, 2), "weak_used")
        self.assertEqual(reliance_label("used_candidate", 4, 4, 1, 3), "weak_used")
        self.assertEqual(reliance_label("used_candidate", 4, 4, 0, 4), "mixed")
        self.assertEqual(reliance_label("control", 4, 4, 0, 4), "robust_ignored")
        self.assertEqual(reliance_label("control", 4, 4, 0, 3), "robust_ignored")
        self.assertEqual(reliance_label("control", 4, 4, 0, 2), "mixed")
        self.assertEqual(reliance_label("control", 4, 4, 1, 3), "mixed")
        self.assertIsNone(reliance_label("control", 3, 4, 0, 3))
        with self.assertRaises(ValueError):
            reliance_label("judge", 4, 4, 0, 0)

    def test_reliance_labels_counts(self):
        df = manifest_rows(2, 0, 1)
        selection, _ = select_resample_set(df, seed=1)
        ids = list(selection["rollout_id"])  # used: idx 0, 1; control: idx 2
        used0, used1 = [i for i in ids if i.endswith(":0")][0], [i for i in ids if i.endswith(":1")][0]
        ctrl = [i for i in ids if i.endswith(":2")][0]
        res = rerolls(ids, {used0: ["B", "B", "B", "C"], used1: ["B", None, "A", "A"], ctrl: ["A", "A", "A", "C"]},
                      {used0: [0, 1, 1, None]})
        q = reliance_labels(selection, res, k=4).set_index("rollout_id")
        self.assertEqual(q.loc[used0, "reliance_label"], "robust_used")
        self.assertEqual(q.loc[used0, "k_to_hint_count"], 3)
        self.assertEqual(q.loc[used0, "k_other_count"], 1)
        self.assertEqual(q.loc[used0, "k_judged"], 3)
        self.assertEqual(q.loc[used0, "k_unfaithful"], 1)
        self.assertEqual(q.loc[used1, "reliance_label"], "weak_used")
        self.assertEqual(q.loc[used1, "k_truncated"], 1)
        self.assertEqual(q.loc[used1, "k_modal_count"], 2)
        self.assertEqual(q.loc[ctrl, "reliance_label"], "robust_ignored")

    def test_incomplete_rerolls_stay_unlabeled(self):
        df = manifest_rows(1, 0, 0)
        selection, _ = select_resample_set(df, seed=1)
        sid = selection["rollout_id"].iloc[0]
        q = reliance_labels(selection, rerolls([sid], {sid: ["B", "B"]}), k=4)
        self.assertEqual(q["k_n"].iloc[0], 2)
        self.assertTrue(pd.isna(q["reliance_label"].iloc[0]))


class ContrastTest(unittest.TestCase):
    def test_policy(self):
        df = manifest_rows(3, 0, 1)
        selection, _ = select_resample_set(df, seed=1)
        ids = {sid.rsplit(":", 1)[1]: sid for sid in selection["rollout_id"]}
        res = rerolls(
            list(ids.values()),
            {ids["0"]: ["B", "B", "B", "A"],   # robust_used, one modal re-roll -> paired_a
             ids["1"]: ["B", "A", "A", "A"],   # weak_used
             ids["2"]: ["A", "A", "A", "A"],   # used candidate that never re-flips -> mixed
             ids["3"]: ["A", "A", "A", "C"]},  # control -> robust_ignored
            {ids["0"]: [0, 1, 0, None], ids["1"]: [1, None, None, None]},
        )
        questions = reliance_labels(selection, res, k=4)
        original = df[df["rollout_id"].isin(selection["rollout_id"])].assign(
            source_rollout_id=lambda d: d["rollout_id"])
        rollouts = assign_contrasts(pd.concat([original, res], ignore_index=True), questions)
        by = rollouts.set_index("rollout_id")

        # Contrast A: robust_used to-target rows train as used (original + 3 re-rolls).
        used_train = rollouts[(rollouts["contrast_a"] == "used") & (rollouts["contrast_a_set"] == "train")]
        self.assertEqual(sorted(used_train["rollout_id"]), sorted([ids["0"]] + [f"{ids['0']}#rs{s}" for s in (43, 44, 45)]))
        # ... its one modal re-roll is an ignored row in the challenge set, and the pair is flagged.
        self.assertEqual(by.loc[f"{ids['0']}#rs46", "contrast_a"], "ignored")
        self.assertEqual(by.loc[f"{ids['0']}#rs46", "contrast_a_set"], "challenge")
        self.assertTrue(bool(by.loc[f"{ids['0']}#rs46", "paired_a"]))
        self.assertTrue(bool(by.loc[ids["0"], "paired_a"]))
        # robust_ignored modal rows train as ignored; the third-option re-roll is nothing.
        ign_train = rollouts[(rollouts["contrast_a"] == "ignored") & (rollouts["contrast_a_set"] == "train")]
        self.assertEqual(sorted(ign_train["rollout_id"]), sorted([ids["3"]] + [f"{ids['3']}#rs{s}" for s in (43, 44, 45)]))
        self.assertTrue(pd.isna(by.loc[f"{ids['3']}#rs46", "contrast_a"]))
        self.assertFalse(bool(by.loc[ids["3"], "paired_a"]))
        # weak_used and mixed questions only feed the challenge set.
        for sid in (ids["1"], ids["2"]):
            rows = rollouts[rollouts["source_rollout_id"] == sid]
            self.assertTrue((rows["contrast_a_set"].dropna() == "challenge").all())
            self.assertGreater(rows["contrast_a"].notna().sum(), 0)
        # Contrast B: judged to-target rollouts of robust_used questions; both verdicts -> paired.
        b = rollouts[rollouts["contrast_b"].fillna(False)]
        self.assertEqual(sorted(b["rollout_id"]), sorted([ids["0"], f"{ids['0']}#rs43", f"{ids['0']}#rs44", f"{ids['0']}#rs45"]))
        self.assertTrue(b["paired_b"].all())
        self.assertFalse(rollouts.loc[rollouts["source_rollout_id"] == ids["1"], "contrast_b"].any())


def label_config_frame() -> tuple[pd.DataFrame, dict[str, str]]:
    """``assign_contrasts`` output over a positive and a negative cell with ``provenance`` set → ``(rollouts, ids)``."""
    df = pd.concat([manifest_rows(3, 0, 2), manifest_rows(1, 0, 1, case="negative", start=10)], ignore_index=True)
    selection, _ = select_resample_set(df, seed=1)
    ids = {sid.rsplit(":", 1)[1]: sid for sid in selection["rollout_id"]}
    assert set(ids) == {"0", "1", "2", "3", "4", "10", "11"}, sorted(ids)
    res = rerolls(
        list(ids.values()),
        {ids["0"]: ["B", "B", "B", "A"], ids["1"]: ["B", "B", "B", "B"], ids["2"]: ["B", "A", "A", "A"],
         ids["3"]: ["A", "A", "A", "C"], ids["4"]: ["A", "A", "C", "C"],
         ids["10"]: ["B", "B", "B", "B"], ids["11"]: ["A", "A", "A", "A"]},
        {ids["0"]: [1, 0, -1, None], ids["1"]: [None, 1, 0, 1], ids["2"]: [1, None, None, None],
         ids["10"]: [0, 0, 1, 1]},
    )
    questions = reliance_labels(selection, res, k=4)
    original = df[df["rollout_id"].isin(selection["rollout_id"])].assign(
        source_rollout_id=lambda d: d["rollout_id"], provenance="hinted_once")
    res["provenance"] = "resample_k4"
    for col in original.columns:
        if col not in res.columns:
            res[col] = original.set_index("rollout_id").loc[res["source_rollout_id"], col].to_numpy()
    # The production path (derive_rollout_rows) stamps the judge's -1 as exclude_reason == incoherent.
    res["exclude_reason"] = res["exclude_reason"].astype(object).where(res["judge_label_final"] != -1, "incoherent")
    rollouts = assign_contrasts(pd.concat([original, res], ignore_index=True), questions)
    return rollouts, ids


class LabelConfigColumnsTest(unittest.TestCase):
    def setUp(self):
        self.rollouts, self.ids = label_config_frame()
        self.by = self.rollouts.set_index("rollout_id")

    def vr(self, sid, seed=None):
        return self.by.loc[f"{sid}#rs{seed}" if seed else sid, "label_verbalised_rest"]

    def ui(self, sid, seed=None):
        return self.by.loc[f"{sid}#rs{seed}" if seed else sid, "label_used_ignored"]

    def test_label_verbalised_rest_classes(self):
        self.assertEqual(VERBALISED_REST_CLASSES, ("verbalised", "rest"))
        q0, q1, q2, q3, q10 = (self.ids[k] for k in ("0", "1", "2", "3", "10"))
        self.assertEqual(self.vr(q0, 43), "verbalised")     # used & 1
        self.assertEqual(self.vr(q0, 44), "rest")           # used & 0
        self.assertTrue(pd.isna(self.vr(q0, 45)))           # used & -1
        self.assertTrue(pd.isna(self.vr(q1, 43)))           # used & null verdict
        self.assertEqual(self.vr(q1, 44), "verbalised")
        self.assertEqual(self.vr(q1, 45), "rest")
        for seed in (43, 44, 45):
            self.assertEqual(self.vr(q3, seed), "rest")     # ignored
        self.assertEqual([self.vr(q10, s) for s in (43, 44, 45, 46)], ["rest", "rest", "verbalised", "verbalised"])
        # weak_used / mixed questions never enter, whatever their rows did.
        for sid in (q2, self.ids["4"]):
            rows = self.rollouts[self.rollouts["source_rollout_id"] == sid]
            self.assertTrue(rows["label_verbalised_rest"].isna().all())
            self.assertTrue(rows["label_used_ignored"].isna().all())
        # The hinted-once original is never a dataset row.
        originals = self.rollouts[self.rollouts["provenance"] == "hinted_once"]
        self.assertEqual(len(originals), 7)
        self.assertTrue(originals["label_verbalised_rest"].isna().all())
        self.assertTrue(originals["label_used_ignored"].isna().all())
        self.assertTrue(self.rollouts["label_verbalised_rest"].dropna().isin(VERBALISED_REST_CLASSES).all())

    def test_label_used_ignored_is_verdict_independent(self):
        self.assertEqual(USED_IGNORED_CLASSES, ("used", "ignored"))
        q0, q1, q3, q11 = (self.ids[k] for k in ("0", "1", "3", "11"))
        # Verdict-independent: the -1 used row and the unjudged one are both "used".
        self.assertEqual(self.ui(q0, 45), "used")   # verdict -1
        self.assertEqual(self.by.loc[f"{q0}#rs45", "exclude_reason"], "incoherent")
        self.assertEqual(self.ui(q1, 43), "used")   # verdict null
        self.assertTrue(pd.isna(self.by.loc[f"{q1}#rs43", "exclude_reason"]))
        for seed in (43, 44):
            self.assertEqual(self.ui(q0, seed), "used")
        for seed in (43, 44, 45, 46):
            self.assertEqual(self.ui(q1, seed), "used")
            self.assertEqual(self.ui(q11, seed), "ignored")
        for seed in (43, 44, 45):
            self.assertEqual(self.ui(q3, seed), "ignored")
        self.assertTrue(self.rollouts["label_used_ignored"].dropna().isin(USED_IGNORED_CLASSES).all())

    def test_ignored_requires_modal_answer(self):
        # q3 is robust_ignored (3/4 modal); its third-option re-roll (seed 46) is neither class.
        q3 = self.ids["3"]
        self.assertEqual(self.by.loc[f"{q3}#rs46", "reliance_label"], "robust_ignored")
        self.assertEqual(self.by.loc[f"{q3}#rs46", "model_answer"], "C")
        self.assertTrue(pd.isna(self.vr(q3, 46)))
        self.assertTrue(pd.isna(self.ui(q3, 46)))

    def test_used_requires_to_hint(self):
        # q0 is robust_used (3/4 to target); its off-pattern modal re-roll (seed 46) is neither class.
        q0 = self.ids["0"]
        self.assertEqual(self.by.loc[f"{q0}#rs46", "reliance_label"], "robust_used")
        self.assertFalse(bool(self.by.loc[f"{q0}#rs46", "to_hint"]))
        self.assertTrue(pd.isna(self.vr(q0, 46)))
        self.assertTrue(pd.isna(self.ui(q0, 46)))

    def test_reroll_detection_without_provenance_column(self):
        # Without ``provenance`` (or ``is_resample``) a re-roll is a row whose id differs from its source's.
        rollouts, ids = label_config_frame()
        again = assign_contrasts(rollouts.drop(columns=["provenance", "label_verbalised_rest", "label_used_ignored"]),
                                 rollouts.drop_duplicates("source_rollout_id")[["source_rollout_id", "reliance_label"]]
                                 .rename(columns={"source_rollout_id": "rollout_id"}))
        for col in ("label_verbalised_rest", "label_used_ignored"):
            pd.testing.assert_series_equal(again[col], rollouts[col], check_names=False)


class SurvivalTest(unittest.TestCase):
    def test_table_and_interval(self):
        df = pd.concat([manifest_rows(4, 0, 2), manifest_rows(2, 0, 2, hint="consensus", start=50)], ignore_index=True)
        selection, _ = select_resample_set(df, seed=1)
        answers = {}
        for sid in selection["rollout_id"]:
            idx = int(sid.rsplit(":", 1)[1])
            role = selection.set_index("rollout_id").loc[sid, "role"]
            if role == "control":
                answers[sid] = ["A"] * 4
            else:
                answers[sid] = ["B", "B", "B", "B"] if idx % 2 == 0 else ["B", "A", "A", "A"]
        questions = reliance_labels(selection, rerolls(list(answers), answers), k=4)
        table = survival_table(questions).set_index("hint_style")
        self.assertEqual(table.loc["metadata", "n_single_used"], 4)
        self.assertEqual(table.loc["metadata", "n_robust_used"], 2)
        self.assertEqual(table.loc["metadata", "n_weak_used"], 2)
        self.assertAlmostEqual(table.loc["metadata", "survival_rate"], 0.5)
        self.assertEqual(table.loc["metadata", "n_robust_ignored"], 2)
        self.assertAlmostEqual(table.loc["metadata", "ignored_rate"], 1.0)
        self.assertEqual(table.loc["consensus", "n_single_used"], 2)
        pooled = survival_table(questions, [])
        self.assertEqual(pooled["n_single_used"].iloc[0], 6)
        lo, hi = wilson_interval(3, 6)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertTrue(np.isnan(wilson_interval(0, 0)[0]))


if __name__ == "__main__":
    unittest.main()
