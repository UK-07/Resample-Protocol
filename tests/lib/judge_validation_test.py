"""Tests for src/lib/judge_validation.py — sampling, metrics, decision rules, gap audit, review."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.lib.judge_validation import (
    CELL_KEYS,
    agreement_block,
    agreement_breakdowns,
    allocate_quota,
    apply_decision_rules,
    assign_strata,
    build_review_sheet,
    cell_counts,
    cohen_kappa,
    draw_sample,
    gap_audit,
    plan_quotas,
    population_rates,
    render_report,
    score_review,
    weighted_rate_bootstrap,
)


def make_frame(n_per_cell: dict, *, model="nemotron", run="mmlu_pro") -> pd.DataFrame:
    """A frame with ``n_per_cell[(stratum, style)]`` rows in every cell."""
    rows = []
    for (stratum, style), n in n_per_cell.items():
        label = 0 if stratum.startswith("unfaithful") else 1
        mention = 1 if stratum == "unfaithful_mention" else 0
        conf = 0.6 if stratum == "faithful_low_conf" else 0.95
        for i in range(n):
            rows.append({
                "rollout_id": f"{model}:{run}:{style}:{stratum}:{i}", "subject_model": model, "run": run,
                "hint_style": style, "case": "positive" if i % 2 == 0 else "negative",
                "judge_label": label, "mention": mention if label == 0 else np.nan, "judge_confidence": conf,
            })
    frame = pd.DataFrame(rows)
    frame["stratum"] = assign_strata(frame)
    return frame


class StrataTest(unittest.TestCase):
    def test_assignment_by_label_mention_and_confidence(self):
        frame = pd.DataFrame({
            "judge_label": [0, 0, 0, 1, 1, 1, -1, None],
            "mention": [1, 0, np.nan, np.nan, np.nan, np.nan, 1, 0],
            "judge_confidence": [0.9, 0.9, 0.9, 0.7, 0.71, np.nan, 0.9, 0.9],
        })
        self.assertEqual(list(assign_strata(frame)), [
            "unfaithful_mention", "unfaithful_no_mention", "unfaithful_no_mention",
            "faithful_low_conf", "faithful_high_conf", "faithful_high_conf", None, None,
        ])

    def test_cell_counts_keys(self):
        frame = make_frame({("unfaithful_mention", "post_hoc"): 3, ("faithful_high_conf", "post_hoc"): 2})
        counts = cell_counts(frame)
        self.assertEqual(list(counts.columns), CELL_KEYS + ["n_available"])
        self.assertEqual(counts["n_available"].tolist(), [2, 3])


class QuotaTest(unittest.TestCase):
    def test_take_everything_when_short(self):
        self.assertEqual(allocate_quota({"a": 3, "b": 4}, 10), {"a": 3, "b": 4})

    def test_proportional_with_weights(self):
        # Shares follow weight x availability: b counts double.
        take = allocate_quota({"a": 100, "b": 20, "c": 50}, 60, weights={"b": 2.0})
        self.assertEqual(sum(take.values()), 60)
        self.assertEqual(take, {"a": 31, "b": 13, "c": 16})
        plain = allocate_quota({"a": 100, "b": 20, "c": 50}, 60)
        self.assertEqual(plain, {"a": 34, "b": 8, "c": 18})

    def test_every_cell_is_covered_when_the_target_allows(self):
        # A cell drawn 0 times would have no design weight and vanish from the reweighted rates.
        take = allocate_quota({"big": 1000, "small": 3}, 20)
        self.assertEqual(take["small"], 1)
        self.assertEqual(sum(take.values()), 20)
        take = allocate_quota({"a": 10, "b": 10, "c": 10}, 2)  # fewer draws than cells: coverage is impossible
        self.assertEqual(sum(take.values()), 2)

    def test_overflowing_cell_is_capped_and_the_rest_reshared(self):
        take = allocate_quota({"a": 100, "b": 5, "c": 50}, 140, weights={"b": 2.0})
        self.assertEqual(take, {"a": 90, "b": 5, "c": 45})

    def test_floor_keeps_rare_cells(self):
        take = allocate_quota({"a": 1000, "b": 2, "c": 1}, 20, min_per_cell=2)
        self.assertEqual((take["b"], take["c"]), (2, 1))
        self.assertEqual(sum(take.values()), 20)

    def test_floors_exceeding_target_fall_back_to_proportional(self):
        take = allocate_quota({"a": 10, "b": 10, "c": 10}, 4, min_per_cell=3)
        self.assertEqual(sum(take.values()), 4)

    def test_zero_or_empty(self):
        self.assertEqual(allocate_quota({"a": 5}, 0), {})
        self.assertEqual(allocate_quota({}, 5), {})

    def test_plan_spills_shortfall_between_partner_strata(self):
        frame = make_frame({
            ("unfaithful_mention", "post_hoc"): 20, ("unfaithful_mention", "consensus"): 10,
            ("unfaithful_no_mention", "post_hoc"): 200, ("unfaithful_no_mention", "metadata"): 100,
            ("faithful_low_conf", "post_hoc"): 500, ("faithful_high_conf", "post_hoc"): 500,
        })
        plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 120, "unfaithful_no_mention": 60,
                                                 "faithful_low_conf": 80, "faithful_high_conf": 40})
        by_stratum = plan.groupby("stratum")["n_take"].sum().to_dict()
        # 30 available in the mention stratum: all taken, the 90 shortfall spills into no_mention.
        self.assertEqual(by_stratum["unfaithful_mention"], 30)
        self.assertEqual(by_stratum["unfaithful_no_mention"], 150)
        self.assertEqual(by_stratum["faithful_low_conf"], 80)
        self.assertEqual(by_stratum["faithful_high_conf"], 40)
        self.assertTrue((plan["n_take"] <= plan["n_available"]).all())

    def test_plan_oversamples_named_styles(self):
        frame = make_frame({("unfaithful_mention", "post_hoc"): 100, ("unfaithful_mention", "consensus"): 100})
        plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 60, "unfaithful_no_mention": 0,
                                                 "faithful_low_conf": 0, "faithful_high_conf": 0},
                           oversample={"unfaithful_mention": ("post_hoc",)}, oversample_factor=2.0, spill=())
        take = plan.set_index("hint_style")["n_take"].to_dict()
        self.assertEqual(take, {"consensus": 20, "post_hoc": 40})


class DrawTest(unittest.TestCase):
    def test_draw_is_seeded_per_cell_and_weighted(self):
        frame = make_frame({("unfaithful_mention", "post_hoc"): 50, ("faithful_high_conf", "post_hoc"): 10})
        plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 10, "unfaithful_no_mention": 0,
                                                 "faithful_low_conf": 0, "faithful_high_conf": 5}, spill=())
        a = draw_sample(frame, plan, seed=1)
        b = draw_sample(frame, plan, seed=1)
        c = draw_sample(frame, plan, seed=2)
        self.assertEqual(a["rollout_id"].tolist(), b["rollout_id"].tolist())
        self.assertNotEqual(a["rollout_id"].tolist(), c["rollout_id"].tolist())
        self.assertEqual(len(a), 15)
        self.assertFalse(a["rollout_id"].duplicated().any())
        weights = a.groupby("stratum")["design_weight"].first().to_dict()
        self.assertAlmostEqual(weights["unfaithful_mention"], 5.0)
        self.assertAlmostEqual(weights["faithful_high_conf"], 2.0)
        # Changing another cell's target leaves this cell's draw untouched.
        plan2 = plan.copy()
        plan2.loc[plan2["stratum"] == "faithful_high_conf", "n_take"] = 3
        d = draw_sample(frame, plan2, seed=1)
        self.assertEqual(sorted(d[d["stratum"] == "unfaithful_mention"]["rollout_id"]),
                         sorted(a[a["stratum"] == "unfaithful_mention"]["rollout_id"]))
        # Raising a cell's own target only adds rows: verdicts already bought stay in the sample.
        plan3 = plan.copy()
        plan3.loc[plan3["stratum"] == "unfaithful_mention", "n_take"] = 20
        e = draw_sample(frame, plan3, seed=1)
        self.assertTrue(set(a[a["stratum"] == "unfaithful_mention"]["rollout_id"])
                        <= set(e[e["stratum"] == "unfaithful_mention"]["rollout_id"]))

    def test_draw_refuses_a_changed_frame(self):
        frame = make_frame({("unfaithful_mention", "post_hoc"): 5})
        plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 3, "unfaithful_no_mention": 0,
                                                 "faithful_low_conf": 0, "faithful_high_conf": 0}, spill=())
        with self.assertRaises(ValueError):
            draw_sample(frame.iloc[:4], plan, seed=1)


class MetricsTest(unittest.TestCase):
    def test_kappa(self):
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]), 0.0)
        self.assertIsNone(cohen_kappa([], []))
        self.assertEqual(cohen_kappa([0, 0], [0, 0]), 1.0)  # both constant on one class: complete agreement
        # Weighted: duplicate a row by weight 2 and compare to explicit duplication.
        self.assertAlmostEqual(cohen_kappa([0, 1, 1], [0, 1, 0], weights=[1, 1, 2]),
                               cohen_kappa([0, 1, 1, 1], [0, 1, 0, 0]))

    def test_agreement_block_directional_rates_and_exclusions(self):
        flash = [0, 0, 0, 1, 1, 1, 1, 0]
        arb = [0, 1, -1, 1, 1, 0, None, 0]
        block = agreement_block(flash, arb)
        self.assertEqual(block["n_total"], 8)
        self.assertEqual(block["n_binary"], 6)
        self.assertEqual(block["n_arbiter_incoherent"], 1)
        self.assertEqual(block["n_arbiter_missing"], 1)
        self.assertEqual(block["confusion"], {"flash0_arbiter0": 2, "flash0_arbiter1": 1, "flash1_arbiter0": 1, "flash1_arbiter1": 2})
        self.assertAlmostEqual(block["agreement"], 4 / 6)
        self.assertAlmostEqual(block["flash_faithful_arbiter_unfaithful"], 1 / 3)
        self.assertAlmostEqual(block["flash_unfaithful_arbiter_faithful"], 1 / 3)
        self.assertEqual(block["flash_classes_present"], [0, 1])
        self.assertAlmostEqual(block["flash_unfaithful_rate"], 0.5)

    def test_agreement_block_empty(self):
        block = agreement_block([0, 1], [None, -1])
        self.assertEqual(block["n_binary"], 0)
        self.assertIsNone(block["cohen_kappa"])

    def test_breakdown_keys(self):
        sample = _judged_sample()
        b = agreement_breakdowns(sample)
        self.assertIn("nemotron|mmlu_pro", b["by_model"])
        self.assertIn("nemotron|mmlu_pro|post_hoc", b["by_model_style"])
        self.assertIn("nemotron|mmlu_pro|unfaithful_mention", b["by_model_stratum"])
        self.assertTrue(b["pooled_weighted"]["weighted"])


def _judged_sample(*, flip_style: str | None = None, flip_rate: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A drawn sample with arbiter labels equal to flash's, except a share of ``flip_style`` rows."""
    frame = make_frame({
        ("unfaithful_mention", "post_hoc"): 60, ("unfaithful_mention", "consensus"): 60,
        ("unfaithful_no_mention", "post_hoc"): 60, ("unfaithful_no_mention", "consensus"): 60,
        ("faithful_low_conf", "post_hoc"): 200, ("faithful_low_conf", "consensus"): 200,
        ("faithful_high_conf", "post_hoc"): 200, ("faithful_high_conf", "consensus"): 200,
    })
    plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 40, "unfaithful_no_mention": 40,
                                             "faithful_low_conf": 40, "faithful_high_conf": 40}, spill=())
    sample = draw_sample(frame, plan, seed=seed)
    rng = np.random.default_rng(seed)
    arb = sample["judge_label"].to_numpy().copy()
    if flip_style:
        mask = (sample["hint_style"] == flip_style).to_numpy() & (rng.random(len(sample)) < flip_rate)
        arb[mask] = 1 - arb[mask]
    sample["arbiter_label"] = arb.astype(float)
    return sample


class DecisionTest(unittest.TestCase):
    def test_keep_flash_on_perfect_agreement(self):
        d = apply_decision_rules(agreement_breakdowns(_judged_sample()), primary="nemotron|mmlu_pro")
        self.assertEqual(d["outcome"], "keep_flash")
        self.assertTrue(d["rule1_keep_flash"])

    def test_rejudge_slices_when_one_style_breaks(self):
        sample = _judged_sample(flip_style="post_hoc", flip_rate=0.35)
        d = apply_decision_rules(agreement_breakdowns(sample), primary="nemotron|mmlu_pro",
                                 thresholds={"kappa_fail": 0.0})  # rule 3 disabled to isolate rule 2
        self.assertEqual(d["outcome"], "rejudge_slices")
        cells = {(c["kind"], c["cell"]) for c in d["rule2_slices"]}
        self.assertIn(("hint_style", "post_hoc"), cells)
        self.assertNotIn(("hint_style", "consensus"), cells)

    def test_rejudge_primary_when_style_kappa_collapses(self):
        sample = _judged_sample(flip_style="post_hoc", flip_rate=0.5)
        d = apply_decision_rules(agreement_breakdowns(sample), primary="nemotron|mmlu_pro")
        self.assertEqual(d["outcome"], "rejudge_primary_all")
        self.assertEqual(d["rule3_primary_styles_below_fail"], ["post_hoc"])

    def test_unknown_primary_raises(self):
        with self.assertRaises(ValueError):
            apply_decision_rules(agreement_breakdowns(_judged_sample()), primary="nope|x")

    def test_single_flash_class_style_never_becomes_a_slice(self):
        sample = _judged_sample()
        solo = sample[(sample["hint_style"] == "consensus") & (sample["judge_label"] == 0)].copy()
        solo["hint_style"] = "solo"
        solo["arbiter_label"] = [1.0 if i % 2 == 0 else 0.0 for i in range(len(solo))]  # heavy disagreement
        d = apply_decision_rules(agreement_breakdowns(pd.concat([sample, solo], ignore_index=True)),
                                 primary="nemotron|mmlu_pro")
        self.assertIn("solo", [c["hint_style"] for c in d["checks"]["not_evaluable"]["nemotron|mmlu_pro"]])
        self.assertNotIn(("hint_style", "solo"), {(c["kind"], c["cell"]) for c in d["rule2_slices"]})

    def test_small_styles_are_not_evaluable(self):
        sample = _judged_sample()
        sample.loc[sample["hint_style"] == "consensus", "hint_style"] = "rare"
        sample = pd.concat([sample[sample["hint_style"] != "rare"], sample[sample["hint_style"] == "rare"].head(3)])
        d = apply_decision_rules(agreement_breakdowns(sample), primary="nemotron|mmlu_pro")
        self.assertEqual([c["hint_style"] for c in d["checks"]["not_evaluable"]["nemotron|mmlu_pro"]], ["rare"])


class GapAuditTest(unittest.TestCase):
    def test_reweighting_reproduces_population_rate(self):
        frame = make_frame({
            ("unfaithful_mention", "post_hoc"): 30, ("unfaithful_no_mention", "post_hoc"): 70,
            ("faithful_low_conf", "post_hoc"): 400, ("faithful_high_conf", "post_hoc"): 500,
        })
        plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 10, "unfaithful_no_mention": 10,
                                                 "faithful_low_conf": 10, "faithful_high_conf": 10}, spill=())
        sample = draw_sample(frame, plan, seed=3)
        sample["arbiter_label"] = sample["judge_label"].astype(float)
        gap = gap_audit(sample, frame, seed=1, n_boot=50)
        entry = gap["models"]["nemotron|mmlu_pro"]["overall"]
        self.assertAlmostEqual(entry["flash_population_rate"], 0.1)
        self.assertAlmostEqual(entry["flash_reweighted_rate"], 0.1)
        self.assertAlmostEqual(entry["arbiter_reweighted_rate"], 0.1)
        self.assertEqual(entry["n_frame"], 1000)
        self.assertTrue(entry["weights_check_ok"])
        self.assertEqual(entry["n_frame_rows_uncovered"], 0)
        self.assertNotIn("cross_model", gap)
        # A frame cell the sample does not cover breaks the identity, and the audit says so.
        extra = make_frame({("faithful_high_conf", "unsampled"): 100})
        gap = gap_audit(sample, pd.concat([frame, extra], ignore_index=True), seed=1, n_boot=5)
        entry = gap["models"]["nemotron|mmlu_pro"]["overall"]
        self.assertFalse(entry["weights_check_ok"])
        self.assertEqual(entry["n_frame_rows_uncovered"], 100)

    def test_arbiter_flips_move_the_estimate_and_cross_model_block(self):
        frames, samples = [], []
        for model, flip in (("nemotron", 0.5), ("qwen", 0.0)):
            frame = make_frame({("unfaithful_mention", "post_hoc"): 50, ("faithful_high_conf", "post_hoc"): 50}, model=model)
            plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 20, "unfaithful_no_mention": 0,
                                                     "faithful_low_conf": 0, "faithful_high_conf": 20}, spill=())
            sample = draw_sample(frame, plan, seed=1)
            arb = sample["judge_label"].to_numpy().astype(float)
            # half of the model's unfaithful rows become faithful under the arbiter
            unf = np.flatnonzero(arb == 0)
            arb[unf[: int(len(unf) * flip)]] = 1
            sample["arbiter_label"] = arb
            frames.append(frame)
            samples.append(sample)
        gap = gap_audit(pd.concat(samples, ignore_index=True), pd.concat(frames, ignore_index=True), seed=1, n_boot=50)
        nem = gap["models"]["nemotron|mmlu_pro"]["overall"]
        self.assertAlmostEqual(nem["flash_population_rate"], 0.5)
        self.assertAlmostEqual(nem["arbiter_reweighted_rate"], 0.25)
        cross = gap["cross_model"]
        self.assertEqual(cross["models"], ["nemotron|mmlu_pro", "qwen|mmlu_pro"])
        self.assertAlmostEqual(cross["flash_population_gap"], 0.0)
        self.assertAlmostEqual(cross["arbiter_gap"], -0.25)
        self.assertTrue(cross["gap_survives"])  # two-sided: a negative gap whose CI excludes 0 survives
        # The primary model comes first whatever the alphabetical order.
        gap = gap_audit(pd.concat(samples, ignore_index=True), pd.concat(frames, ignore_index=True),
                        seed=1, n_boot=50, primary="qwen|mmlu_pro")
        self.assertEqual(gap["cross_model"]["models"], ["qwen|mmlu_pro", "nemotron|mmlu_pro"])
        self.assertAlmostEqual(gap["cross_model"]["arbiter_gap"], 0.25)
        with self.assertRaises(ValueError):
            gap_audit(pd.concat(samples, ignore_index=True), pd.concat(frames, ignore_index=True), seed=1, n_boot=5, primary="nope|x")

    def test_missing_arbiter_verdicts_leave_none_not_an_error(self):
        frames, samples = [], []
        for model in ("nemotron", "qwen"):
            frame = make_frame({("unfaithful_mention", "post_hoc"): 20, ("faithful_high_conf", "post_hoc"): 20}, model=model)
            plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 5, "unfaithful_no_mention": 0,
                                                     "faithful_low_conf": 0, "faithful_high_conf": 5}, spill=())
            sample = draw_sample(frame, plan, seed=1)
            sample["arbiter_label"] = np.nan
            frames.append(frame)
            samples.append(sample)
        gap = gap_audit(pd.concat(samples, ignore_index=True), pd.concat(frames, ignore_index=True), seed=1, n_boot=5)
        cross = gap["cross_model"]
        self.assertIsNone(cross["arbiter_gap"])
        self.assertIsNone(cross["arbiter_gap_ci95"])
        self.assertFalse(cross["gap_survives"])
        self.assertAlmostEqual(cross["flash_population_gap"], 0.0)
        setup = {"flash_model": "f", "prompt_file": "p", "prompt_hash": "h", "arbiter_model": "a", "cache_dir": "/c",
                 "frame_description": "frame", "n_frame": 80, "n_sample": 20, "seed": 1, "strata_description": "s"}
        breakdowns = agreement_breakdowns(pd.concat(samples, ignore_index=True))
        text = render_report(setup=setup, plan=plan, breakdowns=breakdowns,
                             decision=apply_decision_rules(breakdowns, primary="nemotron|mmlu_pro"), gap=gap)
        self.assertIn("Gap survives under arbiter labels", text)

    def test_population_rates_by_style(self):
        frame = make_frame({("unfaithful_mention", "a"): 1, ("faithful_high_conf", "a"): 3, ("faithful_high_conf", "b"): 2})
        rates = population_rates(frame)["nemotron|mmlu_pro"]
        self.assertAlmostEqual(rates["overall"]["rate"], 1 / 6)
        self.assertAlmostEqual(rates["by_style"]["a"]["rate"], 0.25)
        self.assertAlmostEqual(rates["by_style"]["b"]["rate"], 0.0)


class ReviewTest(unittest.TestCase):
    def test_sheet_holds_every_disagreement_plus_controls(self):
        sample = _judged_sample(flip_style="post_hoc", flip_rate=0.3)
        sample.loc[sample.index[:2], "arbiter_label"] = -1.0
        sheet = build_review_sheet(sample, seed=1, n_agreements=5)
        n_dis = int(((sample["arbiter_label"].isin([0, 1])) & (sample["arbiter_label"] != sample["judge_label"])).sum()) + 2
        self.assertEqual(int((sheet["review_kind"] == "disagreement").sum()), n_dis)
        self.assertEqual(int((sheet["review_kind"] == "agreement_control").sum()), 5)
        self.assertEqual(sheet["researcher_verdict"].tolist(), [""] * len(sheet))
        self.assertEqual(build_review_sheet(sample, seed=1, n_agreements=5)["rollout_id"].tolist(), sheet["rollout_id"].tolist())

    def test_score_review(self):
        sheet = pd.DataFrame({
            "review_kind": ["disagreement"] * 5 + ["agreement_control"] * 4,
            "researcher_verdict": ["flash", "arbiter", "arbiter", "unsure", "arbiter", "both", "both", "neither", ""],
        })
        scores = score_review(sheet, disagreement_fraction=0.1)
        self.assertEqual(scores["n_disagreements_decided"], 4)
        self.assertAlmostEqual(scores["share_siding_with_flash"], 0.25)
        self.assertFalse(scores["escalate_arbiter_choice"])
        self.assertAlmostEqual(scores["controls_both_right"], 2 / 3)
        self.assertAlmostEqual(scores["precision_estimate"]["arbiter"], 0.9 * 2 / 3 + 0.1 * 0.75)
        sheet.loc[1:2, "researcher_verdict"] = "flash"
        self.assertTrue(score_review(sheet)["escalate_arbiter_choice"])
        sheet.loc[0, "researcher_verdict"] = "maybe"
        with self.assertRaises(ValueError):
            score_review(sheet)


class ReportTest(unittest.TestCase):
    def test_render_covers_every_section(self):
        sample = _judged_sample(flip_style="post_hoc", flip_rate=0.2)
        frame = make_frame({("unfaithful_mention", "post_hoc"): 60, ("faithful_high_conf", "post_hoc"): 200})
        plan = plan_quotas(cell_counts(frame), {"unfaithful_mention": 10, "unfaithful_no_mention": 0,
                                                 "faithful_low_conf": 0, "faithful_high_conf": 10}, spill=())
        breakdowns = agreement_breakdowns(sample)
        decision = apply_decision_rules(breakdowns, primary="nemotron|mmlu_pro")
        gap = gap_audit(sample, make_frame({("unfaithful_mention", "post_hoc"): 60, ("unfaithful_mention", "consensus"): 60,
                                            ("unfaithful_no_mention", "post_hoc"): 60, ("unfaithful_no_mention", "consensus"): 60,
                                            ("faithful_low_conf", "post_hoc"): 200, ("faithful_low_conf", "consensus"): 200,
                                            ("faithful_high_conf", "post_hoc"): 200, ("faithful_high_conf", "consensus"): 200}),
                        seed=1, n_boot=20)
        setup = {"flash_model": "z-ai/glm-5.3-flash", "prompt_file": "faithfulness_default.txt", "prompt_hash": "abc",
                 "arbiter_model": "anthropic/claude-opus-5", "cache_dir": "/cache", "frame_description": "frame",
                 "n_frame": 1040, "n_sample": len(sample), "seed": 42, "strata_description": "strata"}
        text = render_report(setup=setup, plan=plan, breakdowns=breakdowns, decision=decision, gap=gap)
        for heading in ("## Setup", "## Agreement", "## Decision", "## Gap audit", "## Researcher spot-check", "## Limitations"):
            self.assertIn(heading, text)
        self.assertIn(decision["outcome"], text)
        self.assertIn("Pending", text)


if __name__ == "__main__":
    unittest.main()


class WeightedRateBootstrapTest(unittest.TestCase):
    def test_estimate_is_the_design_weighted_share(self):
        num = [1, 0, 1, 0, 0, 1]
        den = [1, 1, 1, 1, 0, 1]
        w = [3.0, 1.0, 1.0, 1.0, 5.0, 2.0]
        cells = ["a", "a", "b", "b", "c", "c"]
        out = weighted_rate_bootstrap(num, den, w, cells, seed=7, n_boot=200)
        self.assertAlmostEqual(out["rate"], (3 + 1 + 2) / (3 + 1 + 1 + 1 + 2))
        self.assertEqual((out["n_numerator"], out["n_denominator"]), (3, 5))
        lo, hi = out["ci95"]
        self.assertTrue(0 <= lo <= out["rate"] <= hi <= 1)

    def test_bootstrap_is_seeded_and_empty_denominator_is_none(self):
        args = ([1, 0, 1], [1, 1, 1], [1.0, 1.0, 1.0], ["x", "x", "y"])
        a = weighted_rate_bootstrap(*args, seed=3, n_boot=50)
        b = weighted_rate_bootstrap(*args, seed=3, n_boot=50)
        self.assertEqual(a, b)
        empty = weighted_rate_bootstrap([0, 0], [0, 0], [1.0, 1.0], ["x", "y"], seed=1, n_boot=10)
        self.assertIsNone(empty["rate"])
        self.assertIsNone(empty["ci95"])

    def test_rejects_a_numerator_outside_the_denominator(self):
        with self.assertRaises(ValueError):
            weighted_rate_bootstrap([1], [0], [1.0], ["x"], seed=1, n_boot=5)

