"""Tests for src/scripts/visualizations/judge_validation_plots.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.visualizations import judge_validation_plots as mod

FLASH = "z-ai/glm-5.3-flash"
ARBITER = "openai/gpt-5.6-terra"
PROMPT_HASH = "4c9dba9479"
NEMO_SRC = "nemotron-nano-9b-v2_mmlu_pro-test_0909_baseline_hinted_rollouts_judged.csv"
QWEN_SRC = "qwen3.5-9b_mmlu-pro_hinted-rollouts-0828_merged_judged.csv"


def synthetic_inputs(root: Path, seed: int = 0) -> dict:
    """A validation run, shared manifest, backfill cache, reliance, overrides, cueball tree and golden set."""
    rng = np.random.default_rng(seed)
    roles = np.array(mod.ROLE_ORDER)
    # --- the validation sample: two models, every cue, four strata -------------------------
    rows = []
    i = 0
    for model, run, src in [("nemotron-nano-9b-v2", "mmlu_pro-test_0909", NEMO_SRC), ("qwen3.5-9b", "mmlu-pro", QWEN_SRC)]:
        for cue in mod.CUE_ORDER:
            for stratum in ("unfaithful_mention", "unfaithful_no_mention", "faithful_low_conf", "faithful_high_conf"):
                for _ in range(4):
                    label = 0 if stratum.startswith("unfaithful") else 1
                    role = "credited" if label == 1 else str(rng.choice(["verification_only", "rejected", "neutral", "none"]))
                    arb_role = role if rng.random() < 0.8 else str(rng.choice(roles))
                    rows.append({
                        "rollout_id": f"{model}:{run}:{cue}:{i}", "question_id": f"mmlu_pro:test:{i}", "subject_model": model,
                        "subject_model_id": model, "run": run, "hint_style": cue, "case": "positive", "original_index": i,
                        "source_csv": src, "target_option": "C", "model_answer": "C", "judge_model": FLASH, "judge_prompt": "p",
                        "judge_label": label, "judge_confidence": 0.9, "trace_token_len": 500, "mention": int(role != "none"),
                        "stratum": stratum, "cell_n_available": 40, "cell_n_take": 4, "design_weight": 10.0,
                        "arbiter_label": int(arb_role == "credited") if rng.random() > 0.02 else -1, "arbiter_confidence": 0.9,
                        "arbiter_hint_role": arb_role, "arbiter_hint_quote": "", "arbiter_reasoning": "", "arbiter_error": None,
                        "arbiter_cost_usd": 0.01, "arbiter_prompt_tokens": 100, "arbiter_completion_tokens": 50,
                        "arbiter_model": ARBITER, "arbiter_prompt_hash": PROMPT_HASH, "role_true": role,
                    })
                    i += 1
    sample = pd.DataFrame(rows)
    vdir = root / "judge_validation"; vdir.mkdir(parents=True)
    sample.drop(columns=["role_true"]).to_csv(vdir / "sample_judged.csv", index=False)
    sample.drop(columns=["role_true"]).to_csv(vdir / "frame.csv", index=False)
    # the review sheet: every disagreement plus a few agreement controls, verdicts filled in
    binary = sample["arbiter_label"].isin([0, 1])
    dis = sample[binary & (sample["judge_label"] != sample["arbiter_label"])].assign(review_kind="disagreement")
    ctl = sample[binary & (sample["judge_label"] == sample["arbiter_label"])].head(6).assign(review_kind="agreement_control")
    sheet = pd.concat([dis, ctl], ignore_index=True).drop(columns=["role_true"])
    sheet["researcher_verdict"] = [("flash" if rng.random() < 0.6 else "arbiter") if k == "disagreement" else "both" for k in sheet["review_kind"]]
    sheet.loc[sheet.index[:1], "researcher_verdict"] = ""     # one undecided row
    sheet["researcher_notes"] = ""
    sheet.to_csv(vdir / "review_sheet.csv", index=False)
    # the shared manifest carries the primary judge's roles (nemotron rows only) and the cueball manifest the grid
    shared = sample[["rollout_id", "subject_model", "run", "judge_label", "role_true"]].rename(columns={"role_true": "judge_role"})
    shared.loc[shared["subject_model"] == "qwen3.5-9b", "judge_role"] = None
    (root / "hinted_rollouts").mkdir()
    shared.to_parquet(root / "hinted_rollouts" / "rollout_manifest.parquet", index=False)
    # the backfill cache: the primary judge re-run on the corrected prompt for the qwen rows
    stem = QWEN_SRC.removesuffix("_judged.csv")
    cdir = root / "judge_cache_prompt_backfill" / stem; cdir.mkdir(parents=True)
    slug = FLASH.replace("/", "_")
    with (cdir / f"{slug}_binary_{PROMPT_HASH}.jsonl").open("w") as f:
        for r in sample[sample["subject_model"] == "qwen3.5-9b"].itertuples():
            f.write(json.dumps({"original_index": int(r.original_index), "hint_name": r.hint_style, "judge_model": FLASH,
                                "prompt_hash": PROMPT_HASH, "label": int(r.judge_label), "confidence": 0.9,
                                "hint_role": r.role_true, "hint_quote": "", "reasoning": "", "error": None}) + "\n")
    # reliance labels for the nemotron rows, reviewer overrides for the reviewed rows
    (root / "resample").mkdir()
    nem = sample[sample["subject_model"] == "nemotron-nano-9b-v2"]
    pd.DataFrame({"rollout_id": nem["rollout_id"], "reliance_label": rng.choice(["robust_used", "weak_used", "mixed"], size=len(nem))}) \
        .to_csv(root / "resample" / "question_reliance.csv", index=False)
    decided = sheet[sheet["researcher_verdict"].isin(["flash", "arbiter", "both"])]
    final = np.where(decided["researcher_verdict"] == "arbiter", decided["arbiter_label"], decided["judge_label"])
    pd.DataFrame({"rollout_id": decided["rollout_id"], "judge_label_final": final, "judge_label_final_model": "researcher",
                  "judge_confidence_final": 0.9, "judge_role_final": None, "hint_quote_final": None,
                  "override_reason": "researcher", "judged_utc": "2026-09-16T00:00:00+00:00"}) \
        .to_csv(root / "hinted_rollouts" / "judge_label_overrides.csv", index=False)
    # the cueball tree: a manifest over the five paper models and one binary-judge file per cell
    cueball = root / "cueball"; (cueball / "hinted_rollouts").mkdir(parents=True); (cueball / "binary_judge").mkdir()
    grid = []
    j = 0
    for model in mod.PAPER_MODELS:
        for dataset in ("commonsense_qa", "medqa"):
            src = f"{model}_{dataset}_cueball_baseline_hinted_rollouts_judged.csv"
            cell = []
            for cue in mod.CUE_ORDER:
                for _ in range(6):
                    role = str(rng.choice(roles, p=[0.6, 0.1, 0.1, 0.1, 0.1]))
                    switched = bool(rng.random() < 0.8)
                    grid.append({"rollout_id": f"{model}:{dataset}:{cue}:{j}", "subject_model": model, "dataset": dataset,
                                 "hint_style": cue, "case": "positive", "original_index": j, "to_hint": switched,
                                 "judge_label": (1 if role == "credited" else 0) if switched else None,
                                 "judge_role": role if switched else None, "exclude_reason": None, "source_csv": src})
                    if switched:
                        cell.append({"original_index": j, "sample_type": "positive", "hint_name": cue, "prompt": "q", "rollout": "r",
                                     "judge_model": FLASH, "judge_label": int(rng.random() < (0.95 if role == "credited" else 0.3)),
                                     "judge_confidence": 0.9, "judge_reasoning": "", "judge_role": None, "judge_hint_quote": None})
                    j += 1
            pd.DataFrame(cell).to_csv(cueball / "binary_judge" / src.replace("_judged.csv", "_binary_judged.csv"), index=False)
    pd.DataFrame(grid).to_parquet(cueball / "hinted_rollouts" / "rollout_manifest.parquet", index=False)
    # the golden set
    g = pd.DataFrame({"model": rng.choice(["qwen3.5-9b", "olmo3-7b-think", "nemotron-nano-9b-v2"], size=30), "dataset": "mmlu-test",
                      "hint_name": rng.choice(mod.CUE_ORDER, size=30), "original_index": np.arange(30),
                      "judge_label": rng.integers(0, 2, size=30)})
    g["manual_label"] = np.where(rng.random(30) < 0.85, g["judge_label"], 1 - g["judge_label"])
    g["manual_label_blind"] = np.where(rng.random(30) < 0.8, g["judge_label"], 1 - g["judge_label"])
    g.to_csv(root / "golden.csv", index=False)
    return {"validation_dir": vdir, "shared_manifest": root / "hinted_rollouts" / "rollout_manifest.parquet",
            "backfill": root / "judge_cache_prompt_backfill", "reliance": root / "resample" / "question_reliance.csv",
            "overrides": root / "hinted_rollouts" / "judge_label_overrides.csv", "cueball": cueball, "golden": root / "golden.csv"}


def argv_for(paths: dict, out: Path, **extra) -> list[str]:
    argv = ["--validation-dir", str(paths["validation_dir"]), "--cueball-dir", str(paths["cueball"]),
            "--shared-manifest", str(paths["shared_manifest"]), "--backfill-cache-dir", str(paths["backfill"]),
            "--reliance", str(paths["reliance"]), "--overrides", str(paths["overrides"]), "--golden", str(paths["golden"]),
            "--out", str(out), "--formats", "png", "--n-boot", "20"]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


class JudgeValidationPlotsTest(unittest.TestCase):
    def test_every_variant_draws_and_is_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic_inputs(Path(tmp))
            out = Path(tmp) / "figs"
            self.assertEqual(mod.main(argv_for(paths, out)), 0)
            pngs = sorted(p.name for p in out.glob("*.png"))
            self.assertEqual(pngs, [f"{k}.png" for k in ["J1a", "J1b", "J1c", "J2a", "J2b", "J2c", "J3a", "J3b", "J4a", "J4b", "J4c"]])
            index = (out / "figures_judge.md").read_text()
            for p in pngs:
                self.assertIn(f"**{p.removesuffix('.png')}**", index)
            self.assertIn("corrected-prompt verdicts used for", index)
            self.assertTrue((out / "_cache" / "binary_judge_labels.parquet").exists())
            # a second run reuses the cached binary labels and produces the same index
            self.assertEqual(mod.main(argv_for(paths, out, only="J2b")), 0)

    def test_sample_labels_come_from_the_backfill_cache_where_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic_inputs(Path(tmp))
            notes: list[str] = []
            s = mod.load_sample(paths["validation_dir"], paths["shared_manifest"], paths["backfill"], paths["reliance"], paths["overrides"], notes)
            self.assertEqual(set(s.loc[s["subject_model"] == "qwen3.5-9b", "primary_source"]), {"corrected_prompt"})
            self.assertEqual(set(s.loc[s["subject_model"] == "nemotron-nano-9b-v2", "primary_source"]), {"manifest"})
            self.assertTrue(s["primary_role"].isin(mod.JUDGE_ROLES).all())        # both sources supply a role
            self.assertTrue(s.loc[s["subject_model"] == "nemotron-nano-9b-v2", "reliance_label"].notna().all())
            self.assertTrue(s.loc[s["subject_model"] == "qwen3.5-9b", "reliance_label"].isna().all())
            reviewed = s[s["reviewed"]]
            self.assertGreater(len(reviewed), 0)
            unreviewed = s[~s["reviewed"]]
            self.assertTrue((unreviewed["reviewer_label"] == unreviewed["primary_label"]).all())

    def test_kappa_and_precision_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic_inputs(Path(tmp))
            notes: list[str] = []
            s = mod.load_sample(paths["validation_dir"], paths["shared_manifest"], paths["backfill"], paths["reliance"], paths["overrides"], notes)
            kappa, n = mod.kappa_grid(s, mod.PAPER_CUES, ["nemotron-nano-9b-v2", "qwen3.5-9b"])
            self.assertEqual(list(kappa.index), mod.PAPER_CUES + ["all"])
            self.assertEqual(int(n.loc["all"].sum()), int(s["both_binary"].sum()))
            self.assertTrue(((kappa.fillna(0) >= -1) & (kappa.fillna(0) <= 1)).all().all())
            rk, rn = mod.role_kappa_grid(s, ["nemotron-nano-9b-v2"])
            self.assertEqual(int(rn.loc["all", "nemotron-nano-9b-v2"]), int((s["subject_model"] == "nemotron-nano-9b-v2").sum()))
            sheet = mod.load_sheet(paths["validation_dir"], s, notes)
            prec = mod.role_precision(sheet, s)
            allrow = prec[prec["role"] == "all"].iloc[0]
            self.assertEqual(int(allrow["n"]), int(sheet["decided"].sum()))
            self.assertTrue(0 <= allrow["lo"] <= allrow["precision"] <= allrow["hi"] <= 1)
            self.assertEqual(prec.attrs["controls_both_right"], 1.0)
            metrics = mod.weighted_metrics(s, ["nemotron-nano-9b-v2", "qwen3.5-9b"], seed=1, n_boot=10)
            self.assertIn("false_claim_rate", set(metrics.loc[metrics["subject_model"] == "nemotron-nano-9b-v2", "metric"]))
            self.assertNotIn("false_claim_rate", set(metrics.loc[metrics["subject_model"] == "qwen3.5-9b", "metric"]))
            self.assertEqual(set(metrics["label_set"]), set(mod.LABEL_SETS))

    def test_single_model_run_facets_the_binary_figure_by_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic_inputs(Path(tmp))
            out = Path(tmp) / "figs"
            self.assertEqual(mod.main(argv_for(paths, out, models="nemotron-nano-9b-v2")), 0)
            pngs = sorted(p.name for p in out.glob("*.png"))
            self.assertEqual(len(pngs), 11, pngs)
            index = (out / "figures_judge.md").read_text()
            self.assertIn("models: Nemotron-Nano-9B —", index)
            self.assertIn("one panel per dataset", index)
            self.assertNotIn("Qwen3.5-9B κ", index)
            self.assertNotIn("corrected-prompt verdicts used for", index)   # no Qwen rows, so no backfill note
            self.assertIn("pooled over the four datasets of Nemotron-Nano-9B", index)

    def test_paper_cues_are_the_eight_cue_styles(self):
        self.assertEqual(mod.PAPER_CUES, mod.CUE_ORDER)
        self.assertEqual(len(mod.PAPER_CUES), 8)
        self.assertIn("grader_hacking", mod.PAPER_CUES)

    def test_kappa_pooled_row_covers_only_the_drawn_cues(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic_inputs(Path(tmp))
            s = mod.load_sample(paths["validation_dir"], paths["shared_manifest"], paths["backfill"], paths["reliance"], paths["overrides"], [])
            cues = ["expert_opinion", "post_hoc"]
            kappa, n = mod.kappa_grid(s, cues, ["nemotron-nano-9b-v2"])
            self.assertEqual(list(kappa.index), cues + ["all"])
            drawn = s[(s["subject_model"] == "nemotron-nano-9b-v2") & s["both_binary"] & s["hint_style"].isin(cues)]
            self.assertEqual(int(n.loc["all", "nemotron-nano-9b-v2"]), len(drawn))
            self.assertEqual(int(n.loc[cues, "nemotron-nano-9b-v2"].sum()), len(drawn))

    def test_explicit_cue_list_restricts_the_rows_and_bad_lists_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic_inputs(Path(tmp))
            out = Path(tmp) / "figs"
            # duplicated and out-of-order ids are canonicalised; the caption counts only the drawn cues
            self.assertEqual(mod.main(argv_for(paths, out, only="J1a", cues="post_hoc, expert_opinion,post_hoc")), 0)
            s = mod.load_sample(paths["validation_dir"], paths["shared_manifest"], paths["backfill"], paths["reliance"], paths["overrides"], [])
            expected = int((s["both_binary"] & s["hint_style"].isin(["expert_opinion", "post_hoc"])).sum())
            self.assertLess(expected, int(s["both_binary"].sum()))
            self.assertIn(f"{expected} rows both judges labelled 0/1", (out / "figures_judge.md").read_text())
            for bad in ("expert_opinion,not-a-cue", ",", ""):
                with self.assertRaises(SystemExit):
                    mod.main(argv_for(paths, out, cues=bad))

    def test_unknown_model_id_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic_inputs(Path(tmp))
            with self.assertRaises(SystemExit):
                mod.main(argv_for(paths, Path(tmp) / "figs", models="not-a-model"))

    def test_missing_optional_inputs_skip_their_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic_inputs(Path(tmp))
            out = Path(tmp) / "figs"
            argv = argv_for(paths, out)
            argv[argv.index("--golden") + 1] = str(Path(tmp) / "missing.csv")
            argv[argv.index("--cueball-dir") + 1] = str(Path(tmp) / "no_tree")
            self.assertEqual(mod.main(argv), 0)
            pngs = sorted(p.name for p in out.glob("*.png"))
            self.assertNotIn("J2a.png", pngs)
            self.assertNotIn("J3b.png", pngs)
            self.assertIn("J3a.png", pngs)
            self.assertIn("golden set missing", (out / "figures_judge.md").read_text())


if __name__ == "__main__":
    unittest.main()
