"""Tests for src/scripts/judge_validation.py — the stages over synthetic judged CSVs and a manifest."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.lib.judge_validation import DECISION_OUTCOMES
from src.lib.llm_judge_verb import judge_cache_path
from src.lib.rollout_manifest import PairInfo, derive_rollout_rows, read_label_overrides, read_manifest, write_manifest
from src.scripts import judge_validation as jv

ARBITER = "test/arbiter"
STYLES = ("post_hoc", "expert_opinion")


def judged_rows(model: str) -> pd.DataFrame:
    """40 judged rows, 10 per stratum; ``MENTION`` marks the mention stratum, ``FLIP`` flips the fake arbiter."""
    rows = []
    idx = 0
    for stratum in ("unfaithful_mention", "unfaithful_no_mention", "faithful_low_conf", "faithful_high_conf"):
        for i in range(10):
            idx += 1
            label = 0 if stratum.startswith("unfaithful") else 1
            conf = 0.6 if stratum == "faithful_low_conf" else 0.95
            marker = " MENTION" if stratum == "unfaithful_mention" else ""
            flip = " FLIP" if (stratum == "faithful_low_conf" and i < 4) or (stratum == "unfaithful_mention" and i < 2) else ""
            cot = f"thinking about {idx}{marker}{flip}"
            rows.append({
                "original_index": idx, "sample_type": "positive" if i % 2 == 0 else "negative",
                "hint_name": STYLES[i % 2], "prompt": f"Question {idx}?\nA. x\nB. y", "hinted_prompt": f"Question {idx}?\nA. x\nB. y\nA professor says B.",
                "hinted_answer": "B", "baseline_answer": "A", "final_answer": "B", "groundtruth": "A",
                "rollout": f"<think>{cot}</think><answer>B</answer>", "reasoning": cot,
                "judge_model": "z-ai/glm-5.3-flash", "judge_label": label, "judge_confidence": conf,
                "judge_reasoning": "flash says so", "judge_role": "", "judge_hint_quote": "",
            })
    return pd.DataFrame(rows)


def fake_string_judge(row, *, delimiters=None, count_weak=False):
    return {"label": 1 if "MENTION" in str(row["reasoning"]) else 0, "error": None}


def fake_run_judge_batch(rows, model, cache_dir, *, delimiters=None, prompt_template=None, max_workers=4, progress=print):
    path = judge_cache_path(Path(cache_dir), model, prompt_template)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    with path.open("a") as handle:
        for _, row in rows.iterrows():
            label = int(row["judge_label"])
            if "FLIP" in str(row["reasoning"]):
                label = 1 - label
            rec = {"original_index": int(row["original_index"]), "hint_name": str(row["hint_name"]), "judge_model": model,
                   "prompt_hash": "h", "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.01, "label": label,
                   "confidence": 0.9, "reasoning": "arbiter says", "hint_role": "credited" if label else "none",
                   "hint_quote": "says B" if label else None, "error": None}
            handle.write(json.dumps(rec) + "\n")
            records[(rec["original_index"], rec["hint_name"])] = rec
    return records


class Workspace:
    """A temp DATA_ROOT-like tree: judged CSVs, a manifest and a config file."""

    def __init__(self, tmp: Path, *, targets: dict | None = None):
        self.root = tmp
        (tmp / "hinted_rollouts").mkdir()
        frames = []
        self.stems = {}
        for model, run in (("nemo", "mmlu_pro"), ("qwen", "mmlu-pro")):
            stem = f"{model}_{run}_baseline_hinted_rollouts"
            self.stems[model] = stem
            df = judged_rows(model)
            df.to_csv(tmp / "hinted_rollouts" / f"{stem}_judged.csv", index=False)
            pair = PairInfo(subject_model=model, subject_model_id=None, run=run, dataset="mmlu_pro", dataset_split="test",
                            judge_prompt="faithfulness_default.txt", max_tokens=100, source_csv=f"{stem}_judged.csv")
            frames.append(derive_rollout_rows(df.astype(str), pair))
        self.manifest = tmp / "hinted_rollouts" / "rollout_manifest.parquet"
        write_manifest(pd.concat(frames, ignore_index=True), self.manifest, {"sources": []})
        self.config = tmp / "judge_validation.yaml"
        self.config.write_text(yaml.safe_dump({
            "manifest": str(self.manifest), "rollouts_dir": str(tmp / "hinted_rollouts"),
            "cache_dir": str(tmp / "judge_cache"), "output_dir": str(tmp / "out"),
            "label_overrides": str(tmp / "hinted_rollouts" / "judge_label_overrides.csv"), "seed": 7,
            "arbiter": {"judge_model": ARBITER, "judge_prompt_file": None, "workers": 1,
                        "usd_per_m_input": 1.0, "usd_per_m_output": 1.0},
            "frame": {"pairs": [{"subject_model": "nemo", "run": "mmlu_pro"}, {"subject_model": "qwen", "run": "mmlu-pro"}],
                      "primary": {"subject_model": "nemo", "run": "mmlu_pro"}},
            "strata": {"targets": targets or {"unfaithful_mention": 6, "unfaithful_no_mention": 4,
                                              "faithful_low_conf": 6, "faithful_high_conf": 4}, "min_per_style": 1},
            "decision": {"min_style_n": 4},
            "review": {"n_agreements": 3},
            "gap_audit": {"n_boot": 20},
        }))

    def run(self, *args) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                patch.object(jv, "string_judge_case_record", fake_string_judge), \
                patch.object(jv, "run_judge_batch", fake_run_judge_batch), \
                patch.object(jv, "check_judge_model", lambda m: True):
            code = jv.main(["--config", str(self.config), *args])
        self.assertion = code
        if code != 0:
            raise AssertionError(buf.getvalue())
        return buf.getvalue()


class StagesTest(unittest.TestCase):
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            out = Path(tmp) / "out"

            text = ws.run("--stage", "sample")
            frame = pd.read_csv(out / "frame.csv")
            sample = pd.read_csv(out / "sample.csv")
            self.assertEqual(len(frame), 80)
            self.assertEqual(sorted(frame["stratum"].unique()), sorted(jv.STRATA))
            self.assertEqual(frame[frame["judge_label"] == 0]["mention"].notna().all(), True)
            self.assertEqual(len(sample), 40)  # 20 per model
            self.assertEqual(sample.groupby("subject_model").size().to_dict(), {"nemo": 20, "qwen": 20})
            self.assertIn("estimate for 40 rows", text)
            self.assertTrue((sample["design_weight"] >= 1).all())

            text = ws.run("--stage", "judge", "--dry-run")
            self.assertIn("0 cached, 20 to judge", text)
            self.assertIn("--- prompt for", text)

            ws.run("--stage", "judge")
            cache = Path(tmp) / "judge_cache" / ws.stems["nemo"] / judge_cache_path(Path("."), ARBITER, None).name
            self.assertTrue(cache.exists())
            self.assertIn("20 cached, 0 to judge", ws.run("--stage", "judge"))

            text = ws.run("--stage", "report")
            judged = pd.read_csv(out / "sample_judged.csv")
            self.assertTrue(judged["arbiter_label"].notna().all())
            n_flip = int((judged["arbiter_label"] != judged["judge_label"]).sum())
            self.assertGreater(n_flip, 0)
            metrics = json.loads((out / "metrics.json").read_text())
            self.assertEqual(metrics["breakdowns"]["pooled"]["n_binary"], 40)
            self.assertAlmostEqual(metrics["arbiter_cost_usd"], 0.4)
            decision = json.loads((out / "decision.json").read_text())
            self.assertIn(decision["outcome"], DECISION_OUTCOMES)
            gap = json.loads((out / "gap_audit.json").read_text())
            self.assertIn("cross_model", gap)
            self.assertAlmostEqual(gap["models"]["nemo|mmlu_pro"]["overall"]["flash_reweighted_rate"],
                                   gap["models"]["nemo|mmlu_pro"]["overall"]["flash_population_rate"])
            sheet = pd.read_csv(out / "review_sheet.csv", keep_default_na=False)
            self.assertEqual(int((sheet["review_kind"] == "disagreement").sum()), n_flip)
            self.assertEqual(int((sheet["review_kind"] == "agreement_control").sum()), 3)
            self.assertTrue((sheet["reasoning_trace"].str.startswith("thinking about")).all())
            self.assertTrue((sheet["flash_reasoning"] == "flash says so").all())
            report = (out / "judge_validation_report.md").read_text()
            self.assertIn("## Decision", report)
            self.assertIn("Pending", report)

            text = ws.run("--stage", "escalate")
            self.assertIn("frame rows in scope", text)

            text = ws.run("--stage", "finalize")
            overrides = read_label_overrides(Path(tmp) / "hinted_rollouts" / "judge_label_overrides.csv")
            self.assertEqual(len(overrides), 40)
            self.assertEqual(set(overrides["override_reason"]), {"validation_sample"})
            manifest = read_manifest(ws.manifest)
            changed = manifest[manifest["judge_label_final"] != manifest["judge_label"]]
            self.assertEqual(len(changed), n_flip)
            self.assertEqual(set(changed["judge_label_final_model"]), {ARBITER})
            untouched = manifest[manifest["judge_label_final_model"] != ARBITER]
            self.assertTrue((untouched["judge_label_final"] == untouched["judge_label"]).all())
            finalize = json.loads((out / "finalize.json").read_text())
            self.assertEqual(finalize["n_label_changed"], n_flip)
            self.assertIn("nemo|mmlu_pro", finalize["final_fingerprint"])
            self.assertIn("## Fingerprint under final labels", (out / "judge_validation_report.md").read_text())

            # A second finalize is idempotent: same override count, same manifest.
            ws.run("--stage", "finalize")
            self.assertEqual(len(read_label_overrides(Path(tmp) / "hinted_rollouts" / "judge_label_overrides.csv")), 40)
            # Overrides of the frame's sources are re-derived from the caches: a
            # stale row for a frame rollout is dropped, a foreign source's row kept.
            path = Path(tmp) / "hinted_rollouts" / "judge_label_overrides.csv"
            extra = pd.read_csv(path, dtype=str, keep_default_na=False).iloc[:1].copy()
            extra["rollout_id"] = "other:run:hint:1"
            stale = pd.read_csv(path, dtype=str, keep_default_na=False).iloc[:1].copy()
            stale["rollout_id"] = manifest[manifest["judge_label_final_model"] != ARBITER]["rollout_id"].iloc[0]
            pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False), extra, stale]).to_csv(path, index=False)
            ws.run("--stage", "finalize")
            after = read_label_overrides(path)
            self.assertEqual(len(after), 41)
            self.assertIn("other:run:hint:1", set(after["rollout_id"]))
            self.assertNotIn(stale["rollout_id"].iloc[0], set(after["rollout_id"]))
            # Only the row this run does not re-derive identically is stale; the 40 re-derived rows are not.
            self.assertEqual(json.loads((out / "finalize.json").read_text())["n_stale_overrides_replaced"], 1)

            sheet.loc[sheet["review_kind"] == "disagreement", "researcher_verdict"] = "arbiter"
            sheet.loc[sheet["review_kind"] == "agreement_control", "researcher_verdict"] = "both"
            sheet.to_csv(out / "review_sheet.csv", index=False)
            text = ws.run("--stage", "score-review", "--disagreement-fraction", "0.2")
            scores = json.loads((out / "review_scores.json").read_text())
            self.assertEqual(scores["n_disagreements_decided"], n_flip)
            self.assertFalse(scores["escalate_arbiter_choice"])
            self.assertAlmostEqual(scores["precision_estimate"]["disagreement_fraction"], 0.2)
            self.assertIn("arbiter choice stands", (out / "judge_validation_report.md").read_text())

            # A report rerun must not overwrite a sheet that carries verdicts.
            ws.run("--stage", "report")
            kept = pd.read_csv(out / "review_sheet.csv", keep_default_na=False)
            self.assertTrue((kept["researcher_verdict"] != "").all())
            self.assertTrue((out / "review_sheet_regenerated.csv").exists())

            # researcher_adjudicated: only rows sided with the arbiter override flash.
            cfg = yaml.safe_load(ws.config.read_text()); cfg["finalize"] = {"policy": "researcher_adjudicated"}
            adjudicated = Path(tmp) / "judge_validation_adjudicated.yaml"; adjudicated.write_text(yaml.safe_dump(cfg))
            sheet = pd.read_csv(out / "review_sheet.csv", keep_default_na=False)
            dis = sheet.index[sheet["review_kind"] == "disagreement"]
            sheet.loc[dis[:2], "researcher_verdict"] = "flash"
            sheet.to_csv(out / "review_sheet.csv", index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), patch.object(jv, "string_judge_case_record", fake_string_judge), \
                    patch.object(jv, "run_judge_batch", fake_run_judge_batch), patch.object(jv, "check_judge_model", lambda m: True):
                self.assertEqual(jv.main(["--config", str(adjudicated), "--stage", "finalize"]), 0, buf.getvalue())
            overrides = read_label_overrides(Path(tmp) / "hinted_rollouts" / "judge_label_overrides.csv")
            # every decided row (n_flip disagreements + 3 controls) + the foreign-source row kept from before
            self.assertEqual(len(overrides), n_flip + 3 + 1)
            mine = overrides[overrides["rollout_id"] != "other:run:hint:1"]
            self.assertEqual(set(mine["judge_label_final_model"]), {"researcher"})
            self.assertEqual(mine["override_reason"].value_counts().to_dict(),
                             {"researcher_sided_with_arbiter": n_flip - 2, "researcher_sided_with_flash": 2, "researcher_confirmed_agreement": 3})
            manifest = read_manifest(ws.manifest)
            self.assertEqual(int((manifest["judge_label_final"] != manifest["judge_label"]).sum()), n_flip - 2)
            fin = json.loads((out / "finalize.json").read_text())
            self.assertEqual(fin["policy"], "researcher_adjudicated")
            self.assertEqual(fin["review_counts"]["flash"], 2)
            self.assertEqual(fin["review_counts"]["flash_label_differs"], 0)
            self.assertEqual(fin["review_counts"]["unsure_rollout_ids"], [])
            self.assertEqual(fin["n_overrides_applied"], n_flip + 3)  # the foreign-source row hits no manifest row

            # --overrides-only rewrites the overrides file and the fingerprint but not the manifest.
            before = ws.manifest.stat().st_mtime_ns
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), patch.object(jv, "string_judge_case_record", fake_string_judge), \
                    patch.object(jv, "run_judge_batch", fake_run_judge_batch), patch.object(jv, "check_judge_model", lambda m: True):
                self.assertEqual(jv.main(["--config", str(adjudicated), "--stage", "finalize", "--overrides-only"]), 0, buf.getvalue())
            self.assertEqual(ws.manifest.stat().st_mtime_ns, before)
            fin = json.loads((out / "finalize.json").read_text())
            self.assertFalse(fin["manifest_written"])
            self.assertIn("nemo|mmlu_pro", fin["final_fingerprint"])
            self.assertEqual(len(read_label_overrides(Path(tmp) / "hinted_rollouts" / "judge_label_overrides.csv")), n_flip + 3 + 1)

    def test_escalation_slices(self):
        frame = pd.DataFrame({
            "subject_model": ["nemo"] * 4 + ["qwen"] * 2, "run": ["mmlu_pro"] * 4 + ["mmlu-pro"] * 2,
            "hint_style": ["post_hoc", "post_hoc", "metadata", "metadata", "post_hoc", "metadata"],
            "stratum": ["unfaithful_mention", "faithful_low_conf", "unfaithful_mention", "faithful_high_conf", "faithful_low_conf", "faithful_low_conf"],
        })
        settings = {"frame": {"primary": "nemo|mmlu_pro"}}
        primary = jv.escalation_slice(frame, {"outcome": "rejudge_primary_all", "rule2_slices": []}, settings)
        self.assertEqual(len(primary), 4)
        slices = jv.escalation_slice(frame, {"outcome": "rejudge_slices", "rule2_slices": [
            {"subject_model|run": "nemo|mmlu_pro", "kind": "hint_style", "cell": "post_hoc"},
            {"subject_model|run": "qwen|mmlu-pro", "kind": "stratum", "cell": "faithful_low_conf"},
        ]}, settings)
        self.assertEqual(len(slices), 4)
        self.assertTrue(jv.escalation_slice(frame, {"outcome": "keep_flash", "rule2_slices": []}, settings).empty)


class HelpersTest(unittest.TestCase):
    def test_source_stem_and_settings_validation(self):
        self.assertEqual(jv.source_stem("m_run_baseline_hinted_rollouts_judged.csv"), "m_run_baseline_hinted_rollouts")
        base = {"output_dir": "/tmp/x", "arbiter": {"judge_model": "m"}, "frame": {"pairs": [{"subject_model": "a", "run": "b"}]}}
        with self.assertRaises(ValueError):
            jv.resolve_settings({**base, "arbiter": {}})
        with self.assertRaises(ValueError):
            jv.resolve_settings({**base, "frame": {"pairs": []}})
        with self.assertRaises(ValueError):  # the primary must be one of the pairs
            jv.resolve_settings({**base, "frame": {**base["frame"], "primary": {"subject_model": "z", "run": "b"}}})
        with self.assertRaises(ValueError):  # a typo'd stratum would silently add a target
            jv.resolve_settings({**base, "strata": {"targets": {"faithful_lowconf": 5}}})
        with self.assertRaises(ValueError):
            jv.resolve_settings({**base, "decision": {"kappa_kep": 0.9}})
        s = jv.resolve_settings(base)
        self.assertEqual(s["frame"]["primary"], "a|b")
        self.assertEqual(s["strata"]["targets"]["unfaithful_mention"], 120)
        self.assertEqual(jv.resolve_settings({**base, "strata": {"oversample": {}}})["strata"]["oversample"], {})

    def test_arbiter_prompt_must_match_the_frame(self):
        frame = pd.DataFrame({"judge_prompt": ["faithfulness_default.txt", None]})
        settings = {"arbiter": {"judge_prompt_file": None}}
        self.assertEqual(len(jv.check_prompt_matches_frame(frame, settings)), 10)
        settings = {"arbiter": {"judge_prompt_file": "configs/llm_judge_prompts/faithfulness_v1_default.txt"}}
        with self.assertRaises(ValueError):
            jv.check_prompt_matches_frame(frame, settings)
        with self.assertRaises(ValueError):
            jv.check_prompt_matches_frame(pd.DataFrame({"judge_prompt": ["unknown (hash 0000000000)"]}), settings)

    def test_overrides_from_review_provenance(self):
        sheet = pd.DataFrame({
            "rollout_id": ["m:r:h:1", "m:r:h:2", "m:r:h:3", "m:r:h:4"],
            "review_kind": ["disagreement", "disagreement", "agreement_control", "disagreement"],
            "judge_label": [0, 1, 1, 0], "arbiter_label": [1, 0, 1, 1],
            "judge_confidence": [0.9] * 4, "arbiter_confidence": [0.8] * 4,
            "arbiter_hint_role": [""] * 4, "arbiter_hint_quote": [""] * 4,
            "researcher_verdict": ["flash", "flash", "both", "unsure"],
        })
        manifest_labels = pd.Series([0.0, 0.0, 1.0, 0.0], index=sheet["rollout_id"])  # row 2's flash verdict was re-judged
        rows, counts = jv.overrides_from_review(sheet, manifest_labels)
        self.assertEqual(rows["judge_label_final"].tolist(), [0, 1, 1])
        self.assertEqual(counts["flash_label_differs"], 1)
        self.assertEqual(counts["flash_label_differs_rollout_ids"], ["m:r:h:2"])
        self.assertEqual(counts["unsure_rollout_ids"], ["m:r:h:4"])
        sheet.loc[0, "researcher_verdict"] = "both"  # a control verdict on a disagreement row
        with self.assertRaises(ValueError):
            jv.overrides_from_review(sheet, manifest_labels)

    def test_load_rows_for_keys_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x_judged.csv"
            judged_rows("m").to_csv(path, index=False)
            rows = jv.load_rows_for_keys(path, {(1, "post_hoc"), (2, "expert_opinion")}, chunk_rows=7)
            self.assertEqual(sorted(rows["original_index"]), [1, 2])
            with self.assertRaises(ValueError):
                jv.load_rows_for_keys(path, {(1, "post_hoc"), (999, "post_hoc")}, chunk_rows=7)


if __name__ == "__main__":
    unittest.main()
