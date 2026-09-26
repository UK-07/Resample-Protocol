"""Raw-tree preparation tests with independently specified stored outcomes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.lib.paper_figures.inputs import prepare
from src.scripts.visualizations.common.ssp_common import resolve_baseline_csv


def make_release(root: Path, *, relocated: bool = True) -> None:
    for folder in ("baselines", "binary_judge", "hinted_rollouts", "resample"):
        (root / folder).mkdir(parents=True)
    model, run = "qwen3-8b", "mmlu_pro-test_fixture"
    stem = f"{model}_{run}_baseline_hinted_rollouts"
    baseline = root / "baselines" / f"{model}_{run}_baseline.csv"
    samples = [["I"] * 6 + ["J", None], ["G"] * 8, ["A"] * 8]
    modal, target = ["I", "G", "A"], ["J", "H", "B"]
    answer = ["J", "H", "A"]
    baseline_rows, original, binary = [], [], []
    for index in range(3):
        baseline_rows.append({
            "original_index": index, "question": f"Question {index}",
            "choices": json.dumps([f"Option {c}" for c in "ABCDEFGHIJ"]),
            "groundtruth": modal[index], "baseline_answer": modal[index],
            "baseline_status": "correct", "sample_answers": json.dumps(samples[index]),
            # A distracting answer inside custom reasoning must be ignored.
            "sample_rollouts": json.dumps([
                f"<reason><answer>B</answer></reason><answer>{a}</answer>"
                if a is not None else "" for a in samples[index]
            ]),
            "n_top_votes": [6, 8, 8][index],
        })
        original.append({
            "rollout_id": f"p{index}", "question_id": f"q{index}", "subject_model": model,
            "run": run, "dataset": "mmlu_pro", "dataset_split": "test", "original_index": index,
            "hint_style": "expert_opinion", "case": "positive", "target_option": target[index],
            "groundtruth": modal[index], "baseline_modal_answer": modal[index],
            "baseline_stability": [6, 8, 8][index], "model_answer": answer[index],
            "to_hint": index == 0, "truncated": index == 1,
            "exclude_reason": "truncated" if index == 1 else None,
            "source_csv": f"{stem}_judged.csv", "judge_label_final": [-1, 0, None][index],
            "judge_label_final_model": "stored-role-judge", "judge_role": ["rejected", "neutral", None][index],
            "judge_confidence": 0.9, "baseline_hint_votes": [1, 0, 0][index],
            "baseline_n_samples": 8, "split": "train", "trace_token_len": 20000 if index == 1 else 100,
            "max_tokens_used": 16384,
        })
        binary.append({
            "original_index": index, "sample_type": "positive", "hint_name": "expert_opinion",
            "hinted_answer": target[index], "groundtruth": modal[index], "final_answer": answer[index],
            "judge_model": "stored-binary-judge", "judge_label": 0, "judge_confidence": 0.9,
        })
    pd.DataFrame(baseline_rows).to_csv(baseline, index=False)
    baseline.with_suffix(".meta.json").write_text(json.dumps({
        "reasoning_delimiters": ["<reason>", "</reason>"], "thinking": True,
        "top_p": 0.95, "top_k": 20,
    }))
    recorded = Path("/previous-machine/released/baselines") / baseline.name if relocated else baseline
    (root / "hinted_rollouts" / f"{stem}.meta.json").write_text(json.dumps({"baseline_csv": str(recorded)}))
    pd.DataFrame(original).to_parquet(root / "hinted_rollouts/rollout_manifest.parquet", index=False)
    pd.DataFrame(binary).to_csv(root / "binary_judge" / f"{stem}_binary_judged.csv", index=False)
    reliance, rerolls = [], []
    # Pair 0 persists 3/4; one of those outcomes is incoherent. Pair 1 has
    # a truncated target answer, an unparsed trace, a modal answer and a hit.
    outcomes = [
        [("J", False, True, 0, None), ("J", False, True, -1, "incoherent"),
         ("J", False, True, 1, None), ("I", False, False, 1, None)],
        [("H", True, False, None, "truncated"), (None, False, False, None, "unanswered"),
         ("G", False, False, 1, None), ("H", False, True, 1, None)],
    ]
    for index, draws in enumerate(outcomes):
        reliance_label = "robust_used" if index == 0 else "weak_used"
        reliance.append({
            "rollout_id": f"p{index}", "subject_model": model, "run": run, "original_index": index,
            "hint_style": "expert_opinion", "case": "positive", "role": "used_candidate",
            "reliance_label": reliance_label, "k_n": 4, "k_to_hint_count": 3 if index == 0 else 1,
            "k_truncated": int(index == 1),
        })
        for draw, (letter, truncated, hit, label, exclusion) in enumerate(draws):
            rerolls.append({
                "rollout_id": f"p{index}-r{draw}", "source_rollout_id": f"p{index}",
                "subject_model": model, "run": run, "dataset": "mmlu_pro", "original_index": index,
                "hint_style": "expert_opinion", "case": "positive", "provenance": "resample_k4",
                "reliance_label": reliance_label, "role": "used_candidate", "to_hint": hit,
                "model_answer": letter, "truncated": truncated, "judge_label_final": label,
                "judge_role": "credited" if label == 1 else "rejected", "exclude_reason": exclusion,
                "sample_seed": 43 + draw, "trace_token_len": 20000 if truncated else 100,
                "max_tokens_used": 16384, "split": "train", "target_option": target[index],
                "baseline_modal_answer": modal[index],
            })
    # Original rows in the resample manifest must not be counted as fresh draws.
    excluded = dict(rerolls[0], rollout_id="original-copy", provenance="hinted_once")
    pd.DataFrame([*rerolls, excluded]).to_parquet(root / "resample/resample_manifest.parquet", index=False)
    pd.DataFrame(reliance).to_csv(root / "resample/question_reliance.csv", index=False)


class PrepareInputsTest(unittest.TestCase):
    def test_named_dataset_options_need_no_dataset_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "release", Path(temporary) / "outputs"
            make_release(root)
            sidecar = next((root / "baselines").glob("*.meta.json"))
            meta = json.loads(sidecar.read_text())
            meta["dataset"] = {"name": "mmlu_pro"}
            sidecar.write_text(json.dumps(meta))
            with patch("src.lib.dataset.load_dataset", side_effect=AssertionError("No dataset download")):
                paths = prepare(root, output)
            self.assertEqual(pd.read_parquet(paths["pairs"]).iloc[0].b0, "I")

    def test_raw_release_relocation_and_stored_outcome_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "moved-release", Path(temporary) / "outputs"
            make_release(root)
            def source_hashes():
                return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob("*") if p.is_file()}
            before = source_hashes()
            paths = prepare(root, output)
            self.assertEqual(before, source_hashes())
            self.assertEqual({p.name for p in output.iterdir()}, {"pairs_master.parquet", "rerolls_long.parquet"})
            self.assertFalse((root / "plots").exists())
            pairs = pd.read_parquet(paths["pairs"]).set_index("rollout_id")
            rerolls = pd.read_parquet(paths["rerolls"]).set_index("rollout_id")
            self.assertTrue({"source_csv", "binary_csv", "baseline_csv"}.isdisjoint(pairs.columns))
            for table in (pairs, rerolls):
                self.assertNotIn(str(root), table.to_json())
                self.assertNotIn("/previous-machine/", table.to_json())
            self.assertEqual(len(pairs), 3)
            self.assertEqual(len(rerolls), 8)
            first = pairs.loc["p0"]
            self.assertEqual(first.b0, "I")
            self.assertEqual(first.n_options, 10)
            self.assertEqual(first.bl_n_answered, 7)
            self.assertEqual(first.bl_n_target, 1)
            self.assertEqual(first.bl_n_samples, 8)
            self.assertAlmostEqual(first.bl_frac_target, 1 / 7)
            self.assertEqual(first.qkey, "mmlu_pro:0")
            self.assertTrue(first.ssp_flip)
            self.assertTrue(first.ssp_unfaithful)
            self.assertEqual(first.cat_orig, "incoherent")
            self.assertEqual(first.role_orig, "rejected")
            self.assertEqual(first.k_hit, 3)
            self.assertTrue(first.persist3)
            self.assertEqual(first.k_rsp_pool, 3)
            self.assertEqual(first.k_rsp_unf, 2)
            self.assertEqual(first.k_rsp_inc, 1)
            self.assertTrue(rerolls.loc["p0-r1", "hit_rsp_unf"])
            second = pairs.loc["p1"]
            self.assertFalse(second.ssp_flip)
            self.assertTrue(second.ssp_flip_truncated)
            self.assertEqual(second.k_hit, 1)
            self.assertEqual(second.k_hit_any, 2)
            self.assertEqual(second.k_hit_trunc, 1)
            self.assertEqual(second.k_missing, 2)
            self.assertEqual(second.k_eff, 2)
            self.assertEqual(second.k_off, 1)
            self.assertEqual(second.k_modal, 1)
            self.assertEqual(second.k_over16k, 1)
            self.assertFalse(rerolls.loc["p1-r0", "hit"])
            self.assertTrue(rerolls.loc["p1-r0", "hit_any"])
            self.assertEqual(pairs.loc["p2", "cat_orig"], "unjudged")
            self.assertTrue(pd.isna(pairs.loc["p2", "k_n"]))
            self.assertFalse(pairs.loc["p2", "persist1"])

    def test_inconsistent_baseline_votes_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "release", Path(temporary) / "output"
            make_release(root, relocated=False)
            path = root / "hinted_rollouts/rollout_manifest.parquet"
            manifest = pd.read_parquet(path)
            manifest.loc[0, "baseline_hint_votes"] = 2
            manifest.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "Baseline target counts disagree"):
                prepare(root, output)
            self.assertFalse(output.exists())

    def test_extra_model_sources_do_not_enter_paper_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "release", Path(temporary) / "output"
            make_release(root)
            # Unselected model artifacts must not even be opened. Its full
            # rows may coexist in both manifests in the six-model release.
            (root / "binary_judge/qwen3.6-27b_extra_baseline_hinted_rollouts_binary_judged.csv").write_text("unused")
            for filename in ("hinted_rollouts/rollout_manifest.parquet", "resample/resample_manifest.parquet"):
                path = root / filename
                manifest = pd.read_parquet(path)
                extra = manifest.iloc[[0]].copy()
                extra["subject_model"] = "qwen3.6-27b"
                extra["rollout_id"] = "extra-model"
                if "source_csv" in extra:
                    extra["source_csv"] = "qwen3.6-27b_extra_baseline_hinted_rollouts_judged.csv"
                pd.concat([manifest, extra], ignore_index=True).to_parquet(path, index=False)
            paths = prepare(root, output)
            self.assertEqual(set(pd.read_parquet(paths["pairs"]).subject_model), {"qwen3-8b"})
            self.assertEqual(set(pd.read_parquet(paths["rerolls"]).subject_model), {"qwen3-8b"})

    def test_cannot_write_into_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for output in (root, root / "outputs"):
                with self.assertRaisesRegex(ValueError, "read-only"):
                    prepare(root, output)
            self.assertFalse((root / "outputs").exists())

    def test_relocated_recipe_never_reads_external_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            root.mkdir()
            external = Path(temporary) / "old" / "baselines" / "baseline.csv"
            external.parent.mkdir(parents=True)
            external.write_text("external")
            with self.assertRaisesRegex(FileNotFoundError, "missing from the release"):
                resolve_baseline_csv(external, root)
            (root / "baselines").mkdir()
            local = root / "baselines/baseline.csv"
            local.write_text("local")
            self.assertEqual(resolve_baseline_csv(external, root), local)
            self.assertEqual(resolve_baseline_csv("baselines/baseline.csv", root), local)
            with self.assertRaisesRegex(ValueError, "outside"):
                resolve_baseline_csv("/unrelated/baseline.csv", root)


if __name__ == "__main__":
    unittest.main()
