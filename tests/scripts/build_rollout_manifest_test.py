"""Tests for src/scripts/build_rollout_manifest.py — the backfill from judged CSVs."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.lib.rollout_manifest import MANIFEST_COLUMNS, read_manifest, read_manifest_meta, write_manifest
from src.scripts.build_rollout_manifest import (
    baseline_path_for, build_one, judge_models_in, main, make_token_counter, model_id_for,
)
from tests.lib.rollout_manifest_test import make_baseline, make_chunk

STEM = "qwen3.5-9b_medqa-test_0911_baseline_hinted_rollouts"


def write_sources(tmp: Path, *, sidecar: bool = True, stem: str = STEM,
                  drop_columns: tuple[str, ...] = (), relabel: dict | None = None) -> Path:
    """A judged CSV (+ recipe sidecar, judge summary) and its baseline under ``tmp``.

    ``drop_columns`` leaves judged-CSV columns out (an old-schema file);
    ``relabel`` overrides ``baseline_status`` per original_index.
    """
    rollouts = tmp / "hinted_rollouts"
    baselines = tmp / "baselines"
    rollouts.mkdir(exist_ok=True)
    baselines.mkdir(exist_ok=True)
    judged = rollouts / f"{stem}_judged.csv"
    chunk = make_chunk()
    chunk.insert(3, "prompt", "Q?")
    chunk.insert(4, "hinted_prompt", "hinted Q?")
    chunk["additional_fields"] = "{}"
    chunk["judge_reasoning"] = "because"
    chunk.drop(columns=list(drop_columns)).to_csv(judged, index=False)
    baseline_csv = baselines / f"{stem.removesuffix('_hinted_rollouts')}.csv"
    base = make_baseline().reset_index()
    base["prompt"] = "Q?"
    for idx, status in (relabel or {}).items():
        base.loc[base["original_index"] == idx, "baseline_status"] = status
    base.to_csv(baseline_csv, index=False)
    baseline_csv.with_suffix(".meta.json").write_text(json.dumps({
        "model_name": "Qwen/Qwen3.5-9B", "dataset": {"name": "medqa", "params": {"split": "test"}},
        "consistency_samples": 8,
    }))
    if sidecar:
        (rollouts / f"{stem}.meta.json").write_text(json.dumps({
            "model_name": "Qwen/Qwen3.5-9B", "baseline_csv": str(baseline_csv),
            "max_tokens": 16384, "temperature": 0.7,
        }))
    (rollouts / f"{stem}_judged_summary.json").write_text(json.dumps({
        "judge_model": "z-ai/glm-5.3-flash", "judge_prompt_hash": "",
    }))
    return judged


class TestHelpers(unittest.TestCase):
    def test_judge_models_in_chunk(self):
        chunk = pd.DataFrame({"judge_model": ["a/x", None, " ", "b/y", "a/x"]})
        self.assertEqual(judge_models_in(chunk), {"a/x", "b/y"})
        self.assertEqual(judge_models_in(pd.DataFrame({"other": [1]})), set())

    def test_token_counter_batches_calls(self):
        calls = []

        def fake_tokenizer(texts, **kwargs):
            calls.append(len(texts))
            return {"length": [len(t) for t in texts]}

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer):
            count = make_token_counter("org/M", batch=2)
        self.assertEqual(count(["a", "bb", "ccc", "dddd", "eeeee"]), [1, 2, 3, 4, 5])
        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual(count([]), [])

    def test_model_id_prefers_sidecar_then_registry(self):
        self.assertEqual(model_id_for("whatever", {"model_name": "org/M"}), "org/M")
        self.assertEqual(model_id_for("qwen3.5-9b", {}), "Qwen/Qwen3.5-9B")
        self.assertIsNone(model_id_for("no-such-model", {}))

    def test_baseline_path_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            expected = Path(tmp) / "baselines" / "qwen3.5-9b_medqa-test_0911_baseline.csv"
            self.assertEqual(baseline_path_for(judged, {"baseline_csv": str(expected)}, "x", "y"), expected)
            with patch.dict("os.environ", {"DATA_ROOT": tmp}):
                self.assertEqual(baseline_path_for(judged, {}, "qwen3.5-9b", "medqa-test_0911"), expected)
                self.assertIsNone(baseline_path_for(judged, {"baseline_csv": "/nope.csv"}, "a", "b"))


class TestBuildOne(unittest.TestCase):
    def test_rows_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            rows, record = build_one(judged, chunk_rows=3, token_len=False, noise_flip_min_votes=1)
        self.assertEqual(list(rows.columns), list(MANIFEST_COLUMNS))
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows["run"]), {"medqa-test_0911"})
        self.assertEqual(set(rows["dataset"]), {"medqa"})
        self.assertEqual(set(rows["dataset_split"]), {"test"})
        self.assertEqual(set(rows["subject_model_id"]), {"Qwen/Qwen3.5-9B"})
        self.assertEqual(set(rows["baseline_n_samples"].dropna()), {8})
        self.assertEqual(set(rows["max_tokens_used"]), {16384})
        self.assertEqual(int(rows["to_hint"].sum()), 4)
        self.assertTrue(rows["trace_token_len"].isna().all())
        self.assertEqual(rows.set_index("original_index").loc[4, "baseline_stability"], 5)
        # Chunking never splits a row's derivation: the same result as one chunk.
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            whole, _ = build_one(judged, chunk_rows=100, token_len=False, noise_flip_min_votes=1)
        pd.testing.assert_frame_equal(rows, whole)
        self.assertEqual(record["judge_model"], "z-ai/glm-5.3-flash")
        self.assertEqual(record["judge_prompt"], "unknown")
        self.assertEqual(record["n_rows"], 8)
        self.assertEqual(record["n_switched"], 4)
        self.assertEqual(record["n_baseline_answer_mismatch"], 0)
        self.assertEqual(record["n_baseline_relabelled"], 0)
        self.assertFalse(record["judge_input_degraded"])
        self.assertEqual(record["source_csv"], judged.name)

    def test_relabelled_question_is_excluded_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp), relabel={1: "inconsistent", 4: "correct"})
            rows, record = build_one(judged, chunk_rows=3, token_len=False, noise_flip_min_votes=1)
        by_idx = rows.set_index("original_index")["exclude_reason"]
        self.assertEqual(by_idx.loc[1], "baseline_relabelled")
        self.assertEqual(by_idx.loc[4], "baseline_relabelled")  # negative case, now correct
        self.assertTrue(pd.isna(by_idx.loc[7]))
        self.assertEqual(record["n_baseline_relabelled"], 2)
        self.assertEqual(record["n_baseline_answer_mismatch"], 0)  # the answer itself did not move

    def test_source_without_reasoning_column_is_degraded(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp), drop_columns=("reasoning",))
            rows, record = build_one(judged, chunk_rows=100, token_len=False, noise_flip_min_votes=1)
        self.assertTrue(record["judge_input_degraded"])
        self.assertEqual(set(rows["exclude_reason"]), {"judge_input_degraded"})
        # Restoring the column clears the flag on the next build.
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            rows, record = build_one(judged, chunk_rows=100, token_len=False, noise_flip_min_votes=1)
        self.assertFalse(record["judge_input_degraded"])
        self.assertNotIn("judge_input_degraded", set(rows["exclude_reason"].dropna()))

    def test_without_sidecar_falls_back_to_baseline_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp), sidecar=False)
            with patch.dict("os.environ", {"DATA_ROOT": tmp}):
                rows, record = build_one(judged, chunk_rows=100, token_len=False, noise_flip_min_votes=1)
        self.assertEqual(record["baseline_csv"], str(Path(tmp) / "baselines" / "qwen3.5-9b_medqa-test_0911_baseline.csv"))
        self.assertTrue(rows["max_tokens_used"].isna().all())
        self.assertEqual(set(rows["dataset"]), {"medqa"})
        self.assertEqual(set(rows["subject_model_id"]), {"Qwen/Qwen3.5-9B"})  # registry lookup

    def test_token_len_uses_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            with patch("src.scripts.build_rollout_manifest.make_token_counter",
                       return_value=lambda texts: [len(t) for t in texts]):
                rows, record = build_one(judged, chunk_rows=100, token_len=True, noise_flip_min_votes=1)
        self.assertTrue(record["trace_token_len"])
        self.assertEqual(rows.set_index("original_index").loc[1, "trace_token_len"], len("the key says B"))


class TestMain(unittest.TestCase):
    def test_writes_manifest_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            out = Path(tmp) / "m.parquet"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["--dir", str(judged.parent), "--output", str(out), "--no-token-len",
                             "--noise-flip-min-votes", "2"])
            self.assertEqual(code, 0, buf.getvalue())
            manifest = read_manifest(out)
            meta = read_manifest_meta(out)
        self.assertEqual(len(manifest), 8)
        self.assertEqual(meta["noise_flip_min_votes"], 2)
        self.assertEqual(meta["sources"][0]["source_csv"], judged.name)
        self.assertIn("8 rollouts from 1 judged CSV(s)", buf.getvalue())

    def test_rebuild_carries_the_split_over(self):
        from src.lib.splits import assign_splits
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            out = Path(tmp) / "m.parquet"
            argv = ["--dir", str(judged.parent), "--output", str(out), "--no-token-len"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            first = assign_splits(read_manifest(out), seed=5)
            write_manifest(first, out, {**read_manifest_meta(out), "split": {"seed": 5}})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(main(argv), 0)
            rebuilt = read_manifest(out)
            meta = read_manifest_meta(out)
        self.assertEqual(list(rebuilt["split"]), list(first.set_index("rollout_id").loc[rebuilt["rollout_id"], "split"]))
        self.assertEqual(meta["split"]["seed"], 5)
        self.assertEqual(meta["split"]["carried_over"]["n_unassigned"], 0)
        self.assertIn("split carried over", buf.getvalue())

    def test_smoke_runs_skipped_unless_included(self):
        smoke = "qwen3.5-9b_medqa-test_smoke_baseline_hinted_rollouts"
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            write_sources(Path(tmp), stem=smoke)
            out = Path(tmp) / "m.parquet"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(main(["--dir", str(judged.parent), "--output", str(out), "--no-token-len"]), 0)
            manifest = read_manifest(out)
            meta = read_manifest_meta(out)
            self.assertEqual(set(manifest["run"]), {"medqa-test_0911"})
            self.assertEqual([s["source_csv"] for s in meta["sources"]], [judged.name])
            self.assertEqual(meta["skipped_sources"], [{"source_csv": f"{smoke}_judged.csv", "reason": "smoke"}])
            self.assertIn("skipping 1 smoke run(s)", buf.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--dir", str(judged.parent), "--output", str(out), "--no-token-len",
                                       "--include-smoke"]), 0)
            manifest = read_manifest(out)
            self.assertEqual(set(manifest["run"]), {"medqa-test_0911", "medqa-test_smoke"})
            self.assertEqual(read_manifest_meta(out)["skipped_sources"], [])

    def test_rejects_noise_flip_min_votes_below_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf), self.assertRaises(SystemExit):
                main(["--dir", str(judged.parent), "--output", str(Path(tmp) / "m.parquet"),
                      "--noise-flip-min-votes", "0"])
        self.assertIn("must be >= 1", buf.getvalue())

    def test_only_filter_and_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                code = main(["--dir", str(judged.parent), "--output", str(Path(tmp) / "m.parquet"),
                             "--only", "no-such-file"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()


class TestLabelOverrides(unittest.TestCase):
    def test_overrides_applied_at_build_and_skippable(self):
        from src.lib.rollout_manifest import OVERRIDE_COLUMNS

        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            overrides = Path(tmp) / "judge_label_overrides.csv"
            pd.DataFrame([{
                "rollout_id": "qwen3.5-9b:medqa-test_0911:expert_opinion:1", "judge_label_final": 0,
                "judge_label_final_model": "anthropic/claude-opus-5", "judge_confidence_final": 0.9,
                "judge_role_final": "none", "hint_quote_final": "", "override_reason": "validation_sample",
                "judged_utc": "2026-09-15T00:00:00+00:00",
            }, {
                "rollout_id": "other:run:hint:7", "judge_label_final": 1,
                "judge_label_final_model": "anthropic/claude-opus-5", "judge_confidence_final": 0.9,
                "judge_role_final": "", "hint_quote_final": "", "override_reason": "arbiter_cache",
                "judged_utc": "2026-09-15T00:00:00+00:00",
            }], columns=OVERRIDE_COLUMNS).to_csv(overrides, index=False)
            out = Path(tmp) / "m.parquet"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["--dir", str(judged.parent), "--output", str(out), "--no-token-len",
                             "--label-overrides", str(overrides)])
            self.assertEqual(code, 0, buf.getvalue())
            manifest = read_manifest(out)
            meta = read_manifest_meta(out)
            row = manifest[manifest["original_index"] == 1].iloc[0]
            self.assertEqual((int(row["judge_label"]), int(row["judge_label_final"])), (1, 0))
            self.assertEqual(row["judge_label_final_model"], "anthropic/claude-opus-5")
            self.assertEqual((meta["label_overrides"]["n_applied"], meta["label_overrides"]["n_unknown"]), (1, 1))
            self.assertIn("label overrides: 1 of 2 applied", buf.getvalue())

            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--dir", str(judged.parent), "--output", str(out), "--no-token-len",
                             "--label-overrides", str(overrides), "--no-label-overrides"])
            self.assertEqual(code, 0)
            manifest = read_manifest(out)
            row = manifest[manifest["original_index"] == 1].iloc[0]
            self.assertEqual(int(row["judge_label_final"]), 1)
            self.assertIsNone(read_manifest_meta(out)["label_overrides"])

    def test_missing_overrides_after_a_build_that_applied_them_is_refused(self):
        from src.lib.rollout_manifest import OVERRIDE_COLUMNS

        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            overrides = Path(tmp) / "judge_label_overrides.csv"
            pd.DataFrame([{
                "rollout_id": "qwen3.5-9b:medqa-test_0911:expert_opinion:1", "judge_label_final": 0,
                "judge_label_final_model": "anthropic/claude-opus-5", "judge_confidence_final": 0.9,
                "judge_role_final": "none", "hint_quote_final": "", "override_reason": "validation_sample",
                "judged_utc": "2026-09-15T00:00:00+00:00",
            }], columns=OVERRIDE_COLUMNS).to_csv(overrides, index=False)
            out = Path(tmp) / "m.parquet"
            common = ["--dir", str(judged.parent), "--output", str(out), "--no-token-len"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([*common, "--label-overrides", str(overrides)]), 0)
            overrides.unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "does not exist"):
                    main([*common, "--label-overrides", str(overrides)])
                self.assertEqual(main([*common, "--label-overrides", str(overrides), "--no-label-overrides"]), 0)
            self.assertIsNone(read_manifest_meta(out)["label_overrides"])

    def test_missing_overrides_file_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged = write_sources(Path(tmp))
            out = Path(tmp) / "m.parquet"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--dir", str(judged.parent), "--output", str(out), "--no-token-len",
                             "--label-overrides", str(Path(tmp) / "absent.csv")])
            self.assertEqual(code, 0)
            meta = read_manifest_meta(out)
            self.assertFalse(meta["label_overrides"]["present"])
            self.assertEqual(meta["label_overrides"]["n_applied"], 0)
