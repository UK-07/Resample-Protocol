"""Tests for src/scripts/resample_hinted_rollouts.py (vLLM and the model mocked)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml

from src.scripts import resample_hinted_rollouts as mod
from src.scripts.select_resample_set import main as select_main
from tests.scripts.resample_fixtures import MODEL_ID, STEM, make_data_root

ROLLOUT = "<think>cot</think><answer>B</answer>"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + f"|{kwargs}"


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, params):
        self.calls.append((list(prompts), list(params)))
        return [SimpleNamespace(outputs=[SimpleNamespace(text=ROLLOUT)]) for _ in prompts]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def write_config(root: Path, repo_configs: Path) -> Path:
    cfg = {
        "items_dir": str(root / "resample" / "items"),
        "output_dir": str(root / "resample" / "rollouts"),
        "model_bases": {MODEL_ID: str(repo_configs / "qwen_3-8b_base.yaml")},
        "sample_seeds": [43, 44], "temperature": 0.7, "chunk_size": 5,
        "inference": {"max_tokens": 24576, "max_model_len": 32768},
    }
    path = root / "resample.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class HelpersTest(unittest.TestCase):
    def test_item_choices_optional(self):
        # Item files written before the `choices` column parse without the check.
        self.assertEqual(mod.item_choices({"choices": '["a", "b"]'}), ["a", "b"])
        self.assertIsNone(mod.item_choices({"choices": float("nan")}))
        self.assertIsNone(mod.item_choices({"choices": ""}))
        self.assertIsNone(mod.item_choices({"hinted_prompt": "q"}))
        self.assertIsNone(mod.item_choices(pd.Series({"choices": "not json"})))

    def test_chat_messages_single_and_multi_turn(self):
        single = mod.chat_messages("hinted Q", "SYS")
        self.assertEqual(single, [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hinted Q"}])
        turns = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "sure?"}]
        multi = mod.chat_messages(json.dumps(turns), "SYS")
        self.assertEqual(multi, [{"role": "system", "content": "SYS"}] + turns)
        # A user prompt that merely starts with "[" is not a message list.
        self.assertEqual(mod.chat_messages("[Note] Q", "SYS")[1]["content"], "[Note] Q")

    def test_resolve_base_checks_model_and_merges_inference(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base.yaml"
            base.write_text(yaml.safe_dump({"model_name": "org/M", "thinking": True,
                                            "inference": {"max_tokens": 8192, "max_num_seqs": 256}}))
            cfg_path = Path(td) / "cfg.yaml"
            cfg = {"model_bases": {"org/M": "base.yaml"}, "inference": {"max_tokens": 24576}}
            merged = mod.resolve_base(cfg, cfg_path, "org/M")
            self.assertEqual(merged["inference"], {"max_tokens": 24576, "max_num_seqs": 256})
            with self.assertRaises(ValueError):
                mod.resolve_base(cfg, cfg_path, "org/Other")
            cfg["model_bases"] = {"org/Other": "base.yaml"}
            with self.assertRaises(ValueError):
                mod.resolve_base(cfg, cfg_path, "org/Other")

    def test_build_items_skips_done_pairs_and_shares_chat_text(self):
        rows = pd.DataFrame({
            "rollout_id": ["a", "b"], "role": ["used_candidate", "control"], "n_options": [4, 4],
            "original_index": [1, 2], "sample_type": "positive", "hint_name": "metadata", "prompt": "q",
            "hinted_prompt": ["hq1", "hq2"], "hinted_answer": "B", "baseline_answer": "A", "groundtruth": "A",
            "additional_fields": "{}",
        })
        thinking = SimpleNamespace(system_prompt="SYS", enable_thinking=True)
        items = mod.build_items(rows, [43, 44], {43: {(1, "metadata")}}, FakeTokenizer(), thinking)
        self.assertEqual([(it["original_index"], it["seed"]) for it in items], [(1, 44), (2, 43), (2, 44)])
        self.assertEqual(items[1]["chat_text"], items[2]["chat_text"])
        self.assertIn("system:SYS|user:hq2", items[1]["chat_text"])
        self.assertEqual(items[0]["option_letters"], ["A", "B", "C", "D"])


class MainTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = make_data_root(Path(self.td.name))
        with patch.dict(os.environ, {"DATA_ROOT": str(self.root)}), contextlib.redirect_stdout(io.StringIO()):
            select_main([])
        self.config = write_config(self.root, Path(mod.__file__).resolve().parents[2] / "configs")

    def run_main(self, argv):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = FakeSamplingParams
        llm = FakeLLM()
        out = io.StringIO()
        with patch.dict(os.environ, {"DATA_ROOT": str(self.root)}), \
                patch.dict(sys.modules, {"vllm": fake_vllm}), \
                patch("src.lib.model_utils.load_model_vLLM", return_value=(llm, FakeTokenizer())), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = mod.main(argv)
        return code, llm, out.getvalue()

    def test_source_thinking_mismatch_is_refused(self):
        sidecar = next((self.root / "resample" / "items").glob("*_resample_items.meta.json"))
        meta = json.loads(sidecar.read_text())
        meta["source_recipe"]["thinking"] = False
        sidecar.write_text(json.dumps(meta))
        with self.assertRaises(ValueError) as ctx:
            self.run_main(["--config", str(self.config), "--dry-run"])
        self.assertIn("thinking", str(ctx.exception))

    def test_dry_run_plans_without_loading(self):
        code, llm, out = self.run_main(["--config", str(self.config), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(llm.calls, [])
        self.assertIn("16 items × 2 seeds", out)
        self.assertIn("32 rollouts to generate", out)
        self.assertIn("max_tokens=24576", out)

    def test_generates_one_csv_per_seed_and_resumes(self):
        code, llm, out = self.run_main(["--config", str(self.config)])
        self.assertEqual(code, 0)
        self.assertEqual(sum(len(p) for p, _ in llm.calls), 32)
        # Every request carried its seed's sampling params.
        seeds = {p.kwargs["seed"] for _, params in llm.calls for p in params}
        self.assertEqual(seeds, {43, 44})
        self.assertTrue(all(p.kwargs["max_tokens"] == 24576 for _, params in llm.calls for p in params))
        out_dir = self.root / "resample" / "rollouts"
        for seed in (43, 44):
            csv = out_dir / f"{STEM}_rs{seed}.csv"
            df = pd.read_csv(csv)
            self.assertEqual(list(df.columns), mod.OUTPUT_COLUMNS)
            self.assertEqual(len(df), 16)
            self.assertEqual(set(df["final_answer"]), {"B"})
            self.assertEqual(set(df["reasoning"]), {"cot"})
            meta = json.loads(csv.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["seed"], seed)
            self.assertEqual(meta["max_tokens"], 24576)
            self.assertEqual(meta["resample"]["source_csv"], f"{STEM}_judged.csv")
            self.assertEqual(meta["resample"]["k"], 2)
            self.assertEqual(meta["model_name"], MODEL_ID)
        # A rerun has nothing to do; after dropping seed 44's file only that seed regenerates.
        code, llm, out = self.run_main(["--config", str(self.config)])
        self.assertEqual(llm.calls, [])
        (out_dir / f"{STEM}_rs44.csv").unlink()
        code, llm, out = self.run_main(["--config", str(self.config)])
        self.assertEqual(sum(len(p) for p, _ in llm.calls), 16)
        self.assertEqual(len(pd.read_csv(out_dir / f"{STEM}_rs44.csv")), 16)

    def test_seed_subset_generates_only_those_seeds_and_keeps_k(self):
        # Disjoint --seeds processes write only their seeds' CSVs; the sidecar's k counts every configured seed.
        code, llm, out = self.run_main(["--config", str(self.config), "--seeds", "44"])
        self.assertEqual(code, 0)
        self.assertIn("16 rollouts to generate", out)
        self.assertIn("seeds=[44] (of [43, 44])", out)
        self.assertEqual({p.kwargs["seed"] for _, params in llm.calls for p in params}, {44})
        out_dir = self.root / "resample" / "rollouts"
        self.assertFalse((out_dir / f"{STEM}_rs43.csv").exists())
        self.assertEqual(len(pd.read_csv(out_dir / f"{STEM}_rs44.csv")), 16)
        meta = json.loads((out_dir / f"{STEM}_rs44.meta.json").read_text())
        self.assertEqual(meta["resample"]["k"], 2)
        self.assertEqual(meta["seed"], 44)
        # The other half picks up only seed 43; the full run afterwards has nothing left.
        code, llm, out = self.run_main(["--config", str(self.config), "--seeds", "43"])
        self.assertEqual({p.kwargs["seed"] for _, params in llm.calls for p in params}, {43})
        code, llm, out = self.run_main(["--config", str(self.config)])
        self.assertEqual(llm.calls, [])
        # A seed outside the config is a usage error, not a silent no-op.
        with self.assertRaises(SystemExit):
            self.run_main(["--config", str(self.config), "--seeds", "45"])

    def test_schema_drift_is_refused(self):
        out_dir = self.root / "resample" / "rollouts"
        out_dir.mkdir(parents=True)
        pd.DataFrame({"original_index": [1], "hint_name": ["metadata"]}).to_csv(out_dir / f"{STEM}_rs43.csv", index=False)
        with self.assertRaises(ValueError):
            self.run_main(["--config", str(self.config), "--dry-run"])

    def test_model_filter(self):
        code, _, out = self.run_main(["--config", str(self.config), "--dry-run", "--model", "qwen3-8b"])
        self.assertEqual(code, 0)
        code, _, out = self.run_main(["--config", str(self.config), "--dry-run", "--model", "org/None"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
