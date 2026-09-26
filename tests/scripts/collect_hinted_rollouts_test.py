import contextlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import yaml

import src.lib.hinted_rollouts as hr_lib
from src.lib.baseline import encode_additional_fields
from src.lib.hints import HintResult

MODULE_NAME = "src.scripts.collect_hinted_rollouts"

mod = importlib.import_module(MODULE_NAME)

DEFAULT_ROLLOUT = "<think>thinking</think>\n<answer>B</answer>"


class FakeTokenizer:
    """apply_chat_template returns the last message's content: a unique key for FakeLLM scripting."""

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append({"messages": messages, "kwargs": kwargs})
        return messages[-1]["content"]


class FakeLLM:
    def __init__(self, script=None):
        self.script = dict(script or {})
        self.calls = []
        self.sampling_params = None

    def generate(self, prompts, sampling_params):
        self.calls.append(list(prompts))
        self.sampling_params = sampling_params
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text=self.script.get(p, DEFAULT_ROLLOUT))]
            )
            for p in prompts
        ]


class PipelineHarness(unittest.TestCase):
    """Shared fixtures for end-to-end main() tests (vLLM, hints, model mocked)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.baseline_csv = self.root / "baseline.csv"
        self.output_csv = self.root / "out" / "hinted_rollouts.csv"

    def _write_baseline(self, correct_indices, incorrect_indices=()):
        rows = []
        for idx in list(correct_indices) + list(incorrect_indices):
            correct = idx in correct_indices
            rows.append(
                {
                    "original_index": idx,
                    "question": f"q{idx}",
                    "choices": json.dumps(["w", "x", "y", "z"]),
                    "prompt": f"Q{idx}",
                    "groundtruth": "A",
                    "baseline_answer": "A" if correct else "B",
                    "correct": correct,
                    "additional_fields": encode_additional_fields(
                        {"subject": "s", "answers_order": "1,0,2,3"}
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(self.baseline_csv, index=False)

    def _write_config(self, **overrides):
        cfg = {
            "model_name": "Qwen/Qwen3-8B",
            "baseline_csv": str(self.baseline_csv),
            "output_csv": str(self.output_csv),
            "hints": ["authority", "metadata"],
            "max_tokens": 64,
            "temperature": 0.0,
            "chunk_size": 512,
        }
        cfg.update(overrides)
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def _fake_get_hinted(self, calls):
        def fake(hint_input, hint_names, hint_n_examples=None):
            name = hint_names[0]
            calls.append({"input": hint_input, "hint": name})
            if hint_input.suggested_answer:
                return {
                    name: HintResult(
                        prompt=f"{name}::{hint_input.prompt}",
                        hinted_answer=hint_input.suggested_answer,
                        groundtruth=hint_input.groundtruth,
                    )
                }
            if name in ("pushback", "post_hoc"):
                messages = [
                    {"role": "user", "content": hint_input.prompt},
                    {"role": "assistant", "content": "The answer is A."},
                    {"role": "user", "content": f"MT::{name}::{hint_input.prompt}"},
                ]
                return {
                    name: HintResult(
                        prompt="",
                        hinted_answer="B",
                        groundtruth=hint_input.groundtruth,
                        messages=messages,
                    )
                }
            return {
                name: HintResult(
                    prompt=f"{name}::{hint_input.prompt}",
                    hinted_answer="B",
                    groundtruth=hint_input.groundtruth,
                )
            }

        return fake

    def _run_main(self, cfg_path, llm, hint_calls, argv_extra=()):
        tok = FakeTokenizer()
        self.tok = tok
        with contextlib.ExitStack() as stack:
            self.load_model = stack.enter_context(
                mock.patch.object(mod, "load_model_vLLM", return_value=(llm, tok))
            )
            stack.enter_context(mock.patch.object(mod, "load_tokenizer", return_value=tok))
            stack.enter_context(
                mock.patch.object(
                    hr_lib, "get_hinted_prompts", self._fake_get_hinted(hint_calls)
                )
            )
            stack.enter_context(
                mock.patch.object(
                    sys,
                    "argv",
                    ["collect_hinted_rollouts", "--config", str(cfg_path), *argv_extra],
                )
            )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            mod.main()

    def _read_output(self):
        return pd.read_csv(self.output_csv, keep_default_na=False)


class TestMainPipeline(PipelineHarness):
    """End-to-end tests of main() with vLLM, hints, and the model mocked."""

    def test_only_correct_rows_and_requested_hints(self):
        self._write_baseline([0, 1], incorrect_indices=[2])
        cfg = self._write_config()
        hint_calls = []
        self._run_main(cfg, FakeLLM(), hint_calls)

        out = self._read_output()
        self.assertEqual(list(out.columns), mod.OUTPUT_COLUMNS)
        self.assertEqual(len(out), 4)  # 2 correct questions x 2 hints
        self.assertEqual(
            sorted(zip(out["original_index"], out["hint_name"])),
            [(0, "authority"), (0, "metadata"), (1, "authority"), (1, "metadata")],
        )
        self.assertNotIn(2, out["original_index"].tolist())
        self.assertEqual(out["sample_type"].unique().tolist(), ["positive"])
        self.assertEqual(out["final_answer"].unique().tolist(), ["B"])
        self.assertEqual(out["rollout"].unique().tolist(), [DEFAULT_ROLLOUT])
        self.assertEqual(out["reasoning"].unique().tolist(), ["thinking"])
        self.assertEqual(out["baseline_answer"].unique().tolist(), ["A"])
        self.assertEqual(out["groundtruth"].unique().tolist(), ["A"])
        self.assertEqual(out["hinted_answer"].unique().tolist(), ["B"])
        row0 = out[(out["original_index"] == 0) & (out["hint_name"] == "authority")]
        self.assertEqual(row0["hinted_prompt"].item(), "authority::Q0")
        self.assertEqual(
            out["additional_fields"].unique().tolist(),
            [encode_additional_fields({"subject": "s", "answers_order": "1,0,2,3"})],
        )

    def test_resume_refuses_csv_with_old_header(self):
        self._write_baseline([0, 1])
        cfg = self._write_config()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        old_cols = [c for c in mod.OUTPUT_COLUMNS if c not in ("prompt", "reasoning")]
        pre = {**{c: "old" for c in old_cols}, "original_index": 0, "hint_name": "authority"}
        pd.DataFrame([pre], columns=old_cols).to_csv(
            self.output_csv, index=False
        )
        with self.assertRaises(ValueError) as ctx:
            self._run_main(cfg, FakeLLM(), [])
        self.assertIn("prompt", str(ctx.exception))
        self.assertEqual(len(pd.read_csv(self.output_csv)), 1)  # untouched

    def test_resume_skips_existing_pairs(self):
        self._write_baseline([0, 1])
        cfg = self._write_config()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pre = pd.DataFrame(
            [
                {
                    **{c: "old" for c in mod.OUTPUT_COLUMNS},
                    "original_index": 0,
                    "hint_name": "authority",
                }
            ],
            columns=mod.OUTPUT_COLUMNS,
        )
        pre.to_csv(self.output_csv, index=False)

        hint_calls = []
        self._run_main(cfg, FakeLLM(), hint_calls)

        out = self._read_output()
        self.assertEqual(len(out), 4)  # 1 pre-existing + 3 new
        pairs = sorted(zip(out["original_index"].astype(int), out["hint_name"]))
        self.assertEqual(
            pairs,
            [(0, "authority"), (0, "metadata"), (1, "authority"), (1, "metadata")],
        )
        built = {(int(c["input"].prompt[1:]), c["hint"]) for c in hint_calls}
        self.assertNotIn((0, "authority"), built)
        # No duplicated header row: every original_index parses as an int.
        self.assertTrue(out["original_index"].astype(str).str.isdigit().all())

    def test_nothing_pending_never_loads_the_model(self):
        self._write_baseline([0, 1], incorrect_indices=[2])
        cfg_path = self._write_config()
        hint_calls = []
        self._run_main(cfg_path, FakeLLM(), hint_calls)
        n_before = len(self._read_output())
        self.assertTrue(self.load_model.called)
        self._run_main(cfg_path, FakeLLM(), hint_calls)
        self.assertFalse(self.load_model.called)
        self.assertEqual(len(self._read_output()), n_before)

    def test_multi_turn_hint_serializes_messages(self):
        self._write_baseline([0])
        cfg = self._write_config(hints=["pushback"])
        hint_calls = []
        self._run_main(cfg, FakeLLM(), hint_calls)

        out = self._read_output()
        self.assertEqual(len(out), 1)
        messages = json.loads(out["hinted_prompt"].item())
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[0]["content"], "Q0")
        # The templated chat prepends the system prompt to the hint's turns.
        templated = self.tok.template_calls[0]["messages"]
        self.assertEqual(
            [m["role"] for m in templated], ["system", "user", "assistant", "user"]
        )

    def test_unparseable_rollout_records_empty_final_answer(self):
        self._write_baseline([0])
        cfg = self._write_config(hints=["authority"])
        llm = FakeLLM(script={"authority::Q0": "<think>no closing answer"})
        self._run_main(cfg, llm, [])

        out = self._read_output()
        self.assertEqual(out["final_answer"].item(), "")
        self.assertEqual(out["rollout"].item(), "<think>no closing answer")

    def test_chunking_appends_per_chunk(self):
        self._write_baseline([0, 1, 2])
        cfg = self._write_config(hints=["authority"], chunk_size=2)
        llm = FakeLLM()
        self._run_main(cfg, llm, [])

        self.assertEqual([len(c) for c in llm.calls], [2, 1])
        out = self._read_output()
        self.assertEqual(len(out), 3)

    def test_example_pool_excludes_evaluated_question(self):
        self._write_baseline([0, 1, 2])
        cfg = self._write_config(hints=["visual_pattern"])
        hint_calls = []
        self._run_main(cfg, FakeLLM(), hint_calls)

        self.assertEqual(len(hint_calls), 3)
        for call in hint_calls:
            pool_questions = call["input"].example_pool["question"].tolist()
            self.assertNotIn(call["input"].question, pool_questions)
            self.assertTrue(pool_questions)

    def test_unknown_hint_raises(self):
        self._write_baseline([0])
        cfg = self._write_config(hints=["not_a_hint"])
        with self.assertRaises(ValueError):
            self._run_main(cfg, FakeLLM(), [])

    def test_sampling_knobs_from_config(self):
        self._write_baseline([0])
        # top_p/top_k omitted -> the 0.95/20 defaults.
        cfg = self._write_config(hints=["authority"], temperature=0.7)
        llm = FakeLLM()
        self._run_main(cfg, llm, [])
        self.assertEqual(llm.sampling_params.temperature, 0.7)
        self.assertEqual(llm.sampling_params.top_p, 0.95)
        self.assertEqual(llm.sampling_params.top_k, 20)
        # Explicit YAML values win; a fresh output CSV so resume does not skip the pair.
        cfg = self._write_config(
            hints=["authority"], temperature=0.7, top_p=0.9, top_k=40,
            output_csv=str(self.root / "out" / "hinted_rollouts2.csv"),
        )
        llm = FakeLLM()
        self._run_main(cfg, llm, [])
        self.assertEqual(llm.sampling_params.top_p, 0.9)
        self.assertEqual(llm.sampling_params.top_k, 40)


class TestCaseModes(PipelineHarness):
    """The `cases` config value / --cases flag select which baseline rows get hinted."""

    def test_negative_cases_hints_point_at_groundtruth(self):
        self._write_baseline([0, 1], incorrect_indices=[2])
        cfg = self._write_config(cases="negative_cases", hints=["authority"])
        hint_calls = []
        self._run_main(cfg, FakeLLM(), hint_calls)

        out = self._read_output()
        self.assertEqual(out["original_index"].tolist(), [2])
        self.assertEqual(out["sample_type"].unique().tolist(), ["negative"])
        # Baseline answer B, groundtruth A: the hint is forced onto the groundtruth.
        self.assertEqual(out["hinted_answer"].item(), "A")
        self.assertEqual(out["baseline_answer"].item(), "B")
        self.assertEqual([c["input"].suggested_answer for c in hint_calls], ["A"])

    def test_both_cases_tags_sample_type(self):
        self._write_baseline([0], incorrect_indices=[2])
        cfg = self._write_config(cases="both", hints=["authority"])
        self._run_main(cfg, FakeLLM(), [])

        out = self._read_output()
        by_idx = dict(zip(out["original_index"], out["sample_type"]))
        self.assertEqual(by_idx, {0: "positive", 2: "negative"})

    def test_cli_cases_overrides_config(self):
        self._write_baseline([0], incorrect_indices=[2])
        cfg = self._write_config(cases="positive_cases", hints=["authority"])
        self._run_main(cfg, FakeLLM(), [], argv_extra=["--cases", "negative_cases"])

        out = self._read_output()
        self.assertEqual(out["original_index"].tolist(), [2])

    def test_invalid_cases_raises(self):
        self._write_baseline([0])
        cfg = self._write_config(cases="everything")
        with self.assertRaises(ValueError):
            self._run_main(cfg, FakeLLM(), [])


class TestLoadDonePairs(unittest.TestCase):
    """load_done_pairs tolerates missing, empty, and schema-less files."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = Path(self.tmpdir.name) / "out.csv"

    def test_missing_file(self):
        self.assertEqual(mod.load_done_pairs(self.path), set())

    def test_empty_file(self):
        self.path.write_text("")
        self.assertEqual(mod.load_done_pairs(self.path), set())

    def test_wrong_schema(self):
        pd.DataFrame([{"foo": 1}]).to_csv(self.path, index=False)
        self.assertEqual(mod.load_done_pairs(self.path), set())

    def test_pairs(self):
        pd.DataFrame(
            [
                {"original_index": 3, "hint_name": "authority"},
                {"original_index": 4, "hint_name": "metadata"},
            ]
        ).to_csv(self.path, index=False)
        self.assertEqual(
            mod.load_done_pairs(self.path), {(3, "authority"), (4, "metadata")}
        )


if __name__ == "__main__":
    unittest.main()
