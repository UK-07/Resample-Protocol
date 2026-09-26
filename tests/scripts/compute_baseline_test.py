import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.lib.baseline import build_verified_pool, load_baseline
from src.lib.config import DEFAULT_TOP_K, DEFAULT_TOP_P
from src.lib.prompts import SYSTEM_PROMPT_PLAIN
from src.lib.paths import resolve_data_path
from src.scripts import compute_baseline as cb

CANONICAL_COLUMNS = [
    "original_index",
    "question",
    "choices",
    "prompt",
    "groundtruth",
    "baseline_answer",
    "correct",
    "baseline_status",
    "additional_fields",
]

SAMPLING_COLUMNS = [
    "sample_answers",
    "sample_rollouts",
    "n_top_votes",
    "self_consistency",
    "n_correct_samples",
]


def rollout_text(letter):
    if letter == "":
        return "<think>unsure</think> no committed choice"
    return f"<think>step by step</think>\n<answer>{letter}</answer>"


def make_generator(spec_by_prompt):
    def fake(model, tokenizer, batch_messages, *, n=1, **kwargs):
        out = []
        for messages in batch_messages:
            spec = spec_by_prompt[messages[-1]["content"]]
            if isinstance(spec, list):
                out.append([rollout_text(s) for s in spec])
            else:
                out.append([rollout_text(spec)] * n)
        return out
    return fake


def subject_df():
    return pd.DataFrame({
        "prompt": ["P0", "P1", "P2", "P3"],
        "groundtruth": ["A", "B", "C", "D"],
        "additional_fields": [
            {"subject": "hist"},
            {"subject": "hist"},
            {"subject": "math"},
            {"subject": "math"},
        ],
        "question": ["q0", "q1", "q2", "q3"],
        "choices": [["w", "x", "y", "z"]] * 4,
        "answer_idx": [0, 1, 2, 3],
        "original_index": [10, 11, 12, 13],
    })


def subjectless_df():
    df = subject_df()
    df["additional_fields"] = [{} for _ in range(len(df))]
    return df


class ResolveSamplingTest(unittest.TestCase):

    def test_none_is_single_greedy_pass(self):
        self.assertEqual((1, 0.0, False), cb.resolve_sampling(None, None))

    def test_one_is_single_greedy_pass(self):
        self.assertEqual((1, 0.0, False), cb.resolve_sampling(1, None))

    def test_k_with_default_temperature(self):
        self.assertEqual(
            (5, cb.DEFAULT_SAMPLING_TEMPERATURE, True), cb.resolve_sampling(5, None)
        )

    def test_k_with_explicit_temperature(self):
        self.assertEqual((3, 0.2, True), cb.resolve_sampling(3, 0.2))

    def test_temperature_without_sampling_rejected(self):
        with self.assertRaises(ValueError):
            cb.resolve_sampling(None, 0.7)

    def test_temperature_with_k_one_rejected(self):
        with self.assertRaises(ValueError):
            cb.resolve_sampling(1, 0.7)

    def test_zero_samples_rejected(self):
        with self.assertRaises(ValueError):
            cb.resolve_sampling(0, None)

    def test_negative_samples_rejected(self):
        with self.assertRaises(ValueError):
            cb.resolve_sampling(-2, None)


class ResolveThresholdTest(unittest.TestCase):

    def test_greedy_unset_returns_none(self):
        self.assertIsNone(cb.resolve_threshold(None, 1, False))

    def test_greedy_with_threshold_rejected(self):
        with self.assertRaises(ValueError):
            cb.resolve_threshold(1, 1, False)

    def test_default_majority_odd_k(self):
        self.assertEqual(3, cb.resolve_threshold(None, 5, True))

    def test_default_majority_even_k(self):
        self.assertEqual(3, cb.resolve_threshold(None, 4, True))

    def test_default_majority_k_two(self):
        self.assertEqual(2, cb.resolve_threshold(None, 2, True))

    def test_explicit_threshold_within_bounds(self):
        self.assertEqual(1, cb.resolve_threshold(1, 5, True))
        self.assertEqual(5, cb.resolve_threshold(5, 5, True))

    def test_threshold_below_one_rejected(self):
        with self.assertRaises(ValueError):
            cb.resolve_threshold(0, 5, True)

    def test_threshold_above_n_rejected(self):
        with self.assertRaises(ValueError):
            cb.resolve_threshold(6, 5, True)


class ThresholdVoteTest(unittest.TestCase):
    """threshold_vote as re-exported through the script module."""

    def test_empty_list(self):
        self.assertEqual(("", 0, 0.0), cb.threshold_vote([], 1))

    def test_all_unparsed(self):
        self.assertEqual(("", 0, 0.0), cb.threshold_vote(["", "", ""], 1))

    def test_majority_accepted(self):
        answer, top, sc = cb.threshold_vote(["A", "A", "B"], 2)
        self.assertEqual("A", answer)
        self.assertEqual(2, top)
        self.assertAlmostEqual(2 / 3, sc)

    def test_below_threshold_rejected(self):
        answer, top, sc = cb.threshold_vote(["A", "B", "C"], 2)
        self.assertEqual("", answer)
        self.assertEqual(1, top)
        self.assertAlmostEqual(1 / 3, sc)

    def test_tie_breaks_lexicographically(self):
        answer, top, sc = cb.threshold_vote(["B", "B", "A", "A"], 2)
        self.assertEqual("A", answer)
        self.assertEqual(2, top)
        self.assertAlmostEqual(0.5, sc)

    def test_unparsed_count_in_denominator(self):
        answer, top, sc = cb.threshold_vote(["A", "A", "", ""], 2)
        self.assertEqual("A", answer)
        self.assertEqual(2, top)
        self.assertAlmostEqual(0.5, sc)

    def test_single_answer_threshold_one(self):
        self.assertEqual(("C", 1, 1.0), cb.threshold_vote(["C"], 1))


class AssertColumnsCompatibleTest(unittest.TestCase):
    """The resume corruption guard."""

    def _write(self, tmpdir, columns):
        path = Path(tmpdir) / "b.csv"
        pd.DataFrame({c: ["x"] for c in columns}).to_csv(path, index=False)
        return path

    def test_new_csv_ok(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, ["original_index", "correct", "baseline_status"])
            cb.assert_columns_compatible(path)  # no raise

    def test_legacy_csv_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, ["original_index", "correct"])
            with self.assertRaises(ValueError) as ctx:
                cb.assert_columns_compatible(path)
            self.assertIn("baseline_status", str(ctx.exception))

    def test_empty_csv_ok(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.csv"
            path.write_text("")
            cb.assert_columns_compatible(path)  # no raise


class AssertStoredSamplesHaveDelimiterTest(unittest.TestCase):
    """The stripped-delimiter resume guard."""

    def _write(self, tmpdir, samples):
        path = Path(tmpdir) / "b.csv"
        pd.DataFrame({
            "original_index": range(len(samples)),
            "sample_rollouts": [json.dumps(s) for s in samples],
        }).to_csv(path, index=False)
        return path

    def test_delimited_samples_ok(self):
        from src.lib.model_utils import GEMMA4_DELIMITERS

        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, [["<|channel>thought\nx<channel|><answer>A</answer>"]] * 3)
            cb.assert_stored_samples_have_delimiter(path, GEMMA4_DELIMITERS, thinking_enabled=True)

    def test_stripped_samples_rejected(self):
        from src.lib.model_utils import GEMMA4_DELIMITERS

        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, [["thought\nx<answer>A</answer>"]] * 3)
            with self.assertRaises(ValueError) as ctx:
                cb.assert_stored_samples_have_delimiter(path, GEMMA4_DELIMITERS, thinking_enabled=True)
            self.assertIn("special tokens stripped", str(ctx.exception))
            self.assertIn("<channel|>", str(ctx.exception))

    def test_thinking_off_or_empty_ok(self):
        from src.lib.model_utils import GEMMA4_DELIMITERS

        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, [["thought\nx<answer>A</answer>"]] * 3)
            cb.assert_stored_samples_have_delimiter(path, GEMMA4_DELIMITERS, thinking_enabled=False)
            empty = Path(d) / "empty.csv"
            empty.write_text("")
            cb.assert_stored_samples_have_delimiter(empty, GEMMA4_DELIMITERS, thinking_enabled=True)
            no_col = Path(d) / "nocol.csv"
            pd.DataFrame({"original_index": [1]}).to_csv(no_col, index=False)
            cb.assert_stored_samples_have_delimiter(no_col, GEMMA4_DELIMITERS, thinking_enabled=True)


class BuildLoaderKwargsTest(unittest.TestCase):

    def test_mmlu_defaults(self):
        name, kwargs = cb.build_loader_kwargs({"name": "mmlu", "params": {}}, 42)
        self.assertEqual("mmlu", name)
        self.assertEqual(
            {"config": "all", "split": "test", "subject_filter": None,
             "exclude_subjects": None, "samples_per_subject": None, "seed": 42,
             "random_answers_order": False},
            kwargs,
        )

    def test_mmlu_train_alias_resolved(self):
        _, kwargs = cb.build_loader_kwargs(
            {"name": "mmlu", "params": {"split": "train"}}, 42
        )
        self.assertEqual("auxiliary_train", kwargs["split"])

    def test_mmlu_subjects_mapped_to_subject_filter(self):
        _, kwargs = cb.build_loader_kwargs(
            {"name": "mmlu", "params": {"subjects": ["philosophy"]}}, 42
        )
        self.assertEqual(["philosophy"], kwargs["subject_filter"])

    def test_mmlu_exclude_subjects_passthrough(self):
        _, kwargs = cb.build_loader_kwargs(
            {"name": "mmlu", "params": {"exclude_subjects": ["misc"]}}, 42
        )
        self.assertEqual(["misc"], kwargs["exclude_subjects"])

    def test_mmlu_samples_per_subject_forwarded(self):
        _, kwargs = cb.build_loader_kwargs(
            {"name": "mmlu", "params": {"samples_per_subject": 7}}, 13
        )
        self.assertEqual(7, kwargs["samples_per_subject"])
        self.assertEqual(13, kwargs["seed"])

    def test_mmlu_pro_defaults(self):
        name, kwargs = cb.build_loader_kwargs({"name": "mmlu_pro", "params": {}}, 42)
        self.assertEqual("mmlu_pro", name)
        self.assertEqual(
            {"split": "test", "category_filter": None, "exclude_categories": None,
             "samples_per_category": None, "require_n_options": None, "seed": 42,
             "random_answers_order": False},
            kwargs,
        )

    def test_mmlu_pro_categories_mapped_and_forwarded(self):
        _, kwargs = cb.build_loader_kwargs(
            {"name": "mmlu_pro", "params": {
                "split": "validation", "categories": ["math"],
                "exclude_categories": ["other"], "samples_per_category": 5,
                "require_n_options": 10,
            }}, 13
        )
        self.assertEqual("validation", kwargs["split"])
        self.assertEqual(["math"], kwargs["category_filter"])
        self.assertEqual(["other"], kwargs["exclude_categories"])
        self.assertEqual(5, kwargs["samples_per_category"])
        self.assertEqual(10, kwargs["require_n_options"])
        self.assertEqual(13, kwargs["seed"])

    def test_gpqa_defaults_with_seed(self):
        name, kwargs = cb.build_loader_kwargs({"name": "gpqa"}, 7)
        self.assertEqual("gpqa", name)
        self.assertEqual(
            {"config": "gpqa_diamond", "split": "train", "seed": 7,
             "random_answers_order": False},
            kwargs,
        )

    def test_random_answers_order_forwarded_to_all_loaders(self):
        for block in (
            {"name": "mmlu", "params": {}},
            {"name": "mmlu_pro", "params": {}},
            {"name": "gpqa"},
        ):
            _, kwargs = cb.build_loader_kwargs(block, 42, random_answers_order=True)
            self.assertTrue(kwargs["random_answers_order"])

    def test_gpqa_explicit_params(self):
        _, kwargs = cb.build_loader_kwargs(
            {"name": "gpqa", "params": {"config": "gpqa_main", "split": "train"}}, 1
        )
        self.assertEqual("gpqa_main", kwargs["config"])

    def test_aqua_defaults_and_split(self):
        name, kwargs = cb.build_loader_kwargs({"name": "aqua"}, 5)
        self.assertEqual("aqua", name)
        self.assertEqual(
            {"split": "train", "seed": 5, "random_answers_order": False}, kwargs
        )
        _, kwargs = cb.build_loader_kwargs(
            {"name": "aqua", "params": {"split": "test"}}, 5
        )
        self.assertEqual("test", kwargs["split"])

    def test_every_registered_dataset_has_a_loader_branch(self):
        from src.lib.dataset import DATASET_REGISTRY
        for ds in DATASET_REGISTRY:
            resolved, kwargs = cb.build_loader_kwargs({"name": ds}, 42)
            self.assertEqual(ds, resolved)
            self.assertIn("seed", kwargs)

    def test_default_splits_agree_with_run_pipeline_tag(self):
        # run_pipeline's default split must agree with build_loader_kwargs' or the file is mislabelled.
        from src.lib.dataset import DATASET_REGISTRY
        from src.scripts.run_pipeline import derive_dataset_tag
        for ds in DATASET_REGISTRY:
            if ds == "gpqa":  # tagged by subset, not split
                continue
            _, kwargs = cb.build_loader_kwargs({"name": ds}, 42)
            self.assertEqual(f"{ds}-{kwargs['split']}", derive_dataset_tag({"name": ds}))

    def test_unsupported_dataset_lists_the_registry(self):
        with self.assertRaises(ValueError) as ctx:
            cb.build_loader_kwargs({"name": "imagenet"}, 42)
        msg = str(ctx.exception)
        for ds in ("mmlu", "aqua"):
            self.assertIn(repr(ds), msg)

    def test_commonsense_qa_defaults_and_split(self):
        name, kwargs = cb.build_loader_kwargs({"name": "commonsense_qa"}, 5)
        self.assertEqual("commonsense_qa", name)
        self.assertEqual(
            {"split": "validation", "seed": 5, "random_answers_order": False}, kwargs
        )
        _, kwargs = cb.build_loader_kwargs(
            {"name": "commonsense_qa", "params": {"split": "train"}}, 5
        )
        self.assertEqual("train", kwargs["split"])

    def test_medqa_defaults_and_split(self):
        name, kwargs = cb.build_loader_kwargs({"name": "medqa"}, 5)
        self.assertEqual("medqa", name)
        self.assertEqual({"split": "test", "seed": 5, "random_answers_order": False}, kwargs)
        _, kwargs = cb.build_loader_kwargs(
            {"name": "medqa", "params": {"split": "train"}}, 5, random_answers_order=True
        )
        self.assertEqual("train", kwargs["split"])
        self.assertTrue(kwargs["random_answers_order"])

    def test_name_is_case_insensitive(self):
        name, _ = cb.build_loader_kwargs({"name": " MMLU ", "params": {}}, 42)
        self.assertEqual("mmlu", name)

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            cb.build_loader_kwargs({"name": "trivia", "params": {}}, 42)


class DeriveOutputPathTest(unittest.TestCase):

    def test_explicit_path_used_verbatim(self):
        self.assertEqual(
            Path("/tmp/x.csv"), cb.derive_output_path("/tmp/x.csv", "m", "mmlu")
        )

    def test_derived_path_uses_short_name_and_dataset(self):
        self.assertEqual(
            resolve_data_path("${DATA_ROOT}/baselines/qwen3-8b_mmlu_baseline.csv"),
            cb.derive_output_path(None, "Qwen/Qwen3-8B", "mmlu"),
        )


class MainFlowTest(unittest.TestCase):
    """End-to-end tests of main() with model, dataset, and generation mocked."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_csv = self.tmp / "baseline.csv"

    def base_cfg(self, **extra):
        cfg = {
            "model_name": "Qwen/Qwen3-8B",
            "dataset": {"name": "mmlu", "params": {}},
            "output_csv": str(self.out_csv),
            "batch_size": 2,
        }
        cfg.update(extra)
        return cfg

    def run_main(self, cfg, df, spec_by_prompt, extra_argv=()):
        argv = ["compute_baseline", "--config", "cfg.yaml", *extra_argv]
        gen = make_generator(spec_by_prompt)
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(cb, "load_config", return_value=cfg), \
                mock.patch.object(
                    cb, "load_model_vLLM",
                    return_value=(mock.MagicMock(), mock.MagicMock()),
                ), \
                mock.patch.object(
                    cb, "load_data", return_value=(df, ["A", "B", "C", "D"])
                ) as data_mock, \
                mock.patch.object(
                    cb, "generate_rollouts_batch", side_effect=gen
                ) as gen_mock, \
                contextlib.redirect_stdout(io.StringIO()):
            cb.main()
        self.load_data_mock = data_mock
        self.generate_mock = gen_mock
        return gen_mock

    def greedy_spec(self):
        return {"P0": "A", "P1": "C", "P2": "", "P3": "D"}

    def test_greedy_writes_canonical_csv(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        out = pd.read_csv(self.out_csv, keep_default_na=False)
        self.assertEqual(CANONICAL_COLUMNS, list(out.columns))
        self.assertEqual([10, 11, 12, 13], out["original_index"].tolist())
        self.assertEqual(["A", "C", "", "D"], out["baseline_answer"].tolist())
        self.assertEqual(
            [True, False, False, True],
            [str(v) == "True" for v in out["correct"].tolist()],
        )
        self.assertEqual(
            ["correct", "incorrect", "unanswered", "correct"],
            out["baseline_status"].tolist(),
        )
        self.assertEqual(["w", "x", "y", "z"], json.loads(out["choices"].iloc[0]))
        self.assertEqual(
            [
                {"subject": "hist"},
                {"subject": "hist"},
                {"subject": "math"},
                {"subject": "math"},
            ],
            [json.loads(v) for v in out["additional_fields"].tolist()],
        )

    def test_greedy_meta_sidecar(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        meta_path = self.out_csv.with_suffix(".meta.json")
        self.assertTrue(meta_path.exists())
        meta = json.loads(meta_path.read_text())
        self.assertEqual("Qwen/Qwen3-8B", meta["model_name"])
        self.assertEqual({"name": "mmlu", "params": {}}, meta["dataset"])
        self.assertEqual(1, meta["consistency_samples"])
        self.assertEqual(0.0, meta["temperature"])
        self.assertFalse(meta["sampling_enabled"])
        self.assertIsNone(meta["vote_threshold"])
        self.assertIsNone(meta["max_samples"])
        self.assertEqual(["subject"], meta["additional_field_keys"])
        self.assertEqual("test", meta["split"])
        self.assertEqual(4, meta["n_rows"])
        self.assertAlmostEqual(0.5, meta["accuracy"])

    def test_greedy_generation_settings(self):
        gen_mock = self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        for call in gen_mock.call_args_list:
            self.assertEqual(1, call.kwargs["n"])
            self.assertEqual(0.0, call.kwargs["temperature"])

    def test_batching_respects_batch_size(self):
        gen_mock = self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        self.assertEqual(2, gen_mock.call_count)

    def test_csv_round_trips_through_load_baseline(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        loaded = load_baseline(self.out_csv)
        self.assertEqual(["w", "x", "y", "z"], loaded["choices"].iloc[0])
        self.assertEqual({"subject": "hist"}, loaded["additional_fields"].iloc[0])
        self.assertEqual({"subject": "math"}, loaded["additional_fields"].iloc[3])
        pool = build_verified_pool(loaded)
        self.assertEqual([10, 13], pool["original_index"].tolist())

    def consistency_spec(self):
        return {
            "P0": ["A", "A", "B"],
            "P1": ["A", "C", ""],
            "P2": ["C", "C", "C"],
            "P3": ["A", "B", "C"],
        }

    def test_consistency_sampling_columns_and_votes(self):
        cfg = self.base_cfg(consistency_samples=3)
        self.run_main(cfg, subject_df(), self.consistency_spec())
        out = pd.read_csv(self.out_csv, keep_default_na=False)
        self.assertEqual(CANONICAL_COLUMNS + SAMPLING_COLUMNS, list(out.columns))
        self.assertEqual(["A", "", "C", ""], out["baseline_answer"].tolist())
        self.assertEqual(
            [True, False, True, False],
            [str(v) == "True" for v in out["correct"].tolist()],
        )
        self.assertEqual(
            ["correct", "inconsistent", "correct", "inconsistent"],
            out["baseline_status"].tolist(),
        )
        self.assertEqual(["A", "C", ""], json.loads(out["sample_answers"].iloc[1]))
        self.assertEqual(3, len(json.loads(out["sample_rollouts"].iloc[0])))
        self.assertEqual([2, 1, 3, 1], out["n_top_votes"].astype(int).tolist())
        self.assertAlmostEqual(2 / 3, float(out["self_consistency"].iloc[0]))
        self.assertAlmostEqual(1.0, float(out["self_consistency"].iloc[2]))
        self.assertEqual([2, 0, 3, 0], out["n_correct_samples"].astype(int).tolist())
        meta = json.loads(self.out_csv.with_suffix(".meta.json").read_text())
        self.assertEqual(3, meta["consistency_samples"])
        self.assertEqual(2, meta["vote_threshold"])
        self.assertTrue(meta["sampling_enabled"])
        self.assertAlmostEqual(
            cb.DEFAULT_SAMPLING_TEMPERATURE, meta["temperature"]
        )

    def test_consistency_generation_settings(self):
        cfg = self.base_cfg(consistency_samples=3)
        gen_mock = self.run_main(cfg, subject_df(), self.consistency_spec())
        for call in gen_mock.call_args_list:
            self.assertEqual(3, call.kwargs["n"])
            self.assertEqual(
                cb.DEFAULT_SAMPLING_TEMPERATURE, call.kwargs["temperature"]
            )

    def test_cli_overrides_yaml(self):
        gen_mock = self.run_main(
            self.base_cfg(), subject_df(), self.consistency_spec(),
            extra_argv=[
                "--consistency-samples", "3",
                "--temperature", "0.2",
                "--vote-threshold", "3",
            ],
        )
        for call in gen_mock.call_args_list:
            self.assertEqual(3, call.kwargs["n"])
            self.assertEqual(0.2, call.kwargs["temperature"])
        out = pd.read_csv(self.out_csv, keep_default_na=False)
        self.assertEqual(["", "", "C", ""], out["baseline_answer"].tolist())
        meta = json.loads(self.out_csv.with_suffix(".meta.json").read_text())
        self.assertEqual(3, meta["vote_threshold"])
        self.assertEqual(0.2, meta["temperature"])

    def test_dataset_params_reach_loader(self):
        cfg = self.base_cfg(seed=13)
        cfg["dataset"] = {"name": "mmlu", "params": {"samples_per_subject": 1}}
        self.run_main(cfg, subject_df(), self.greedy_spec())
        kwargs = self.load_data_mock.call_args.kwargs
        self.assertEqual(1, kwargs["samples_per_subject"])
        self.assertEqual(13, kwargs["seed"])
        self.assertEqual("test", kwargs["split"])

    def test_subjectless_split_with_max_samples(self):
        cfg = self.base_cfg(max_samples=2)
        cfg["dataset"] = {"name": "mmlu", "params": {"split": "train"}}
        self.run_main(cfg, subjectless_df(), self.greedy_spec())
        out = pd.read_csv(self.out_csv)
        self.assertEqual(2, len(out))
        meta = json.loads(self.out_csv.with_suffix(".meta.json").read_text())
        self.assertEqual([], meta["additional_field_keys"])
        self.assertEqual(2, meta["max_samples"])
        self.assertEqual("auxiliary_train", meta["split"])
        loaded = load_baseline(self.out_csv)
        self.assertEqual([{}, {}], loaded["additional_fields"].tolist())

    def test_empty_dataset_rejected(self):
        empty = subject_df().iloc[0:0]
        with self.assertRaises(ValueError):
            self.run_main(self.base_cfg(), empty, {})

    def test_invalid_max_samples_rejected_before_model_load(self):
        cfg = self.base_cfg(max_samples=0)
        argv = ["compute_baseline", "--config", "cfg.yaml"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(cb, "load_config", return_value=cfg), \
                mock.patch.object(cb, "load_model_vLLM") as loader, \
                mock.patch.object(cb, "load_data"), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                cb.main()
        loader.assert_not_called()

    def test_temperature_without_sampling_rejected_in_main(self):
        cfg = self.base_cfg(temperature=0.5)
        with self.assertRaises(ValueError):
            self.run_main(cfg, subject_df(), self.greedy_spec())

    def read_meta(self):
        return json.loads(self.out_csv.with_suffix(".meta.json").read_text())

    def test_random_answers_order_default_off(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        self.assertFalse(self.load_data_mock.call_args.kwargs["random_answers_order"])
        self.assertFalse(self.read_meta()["random_answers_order"])

    def test_random_answers_order_cli_flag(self):
        self.run_main(
            self.base_cfg(), subject_df(), self.greedy_spec(),
            extra_argv=["--random_answers_order"],
        )
        self.assertTrue(self.load_data_mock.call_args.kwargs["random_answers_order"])
        self.assertTrue(self.read_meta()["random_answers_order"])

    def test_random_answers_order_yaml_key(self):
        cfg = self.base_cfg(random_answers_order=True)
        self.run_main(cfg, subject_df(), self.greedy_spec())
        self.assertTrue(self.load_data_mock.call_args.kwargs["random_answers_order"])
        self.assertTrue(self.read_meta()["random_answers_order"])

    def test_resume_rejects_random_order_mismatch(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        with self.assertRaises(ValueError) as ctx:
            self.run_main(
                self.base_cfg(), subject_df(), self.greedy_spec(),
                extra_argv=["--random_answers_order"],
            )
        self.assertIn("random_answers_order", str(ctx.exception))

    def test_resume_accepts_pre_flag_sidecar(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        meta_path = self.out_csv.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        del meta["random_answers_order"]
        meta_path.write_text(json.dumps(meta))
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        out = pd.read_csv(self.out_csv)
        self.assertEqual(4, len(out))

    def test_thinking_recorded_in_sidecar(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        meta = self.read_meta()
        self.assertTrue(meta["thinking"])
        # Informational, not resume-identity.
        self.assertIn("thinking_mode", meta)
        self.assertIn("reasoning_delimiters", meta)
        self.assertNotIn("thinking_mode", cb.RESUME_IDENTITY_FIELDS)
        self.assertNotIn("reasoning_delimiters", cb.RESUME_IDENTITY_FIELDS)
        self.assertIn("thinking", cb.RESUME_IDENTITY_FIELDS)

    def test_thinking_off_uses_plain_prompt(self):
        self.run_main(self.base_cfg(thinking="off"), subject_df(), self.greedy_spec())
        meta = self.read_meta()
        self.assertFalse(meta["thinking"])
        self.assertEqual(meta["thinking_mode"], "none")
        system_prompts = {
            msgs[0]["content"]
            for call in self.generate_mock.call_args_list
            for msgs in call.args[2]
        }
        self.assertEqual(system_prompts, {SYSTEM_PROMPT_PLAIN})
        self.assertIs(
            self.generate_mock.call_args.kwargs["enable_thinking"], False
        )

    def test_resume_rejects_thinking_mismatch(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        with self.assertRaises(ValueError) as ctx:
            self.run_main(
                self.base_cfg(thinking="off"), subject_df(), self.greedy_spec()
            )
        self.assertIn("thinking", str(ctx.exception))

    def test_sampling_truncation_defaults_and_sidecar(self):
        gen_mock = self.run_main(
            self.base_cfg(consistency_samples=3, seed=11), subject_df(), self.consistency_spec()
        )
        for call in gen_mock.call_args_list:
            self.assertEqual(DEFAULT_TOP_P, call.kwargs["top_p"])
            self.assertEqual(DEFAULT_TOP_K, call.kwargs["top_k"])
            self.assertEqual(11, call.kwargs["seed"])
        meta = self.read_meta()
        self.assertEqual((DEFAULT_TOP_P, DEFAULT_TOP_K, 11), (meta["top_p"], meta["top_k"], meta["sampling_seed"]))
        self.assertIn("top_p", cb.RESUME_IDENTITY_FIELDS)
        self.assertIn("top_k", cb.RESUME_IDENTITY_FIELDS)
        self.assertNotIn("sampling_seed", cb.RESUME_IDENTITY_FIELDS)
        self.out_csv.unlink()
        gen_mock = self.run_main(
            self.base_cfg(consistency_samples=3, top_p=1.0, top_k=-1), subject_df(), self.consistency_spec()
        )
        self.assertEqual((1.0, -1), (gen_mock.call_args.kwargs["top_p"], gen_mock.call_args.kwargs["top_k"]))
        self.out_csv.unlink()
        gen_mock = self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        self.assertIsNone(gen_mock.call_args.kwargs["seed"])
        meta = self.read_meta()
        self.assertEqual((DEFAULT_TOP_P, DEFAULT_TOP_K, None), (meta["top_p"], meta["top_k"], meta["sampling_seed"]))

    def test_resume_rejects_legacy_sampling_mismatch(self):
        cfg = self.base_cfg(consistency_samples=3)
        self.run_main(cfg, subject_df(), self.consistency_spec())
        meta_path = self.out_csv.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        del meta["top_p"], meta["top_k"]
        meta_path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError) as ctx:
            self.run_main(cfg, subject_df(), self.consistency_spec())
        msg = str(ctx.exception)
        self.assertIn("top_p: existing=1.0", msg)
        self.assertIn("top_k: existing=-1", msg)
        self.assertIn("'top_p': 1.0, 'top_k': -1", msg)

    def test_resume_accepts_legacy_sampling_when_reproduced(self):
        cfg = self.base_cfg(consistency_samples=3)
        self.run_main(cfg, subject_df(), self.consistency_spec())
        meta_path = self.out_csv.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        del meta["top_p"], meta["top_k"]
        meta_path.write_text(json.dumps(meta))
        self.run_main(self.base_cfg(consistency_samples=3, top_p=1.0, top_k=-1), subject_df(), self.consistency_spec())
        self.assertEqual(4, len(pd.read_csv(self.out_csv)))
        self.assertEqual((1.0, -1), (self.read_meta()["top_p"], self.read_meta()["top_k"]))

    def test_resume_ignores_sampling_knobs_when_greedy(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        meta_path = self.out_csv.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        del meta["top_p"], meta["top_k"]
        meta_path.write_text(json.dumps(meta))
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        self.assertEqual(4, len(pd.read_csv(self.out_csv)))
        current = {**cb.LEGACY_IDENTITY_DEFAULTS, "sampling_enabled": False, "top_p": 0.95, "top_k": 20}
        cb.assert_resume_compatible(cb.apply_legacy_identity_defaults({"sampling_enabled": False}), current)
        with self.assertRaisesRegex(ValueError, "top_p"):
            cb.assert_resume_compatible(
                cb.apply_legacy_identity_defaults({"sampling_enabled": True}),
                {**current, "sampling_enabled": True},
            )

    def test_resume_accepts_pre_thinking_sidecar(self):
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        meta_path = self.out_csv.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        del meta["thinking"]
        meta_path.write_text(json.dumps(meta))
        self.run_main(self.base_cfg(), subject_df(), self.greedy_spec())
        self.assertEqual(4, len(pd.read_csv(self.out_csv)))


if __name__ == "__main__":
    unittest.main()
