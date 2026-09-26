"""Tests for src/scripts/run_pipeline.py (every stage subprocess mocked)."""

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.lib.config import DEFAULT_TOP_K, DEFAULT_TOP_P
from src.lib.hints import HINTS
from src.lib.paths import REPO_ROOT
from src.lib.resample import DROPPED_HINTS
from src.scripts.run_pipeline import (
    BATCH_JUDGE_CONFIG_KEYS,
    BOOL_FLAGS,
    COLLECT_CONFIG_KEYS,
    DEFAULT_STAGES,
    FLAG_KEYS,
    OPTIONAL_STAGES,
    PAPER_HINT_STYLES,
    RESAMPLE_CONFIG_KEYS,
    STAGE_MODULES,
    STAGES,
    Plan,
    apply_cli_overrides,
    stage_env,
    baseline_identity,
    baseline_is_complete,
    build_stage_config,
    derive_dataset_tag,
    flag_argv,
    generation_key,
    main,
    parse_dataset_param,
    resolve_baseline_sampling,
    resolve_stages,
    rollouts_sidecar,
)

MODEL = "Test/Model-1B"
HINTS_USED = ["authority", "metadata"]
LEGACY_STAGES = ["baseline", "rollouts", "judge"]
CELL = "test_baseline_hinted_rollouts"   # the rollouts CSV stem of setup_tree's config


def make_cfg(**overrides) -> dict:
    """A loaded pipeline config (shared keys as the base would supply them)."""
    cfg = {
        "model_name": MODEL,
        "seed": 7,
        "thinking": True,
        "inference": {"max_model_len": 1024, "max_tokens": 256, "gpu_memory_utilization": 0.8},
        "layers": [1, 2],
        "compute_baseline": {
            "dataset": {"name": "gpqa", "params": {"config": "gpqa_diamond", "split": "train"}},
            "consistency_samples": 4, "temperature": 0.5, "vote_threshold": 3,
            "batch_size": 8,
        },
        "collect_hinted_rollouts": {
            "hints": HINTS_USED, "temperature": 0.7, "cases": "both",
            "inference": {"max_tokens": 512},
        },
        "judge_rollouts": {"judge_model": "test/judge", "workers": 2},
    }
    cfg.update(overrides)
    return cfg


def make_args(**kw) -> argparse.Namespace:
    base = dict(model=None, dataset=None, dataset_param=None, run_name=None,
                cases=None, judge_model=None)
    base.update(kw)
    return argparse.Namespace(**base)


def flag(argv, name):
    """The value after ``name`` in an argv (None when absent)."""
    argv = list(argv)
    return argv[argv.index(name) + 1] if name in argv else None


def short_name_patch():
    return patch("src.lib.model_utils.get_model_short_name", return_value="test-1b")


class TestConfigHelpers(unittest.TestCase):
    def test_dataset_tag_encodes_name_and_split_or_subset(self):
        self.assertEqual(derive_dataset_tag({"name": "gpqa", "params": {"config": "gpqa_diamond"}}), "gpqa-diamond")
        self.assertEqual(derive_dataset_tag({"name": "gpqa"}), "gpqa-diamond")
        self.assertEqual(derive_dataset_tag({"name": "mmlu", "params": {"split": "train"}}), "mmlu-train")
        self.assertEqual(derive_dataset_tag({"name": "MMLU"}), "mmlu-test")
        self.assertEqual(derive_dataset_tag({"name": "mmlu_pro", "params": {"split": "validation"}}), "mmlu_pro-validation")
        self.assertEqual(derive_dataset_tag({"name": "commonsense_qa"}), "commonsense_qa-validation")
        self.assertEqual(derive_dataset_tag({"name": "commonsense_qa", "params": {"split": "train"}}), "commonsense_qa-train")
        self.assertEqual(derive_dataset_tag({"name": "medqa"}), "medqa-test")
        self.assertEqual(derive_dataset_tag({"name": "aqua"}), "aqua-train")
        self.assertEqual(derive_dataset_tag({"name": "aqua", "params": {"split": "test"}}), "aqua-test")

    def test_default_split_tag_matches_the_loader_default(self):
        from src.scripts.compute_baseline import build_loader_kwargs
        for name in ("mmlu", "mmlu_pro", "medqa", "aqua", "commonsense_qa"):
            _, kwargs = build_loader_kwargs({"name": name}, 42)
            self.assertEqual(derive_dataset_tag({"name": name}), f"{name}-{kwargs['split']}")

    def test_dataset_param_values_are_yaml_parsed(self):
        self.assertEqual(parse_dataset_param("split=test"), ("split", "test"))
        self.assertEqual(parse_dataset_param("samples_per_subject=25"), ("samples_per_subject", 25))
        self.assertEqual(parse_dataset_param("subjects=[a, b]"), ("subjects", ["a", "b"]))
        self.assertEqual(parse_dataset_param("x="), ("x", None))
        with self.assertRaisesRegex(ValueError, "key=value"):
            parse_dataset_param("nonsense")

    def test_cli_overrides_land_in_the_right_blocks(self):
        cfg = make_cfg()
        out = apply_cli_overrides(cfg, make_args(
            model="Other/Model", dataset="mmlu", dataset_param=["split=train"],
            run_name="0904", cases="positive_cases", judge_model="other/judge",
        ))
        self.assertEqual(out["model_name"], "Other/Model")
        self.assertEqual(out["compute_baseline"]["dataset"], {"name": "mmlu", "params": {"split": "train"}})
        self.assertEqual(out["compute_baseline"]["consistency_samples"], 4)
        self.assertEqual(out["run_name"], "0904")
        self.assertEqual(out["collect_hinted_rollouts"]["cases"], "positive_cases")
        self.assertEqual(out["collect_hinted_rollouts"]["hints"], HINTS_USED)
        self.assertEqual(out["judge_rollouts"]["judge_model"], "other/judge")
        self.assertEqual(out["judge_rollouts"]["workers"], 2)
        self.assertEqual(cfg["model_name"], MODEL)
        self.assertEqual(cfg["compute_baseline"]["dataset"]["name"], "gpqa")

    def test_dataset_param_alone_overlays_the_config_block(self):
        out = apply_cli_overrides(make_cfg(), make_args(dataset_param=["config=gpqa_main"]))
        self.assertEqual(out["compute_baseline"]["dataset"]["params"], {"config": "gpqa_main", "split": "train"})

    def test_resolve_stages_defaults_and_fixed_order(self):
        self.assertEqual(resolve_stages(None), list(DEFAULT_STAGES))
        self.assertEqual(len(DEFAULT_STAGES), 10)
        self.assertEqual(resolve_stages("judge,baseline"), ["baseline", "judge"])
        self.assertEqual(resolve_stages("figures,manifest"), ["manifest", "figures"])
        with self.assertRaisesRegex(ValueError, "Unknown stage"):
            resolve_stages("baseline,nope")
        with self.assertRaisesRegex(ValueError, "No stage"):
            resolve_stages(None, [])

    def test_probe_stages_are_opt_in(self):
        self.assertEqual(OPTIONAL_STAGES, ("splits", "probe_dataset", "activations", "train_probe"))
        self.assertFalse(set(OPTIONAL_STAGES) & set(DEFAULT_STAGES))
        # The config's `stages:` list restricts or extends the default; the CLI wins over it.
        self.assertEqual(resolve_stages(None, ["train_probe", "splits"]), ["splits", "train_probe"])
        self.assertEqual(resolve_stages(None, list(STAGES)), list(STAGES))
        self.assertEqual(resolve_stages("judge", ["splits"]), ["judge"])
        with self.assertRaisesRegex(ValueError, "must be a list"):
            resolve_stages(None, "splits")

    def test_sampling_resolution_mirrors_compute_baseline(self):
        self.assertEqual(resolve_baseline_sampling({}), {
            "consistency_samples": 1, "temperature": 0.0,
            "sampling_enabled": False, "vote_threshold": None,
        })
        self.assertEqual(resolve_baseline_sampling({"consistency_samples": 8}), {
            "consistency_samples": 8, "temperature": 0.7,
            "sampling_enabled": True, "vote_threshold": 5,
        })
        self.assertEqual(
            resolve_baseline_sampling({"consistency_samples": 4, "temperature": 0.5, "vote_threshold": 3})["vote_threshold"],
            3,
        )


class TestStageEnv(unittest.TestCase):
    def test_interpreter_bin_dir_leads_the_path(self):
        bin_dir = str(Path(sys.executable).parent)
        with patch.dict(os.environ, {"PATH": "/usr/bin"}):
            env = stage_env()
            self.assertEqual(env["PATH"].split(os.pathsep)[0], bin_dir)
            self.assertIn("/usr/bin", env["PATH"].split(os.pathsep))
        with patch.dict(os.environ, {"PATH": bin_dir + os.pathsep + "/usr/bin"}):
            self.assertEqual(stage_env()["PATH"], bin_dir + os.pathsep + "/usr/bin")


class TestStageConfigs(unittest.TestCase):
    def test_baseline_block_overlays_the_shared_keys(self):
        gen = build_stage_config(make_cfg(), "baseline")
        self.assertEqual(gen["model_name"], MODEL)
        self.assertEqual(gen["seed"], 7)
        self.assertIs(gen["thinking"], True)
        self.assertNotIn("layers", gen)
        self.assertNotIn("compute_baseline", gen)
        self.assertNotIn("judge_rollouts", gen)
        self.assertEqual(gen["dataset"]["name"], "gpqa")
        self.assertEqual(gen["consistency_samples"], 4)
        self.assertEqual(gen["batch_size"], 8)
        self.assertEqual(gen["inference"], {"max_model_len": 1024, "max_tokens": 256, "gpu_memory_utilization": 0.8})

    def test_pipeline_keys_and_sections_never_reach_the_stage_scripts(self):
        cfg = make_cfg(data_dir="/x", stages=["judge"], build_rollout_manifest={"only": "a"},
                       train_probe={"runs": []})
        for stage in ("baseline", "rollouts"):
            gen = build_stage_config(cfg, stage)
            for key in ("data_dir", "stages", "build_rollout_manifest", "train_probe", "run_name", "pipeline_dir"):
                self.assertNotIn(key, gen)

    def test_rollouts_inference_merges_into_the_shared_block(self):
        gen = build_stage_config(make_cfg(), "rollouts")
        self.assertEqual(gen["model_name"], MODEL)
        self.assertEqual(gen["hints"], HINTS_USED)
        self.assertEqual(gen["cases"], "both")
        self.assertEqual(gen["inference"], {"max_model_len": 1024, "max_tokens": 512, "gpu_memory_utilization": 0.8})
        self.assertEqual(gen["layers"], [1, 2])

    def test_judge_block_is_passed_alone_and_validated(self):
        gen = build_stage_config(make_cfg(), "judge")
        self.assertEqual(gen, {"judge_model": "test/judge", "workers": 2})
        with self.assertRaisesRegex(ValueError, "judge_rollouts"):
            build_stage_config(make_cfg(judge_rollouts={"model": "x"}), "judge")

    def test_identity_matches_what_compute_baseline_records(self):
        ident = baseline_identity(build_stage_config(make_cfg(), "baseline"))
        self.assertEqual(ident["model_name"], MODEL)
        self.assertEqual(ident["dataset"], {"name": "gpqa", "params": {"config": "gpqa_diamond", "split": "train"}})
        self.assertEqual(ident["consistency_samples"], 4)
        self.assertEqual(ident["temperature"], 0.5)
        self.assertTrue(ident["sampling_enabled"])
        self.assertEqual(ident["vote_threshold"], 3)
        self.assertEqual(ident["seed"], 7)
        self.assertEqual(ident["max_tokens"], 256)
        self.assertIsNone(ident["max_samples"])
        self.assertIs(ident["thinking"], True)
        self.assertEqual((ident["top_p"], ident["top_k"]), (DEFAULT_TOP_P, DEFAULT_TOP_K))
        custom = baseline_identity({**build_stage_config(make_cfg(), "baseline"), "top_p": 1.0, "top_k": -1})
        self.assertEqual((custom["top_p"], custom["top_k"]), (1.0, -1))

    def test_generation_key_and_sidecar_track_the_recipe(self):
        gen = build_stage_config(make_cfg(), "rollouts")
        gen.update({"output_csv": "/x/r.csv", "baseline_csv": "/x/b.csv"})
        key = generation_key(gen)
        self.assertEqual(key, "r.csv|cases=both|hints=authority,metadata|temp=0.7|seed=7|max_tokens=512")
        self.assertNotEqual(key, generation_key({**gen, "hints": ["authority"]}))
        self.assertNotEqual(key, generation_key({**gen, "inference": {"max_tokens": 1024}}))
        sidecar = rollouts_sidecar(gen)
        self.assertEqual(sidecar["model_name"], MODEL)
        self.assertEqual(sidecar["baseline_csv"], "/x/b.csv")
        self.assertEqual(sidecar["hints"], HINTS_USED)
        self.assertEqual(sidecar["max_tokens"], 512)
        self.assertEqual(sidecar["max_model_len"], 1024)
        self.assertIs(sidecar["thinking"], True)
        self.assertEqual((sidecar["top_p"], sidecar["top_k"]), (DEFAULT_TOP_P, DEFAULT_TOP_K))


class TestBaselineIsComplete(unittest.TestCase):
    def setUp(self):
        self.identity = {
            "model_name": MODEL,
            "dataset": {"name": "gpqa", "params": {"config": "gpqa_diamond", "split": "train"}},
            "consistency_samples": 4, "temperature": 0.5, "sampling_enabled": True,
            "vote_threshold": 3, "max_samples": None, "seed": 7, "max_tokens": 256,
            "random_answers_order": False, "top_p": DEFAULT_TOP_P, "top_k": DEFAULT_TOP_K,
        }

    def write(self, td: Path, sidecar: dict | None) -> Path:
        csv = td / "b.csv"
        csv.write_text("original_index\n1\n")
        if sidecar is not None:
            csv.with_suffix(".meta.json").write_text(json.dumps(sidecar))
        return csv

    def test_missing_csv_or_sidecar_or_incomplete_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            self.assertFalse(baseline_is_complete(td / "nope.csv", self.identity)[0])
            csv = self.write(td, None)
            self.assertFalse(baseline_is_complete(csv, self.identity)[0])
            csv = self.write(td, {**self.identity, "complete": False})
            self.assertFalse(baseline_is_complete(csv, self.identity)[0])

    def test_matching_complete_sidecar_is_skipped_case_insensitively(self):
        with tempfile.TemporaryDirectory() as raw:
            csv = self.write(Path(raw), {
                **self.identity, "model_name": MODEL.lower(), "complete": True, "n_rows": 3,
            })
            skip, why = baseline_is_complete(csv, self.identity)
        self.assertTrue(skip)
        self.assertIn("3 rows", why)

    def test_identity_mismatch_is_reported_not_skipped(self):
        with tempfile.TemporaryDirectory() as raw:
            csv = self.write(Path(raw), {**self.identity, "complete": True, "vote_threshold": 2})
            skip, why = baseline_is_complete(csv, self.identity)
        self.assertFalse(skip)
        self.assertIn("vote_threshold", why)

    def test_thinking_is_compared_only_when_the_config_pins_it(self):
        with tempfile.TemporaryDirectory() as raw:
            csv = self.write(Path(raw), {**self.identity, "complete": True, "thinking": False})
            self.assertTrue(baseline_is_complete(csv, self.identity)[0])
            skip, why = baseline_is_complete(csv, {**self.identity, "thinking": True})
        self.assertFalse(skip)
        self.assertIn("thinking", why)

    def test_legacy_sidecar_without_shuffle_flag_means_unshuffled(self):
        with tempfile.TemporaryDirectory() as raw:
            sidecar = {**self.identity, "complete": True}
            del sidecar["random_answers_order"]
            csv = self.write(Path(raw), sidecar)
            self.assertTrue(baseline_is_complete(csv, self.identity)[0])

    def test_legacy_sidecar_without_sampling_knobs_means_vllm_defaults(self):
        with tempfile.TemporaryDirectory() as raw:
            sidecar = {**self.identity, "complete": True}
            del sidecar["top_p"], sidecar["top_k"]
            csv = self.write(Path(raw), sidecar)
            skip, why = baseline_is_complete(csv, self.identity)
            self.assertFalse(skip)
            self.assertIn("top_p: existing=1.0", why)
            self.assertIn("top_k: existing=-1", why)
            legacy = {**self.identity, "top_p": 1.0, "top_k": -1}
            self.assertTrue(baseline_is_complete(csv, legacy)[0])
            greedy = {**self.identity, "consistency_samples": 1, "temperature": 0.0,
                      "sampling_enabled": False, "vote_threshold": None}
            csv = self.write(Path(raw), {**greedy, "complete": True, "top_p": 1.0, "top_k": -1})
            self.assertTrue(baseline_is_complete(csv, greedy)[0])


class TestPlan(unittest.TestCase):
    def test_missing_required_fields_and_bad_values_raise(self):
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}):
            with self.assertRaisesRegex(ValueError, "model_name"):
                Plan(make_cfg(model_name=None))
            with self.assertRaisesRegex(ValueError, "dataset.name"):
                Plan(make_cfg(compute_baseline={"dataset": {}}))
            with self.assertRaisesRegex(ValueError, "Unknown hint"):
                Plan(make_cfg(collect_hinted_rollouts={"hints": ["nope"], "output_csv": "/x.csv"},
                              compute_baseline={**make_cfg()["compute_baseline"], "output_csv": "/b.csv"}))

    def test_default_paths_chain_the_stages(self):
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            plan = Plan(make_cfg(run_name="0904"))
        self.assertEqual(str(plan.baseline_csv), "/data/baselines/test-1b_gpqa-diamond_0904_baseline.csv")
        self.assertEqual(plan.baseline_cfg["output_csv"], str(plan.baseline_csv))
        self.assertEqual(
            str(plan.rollouts_csv),
            "/data/hinted_rollouts/test-1b_gpqa-diamond_0904_baseline_hinted_rollouts.csv",
        )
        self.assertEqual(plan.rollouts_cfg["baseline_csv"], str(plan.baseline_csv))
        self.assertEqual(plan.rollouts_cfg["output_csv"], str(plan.rollouts_csv))
        self.assertEqual(plan.judge_cfg["input_csv"], str(plan.rollouts_csv))
        self.assertTrue(str(plan.judged_csv).endswith("_hinted_rollouts_judged.csv"))
        self.assertEqual(plan.judge_cfg["output_csv"], str(plan.judged_csv))
        self.assertEqual(plan.judge_cfg["cache_dir"],
                         "/data/judge_cache/test-1b_gpqa-diamond_0904_baseline_hinted_rollouts")
        self.assertTrue(str(plan.summary_json).endswith("_hinted_rollouts_judged_summary.json"))
        self.assertEqual(str(plan.record_path), "/data/pipeline/test-1b_gpqa-diamond_0904_baseline_pipeline.json")
        self.assertEqual(
            str(plan.generated_config_path("rollouts")),
            "/data/pipeline/generated_configs/test-1b_gpqa-diamond_0904_baseline_collect_hinted_rollouts.yaml",
        )
        self.assertEqual(plan.cases, "both")
        self.assertEqual(plan.hints, HINTS_USED)
        self.assertEqual(plan.judge_model, "test/judge")

    def test_explicit_paths_win_and_skip_the_registry(self):
        cfg = make_cfg()
        cfg["compute_baseline"]["output_csv"] = "${DATA_ROOT}/baselines/mine.csv"
        cfg["collect_hinted_rollouts"]["baseline_csv"] = "${DATA_ROOT}/baselines/other.csv"
        cfg["judge_rollouts"]["output_csv"] = "${DATA_ROOT}/out/j.csv"
        cfg["judge_rollouts"]["cache_dir"] = "${DATA_ROOT}/cache/x"
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}):
            plan = Plan(cfg)
        self.assertEqual(str(plan.baseline_csv), "/data/baselines/mine.csv")
        self.assertEqual(str(plan.rollouts_input), "/data/baselines/other.csv")
        self.assertEqual(str(plan.rollouts_csv), "/data/hinted_rollouts/other_hinted_rollouts.csv")
        self.assertEqual(str(plan.judged_csv), "/data/out/j.csv")
        self.assertEqual(str(plan.summary_json), "/data/out/j_summary.json")
        self.assertEqual(plan.judge_cfg["cache_dir"], "/data/cache/x")

    def test_every_default_path_derives_from_data_dir(self):
        cfg = make_cfg(data_dir="${DATA_ROOT}/tree", run_name="r")
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            plan = Plan(cfg)
            stem = "test-1b_gpqa-diamond_r_baseline"
            cell = f"{stem}_hinted_rollouts"
            self.assertEqual(str(plan.data_dir), "/data/tree")
            self.assertEqual(str(plan.pipeline_dir), "/data/tree/pipeline")
            self.assertEqual(str(plan.baseline_csv), f"/data/tree/baselines/{stem}.csv")
            self.assertEqual(str(plan.rollouts_csv), f"/data/tree/hinted_rollouts/{cell}.csv")
            self.assertEqual(plan.judge_cfg["cache_dir"], f"/data/tree/judge_cache/{cell}")
            self.assertEqual(str(plan.hinted_dir), "/data/tree/hinted_rollouts")
            self.assertEqual(str(plan.manifest), "/data/tree/hinted_rollouts/rollout_manifest.parquet")
            self.assertEqual(str(plan.label_overrides), "/data/tree/hinted_rollouts/judge_label_overrides.csv")
            self.assertEqual(str(plan.resample_dir), "/data/tree/resample")
            self.assertEqual(str(plan.items_csv()), f"/data/tree/resample/items/{cell}_resample_items.csv")
            self.assertEqual(plan.reroll_glob(), f"/data/tree/resample/rollouts/{cell}_rs*.csv")
            self.assertEqual(str(plan.resample_manifest), "/data/tree/resample/resample_manifest.parquet")
            self.assertEqual(str(plan.summary_md), "/data/tree/hinted_rollouts/unfaithfulness_metrics_summary.md")
            self.assertEqual(str(plan.figures_dir), "/data/tree/figures")
            self.assertEqual(str(plan.probe_dataset_parquet("used_vs_ignored")),
                             "/data/tree/probe_datasets/test-1b_used_vs_ignored.parquet")
            self.assertEqual(plan.default_storage(), {"backend": "local", "local_dir": "/data/tree/probe_activations/test-1b"})
            self.assertEqual(str(plan.trained_probes_dir), "/data/tree/trained_probes/probes")
            self.assertEqual(str(plan.record_path), f"/data/tree/pipeline/{stem}_pipeline.json")
            self.assertEqual(str(plan.generated_path("x")), "/data/tree/pipeline/generated_configs/x.yaml")
            self.assertEqual(plan.sample_seeds, [43, 44, 45, 46])
            self.assertEqual(plan.reroll_judge_model, "test/judge")

    def test_explicit_tree_keys_win_and_chain(self):
        cfg = make_cfg(
            build_rollout_manifest={"dir": "${DATA_ROOT}/judged", "output": "${DATA_ROOT}/m/manifest.parquet"},
            select_resample_set={"output_dir": "${DATA_ROOT}/rs", "sample_seeds": "1,2"},
            unfaithfulness_metrics_summary={"output": "${DATA_ROOT}/s.md"},
            paper_plots={"out": "${DATA_ROOT}/figs"},
            batch_judge_rollouts={"judge_model": "other/judge"},
        )
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            plan = Plan(cfg)
            self.assertEqual(str(plan.hinted_dir), "/data/judged")
            self.assertEqual(str(plan.manifest), "/data/m/manifest.parquet")
            self.assertEqual(str(plan.resample_dir), "/data/rs")
            self.assertEqual(str(plan.items_csv()), f"/data/rs/items/{plan.cell_stem}_resample_items.csv")
            self.assertEqual(str(plan.summary_md), "/data/s.md")
            self.assertEqual(str(plan.figures_dir), "/data/figs")
            self.assertEqual(plan.sample_seeds, [1, 2])
            self.assertEqual(plan.reroll_seeds, [1, 2])
            self.assertEqual(plan.reroll_judge_model, "other/judge")
            select = plan.commands("resample_select")[0].argv
            self.assertEqual(flag(select, "--manifest"), "/data/m/manifest.parquet")
            self.assertEqual(flag(select, "--rollouts-dir"), "/data/judged")
            self.assertEqual(flag(select, "--sample-seeds"), "1,2")
            self.assertEqual(flag(select, "--k"), "2")
            relabel = plan.commands("relabel")[1].argv
            self.assertEqual(flag(relabel, "--resample-dir"), "/data/rs")
            verify = plan.commands("summary")[1].argv
            self.assertEqual(flag(verify, "--summary"), "/data/s.md")

    def test_unknown_section_keys_raise_at_plan_time(self):
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}):
            for section, known in (("compute_baseline", "vote_threshold"), ("collect_hinted_rollouts", "chunk_size"),
                                   ("build_rollout_manifest", "chunk_rows"), ("select_resample_set", "extend"),
                                   ("resample_noise_model", "only"), ("resample_relabel", "no_token_len"),
                                   ("unfaithfulness_metrics_summary", "include_smoke"), ("verify_manifest", "summary"),
                                   ("paper_plots", "formats"), ("assign_splits", "fractions"),
                                   ("resample_hinted_rollouts", "chunk_size"), ("batch_judge_rollouts", "judge_workers"),
                                   ("build_probe_dataset", "predicates"), ("collect_probe_activations", "seq_layers"),
                                   ("train_probe", "runs")):
                base = make_cfg().get(section) or {}
                with self.assertRaisesRegex(ValueError, f"`{section}:`.*bogus.*{known}") as ctx:
                    Plan(make_cfg(**{section: {**base, "bogus": 1}}))
                self.assertIn("Known keys", str(ctx.exception))
            with self.assertRaisesRegex(ValueError, "must be a mapping"):
                Plan(make_cfg(paper_plots=[1]))
            # The two cell sections take their scripts' own keys (a valid key passes).
            cfg = make_cfg()
            cfg["compute_baseline"].update({"vote_threshold": 3, "random_answers_order": False, "top_p": 1.0, "top_k": -1})
            cfg["collect_hinted_rollouts"].update({"chunk_size": 64, "hint_n_examples": {"few_shot": 3},
                                                   "exclusion_list": None, "sample_questions_csv": None})
            Plan(cfg)

    def test_omitted_hints_default_to_the_paper_styles(self):
        self.assertEqual(len(PAPER_HINT_STYLES), 8)
        self.assertTrue(set(PAPER_HINT_STYLES) <= set(HINTS))
        self.assertTrue(set(PAPER_HINT_STYLES).isdisjoint(DROPPED_HINTS))
        cfg = make_cfg()
        del cfg["collect_hinted_rollouts"]["hints"]
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            plan = Plan(cfg)
        self.assertEqual(plan.hints, list(PAPER_HINT_STYLES))
        self.assertEqual(plan.rollouts_cfg["hints"], list(PAPER_HINT_STYLES))

    def test_cell_stem_follows_the_judged_csv(self):
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            plan = Plan(make_cfg(), stages=list(STAGES))
            self.assertEqual(plan.cell_stem, plan.rollouts_csv.stem)
            self.assertEqual(plan.cell_stem, "test-1b_gpqa-diamond_baseline_hinted_rollouts")
            # An explicit judged CSV at the default location is fine; elsewhere it desynchronises 4-7.
            same = make_cfg(judge_rollouts={"output_csv": f"${{DATA_ROOT}}/hinted_rollouts/{plan.cell_stem}_judged.csv"})
            self.assertEqual(Plan(same, stages=["rerolls"]).cell_stem, plan.cell_stem)
            moved = make_cfg(judge_rollouts={"output_csv": "${DATA_ROOT}/out/j.csv"})
            self.assertEqual(Plan(moved).cell_stem, "j")
            self.assertEqual(Plan(moved, stages=["baseline", "judge"]).cell_stem, "j")
            for stages in (["manifest"], ["rerolls"], ["judge", "figures"]):
                with self.assertRaisesRegex(ValueError, r"`judge_rollouts.output_csv` must be /data/hinted_rollouts/.*_judged.csv"):
                    Plan(moved, stages=stages)
            elsewhere = make_cfg(build_rollout_manifest={"dir": "${DATA_ROOT}/judged"})
            with self.assertRaisesRegex(ValueError, "judge_rollouts.output_csv"):
                Plan(elsewhere, stages=["manifest"])


class TestFlagStages(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"DATA_ROOT": "/data"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_flag_argv_renders_every_value_kind(self):
        argv = flag_argv("build_rollout_manifest", {
            "dir": "${DATA_ROOT}/hr", "output": None, "only": "nemotron", "chunk_rows": 2000,
            "no_token_len": True, "include_smoke": False, "noise_flip_min_votes": 2,
        })
        self.assertEqual(argv, ["--dir", "/data/hr", "--only", "nemotron", "--chunk-rows", "2000",
                                "--no-token-len", "--noise-flip-min-votes", "2"])
        self.assertEqual(flag_argv("assign_splits", {"fractions": [0.7, 0.15, 0.15], "force": True}),
                         ["--fractions", "0.7,0.15,0.15", "--force"])
        with self.assertRaisesRegex(ValueError, "`paper_plots:`.*colour.*formats"):
            flag_argv("paper_plots", {"colour": "red"})

    def test_tree_stage_argv(self):
        cfg = make_cfg(data_dir="${DATA_ROOT}/t", build_rollout_manifest={"no_token_len": True},
                       resample_noise_model={"skip_width_check": True}, paper_plots={"only": ["F1", "F3c"]},
                       unfaithfulness_metrics_summary={"include_smoke": True})
        with short_name_patch():
            plan = Plan(cfg)
        manifest = "/data/t/hinted_rollouts/rollout_manifest.parquet"
        cmds = {stage: plan.commands(stage) for stage in ("manifest", "resample_select", "relabel", "summary", "figures")}
        (m,) = cmds["manifest"]
        self.assertEqual(m.module, "src.scripts.build_rollout_manifest")
        self.assertEqual(list(m.argv), ["--dir", "/data/t/hinted_rollouts", "--output", manifest, "--no-token-len",
                                        "--label-overrides", "/data/t/hinted_rollouts/judge_label_overrides.csv"])
        self.assertIsNone(m.config)
        (s,) = cmds["resample_select"]
        self.assertEqual(list(s.argv), ["--manifest", manifest, "--output-dir", "/data/t/resample", "--k", "4",
                                        "--sample-seeds", "43,44,45,46", "--rollouts-dir", "/data/t/hinted_rollouts",
                                        "--extend"])
        noise, relabel = cmds["relabel"]
        self.assertEqual(noise.module, "src.scripts.resample_noise_model")
        self.assertEqual(list(noise.argv), ["--manifest", manifest, "--output-dir", "/data/t/resample", "--skip-width-check"])
        self.assertEqual(relabel.module, "src.scripts.resample_relabel")
        self.assertEqual(list(relabel.argv), ["--resample-dir", "/data/t/resample", "--manifest", manifest])
        summary, verify = cmds["summary"]
        md = "/data/t/hinted_rollouts/unfaithfulness_metrics_summary.md"
        self.assertEqual(list(summary.argv), ["--dir", "/data/t/hinted_rollouts", "--output", md, "--include-smoke"])
        self.assertEqual(list(verify.argv), ["--manifest", manifest, "--summary", md])
        (f,) = cmds["figures"]
        self.assertEqual(list(f.argv), ["--cueball-dir", "/data/t", "--out", "/data/t/figures", "--only", "F1,F3c"])
        # The command line every stage prints / runs.
        self.assertEqual(f.cmd()[:3], [sys.executable, "-m", "src.scripts.visualizations.paper_plots"])

    def test_figures_refuse_a_tree_paper_plots_would_not_read(self):
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {"DATA_ROOT": raw}), short_name_patch():
            plan = Plan(make_cfg())
            self.assertIn("re-sample manifest", plan.missing_input("figures"))
            plan.resample_manifest.parent.mkdir(parents=True)
            plan.resample_manifest.write_text("")
            self.assertIsNone(plan.missing_input("figures"))
            for section in ({"build_rollout_manifest": {"output": "${DATA_ROOT}/m/manifest.parquet"}},
                            {"select_resample_set": {"output_dir": "${DATA_ROOT}/rs"}},
                            {"paper_plots": {"cueball_dir": "${DATA_ROOT}/other"}}):
                why = Plan(make_cfg(**section)).missing_input("figures")
                self.assertIn("paper_plots.cueball_dir", why)
                self.assertIn("rollout_manifest.parquet", why)
            # cueball_dir pointing at the tree the manifest and re-sample dir live in passes.
            moved = make_cfg(data_dir="${DATA_ROOT}/tree", paper_plots={"cueball_dir": "${DATA_ROOT}/tree"})
            plan = Plan(moved)
            plan.resample_manifest.parent.mkdir(parents=True)
            plan.resample_manifest.write_text("")
            self.assertIsNone(plan.missing_input("figures"))

    def test_select_extend_can_be_switched_off_explicitly(self):
        with short_name_patch():
            plan = Plan(make_cfg(select_resample_set={"extend": False, "no_items": True, "only": "qwen"}))
        argv = plan.commands("resample_select")[0].argv
        self.assertNotIn("--extend", argv)
        self.assertIn("--no-items", argv)
        self.assertEqual(flag(argv, "--only"), "qwen")

    def test_splits_extend_follows_the_manifest_state(self):
        with tempfile.TemporaryDirectory() as raw:
            with patch.dict(os.environ, {"DATA_ROOT": raw}), short_name_patch():
                plan = Plan(make_cfg(assign_splits={"seed": 3, "fractions": [0.8, 0.1, 0.1]}))
                argv = plan.commands("splits")[0].argv
                self.assertEqual(list(argv), ["--manifest", str(plan.manifest), "--resample-manifest",
                                              str(plan.resample_manifest), "--seed", "3", "--fractions", "0.8,0.1,0.1"])
                plan.manifest.parent.mkdir(parents=True)
                plan.manifest.with_suffix(".meta.json").write_text(json.dumps({"split": {"seed": 3}}))
                cmd = plan.commands("splits")[0]
                self.assertIn("--extend", cmd.argv)
                self.assertIn("already carries a split", cmd.note)
                # The split column alone (no sidecar block) is enough.
                plan.manifest.with_suffix(".meta.json").write_text("{}")
                pd.DataFrame({"split": ["train", None]}).to_parquet(plan.manifest)
                self.assertIn("--extend", plan.commands("splits")[0].argv)
                pd.DataFrame({"split": [None, None]}).to_parquet(plan.manifest)
                self.assertNotIn("--extend", plan.commands("splits")[0].argv)
                # An explicit force / extend in the section is passed as written.
                forced = Plan(make_cfg(assign_splits={"force": True})).commands("splits")[0].argv
                self.assertIn("--force", forced)
                self.assertNotIn("--extend", forced)
                off = Plan(make_cfg(assign_splits={"extend": False})).commands("splits")[0].argv
                self.assertNotIn("--extend", off)


class TestCellStages(unittest.TestCase):
    def write_configs(self, td: Path, base_model: str = MODEL) -> Path:
        (td / "configs").mkdir(exist_ok=True)
        (td / "configs" / "base.yaml").write_text(yaml.safe_dump(
            {"model_name": base_model, "seed": 7, "thinking": True, "inference": {"max_tokens": 256}}))
        cfg = td / "configs" / "pipeline.yaml"
        cfg.write_text(yaml.safe_dump({"extends": "base.yaml", "compute_baseline": {"dataset": {"name": "medqa"}}}))
        return cfg

    def test_reroll_config_defaults(self):
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            td = Path(raw)
            plan = Plan(make_cfg(), config_path=self.write_configs(td))
            (cmd,) = plan.commands("rerolls")
            self.assertEqual(cmd.module, "src.scripts.resample_hinted_rollouts")
            self.assertEqual(list(cmd.argv), ["--model", MODEL, "--only", plan.cell_stem])
            self.assertEqual(str(cmd.config_path),
                             f"/data/pipeline/generated_configs/{plan.stem}_resample_hinted_rollouts.yaml")
            self.assertEqual(cmd.config, {
                "items_dir": "/data/resample/items",
                "output_dir": "/data/resample/rollouts",
                "model_bases": {MODEL: str((td / "configs" / "base.yaml").resolve())},
                "sample_seeds": [43, 44, 45, 46],
                "temperature": 0.7,
                "top_p": DEFAULT_TOP_P,
                "top_k": DEFAULT_TOP_K,
                # The hinted-rollouts stage's generation budget.
                "inference": {"max_model_len": 1024, "max_tokens": 512, "gpu_memory_utilization": 0.8},
                "chunk_size": 512,
            })

    def test_reroll_config_section_keys_win(self):
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            td = Path(raw)
            (td / "configs").mkdir()
            (td / "configs" / "other_base.yaml").write_text("model_name: x\n")
            cfg = make_cfg(
                select_resample_set={"sample_seeds": [1, 2, 3]},
                resample_hinted_rollouts={"model_bases": {MODEL: "other_base.yaml"}, "sample_seeds": [5, 6],
                                          "temperature": 0.9, "top_p": 1.0, "top_k": -1, "chunk_size": 64,
                                          "inference": {"max_tokens": 99}, "items_dir": "${DATA_ROOT}/it"},
            )
            plan = Plan(cfg, config_path=self.write_configs(td))
            gen = plan.commands("rerolls")[0].config
            self.assertEqual(gen["model_bases"], {MODEL: str((td / "configs" / "other_base.yaml").resolve())})
            self.assertEqual(gen["sample_seeds"], [5, 6])
            self.assertEqual((gen["temperature"], gen["top_p"], gen["top_k"], gen["chunk_size"]), (0.9, 1.0, -1, 64))
            self.assertEqual(gen["inference"], {"max_model_len": 1024, "max_tokens": 99, "gpu_memory_utilization": 0.8})
            self.assertEqual(gen["items_dir"], "/data/it")
            self.assertEqual(str(plan.items_csv()), f"/data/it/{plan.cell_stem}_resample_items.csv")

    def test_reroll_config_generates_a_base_when_none_pins_the_model(self):
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            td = Path(raw)
            generated = "/data/pipeline/generated_configs/test-1b_gpqa-diamond_baseline_reroll_base.yaml"
            shared = {"model_name": MODEL, "seed": 7, "thinking": True, "layers": [1, 2],
                      "inference": {"max_model_len": 1024, "max_tokens": 256, "gpu_memory_utilization": 0.8}}
            (cmd,) = Plan(make_cfg()).commands("rerolls")   # no config path
            self.assertEqual(cmd.config["model_bases"], {MODEL: generated})
            self.assertEqual(cmd.files, ((Path(generated), shared),))
            self.assertIn("no base config pins", cmd.note)
            # --model: the extends base pins another model.
            cfg = apply_cli_overrides(make_cfg(), make_args(model="Other/Model"))
            plan = Plan(cfg, config_path=self.write_configs(td, base_model=MODEL))
            (cmd,) = plan.commands("rerolls")
            self.assertEqual(list(cmd.argv), ["--model", "Other/Model", "--only", plan.cell_stem])
            self.assertEqual(cmd.config["model_bases"], {"Other/Model": generated})
            self.assertEqual(cmd.files[0][1]["model_name"], "Other/Model")
            self.assertNotIn("compute_baseline", cmd.files[0][1])
            # A config without `extends` serves as its own base; a pinning base needs no file.
            cfg = td / "configs" / "flat.yaml"
            cfg.write_text(yaml.safe_dump({"model_name": MODEL, "compute_baseline": {"dataset": {"name": "medqa"}}}))
            (cmd,) = Plan(make_cfg(), config_path=cfg).commands("rerolls")
            self.assertEqual(cmd.config["model_bases"], {MODEL: str(cfg.resolve())})
            self.assertEqual(cmd.files, ())
            self.assertIsNone(cmd.note)
            # An explicit model_bases is never replaced.
            (cmd,) = Plan(make_cfg(resample_hinted_rollouts={"model_bases": {MODEL: "b.yaml"}})).commands("rerolls")
            self.assertEqual(cmd.files, ())

    def test_reroll_sampling_inherits_the_hinted_rollouts_stage(self):
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            cfg = make_cfg()
            cfg["collect_hinted_rollouts"].update({"top_p": 1.0, "top_k": -1})
            gen = Plan(cfg).commands("rerolls")[0].config
            self.assertEqual((gen["top_p"], gen["top_k"]), (1.0, -1))
            # An explicit null in the section still inherits; a set value wins per key.
            cfg["resample_hinted_rollouts"] = {"top_p": None, "top_k": 50}
            gen = Plan(cfg).commands("rerolls")[0].config
            self.assertEqual((gen["top_p"], gen["top_k"]), (1.0, 50))
            # The shared top-level knobs reach stage 2 and through it the re-rolls.
            gen = Plan(make_cfg(top_p=0.8)).commands("rerolls")[0].config
            self.assertEqual((gen["top_p"], gen["top_k"]), (0.8, DEFAULT_TOP_K))
            gen = Plan(make_cfg()).commands("rerolls")[0].config
            self.assertEqual((gen["top_p"], gen["top_k"]), (DEFAULT_TOP_P, DEFAULT_TOP_K))

    def test_reroll_judge_config(self):
        with patch.dict(os.environ, {"DATA_ROOT": "/data"}), short_name_patch():
            plan = Plan(make_cfg(judge_rollouts={"judge_model": "test/judge", "workers": 2, "judge_max_tokens": 16384,
                                                 "judge_prompt_file": "p.txt"}))
            (cmd,) = plan.commands("judge_rerolls")
            self.assertEqual(cmd.module, "src.scripts.batch_judge_rollouts")
            self.assertEqual(cmd.argv, ())
            self.assertEqual(str(cmd.config_path), f"/data/pipeline/generated_configs/{plan.stem}_batch_judge_rollouts.yaml")
            self.assertEqual(cmd.config, {
                "rollouts": [f"/data/resample/rollouts/{plan.cell_stem}_rs*.csv"],
                "judge_model": "test/judge", "judge_prompt_file": "p.txt", "judge_workers": 2,
                "judge_max_tokens": 16384,
            })
            plan = Plan(make_cfg(batch_judge_rollouts={"judge_model": "b/judge", "judge_workers": 48,
                                                       "retry_errors": False, "rollouts": ["${DATA_ROOT}/x_rs*.csv"]}))
            gen = plan.commands("judge_rerolls")[0].config
            self.assertEqual(gen["rollouts"], ["${DATA_ROOT}/x_rs*.csv"])
            self.assertEqual((gen["judge_model"], gen["judge_workers"], gen["retry_errors"]), ("b/judge", 48, False))
            self.assertIsNone(gen["judge_prompt_file"])
            self.assertNotIn("judge_max_tokens", gen)
            self.assertEqual(plan.reroll_judge_model, "b/judge")


class TestProbeStages(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"DATA_ROOT": "/data"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.short = short_name_patch()
        self.short.start()
        self.addCleanup(self.short.stop)

    def test_probe_dataset_specs(self):
        plan = Plan(make_cfg(build_probe_dataset={"predicates": ["used_vs_ignored", "verbalised_vs_rest"],
                                                  "runs": ["medqa-test"], "cases": "positive", "seed": 3}))
        cmds = plan.commands("probe_dataset")
        self.assertEqual([c.module for c in cmds], ["src.scripts.build_probe_dataset"] * 2)
        self.assertEqual(str(cmds[0].config_path),
                         "/data/pipeline/generated_configs/test-1b_build_probe_dataset_used_vs_ignored.yaml")
        self.assertEqual(cmds[0].config, {
            "name": "test-1b_used_vs_ignored", "manifest": "/data/resample/resample_manifest.parquet",
            "predicate": "used_vs_ignored", "subject_models": ["test-1b"], "output_dir": "/data/probe_datasets",
            "runs": ["medqa-test"], "cases": "positive", "seed": 3,
        })
        self.assertEqual(cmds[1].config["name"], "test-1b_verbalised_vs_rest")
        self.assertEqual(cmds[0].argv, ())
        self.assertIsNone(cmds[0].note)
        # Default: the superset predicate; force / chunk_rows become flags.
        plan = Plan(make_cfg(build_probe_dataset={"force": True, "chunk_rows": 2000}))
        (cmd,) = plan.commands("probe_dataset")
        self.assertEqual(cmd.config["predicate"], "used_vs_ignored")
        self.assertEqual(list(cmd.argv), ["--force", "--chunk-rows", "2000"])
        self.assertEqual(plan.outputs("probe_dataset"), {"probe_datasets": ["/data/probe_datasets/test-1b_used_vs_ignored.parquet"]})

    def test_probe_dataset_specs_parse(self):
        from src.lib.probe_datasets import parse_spec
        plan = Plan(make_cfg(build_probe_dataset={"predicates": ["verbalised_vs_unverbalised"],
                                                  "balance": {"method": "ratio", "r": 1.0}, "hint_styles": ["metadata"]}))
        spec = parse_spec(plan.commands("probe_dataset")[0].config)
        self.assertEqual(spec.name, "test-1b_verbalised_vs_unverbalised")
        self.assertEqual(spec.subject_models, ("test-1b",))
        self.assertEqual(spec.balance.method, "ratio")

    def test_existing_probe_dataset_is_skipped_unless_forced(self):
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {"DATA_ROOT": raw}):
            plan = Plan(make_cfg())
            parquet = plan.probe_dataset_parquet("used_vs_ignored")
            parquet.parent.mkdir(parents=True)
            parquet.write_text("")
            self.assertIn("skip", plan.commands("probe_dataset")[0].note)
            self.assertIsNone(Plan(make_cfg(build_probe_dataset={"force": True})).commands("probe_dataset")[0].note)

    def test_activations_config(self):
        cfg = make_cfg(top_p=0.9, collect_probe_activations={
            "layers": [1], "seq_layers": [1], "store_folder": "shared", "max_file_gb": 2, "resume": True,
            "only": ["metadata", "gpqa"], "limit": 5, "hf_model_kwargs": {"chunk_size": 64},
        })
        plan = Plan(cfg)
        (cmd,) = plan.commands("activations")
        self.assertEqual(cmd.module, "src.scripts.collect_probe_activations")
        self.assertEqual(str(cmd.config_path), "/data/pipeline/generated_configs/test-1b_collect_probe_activations.yaml")
        self.assertEqual(list(cmd.argv), ["--resume", "--only", "metadata", "--only", "gpqa", "--limit", "5"])
        self.assertEqual(cmd.config, {
            "model_name": MODEL, "seed": 7, "thinking": True,
            "inference": {"max_model_len": 1024, "max_tokens": 256, "gpu_memory_utilization": 0.8},
            "layers": [1], "seq_layers": [1], "store_folder": "shared", "max_file_gb": 2,
            "hf_model_kwargs": {"chunk_size": 64},
            "dataset": "/data/probe_datasets/test-1b_used_vs_ignored.parquet",
            "storage": {"backend": "local", "local_dir": "/data/probe_activations/test-1b"},
        })
        self.assertNotIn("top_p", cmd.config)   # only the collector's keys reach it
        # predicate / dataset / storage / force.
        plan = Plan(make_cfg(collect_probe_activations={
            "predicate": "verbalised_vs_rest", "force": True,
            "storage": {"backend": "hf", "hf_repo_id": "org/repo", "hf_private": True, "local_dir": None}}))
        (cmd,) = plan.commands("activations")
        self.assertEqual(cmd.config["dataset"], "/data/probe_datasets/test-1b_verbalised_vs_rest.parquet")
        self.assertEqual(cmd.config["storage"]["backend"], "hf")
        self.assertEqual(cmd.config["layers"], [1, 2])   # the base's list
        self.assertEqual(list(cmd.argv), ["--force"])
        self.assertEqual(plan.outputs("activations"), {"store": cmd.config["storage"]})
        (cmd,) = Plan(make_cfg(collect_probe_activations={"dataset": "${DATA_ROOT}/ds.parquet"})).commands("activations")
        self.assertEqual(cmd.config["dataset"], "/data/ds.parquet")

    def test_activations_config_parses(self):
        from src.scripts.collect_probe_activations import parse_config
        cfg, _ = Plan(make_cfg()).activations_config()
        parsed = parse_config(cfg)
        self.assertEqual(parsed.layers, [1, 2])
        self.assertEqual(str(parsed.dataset), "/data/probe_datasets/test-1b_used_vs_ignored.parquet")

    def test_train_runs(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            (td / "probe_base.yaml").write_text(yaml.safe_dump({
                "model": {"layers": [1, 2]}, "dataloader": {"num_workers": 0},
                "training": {"epochs": 2}}))
            cfg_path = td / "pipeline.yaml"
            cfg_path.write_text("model_name: x\n")
            cfg = make_cfg(collect_probe_activations={"store_folder": "shared"}, train_probe={"runs": [
                {"extends": "probe_base.yaml", "probe": {"type": "linear"}, "output": {"run_name": "lin"},
                 "dataset": {"predicate": "verbalised_vs_rest", "cases": "positive"}},
                {"model": None, "probe": {"type": "tfidf"}, "device": "cpu", "dataset": {"path": "${DATA_ROOT}/d.parquet"}},
            ]})
            plan = Plan(cfg, config_path=cfg_path)
            lin, tfidf = plan.commands("train_probe")
            self.assertEqual(lin.module, "src.scripts.train_probe")
            self.assertEqual(str(lin.config_path), "/data/pipeline/generated_configs/test-1b_train_probe_lin.yaml")
            self.assertEqual(lin.argv, ())
            self.assertEqual(lin.config, {
                "model": {"layers": [1, 2], "subject_model": "test-1b",
                          "activations": {"backend": "local", "local_dir": "/data/probe_activations/test-1b",
                                          "folder": "shared"}},
                "dataloader": {"num_workers": 0},
                "training": {"epochs": 2},
                "probe": {"type": "linear"},
                "output": {"run_name": "lin", "dir": "/data/trained_probes/probes"},
                "dataset": {"cases": "positive", "path": "/data/probe_datasets/test-1b_verbalised_vs_rest.parquet"},
            })
            self.assertEqual(str(tfidf.config_path), "/data/pipeline/generated_configs/test-1b_train_probe_run2.yaml")
            self.assertEqual(list(tfidf.argv), ["--device", "cpu"])
            self.assertIsNone(tfidf.config["model"])
            self.assertEqual(tfidf.config["dataset"], {"path": "/data/d.parquet"})
            self.assertEqual(tfidf.config["output"], {"dir": "/data/trained_probes/probes"})
            # Without a store_folder the default folder is the collector's dataset name.
            plan = Plan(make_cfg(train_probe={"runs": [{"model": {}, "probe": {"type": "attention"}}]}))
            (cmd,) = plan.commands("train_probe")
            self.assertEqual(cmd.config["model"]["activations"]["folder"], "test-1b_used_vs_ignored")
            self.assertEqual(cmd.config["dataset"]["path"], "/data/probe_datasets/test-1b_used_vs_ignored.parquet")
            with self.assertRaisesRegex(ValueError, r"train_probe.runs\[0\].*bogus"):
                Plan(make_cfg(train_probe={"runs": [{"bogus": 1}]})).commands("train_probe")
            with self.assertRaisesRegex(ValueError, "needs the pipeline config"):
                Plan(make_cfg(train_probe={"runs": [{"extends": "x.yaml"}]})).commands("train_probe")
            self.assertEqual(Plan(make_cfg()).commands("train_probe"), [])

    def test_train_config_parses(self):
        from src.lib.probe_training import parse_train_config
        plan = Plan(make_cfg(train_probe={"runs": [
            {"probe": {"type": "linear"}, "model": {"layers": [1]}},
            {"probe": {"type": "tfidf"}, "model": None},
        ]}))
        linear, tfidf = (parse_train_config(c.config) for c in plan.commands("train_probe"))
        self.assertEqual(linear.model.folder, "test-1b_used_vs_ignored")
        self.assertEqual(linear.model.subject_model, "test-1b")
        self.assertIsNone(tfidf.model)


def script_source(script: str) -> str:
    for rel in (f"src/scripts/{script}.py", f"src/scripts/visualizations/{script}.py"):
        path = REPO_ROOT / rel
        if path.exists():
            return path.read_text()
    raise AssertionError(f"{script}: no script file")


class TestKeySetsMatchTheScripts(unittest.TestCase):
    """The key tables restated here must equal the stage scripts' own contracts."""

    @staticmethod
    def known_config_keys(script: str) -> set:
        tree = ast.parse(script_source(script))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "KNOWN_CONFIG_KEYS" for t in node.targets):
                return set(ast.literal_eval(node.value))
        raise AssertionError(f"{script}: no KNOWN_CONFIG_KEYS")

    @staticmethod
    def argparse_flags(script: str) -> set:
        source = script_source(script)
        return {f[2:].replace("-", "_") for f in re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source)}

    @staticmethod
    def store_true_flags(script: str) -> set:
        tree = ast.parse(script_source(script))
        out = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                continue
            if any(k.arg == "action" and getattr(k.value, "value", None) == "store_true" for k in node.keywords):
                out.add(str(node.args[0].value)[2:].replace("-", "_"))
        return out

    def test_config_style_key_sets(self):
        self.assertEqual(RESAMPLE_CONFIG_KEYS, self.known_config_keys("resample_hinted_rollouts"))
        self.assertEqual(BATCH_JUDGE_CONFIG_KEYS, self.known_config_keys("batch_judge_rollouts"))
        self.assertEqual(COLLECT_CONFIG_KEYS, self.known_config_keys("collect_probe_activations"))

    def test_flag_tables(self):
        for script, keys in FLAG_KEYS.items():
            self.assertEqual(set(keys), self.argparse_flags(script), script)
            self.assertEqual(set(keys) & BOOL_FLAGS, self.store_true_flags(script), script)


class TestMain(unittest.TestCase):
    """End-to-end over mocked stage subprocesses: each stand-in reads the generated config or
    argv the real script would get and writes its artifacts."""

    def setup_tree(self, td: Path, cfg: dict | None = None) -> Path:
        configs = td / "configs"
        configs.mkdir(exist_ok=True)
        base = configs / "test_base.yaml"
        full = cfg or make_cfg()
        shared = {k: full[k] for k in ("model_name", "seed", "thinking", "inference", "layers")}
        base.write_text(yaml.safe_dump(shared))
        rest = {k: v for k, v in full.items() if k not in shared}
        rest["compute_baseline"] = {
            **rest["compute_baseline"],
            "output_csv": rest["compute_baseline"].get("output_csv") or "${DATA_ROOT}/baselines/test_baseline.csv",
        }
        rest.setdefault("stages", LEGACY_STAGES)
        config_path = configs / "pipeline.yaml"
        config_path.write_text(yaml.safe_dump({"extends": "test_base.yaml", **rest}))
        return config_path

    # --- stage stand-ins ---------------------------------------------------
    def fake_baseline(self, gen: dict) -> int:
        self.baseline_cfg = gen
        out = Path(gen["output_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "original_index": [1, 2, 3],
            "baseline_answer": ["A", "B", "C"],
            "correct": [True, True, False],
        }).to_csv(out, index=False)
        out.with_suffix(".meta.json").write_text(json.dumps({
            **baseline_identity(gen), "complete": True, "n_rows": 3, "accuracy": 0.67,
        }))
        return 0

    def fake_rollouts(self, gen: dict) -> int:
        self.generation_cfg = gen
        out = Path(gen["output_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "original_index": [1, 2, 3, 3],
            "sample_type": ["positive", "positive", "negative", "negative"],
            "hint_name": ["authority", "authority", "authority", "metadata"],
            "hinted_answer": ["B", "C", "A", "C"],
            "baseline_answer": ["A", "B", "C", "C"],
            "groundtruth": ["A", "B", "C", "C"],
            "rollout": ["<think>x</think><answer>B</answer>"] * 4,
            "reasoning": ["x"] * 4,
            "final_answer": ["B", "B", "D", "C"],
        }).to_csv(out, index=False)
        return 0

    def fake_judge(self, gen: dict) -> int:
        self.judge_cfg = gen
        df = pd.read_csv(gen["input_csv"])
        switched = df["final_answer"] == df["hinted_answer"]
        for col in ("judge_model", "judge_label", "judge_confidence", "judge_reasoning"):
            df[col] = pd.NA
        df.loc[switched, "judge_model"] = gen["judge_model"]
        df.loc[switched, "judge_label"] = 0
        df.loc[switched, "judge_confidence"] = 0.9
        df.loc[switched, "judge_reasoning"] = "r"
        df.to_csv(gen["output_csv"], index=False)
        return 0

    def fake_tree(self, stage: str, module: str, argv, gen) -> int:
        """Writes what the later stages' input checks look for."""
        script = module.rsplit(".", 1)[-1]
        if script == "build_rollout_manifest":
            out = Path(flag(argv, "--output"))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("")
        elif script == "select_resample_set":
            out = Path(flag(argv, "--output-dir"))
            (out / "items").mkdir(parents=True, exist_ok=True)
            (out / "resample_selection.csv").write_text("")
            (out / "items" / f"{CELL}_resample_items.csv").write_text("")
        elif script == "resample_hinted_rollouts":
            out = Path(gen["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            for seed in gen["sample_seeds"]:
                (out / f"{CELL}_rs{seed}.csv").write_text("")
        elif script == "resample_relabel":
            (Path(flag(argv, "--resample-dir")) / "resample_manifest.parquet").write_text("")
        elif script == "unfaithfulness_metrics_summary":
            Path(flag(argv, "--output")).write_text("")
        elif script == "build_probe_dataset":
            out = Path(gen["output_dir"]) / f"{gen['name']}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("")
        return 0

    def dispatch(self, stage: str, config_path=None, *, module=None, argv=()) -> int:
        gen = None
        if config_path is not None:
            with open(config_path) as f:
                gen = yaml.safe_load(f)
            self.generated_configs[stage] = gen
        self.calls.append((stage, module or STAGE_MODULES[stage], list(argv), gen))
        handler = self.handlers.get(stage)
        if stage in LEGACY_STAGES:
            return (handler or getattr(self, f"fake_{stage}"))(gen)
        return (handler or self.fake_tree)(stage, module, argv, gen)

    def run_main(self, td: Path, *, extra_argv=(), cfg=None, **handlers):
        config_path = self.setup_tree(td, cfg)
        self.handlers = handlers
        self.generated_configs = {}
        self.calls = []
        argv = ["run_pipeline", "--config", str(config_path), *extra_argv]
        with patch.dict(os.environ, {"DATA_ROOT": str(td / "data")}), \
                patch("src.scripts.run_pipeline.run_stage_script", side_effect=self.dispatch) as runner, \
                patch("src.lib.llm_judge_verb.check_judge_model", return_value=True) as preflight, \
                short_name_patch(), patch.object(sys, "argv", argv):
            self.runner, self.preflight = runner, preflight
            main()
        return td / "data"

    def stages_run(self) -> list[str]:
        """The stages that ran, in order (a two-script stage counted once)."""
        out = []
        for call in self.runner.call_args_list:
            if not out or out[-1] != call.args[0]:
                out.append(call.args[0])
        return out

    @staticmethod
    def seed_tree(td: Path) -> None:
        """What the tree stages need when the cell stages did not run in this test."""
        (td / "data" / "hinted_rollouts").mkdir(parents=True, exist_ok=True)
        (td / "data" / "resample").mkdir(parents=True, exist_ok=True)
        (td / "data" / "resample" / "resample_manifest.parquet").write_text("")

    def record(self, data: Path) -> dict:
        return json.loads((data / "pipeline" / "test_baseline_pipeline.json").read_text())

    # --- tests -------------------------------------------------------------
    def test_full_run_chains_the_three_stages(self):
        with tempfile.TemporaryDirectory() as raw:
            data = self.run_main(Path(raw))
            rollouts_dir = data / "hinted_rollouts"
            judged = pd.read_csv(rollouts_dir / "test_baseline_hinted_rollouts_judged.csv")
            summary = json.loads((rollouts_dir / "test_baseline_hinted_rollouts_judged_summary.json").read_text())
            sidecar = json.loads((rollouts_dir / "test_baseline_hinted_rollouts.meta.json").read_text())
            record = self.record(data)
            generated = data / "pipeline" / "generated_configs"
            on_disk = {p.name: yaml.safe_load(p.read_text()) for p in generated.glob("*.yaml")}

        self.assertEqual(self.stages_run(), LEGACY_STAGES)
        self.preflight.assert_called_once_with("test/judge")

        b = self.generated_configs["baseline"]
        self.assertEqual(b["model_name"], MODEL)
        self.assertEqual(b["seed"], 7)
        self.assertEqual(b["dataset"]["name"], "gpqa")
        self.assertEqual(b["consistency_samples"], 4)
        self.assertEqual(b["inference"]["max_tokens"], 256)
        self.assertTrue(b["output_csv"].endswith("baselines/test_baseline.csv"))
        self.assertNotIn("compute_baseline", b)
        self.assertNotIn("stages", b)
        r = self.generated_configs["rollouts"]
        self.assertEqual(r["baseline_csv"], b["output_csv"])
        self.assertEqual(r["hints"], HINTS_USED)
        self.assertEqual(r["cases"], "both")
        self.assertEqual(r["temperature"], 0.7)
        self.assertEqual(r["inference"]["max_tokens"], 512)
        self.assertEqual(r["inference"]["gpu_memory_utilization"], 0.8)
        self.assertEqual(sidecar["model_name"], MODEL)
        self.assertEqual(sidecar["hints"], HINTS_USED)
        j = self.generated_configs["judge"]
        self.assertEqual(j["input_csv"], r["output_csv"])
        self.assertEqual(j["judge_model"], "test/judge")
        self.assertEqual(j["workers"], 2)
        self.assertTrue(j["cache_dir"].endswith("data/judge_cache/test_baseline_hinted_rollouts"))
        self.assertNotIn("model_name", j)
        self.assertNotIn("inference", j)
        self.assertEqual(sorted(on_disk), [
            "test_baseline_collect_hinted_rollouts.yaml",
            "test_baseline_compute_baseline.yaml",
            "test_baseline_judge_rollouts.yaml",
        ])
        self.assertEqual(on_disk["test_baseline_judge_rollouts.yaml"], j)
        self.assertEqual(int(judged.set_index(["original_index", "hint_name"]).loc[(1, "authority"), "judge_label"]), 0)
        self.assertEqual(summary["judge_model"], "test/judge")
        self.assertEqual(summary["provenance"]["hints"], HINTS_USED)
        by_key = {(x["case"], x["hint_name"]): x for x in summary["results"]}
        self.assertEqual(by_key[("positive", "authority")]["n_switched"], 1)
        self.assertEqual(by_key[("positive", "authority")]["n_unfaithful"], 1)
        self.assertEqual(by_key[("negative", "metadata")]["n_switched"], 1)
        self.assertEqual(record["stages"]["baseline"]["status"], "ok")
        self.assertEqual(record["stages"]["baseline"]["n_rows"], 3)
        self.assertEqual(record["stages"]["rollouts"]["status"], "ok")
        self.assertEqual(record["stages"]["judge"]["status"], "ok")
        self.assertEqual(record["stages"]["judge"]["n_judged"], 2)
        self.assertEqual(record["stages_requested"], LEGACY_STAGES)
        self.assertEqual(len(record["generations_complete"]), 1)

    def test_default_run_chains_the_ten_stages(self):
        cfg = make_cfg(stages=None, batch_judge_rollouts={"judge_model": "b/judge"})
        with tempfile.TemporaryDirectory() as raw:
            data = self.run_main(Path(raw), cfg=cfg)
            record = self.record(data)
            reroll_cfg = yaml.safe_load(
                (data / "pipeline" / "generated_configs" / "test_baseline_resample_hinted_rollouts.yaml").read_text())
            judge_cfg = yaml.safe_load(
                (data / "pipeline" / "generated_configs" / "test_baseline_batch_judge_rollouts.yaml").read_text())
        self.assertEqual(self.stages_run(), list(DEFAULT_STAGES))
        self.assertEqual(self.runner.call_count, 12)   # relabel and summary run two scripts each
        self.assertEqual([c.args[0] for c in self.preflight.call_args_list], ["test/judge", "b/judge"])
        modules = [m.rsplit(".", 1)[-1] for _, m, _, _ in self.calls]
        self.assertEqual(modules, [
            "compute_baseline", "collect_hinted_rollouts", "judge_rollouts", "build_rollout_manifest",
            "select_resample_set", "resample_hinted_rollouts", "batch_judge_rollouts", "resample_noise_model",
            "resample_relabel", "unfaithfulness_metrics_summary", "verify_manifest", "paper_plots",
        ])
        by_module = {m.rsplit(".", 1)[-1]: (argv, gen) for _, m, argv, gen in self.calls}
        hinted = str(data / "hinted_rollouts")
        self.assertEqual(flag(by_module["build_rollout_manifest"][0], "--dir"), hinted)
        self.assertIn("--extend", by_module["select_resample_set"][0])
        self.assertEqual(by_module["resample_hinted_rollouts"][0], ["--model", MODEL, "--only", CELL])
        self.assertEqual(reroll_cfg["model_bases"], {MODEL: str((Path(raw) / "configs" / "test_base.yaml").resolve())})
        self.assertEqual(reroll_cfg["output_dir"], str(data / "resample" / "rollouts"))
        self.assertEqual(judge_cfg["rollouts"], [str(data / "resample" / "rollouts" / f"{CELL}_rs*.csv")])
        self.assertEqual(judge_cfg["judge_model"], "b/judge")
        self.assertEqual(flag(by_module["paper_plots"][0], "--cueball-dir"), str(data))
        self.assertEqual(flag(by_module["verify_manifest"][0], "--summary"),
                         str(data / "hinted_rollouts" / "unfaithfulness_metrics_summary.md"))
        for stage in DEFAULT_STAGES:
            self.assertEqual(record["stages"][stage]["status"], "ok", stage)
        self.assertEqual(record["stages"]["manifest"]["manifest"], str(data / "hinted_rollouts" / "rollout_manifest.parquet"))
        self.assertEqual(record["plan"]["data_dir"], str(data))

    def test_optional_stages_run_only_when_named(self):
        cfg = make_cfg(stages=["manifest", "splits", "probe_dataset"], train_probe={"runs": [{"probe": {"type": "tfidf"}}]})
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            self.seed_tree(td)
            data = self.run_main(td, cfg=cfg)
            record = self.record(data)
            spec = yaml.safe_load(
                (data / "pipeline" / "generated_configs" / "test-1b_build_probe_dataset_used_vs_ignored.yaml").read_text())
        self.assertEqual(self.stages_run(), ["manifest", "splits", "probe_dataset"])
        splits_argv = self.calls[1][2]
        self.assertEqual(splits_argv[:2], ["--manifest", str(data / "hinted_rollouts" / "rollout_manifest.parquet")])
        self.assertNotIn("--extend", splits_argv)   # a fresh manifest carries no split
        self.assertEqual(spec["subject_models"], ["test-1b"])
        self.assertEqual(record["stages"]["probe_dataset"]["status"], "ok")
        self.assertEqual(self.preflight.call_count, 0)
        with tempfile.TemporaryDirectory() as raw:
            # A second run skips the existing parquet; --stages on the CLI wins over the config list.
            td = Path(raw)
            self.seed_tree(td)
            self.run_main(td, cfg=cfg)
            data = self.run_main(td, cfg=cfg, extra_argv=["--stages", "probe_dataset,train_probe"])
            record = self.record(data)
        self.assertEqual(self.stages_run(), ["train_probe"])
        self.assertEqual(record["stages"]["probe_dataset"]["status"], "skipped (already complete)")
        self.assertEqual(record["stages"]["train_probe"]["status"], "ok")

    def test_second_run_skips_baseline_and_generation(self):
        def explode(gen):
            raise AssertionError("must not rerun")

        with tempfile.TemporaryDirectory() as raw:
            self.run_main(Path(raw))
            data = self.run_main(Path(raw), baseline=explode, rollouts=explode)
            record = self.record(data)
        self.assertEqual(self.stages_run(), ["judge"])
        self.assertEqual(record["stages"]["baseline"]["status"], "skipped (already complete)")
        self.assertEqual(record["stages"]["rollouts"]["status"], "skipped (already complete)")

    def test_changed_recipe_regenerates(self):
        with tempfile.TemporaryDirectory() as raw:
            self.run_main(Path(raw))
            cfg = make_cfg()
            cfg["collect_hinted_rollouts"]["hints"] = ["authority"]
            self.run_main(Path(raw), cfg=cfg)
        self.assertEqual(self.stages_run(), ["rollouts", "judge"])

    def test_force_flags_rerun_completed_stages(self):
        with tempfile.TemporaryDirectory() as raw:
            self.run_main(Path(raw))
            self.run_main(Path(raw), extra_argv=["--force-baseline", "--force-generate"])
        self.assertEqual(self.stages_run(), LEGACY_STAGES)

    def test_stages_subset_and_judge_needs_rollouts(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            self.run_main(td, extra_argv=["--stages", "baseline"])
            self.assertEqual(self.stages_run(), ["baseline"])
            self.assertEqual(self.preflight.call_count, 0)
            with self.assertRaises(SystemExit) as ctx:
                self.run_main(td, extra_argv=["--stages", "judge"])
            self.assertEqual(ctx.exception.code, 1)
            record = self.record(td / "data")
        self.assertIn("rollouts CSV not found", record["stages"]["judge"]["status"])

    def test_tree_stage_without_its_input_fails_and_stops(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            with self.assertRaises(SystemExit) as ctx:
                self.run_main(td, extra_argv=["--stages", "rerolls,judge_rerolls"])
            self.assertEqual(ctx.exception.code, 1)
            record = self.record(td / "data")
        self.assertEqual(self.stages_run(), [])
        self.assertIn("items file", record["stages"]["rerolls"]["status"])
        self.assertNotIn("judge_rerolls", record["stages"])

    def test_model_override_writes_the_reroll_base_only_when_the_stage_runs(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            items = td / "data" / "resample" / "items"
            items.mkdir(parents=True)
            (items / f"{CELL}_resample_items.csv").write_text("")
            base = td / "data" / "pipeline" / "generated_configs" / "test_baseline_reroll_base.yaml"
            self.run_main(td, extra_argv=["--model", "Other/Model", "--stages", "rerolls", "--dry-run"])
            self.assertFalse(base.exists())
            self.assertEqual(self.stages_run(), [])
            data = self.run_main(td, extra_argv=["--model", "Other/Model", "--stages", "rerolls"])
            record = self.record(data)
            written = yaml.safe_load(base.read_text())
            reroll_cfg = self.generated_configs["rerolls"]
        self.assertEqual(self.stages_run(), ["rerolls"])
        self.assertEqual(record["stages"]["rerolls"]["status"], "ok")
        self.assertEqual(self.calls[0][2], ["--model", "Other/Model", "--only", CELL])
        self.assertEqual(reroll_cfg["model_bases"], {"Other/Model": str(base)})
        self.assertEqual(written["model_name"], "Other/Model")
        self.assertEqual(written["inference"]["max_tokens"], 256)
        self.assertNotIn("compute_baseline", written)
        self.assertNotIn("stages", written)

    def test_failed_baseline_stops_the_pipeline(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(SystemExit) as ctx:
                self.run_main(Path(raw), baseline=lambda gen: 3)
            self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(self.stages_run(), ["baseline"])

    def test_failed_tree_script_is_recorded(self):
        with tempfile.TemporaryDirectory() as raw:
            self.seed_tree(Path(raw))
            with self.assertRaises(SystemExit):
                self.run_main(Path(raw), extra_argv=["--stages", "manifest,resample_select"],
                              manifest=lambda *a: 2)
            record = self.record(Path(raw) / "data")
        self.assertEqual(self.stages_run(), ["manifest"])
        self.assertEqual(record["stages"]["manifest"]["status"], "failed (build_rollout_manifest exit 2)")

    def test_dry_run_touches_nothing(self):
        cfg = make_cfg(stages=list(STAGES), train_probe={"runs": [{"extends": "probe_base.yaml", "probe": {"type": "tfidf"}}]})
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            (td / "configs").mkdir()
            (td / "configs" / "probe_base.yaml").write_text("model: null\n")
            data = self.run_main(td, cfg=cfg, extra_argv=["--dry-run"])
            self.assertFalse(data.exists())
        self.assertEqual(self.stages_run(), [])
        self.assertEqual(self.preflight.call_count, 0)

    def test_cli_overrides_reach_the_generated_configs(self):
        with tempfile.TemporaryDirectory() as raw, short_name_patch():
            td = Path(raw)
            config_path = self.setup_tree(td)
            text = yaml.safe_load(config_path.read_text())
            text["compute_baseline"].pop("output_csv")
            config_path.write_text(yaml.safe_dump(text))
            self.handlers, self.generated_configs, self.calls = {}, {}, []
            argv = [
                "run_pipeline", "--config", str(config_path),
                "--dataset", "mmlu", "--dataset-param", "split=train",
                "--dataset-param", "samples_per_subject=5", "--run-name", "0904",
                "--cases", "positive_cases", "--judge-model", "other/judge",
                "--stages", "baseline,rollouts",
            ]
            with patch.dict(os.environ, {"DATA_ROOT": str(td / "data")}), \
                    patch("src.scripts.run_pipeline.run_stage_script", side_effect=self.dispatch) as runner, \
                    patch.object(sys, "argv", argv):
                self.runner = runner
                main()
            record = json.loads(
                (td / "data" / "pipeline" / "test-1b_mmlu-train_0904_baseline_pipeline.json").read_text()
            )
        self.assertEqual(self.stages_run(), ["baseline", "rollouts"])
        self.assertEqual(self.baseline_cfg["dataset"], {"name": "mmlu", "params": {"split": "train", "samples_per_subject": 5}})
        self.assertTrue(self.baseline_cfg["output_csv"].endswith("test-1b_mmlu-train_0904_baseline.csv"))
        self.assertEqual(self.generation_cfg["cases"], "positive_cases")
        self.assertEqual(record["plan"]["judge_model"], "other/judge")
        self.assertEqual(record["plan"]["reroll_judge_model"], "other/judge")
        self.assertEqual(record["stages_requested"], ["baseline", "rollouts"])


if __name__ == "__main__":
    unittest.main()
