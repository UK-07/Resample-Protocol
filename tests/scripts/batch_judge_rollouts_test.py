"""Tests for src/scripts/batch_judge_rollouts.py (LLM judge mocked — no network)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.lib.llm_judge_verb import JudgeSetupError
from src.scripts.batch_judge_rollouts import (
    aggregate_judged,
    build_targets,
    main,
    resolve_rollouts_entry,
)

HINTS_USED = ["authority", "metadata"]


def make_rollouts_df() -> pd.DataFrame:
    """Two hints, both cases; rows 1, 3, 4 switched to the hint, row 2 kept the baseline."""
    rows = [
        (1, "positive", "authority", "B", "A", "B"),
        (2, "positive", "authority", "B", "A", "A"),
        (3, "positive", "metadata", "C", "A", "C"),
        (4, "negative", "authority", "D", "A", "D"),
    ]
    return pd.DataFrame([
        {
            "original_index": idx,
            "sample_type": sample_type,
            "hint_name": hint,
            "prompt": f"Question {idx}?",
            "hinted_prompt": f"Hinted question {idx}?",
            "hinted_answer": hinted,
            "baseline_answer": baseline,
            "groundtruth": baseline,
            "rollout": f"<think>t{idx}</think><answer>{final}</answer>",
            "reasoning": f"t{idx}",
            "final_answer": final,
            "additional_fields": "{}",
        }
        for idx, sample_type, hint, hinted, baseline, final in rows
    ])


def make_record(idx: int, hint: str, label: int, *, error: str | None = None) -> dict:
    rec = {
        "original_index": idx,
        "hint_name": hint,
        "judge_model": "test/judge",
        "label": label,
        "confidence": 0.9,
        "reasoning": f"because {label}",
        "error": error,
        "cost_usd": 0.0,
    }
    if error is not None:
        for key in ("label", "confidence", "reasoning"):
            del rec[key]
    return rec


def write_rollouts(
    data_root: Path, name: str = "test_rollouts", sidecar: dict | None = None
) -> Path:
    out = data_root / "hinted_rollouts"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{name}.csv"
    make_rollouts_df().to_csv(csv, index=False)
    if sidecar is not None:
        csv.with_suffix(".meta.json").write_text(json.dumps(sidecar))
    return csv


class TestResolveRolloutsEntry(unittest.TestCase):
    def test_bare_filename_lands_under_hinted_rollouts(self):
        with tempfile.TemporaryDirectory() as td:
            csv = write_rollouts(Path(td))
            with patch.dict(os.environ, {"DATA_ROOT": td}):
                self.assertEqual(resolve_rollouts_entry("test_rollouts.csv"), [csv])

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"DATA_ROOT": td}):
                with self.assertRaisesRegex(ValueError, "not found"):
                    resolve_rollouts_entry("absent.csv")

    def test_glob_expands_and_excludes_judged_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            a = write_rollouts(Path(td), name="a_hinted_rollouts")
            b = write_rollouts(Path(td), name="b_hinted_rollouts")
            # This script's own outputs must never be re-judged via a glob.
            write_rollouts(Path(td), name="a_hinted_rollouts_judged")
            with patch.dict(os.environ, {"DATA_ROOT": td}):
                self.assertEqual(resolve_rollouts_entry("*.csv"), [a, b])

    def test_glob_matching_nothing_raises(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "hinted_rollouts").mkdir()
            with patch.dict(os.environ, {"DATA_ROOT": td}):
                with self.assertRaisesRegex(ValueError, "matched no"):
                    resolve_rollouts_entry("*.csv")


class TestBuildTargets(unittest.TestCase):
    def test_dedupes_glob_and_explicit_overlap_and_collects_problems(self):
        with tempfile.TemporaryDirectory() as td:
            csv = write_rollouts(Path(td))
            with patch.dict(os.environ, {"DATA_ROOT": td}):
                targets, problems = build_targets(
                    ["test_rollouts.csv", "*.csv", "absent.csv"]
                )
        self.assertEqual(targets, [csv])
        self.assertEqual(len(problems), 1)
        self.assertIn("absent.csv", problems[0])

    def test_empty_list_raises(self):
        with self.assertRaisesRegex(ValueError, "nothing to judge"):
            build_targets([])


class TestAggregateJudged(unittest.TestCase):
    def test_counts_and_rates_per_case_and_hint(self):
        labeled = make_rollouts_df()
        labeled["judge_label"] = [0.0, None, None, 1.0]  # (3, metadata) judge-errored
        rows = {(r["case"], r["hint_name"]): r for r in aggregate_judged(labeled)}

        pos_auth = rows[("positive", "authority")]
        self.assertEqual(pos_auth["n_rollouts"], 2)
        self.assertEqual(pos_auth["n_switched"], 1)
        self.assertEqual(pos_auth["switch_rate"], 0.5)
        self.assertEqual(pos_auth["n_unfaithful"], 1)
        self.assertEqual(pos_auth["unfaithful_rate"], 1.0)

        pos_meta = rows[("positive", "metadata")]
        self.assertEqual(pos_meta["n_switched"], 1)
        self.assertEqual(pos_meta["n_judged"], 0)
        self.assertEqual(pos_meta["n_judge_errors"], 1)
        self.assertIsNone(pos_meta["unfaithful_rate"])

        neg_auth = rows[("negative", "authority")]
        self.assertEqual(neg_auth["n_switched"], 1)
        self.assertEqual(neg_auth["n_faithful"], 1)
        self.assertEqual(neg_auth["unfaithful_rate"], 0.0)

    def test_incoherent_leaves_the_denominator(self):
        labeled = make_rollouts_df()
        labeled["judge_label"] = [0.0, None, -1.0, 1.0]
        rows = {(r["case"], r["hint_name"]): r for r in aggregate_judged(labeled)}
        pos_meta = rows[("positive", "metadata")]
        self.assertEqual(pos_meta["n_incoherent"], 1)
        self.assertEqual(pos_meta["n_judged"], 1)
        self.assertIsNone(pos_meta["unfaithful_rate"])

    def test_empty_frame(self):
        self.assertEqual(aggregate_judged(pd.DataFrame()), [])


SIDECAR = {
    "model_name": "Test/Model-1B",
    "baseline_csv": "/data/baselines/test_baseline.csv",
    "cases": "both",
    "hints": HINTS_USED,
    "temperature": 0.7,
    "seed": 7,
}


class TestMain(unittest.TestCase):
    def write_config(self, td: Path, rollouts: list) -> Path:
        configs = td / "configs"
        configs.mkdir(exist_ok=True)
        config_path = configs / "batch_judge.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump({"rollouts": rollouts, "judge_model": "test/judge"}, f)
        return config_path

    def run_main(self, td: Path, rollouts: list, *, extra_argv=(), records=None, judge=None):
        config_path = self.write_config(td, rollouts)
        if records is None:
            records = {
                (1, "authority"): make_record(1, "authority", 0),
                (3, "metadata"): make_record(3, "metadata", 1),
                (4, "authority"): make_record(4, "authority", 1),
            }
        judge_patch = (
            {"side_effect": judge} if judge is not None else {"return_value": records}
        )
        argv = ["batch_judge_rollouts", "--config", str(config_path), *extra_argv]
        with patch.dict(os.environ, {"DATA_ROOT": str(td / "data")}), \
                patch("src.scripts.batch_judge_rollouts.run_judge_batch",
                      **judge_patch) as judge_mock, \
                patch("src.scripts.batch_judge_rollouts.check_judge_model",
                      return_value=True) as preflight, \
                patch.object(sys, "argv", argv):
            # Keep the mocks reachable even when main() raises (SystemExit).
            self.judge_mock, self.preflight = judge_mock, preflight
            main()
        return td / "data" / "hinted_rollouts"

    def test_full_run_writes_judged_csv_and_summary_with_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data", sidecar=SIDECAR)
            out = self.run_main(td, ["test_rollouts.csv"])
            judged = pd.read_csv(out / "test_rollouts_judged.csv")
            summary = json.loads((out / "test_rollouts_judged_summary.json").read_text())

        by_idx = judged.set_index("original_index")
        self.assertEqual(by_idx.loc[1, "judge_label"], 0)
        self.assertEqual(by_idx.loc[4, "judge_label"], 1)
        self.assertTrue(pd.isna(by_idx.loc[2, "judge_label"]))

        self.assertEqual(summary["judge_model"], "test/judge")
        self.assertEqual(summary["provenance"], SIDECAR)
        keyed = {(r["case"], r["hint_name"]) for r in summary["results"]}
        self.assertIn(("positive", "authority"), keyed)
        self.assertIn(("negative", "authority"), keyed)

        # Only the switched rows went to the judge.
        judged_rows = self.judge_mock.call_args.args[0]
        self.assertEqual(sorted(judged_rows["original_index"]), [1, 3, 4])

    def test_missing_sidecar_means_empty_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data")  # no sidecar
            out = self.run_main(td, ["test_rollouts.csv"])
            summary = json.loads((out / "test_rollouts_judged_summary.json").read_text())
        self.assertEqual(summary["provenance"], {})

    def test_glob_judges_every_collect_output(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data", name="a_hinted_rollouts")
            write_rollouts(td / "data", name="b_hinted_rollouts")
            out = self.run_main(td, ["*_hinted_rollouts.csv"])
            self.assertTrue((out / "a_hinted_rollouts_judged.csv").exists())
            self.assertTrue((out / "b_hinted_rollouts_judged.csv").exists())
        self.assertEqual(self.judge_mock.call_count, 2)

    def test_judge_setup_error_aborts_the_batch(self):
        def judge(*args, **kwargs):
            raise JudgeSetupError("API key revoked mid-run")

        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data", name="a_hinted_rollouts")
            write_rollouts(td / "data", name="b_hinted_rollouts")
            with self.assertRaisesRegex(JudgeSetupError, "revoked"):
                self.run_main(td, ["*_hinted_rollouts.csv"], judge=judge)
        # The batch aborts on the first CSV; the second is never judged.
        self.assertEqual(self.judge_mock.call_count, 1)

    def test_missing_entry_is_reported_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data")
            with self.assertRaises(SystemExit) as ctx:
                self.run_main(td, ["test_rollouts.csv", "absent.csv"])
            self.assertEqual(ctx.exception.code, 1)
            # The resolvable CSV was still judged.
            self.assertTrue(
                (td / "data" / "hinted_rollouts" / "test_rollouts_judged.csv").exists()
            )

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data")
            out_dir = self.run_main(td, ["test_rollouts.csv"], extra_argv=["--dry-run"])
            self.assertFalse((out_dir / "test_rollouts_judged.csv").exists())
        self.judge_mock.assert_not_called()
        self.preflight.assert_not_called()

    def test_only_filter_matching_nothing_raises(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data")
            with self.assertRaisesRegex(ValueError, "No judgeable"):
                self.run_main(td, ["test_rollouts.csv"], extra_argv=["--only", "nomatch"])

    def test_judge_model_flag_overrides_config(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data")
            out = self.run_main(
                td, ["test_rollouts.csv"], extra_argv=["--judge-model", "prov/flag"]
            )
            summary = json.loads((out / "test_rollouts_judged_summary.json").read_text())
        self.assertEqual(summary["judge_model"], "prov/flag")
        self.assertEqual(self.judge_mock.call_args.args[1], "prov/flag")

    def test_unknown_config_key_raises(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw)
            write_rollouts(td / "data")
            config_path = self.write_config(td, ["test_rollouts.csv"])
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg["baselines"] = ["x.csv"]  # generation key — wrong script
            with open(config_path, "w") as f:
                yaml.safe_dump(cfg, f)
            argv = ["batch_judge_rollouts", "--config", str(config_path)]
            with patch.dict(os.environ, {"DATA_ROOT": str(td / "data")}), \
                    patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "baselines"):
                    main()


if __name__ == "__main__":
    unittest.main()
