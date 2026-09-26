"""Tests for src/scripts/judge_rollouts.py (LLM judge mocked — no network)."""

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.lib.llm_judge_verb import DEFAULT_JUDGE_MODEL
from src.scripts.judge_rollouts import (
    JUDGE_COLUMNS,
    JudgeRolloutsConfig,
    attach_judgements,
    main,
    resolve_config,
    select_judgeable,
)


def make_rollouts_df() -> pd.DataFrame:
    """Rollouts CSV fixture: 1/4 switched to the hint, 2 unchanged, 3 third option, 5 empty."""
    rows = [
        (1, "B", "A", "B", "<think>hint says B</think><answer>B</answer>"),
        (2, "B", "A", "A", "<think>sticking with A</think><answer>A</answer>"),
        (3, "B", "A", "C", "<think>actually C</think><answer>C</answer>"),
        (4, "B", "A", "B", "<think>fine, B</think><answer>B</answer>"),
        (5, "B", "A", "", ""),
    ]
    return pd.DataFrame(
        [
            {
                "original_index": idx,
                "sample_type": "positive",
                "hint_name": "sycophancy",
                "prompt": f"Question {idx}?",
                "hinted_prompt": f"Hinted question {idx}?",
                "hinted_answer": hinted,
                "baseline_answer": baseline,
                "groundtruth": baseline,
                "rollout": rollout,
                "reasoning": rollout.split("</think>")[0].removeprefix("<think>"),
                "final_answer": final,
                "additional_fields": "{}",
            }
            for idx, hinted, baseline, final, rollout in rows
        ]
    )


def make_record(idx: int, label: int, *, error: str | None = None) -> dict:
    rec = {
        "original_index": idx,
        "hint_name": "sycophancy",
        "judge_model": "test/judge",
        "label": label,
        "confidence": 0.9,
        "reasoning": f"because {label}",
        "error": error,
        "cost_usd": 0.001,
    }
    if error is not None:
        for k in ("label", "confidence", "reasoning"):
            del rec[k]
    return rec


class TestSelectJudgeable(unittest.TestCase):
    def test_default_keeps_only_changes_toward_the_hint(self):
        selected = select_judgeable(make_rollouts_df())
        self.assertEqual(sorted(selected["original_index"]), [1, 4])

    def test_all_keeps_every_nonempty_rollout(self):
        selected = select_judgeable(make_rollouts_df(), judge_all=True)
        self.assertEqual(sorted(selected["original_index"]), [1, 2, 3, 4])

    def test_all_includes_unchanged_and_drifted_rows_with_outcome(self):
        """--all returns unchanged and drifted rows too, each with its outcome."""
        out = select_judgeable(make_rollouts_df(), judge_all=True)
        by_idx = out.set_index("original_index")["outcome"].to_dict()
        self.assertEqual(by_idx[1], "switched_to_hint")
        self.assertEqual(by_idx[4], "switched_to_hint")
        self.assertIn("unchanged", by_idx.values())
        self.assertIn("third_option", by_idx.values())

    def test_empty_rollout_never_judged(self):
        selected = select_judgeable(make_rollouts_df(), judge_all=True)
        self.assertNotIn(5, set(selected["original_index"]))


class TestAttachJudgements(unittest.TestCase):
    def test_fills_judged_rows_and_leaves_others_na(self):
        df = make_rollouts_df()
        records = {
            (1, "sycophancy"): make_record(1, 0),
            (4, "sycophancy"): make_record(4, 1, error="boom"),
        }
        out = attach_judgements(df, records)
        self.assertEqual(list(out.columns), list(df.columns) + JUDGE_COLUMNS)

        judged = out[out["original_index"] == 1].iloc[0]
        self.assertEqual(judged["judge_label"], 0)
        self.assertEqual(judged["judge_confidence"], 0.9)
        self.assertEqual(judged["judge_reasoning"], "because 0")
        self.assertEqual(judged["judge_model"], "test/judge")
        # A record cached before the audit keys were kept has no role/quote.
        self.assertTrue(pd.isna(judged["judge_role"]))
        self.assertTrue(pd.isna(judged["judge_hint_quote"]))

        for idx in (2, 3, 4, 5):  # unjudged, out-of-scope, errored, empty
            row = out[out["original_index"] == idx].iloc[0]
            self.assertTrue(
                all(pd.isna(row[c]) for c in JUDGE_COLUMNS),
                f"row {idx} should have empty judge columns",
            )

    def test_audit_keys_written(self):
        df = make_rollouts_df()
        rec = dict(make_record(1, 1), hint_role="credited", hint_quote="hint says B")
        out = attach_judgements(df, {(1, "sycophancy"): rec})
        judged = out[out["original_index"] == 1].iloc[0]
        self.assertEqual(judged["judge_role"], "credited")
        self.assertEqual(judged["judge_hint_quote"], "hint says B")
        # A verdict with no quotable mention stores NA, not the string "None".
        rec = dict(make_record(4, 0), hint_role="none", hint_quote=None)
        out = attach_judgements(df, {(4, "sycophancy"): rec})
        judged = out[out["original_index"] == 4].iloc[0]
        self.assertEqual(judged["judge_role"], "none")
        self.assertTrue(pd.isna(judged["judge_hint_quote"]))

    def test_original_rows_and_order_preserved(self):
        df = make_rollouts_df()
        out = attach_judgements(df, {})
        self.assertEqual(list(out["original_index"]), list(df["original_index"]))

    def test_verdicts_outside_the_scope_are_blanked(self):
        """A cached verdict under an out-of-scope row's key never reaches the CSV."""
        df = make_rollouts_df()
        records = {
            (1, "sycophancy"): make_record(1, 0),
            (2, "sycophancy"): make_record(2, 0),   # unchanged answer: out of scope
            (3, "sycophancy"): make_record(3, 1),   # third option: only under --all
        }
        out = attach_judgements(df, records).set_index("original_index")
        self.assertEqual(out.loc[1, "judge_label"], 0)
        self.assertTrue(pd.isna(out.loc[2, "judge_label"]))
        self.assertTrue(pd.isna(out.loc[3, "judge_label"]))
        # An explicit scope admits only the rows it names.
        scope = select_judgeable(df, judge_all=True).index
        out = attach_judgements(df, records, scope=scope).set_index("original_index")
        self.assertEqual(out.loc[3, "judge_label"], 1)
        self.assertEqual(out.loc[2, "judge_label"], 0)
        out = attach_judgements(df, records, scope=df.index[:0]).set_index("original_index")
        self.assertTrue(out["judge_label"].isna().all())


class TestMain(unittest.TestCase):
    def run_main(self, tmp: Path, extra_args: list[str] | None = None) -> pd.DataFrame:
        input_csv = tmp / "rollouts.csv"
        make_rollouts_df().to_csv(input_csv, index=False)
        records = {
            (1, "sycophancy"): make_record(1, 1),
            (4, "sycophancy"): make_record(4, 0),
        }
        argv = [
            "judge_rollouts",
            "--input", str(input_csv),
            "--cache-dir", str(tmp / "cache"),
            *(extra_args or []),
        ]
        with patch(
            "src.scripts.judge_rollouts.run_judge_batch", return_value=records
        ) as mock_batch, patch.object(sys, "argv", argv):
            main()
        self.mock_batch = mock_batch
        return pd.read_csv(tmp / "rollouts_judged.csv")

    def test_writes_labeled_csv_beside_input(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.run_main(Path(td))
        self.assertEqual(len(out), 5)
        by_idx = out.set_index("original_index")
        self.assertEqual(by_idx.loc[1, "judge_label"], 1)
        self.assertEqual(by_idx.loc[4, "judge_label"], 0)
        self.assertTrue(pd.isna(by_idx.loc[2, "judge_label"]))
        self.assertTrue(pd.isna(by_idx.loc[3, "judge_label"]))
        self.assertTrue(pd.isna(by_idx.loc[5, "judge_label"]))

    def test_judges_only_rows_changed_toward_the_hint(self):
        with tempfile.TemporaryDirectory() as td:
            self.run_main(Path(td))
        judged = self.mock_batch.call_args.args[0]
        self.assertEqual(sorted(judged["original_index"]), [1, 4])

    def test_all_flag_widens_the_judged_set(self):
        with tempfile.TemporaryDirectory() as td:
            self.run_main(Path(td), extra_args=["--all"])
        judged = self.mock_batch.call_args.args[0]
        self.assertEqual(sorted(judged["original_index"]), [1, 2, 3, 4])

    def test_config_file_supplies_keys_and_flags_override(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_csv = tmp / "rollouts.csv"
            make_rollouts_df().to_csv(input_csv, index=False)
            cfg = tmp / "judge.yaml"
            cfg.write_text(
                f"input_csv: {input_csv}\n"
                f"output_csv: {tmp / 'from_cfg.csv'}\n"
                f"cache_dir: {tmp / 'cache'}\n"
                "judge_model: prov/cfg-model\n"
                "judge_all: true\n"
                "workers: 2\n"
            )
            argv = ["judge_rollouts", "--config", str(cfg), "--judge-model", "prov/flag"]
            with patch(
                "src.scripts.judge_rollouts.run_judge_batch", return_value={}
            ) as mock_batch, patch.object(sys, "argv", argv):
                main()
            self.assertTrue((tmp / "from_cfg.csv").exists())
            args, kwargs = mock_batch.call_args
            self.assertEqual(args[1], "prov/flag")          # flag beats YAML
            self.assertEqual(kwargs["max_workers"], 2)      # YAML beats default
            self.assertEqual(sorted(args[0]["original_index"]), [1, 2, 3, 4])  # judge_all

    def test_unknown_config_key_and_missing_input_raise(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = tmp / "judge.yaml"
            cfg.write_text("input: x.csv\n")
            with patch.object(sys, "argv", ["judge_rollouts", "--config", str(cfg)]):
                with self.assertRaises(ValueError) as ctx:
                    main()
            self.assertIn("Unknown key", str(ctx.exception))
            with patch.object(sys, "argv", ["judge_rollouts"]):
                with self.assertRaises(ValueError):
                    main()

    def test_custom_output_path(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_csv = tmp / "rollouts.csv"
            make_rollouts_df().to_csv(input_csv, index=False)
            out_csv = tmp / "custom.csv"
            argv = [
                "judge_rollouts",
                "--input", str(input_csv),
                "--output", str(out_csv),
                "--cache-dir", str(tmp / "cache"),
            ]
            with patch(
                "src.scripts.judge_rollouts.run_judge_batch", return_value={}
            ), patch.object(sys, "argv", argv):
                main()
            self.assertTrue(out_csv.exists())


def write_rollouts(tmp: Path) -> Path:
    csv = tmp / "rollouts.csv"
    make_rollouts_df().to_csv(csv, index=False)
    return csv


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Run the script's own parser over ``argv``."""
    with patch.object(sys, "argv", ["judge_rollouts", *argv]):
        captured = {}

        def capture(args):
            captured["args"] = args
            raise SystemExit(0)

        with patch("src.scripts.judge_rollouts.resolve_config", side_effect=capture):
            with contextlib.suppress(SystemExit):
                main()
    return captured["args"]


class TestResolveConfig(unittest.TestCase):
    """The dataclass config: defaults, precedence, and key validation."""

    def test_defaults_come_from_the_dataclass(self):
        config = resolve_config(parse_args(["--input", "x.csv"]))
        self.assertEqual(config.judge_model, DEFAULT_JUDGE_MODEL)
        self.assertEqual(config.workers, 4)
        self.assertFalse(config.judge_all)
        self.assertIsNone(config.output_csv)

    def test_flag_beats_yaml_beats_default(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = tmp / "judge.yaml"
            cfg.write_text(
                "input_csv: from_yaml.csv\njudge_model: prov/yaml\nworkers: 7\n"
            )
            config = resolve_config(
                parse_args(["--config", str(cfg), "--judge-model", "prov/flag"])
            )
            self.assertEqual(config.judge_model, "prov/flag")   # flag
            self.assertEqual(config.workers, 7)                 # yaml
            self.assertEqual(config.input_csv, "from_yaml.csv")  # yaml
            self.assertFalse(config.judge_all)                  # default

    def test_unknown_key_names_the_known_ones(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "judge.yaml"
            cfg.write_text("input_csv: x.csv\nworkerz: 4\n")
            with self.assertRaises(ValueError) as ctx:
                resolve_config(parse_args(["--config", str(cfg)]))
            message = str(ctx.exception)
            self.assertIn("workerz", message)
            self.assertIn("judge_prompt_file", message)  # known-keys listing

    def test_every_dataclass_field_is_an_accepted_config_key(self):
        """A knob added to the dataclass must be settable from YAML."""
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "judge.yaml"
            body = "".join(
                f"{f.name}: {'true' if f.name == 'judge_all' else 1 if f.name in ('workers', 'judge_max_tokens') else 'v'}\n"
                for f in JudgeRolloutsConfig.__dataclass_fields__.values()
            )
            cfg.write_text(body)
            config = resolve_config(parse_args(["--config", str(cfg)]))
            self.assertTrue(config.judge_all)
            self.assertEqual(config.workers, 1)
            self.assertEqual(config.judge_max_tokens, 1)


class TestDryRun(unittest.TestCase):
    """--dry-run renders one prompt and touches nothing else."""

    def _dry_run(self, tmp: Path, extra: list[str] | None = None) -> str:
        argv = [
            "judge_rollouts",
            "--input", str(write_rollouts(tmp)),
            "--cache-dir", str(tmp / "cache"),
            "--dry-run",
            *(extra or []),
        ]
        buf = io.StringIO()
        with patch(
            "src.scripts.judge_rollouts.run_judge_batch",
            side_effect=AssertionError("the judge must not be called on a dry run"),
        ), patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
            main()
        return buf.getvalue()

    def test_renders_the_first_judgeable_prompt_without_calling_the_api(self):
        with tempfile.TemporaryDirectory() as td:
            out = self._dry_run(Path(td))
        self.assertIn("<reasoning_trace>", out)
        self.assertIn("hint says B", out)          # the first judgeable row's CoT
        self.assertIn("none sent", out)

    def test_writes_no_files(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._dry_run(tmp)
            self.assertFalse((tmp / "rollouts_judged.csv").exists())
            self.assertFalse((tmp / "rollouts_judged.meta.json").exists())

    def test_respects_the_all_flag_when_choosing_the_row(self):
        """--all widens the judgeable set."""
        with tempfile.TemporaryDirectory() as td:
            out = self._dry_run(Path(td), extra=["--all"])
        self.assertIn("judging 4", out)

    def test_reports_when_nothing_is_judgeable(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            csv = tmp / "rollouts.csv"
            df = make_rollouts_df()
            df["final_answer"] = df["baseline_answer"]   # nobody switched
            df.to_csv(csv, index=False)
            buf = io.StringIO()
            argv = ["judge_rollouts", "--input", str(csv),
                    "--cache-dir", str(tmp / "c"), "--dry-run"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
                main()
            self.assertIn("Nothing judgeable", buf.getvalue())


class TestMetaSidecar(unittest.TestCase):
    """A .meta.json beside the labeled CSV records what produced it."""

    def _run(self, tmp: Path, extra: list[str] | None = None) -> dict:
        records = {(1, "sycophancy"): make_record(1, 1),
                   (4, "sycophancy"): make_record(4, 0)}
        argv = [
            "judge_rollouts",
            "--input", str(write_rollouts(tmp)),
            "--cache-dir", str(tmp / "cache"),
            *(extra or []),
        ]
        with patch(
            "src.scripts.judge_rollouts.run_judge_batch", return_value=records
        ), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            main()
        return json.loads((tmp / "rollouts_judged.meta.json").read_text())

    def test_records_run_identity(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            meta = self._run(tmp, extra=["--judge-model", "prov/m"])
        self.assertEqual(meta["judge_model"], "prov/m")
        self.assertFalse(meta["judge_all"])
        self.assertTrue(meta["input_csv"].endswith("rollouts.csv"))
        self.assertTrue(meta["output_csv"].endswith("rollouts_judged.csv"))
        self.assertIn("judged_at", meta)

    def test_records_the_builtin_prompt_by_cache_hash(self):
        from src.lib.llm_judge_verb import prompt_template_hash

        with tempfile.TemporaryDirectory() as td:
            meta = self._run(Path(td))
        self.assertTrue(meta["prompt"]["builtin"])
        self.assertIsNone(meta["prompt"]["path"])
        self.assertEqual(meta["prompt"]["cache_hash"], prompt_template_hash(None))
        # The default is pinned by content too, not just by "builtin: true".
        self.assertTrue(meta["prompt"]["sha256"].startswith(meta["prompt"]["cache_hash"]))

    def test_records_a_custom_prompt_by_content_hash(self):
        import hashlib

        body = "".join("<%s>${%s}</%s>\n" % (p, p, p) for p in (
            "baseline_prompt", "hint_description", "target_option",
            "model_answer", "reasoning_trace",
        ))
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_file = tmp / "custom.txt"
            prompt_file.write_text(body)
            meta = self._run(tmp, extra=["--judge-prompt-file", str(prompt_file)])
        self.assertFalse(meta["prompt"]["builtin"])
        self.assertEqual(meta["prompt"]["style"], "dollar")
        self.assertEqual(
            meta["prompt"]["sha256"], hashlib.sha256(body.encode()).hexdigest()
        )

    def test_records_verdict_counts_and_cost(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._run(Path(td))
        self.assertEqual(meta["counts"]["rows_in_csv"], 5)
        self.assertEqual(meta["counts"]["selected_for_judging"], 2)
        self.assertEqual(meta["counts"]["verdicts"], {"faithful": 1, "unfaithful": 1})
        self.assertAlmostEqual(meta["cost_usd"], 0.002)


if __name__ == "__main__":
    unittest.main()
