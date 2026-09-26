"""Tests for src/scripts/verify_manifest.py."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.lib.rollout_manifest import derive_rollout_rows, pair_counts, write_manifest
from src.scripts.unfaithfulness_metrics_summary import render_report
from src.scripts.verify_manifest import compare, main, parse_summary, render_pair_row, stale_sources
from tests.lib.rollout_manifest_test import PAIR, make_baseline, make_chunk

SOURCE = {
    "subject_model": PAIR.subject_model, "run": PAIR.run,
    "judge_model": "z-ai/glm-5.3-flash", "judge_prompt": PAIR.judge_prompt,
}


def summary_from(manifest: pd.DataFrame, *, judge_model: str = "z-ai/glm-5.3-flash",
                 judge_prompt: str = PAIR.judge_prompt) -> str:
    """Render the summary script's markdown from a manifest's counts."""
    counts = pair_counts(manifest).loc[(PAIR.subject_model, PAIR.run)]
    counts = counts.drop(columns=["n_unjudged"])
    counts.index = counts.index.set_names(["hint_name", "sample_type"]).swaplevel()
    return render_report([{
        "path": Path(PAIR.source_csv), "model": PAIR.subject_model, "dataset": PAIR.run,
        "sidecar": {"model_name": PAIR.subject_model_id, "temperature": 0.7,
                    "thinking": True, "seed": 42},
        "counts": counts, "judge_model": judge_model, "judge_prompt": judge_prompt,
    }])


class TestParseSummary(unittest.TestCase):
    def test_parses_overview_pair_rows_and_provenance(self):
        manifest = derive_rollout_rows(make_chunk(), PAIR, baseline=make_baseline())
        parsed = parse_summary(summary_from(manifest))
        self.assertEqual([r["model"] for r in parsed["overview"]], [PAIR.subject_model])
        self.assertEqual(parsed["overview"][0]["dataset"], PAIR.run)
        self.assertEqual(parsed["overview"][0]["rollouts"], "8")
        self.assertEqual(parsed["overview"][0]["unjudged"], "1")
        self.assertEqual(parsed["overview"][0]["faithful"], "1")
        pair = parsed["pairs"][(PAIR.subject_model, PAIR.run)]
        self.assertEqual(pair["judge"], "z-ai/glm-5.3-flash")
        self.assertEqual(pair["prompt"], PAIR.judge_prompt)
        keys = [(r["hint"], r["case"]) for r in pair["rows"]]
        self.assertEqual(keys, [("expert_opinion", "negative"), ("expert_opinion", "positive"),
                                ("**all**", "**both**")])
        pos = pair["rows"][1]
        self.assertEqual((pos["rollouts"], pos["changed"], pos["switched"]), ("7", "4", "3"))
        self.assertEqual((pos["unfaithful"], pos["faithful"], pos["incoherent"], pos["unjudged"]),
                         ("0", "1", "1", "1"))
        self.assertEqual(pos["unfaithful_rate"], "0.0%")

    def test_ignores_separator_and_header_rows(self):
        text = "## Overview\n\n| model | dataset |\n| --- | --- |\n\n## m — d\n\n- judge: `j`   judge prompt: `p`\n\n| hint style | case |\n| --- | --- |\n"
        parsed = parse_summary(text)
        self.assertEqual(parsed["overview"], [])
        self.assertEqual(parsed["pairs"][("m", "d")]["rows"], [])


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.manifest = derive_rollout_rows(make_chunk(), PAIR, baseline=make_baseline())
        self.meta = {"sources": [SOURCE]}

    def test_matching_summary_has_no_problems(self):
        parsed = parse_summary(summary_from(self.manifest))
        self.assertEqual(compare(parsed, self.manifest, self.meta), [])

    def test_changed_count_is_reported(self):
        parsed = parse_summary(summary_from(self.manifest))
        tampered = self.manifest.copy()
        tampered.loc[tampered["original_index"] == 8, "changed"] = False
        problems = compare(parsed, tampered, self.meta)
        self.assertTrue(any("positive: changed summary='4' manifest='3'" in p for p in problems), problems)
        self.assertTrue(any("overview: changed" in p for p in problems), problems)
        self.assertTrue(any("**all** / **both**: changed" in p for p in problems), problems)

    def test_verdict_and_rate_mismatch_reported(self):
        parsed = parse_summary(summary_from(self.manifest))
        tampered = self.manifest.copy()
        tampered.loc[tampered["original_index"] == 1, "judge_label"] = 0
        problems = compare(parsed, tampered, self.meta)
        self.assertTrue(any("unfaithful summary='0' manifest='1'" in p for p in problems), problems)
        self.assertTrue(any("unfaithful_rate summary='0.0%' manifest='100.0%'" in p for p in problems), problems)

    def test_missing_and_extra_pairs(self):
        parsed = parse_summary(summary_from(self.manifest))
        other = self.manifest.copy()
        other["subject_model"] = "gemma3-4b-it"
        other["rollout_id"] = other["rollout_id"].str.replace(PAIR.subject_model, "gemma3-4b-it")
        problems = compare(parsed, other, self.meta)
        self.assertIn(f"pair {PAIR.subject_model} / {PAIR.run}: in the summary but not in the manifest", problems)
        self.assertIn(f"pair gemma3-4b-it / {PAIR.run}: in the manifest but not in the summary", problems)

    def test_judge_provenance_checked_against_meta(self):
        parsed = parse_summary(summary_from(self.manifest, judge_model="anthropic/claude-opus-4.6"))
        problems = compare(parsed, self.manifest, self.meta)
        self.assertTrue(any("judge summary='anthropic/claude-opus-4.6'" in p for p in problems), problems)
        problems = compare(parse_summary(summary_from(self.manifest)), self.manifest, {"sources": []})
        self.assertTrue(any("no provenance record" in p for p in problems), problems)
        tampered = self.manifest.copy()
        tampered.loc[tampered["original_index"] == 1, "judge_model"] = "other/judge"
        problems = compare(parse_summary(summary_from(self.manifest)), tampered, self.meta)
        self.assertTrue(any("rows judged by" in p for p in problems), problems)

    def test_missing_table_row_reported(self):
        text = summary_from(self.manifest)
        text = "\n".join(l for l in text.splitlines() if not l.startswith("| expert_opinion | negative"))
        problems = compare(parse_summary(text), self.manifest, self.meta)
        self.assertIn(f"{PAIR.subject_model} / {PAIR.run}: row expert_opinion / negative in the manifest "
                      "but not in the summary", problems)

    def test_render_pair_row_formats_like_the_summary(self):
        row = pd.Series({"n_rollouts": 10, "n_changed": 4, "n_switched": 4, "n_unfaithful": 1,
                         "n_faithful": 2, "n_incoherent": 1, "n_unjudged": 0,
                         "n_truncated": 1, "n_unanswered": 1})
        got = render_pair_row("h", "positive", row)
        self.assertEqual(got["switch_rate"], "40.0%")
        self.assertEqual(got["unfaithful_rate"], "33.3%")
        self.assertEqual(got["truncated"], "1")
        self.assertEqual(got["excluded"], "20.0% †")
        zero = render_pair_row("h", "positive", row * 0)
        self.assertEqual((zero["unfaithful_rate"], zero["excluded"]), ("—", "—"))


class TestMain(unittest.TestCase):
    def run_main(self, manifest: pd.DataFrame, text: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            mpath, spath = Path(tmp) / "m.parquet", Path(tmp) / "s.md"
            write_manifest(manifest, mpath, {"sources": [SOURCE]})
            spath.write_text(text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--manifest", str(mpath), "--summary", str(spath)])
        return code, out.getvalue()

    def test_passes_on_agreement_and_fails_on_drift(self):
        manifest = derive_rollout_rows(make_chunk(), PAIR, baseline=make_baseline())
        code, out = self.run_main(manifest, summary_from(manifest))
        self.assertEqual(code, 0, out)
        self.assertIn("OK:", out)
        tampered = manifest.copy()
        tampered.loc[tampered["original_index"] == 7, "to_hint"] = False
        code, out = self.run_main(tampered, summary_from(manifest))
        self.assertEqual(code, 1)
        self.assertIn("disagreement", out)
        self.assertIn("switched summary='3' manifest='2'", out)

    def test_reports_sources_changed_since_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / PAIR.source_csv
            csv.write_text("a\n")
            stat = csv.stat()
            mtime = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(timespec="seconds")
            fresh = {**SOURCE, "source_csv": csv.name, "source_size": stat.st_size,
                     "source_mtime_utc": mtime}
            self.assertEqual(stale_sources({"rollouts_dir": tmp, "sources": [fresh]}), [])
            csv.write_text("a\nb\n")
            notes = stale_sources({"rollouts_dir": tmp, "sources": [fresh]})
            self.assertEqual(len(notes), 1)
            self.assertIn("changed since the manifest was built", notes[0])
            notes = stale_sources({"rollouts_dir": tmp, "sources": [{**fresh, "source_csv": "gone.csv"}]})
            self.assertIn("gone.csv: missing", notes[0])
        self.assertEqual(stale_sources({"sources": [SOURCE]}), [])  # older meta without rollouts_dir


if __name__ == "__main__":
    unittest.main()
