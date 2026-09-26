"""Dispatch and compatibility checks for the shared visualization entry point."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch

from src.scripts.visualizations import make_all
from src.scripts.visualizations import paper_revision


class MakeAllTest(unittest.TestCase):
    def test_revision_mode_delegates_exact_paths_flags_and_exit_code(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                args = ["--paper-revision-bundle", "/inputs/paper bundle.tar.gz",
                        "--out-root", "/scratch/figure outputs"]
                expected = ["--bundle", "/inputs/paper bundle.tar.gz",
                            "--out-root", "/scratch/figure outputs"]
                if dry_run:
                    args.append("--dry-run")
                    expected.append("--dry-run")
                with patch.object(paper_revision, "main", return_value=7) as revision, \
                     patch.object(make_all, "select_plots") as select, \
                     patch.object(make_all, "run_step") as run:
                    self.assertEqual(make_all.main(args), 7)
                revision.assert_called_once_with(expected)
                select.assert_not_called()
                run.assert_not_called()

    def test_revision_mode_requires_an_explicit_output(self):
        with patch.object(paper_revision, "main") as revision, \
             patch.object(make_all, "run_step") as run:
            with self.assertRaisesRegex(SystemExit, "requires a fresh --out-root"):
                make_all.main(["--paper-revision-bundle", "bundle.tar.gz"])
        revision.assert_not_called()
        run.assert_not_called()

    def test_revision_mode_rejects_incompatible_legacy_flags(self):
        conflicts = (["--only", "survival_vs_stability"], ["--gather-only"],
                     ["--plot-only"], ["--cueball-dir", "/inputs/tree"],
                     ["--cache-dir", "/scratch/cache"])
        for conflict in conflicts:
            with self.subTest(conflict=conflict), \
                 patch.object(paper_revision, "main") as revision, \
                 patch.object(make_all, "run_step") as run:
                with self.assertRaisesRegex(SystemExit, "cannot be combined"):
                    make_all.main(["--paper-revision-bundle", "bundle.tar.gz",
                                   "--out-root", "/scratch/output", *conflict])
                revision.assert_not_called()
                run.assert_not_called()

    def test_legacy_subset_still_gathers_and_plots_in_dependency_order(self):
        with redirect_stdout(io.StringIO()), \
             patch.object(paper_revision, "main") as revision, \
             patch.object(make_all, "run_step", return_value=(True, 0.0)) as run:
            code = make_all.main(["--only", "rates_dumbbell,survival_vs_stability",
                                  "--cueball-dir", "/inputs/tree",
                                  "--out-root", "/scratch/legacy",
                                  "--cache-dir", "/scratch/cache"])
        self.assertEqual(code, 0)
        revision.assert_not_called()
        expected = []
        for plot in ("survival_vs_stability", "rates_dumbbell"):
            out = str(Path("/scratch/legacy") / plot)
            expected.extend([
                call([sys.executable, "-m", "src.scripts.visualizations." + plot + ".gather",
                      "--cueball-dir", "/inputs/tree", "--out-dir", out,
                      "--cache-dir", "/scratch/cache"], dry_run=False),
                call([sys.executable, "-m", "src.scripts.visualizations." + plot + ".plot",
                      "--out-dir", out, "--data", out + "/data"], dry_run=False),
            ])
        self.assertEqual(run.call_args_list, expected)

    def test_legacy_dry_run_never_starts_a_subprocess_or_revision(self):
        with redirect_stdout(io.StringIO()) as printed, \
             patch.object(paper_revision, "main") as revision, \
             patch.object(make_all.subprocess, "run") as launch:
            code = make_all.main(["--only", "survival_vs_stability", "--dry-run"])
        self.assertEqual(code, 0)
        launch.assert_not_called()
        revision.assert_not_called()
        self.assertIn("survival_vs_stability.gather", printed.getvalue())
        self.assertIn("survival_vs_stability.plot", printed.getvalue())

    def test_legacy_gather_failure_still_skips_its_plot_and_continues(self):
        with redirect_stdout(io.StringIO()), \
             patch.object(paper_revision, "main") as revision, \
             patch.object(make_all, "run_step", side_effect=[(False, 0.0), (True, 0.0), (True, 0.0)]) as run:
            code = make_all.main(["--only", "survival_vs_stability,rates_dumbbell"])
        self.assertEqual(code, 1)
        revision.assert_not_called()
        modules = [entry.args[0][2] for entry in run.call_args_list]
        self.assertEqual(modules, ["src.scripts.visualizations.survival_vs_stability.gather",
                                  "src.scripts.visualizations.rates_dumbbell.gather",
                                  "src.scripts.visualizations.rates_dumbbell.plot"])


if __name__ == "__main__":
    unittest.main()
