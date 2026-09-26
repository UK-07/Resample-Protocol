"""Input handling for the reviewer-facing figure command."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.lib.paper_figures import workflow
from src.scripts.visualizations import paper_figures


class FigureWorkflowTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "numeric inputs"
        self.source.mkdir()
        for name in workflow.INPUT_FILES:
            (self.source / name).touch()
        self.output = self.root / "figures output"

    def test_dry_run_requires_only_two_tables_and_writes_nothing(self):
        with patch.object(workflow.subprocess, "run") as launch:
            plan = workflow.build(self.source, self.output, dry_run=True)
        self.assertEqual(plan["pdf_count"], 30)
        self.assertEqual(plan["latex_table_count"], 3)
        self.assertEqual(plan["phases"][0], "compute")
        self.assertFalse(self.output.exists())
        launch.assert_not_called()
        self.assertEqual({p.name for p in self.source.iterdir()}, set(workflow.INPUT_FILES))

    def test_missing_required_numeric_file_is_reported(self):
        (self.source / "rerolls_long.parquet").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "rerolls_long.parquet"):
            workflow.build(self.source, self.output, dry_run=True)
        self.assertFalse(self.output.exists())

    def test_source_tree_and_existing_outputs_are_protected(self):
        for output in (self.source, self.source / "nested"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                workflow.build(self.source, output)
        self.output.mkdir()
        sentinel = self.output / "existing.pdf"
        sentinel.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            workflow.build(self.source, self.output)
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_released_tree_dry_run_includes_preparation(self):
        for name in ("hinted_rollouts/rollout_manifest.parquet", "resample/resample_manifest.parquet", "resample/question_reliance.csv"):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for name in ("baselines", "binary_judge"):
            (self.source / name).mkdir()
        plan = workflow.build(self.source, self.output, from_tree=True, dry_run=True)
        self.assertEqual(plan["phases"][:2], ["prepare", "compute"])
        self.assertFalse(self.output.exists())

    def test_failed_calculation_stops_before_rendering(self):
        failure = subprocess.CalledProcessError(1, "compute")
        with patch.object(workflow.subprocess, "run", side_effect=failure) as launch:
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.build(self.source, self.output)
        self.assertEqual(launch.call_count, 1)
        command = launch.call_args.args[0]
        self.assertIn(str(self.source), command)
        self.assertIn(str(self.output), command)
        self.assertFalse((self.output / "figures").exists())

    def test_cli_rejects_two_input_modes(self):
        with self.assertRaises(SystemExit):
            paper_figures.main(["--data", str(self.source), "--cueball-dir", str(self.source),
                                "--out-root", str(self.output)])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
