"""Saved figure inputs must be usable without generation dependencies."""

from pathlib import Path
import subprocess
import sys
import unittest


class FigureImportsTest(unittest.TestCase):
    def test_saved_inputs_do_not_import_openai_or_datasets(self):
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.modules["openai"] = None
sys.modules["datasets"] = None
from src.lib.paper_figures import inputs
from src.lib import dataset, selection
assert selection.CASES == ("positive", "negative")
assert selection.SPLITS == ("train", "val", "test")
"""],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
