"""Tests for src/scripts/assemble_cueball_tree.py (the copy and sidecar helpers)."""

import json
import tempfile
import unittest
from pathlib import Path

from src.scripts.assemble_cueball_tree import Copier, repoint_sidecar


class TestHelpers(unittest.TestCase):
    def test_copy_records_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "shared" / "a.csv"
            src.parent.mkdir()
            src.write_text("x,y\n1,2\n")
            tree = Path(tmp) / "tree"
            cp = Copier(tree, "2000-01-01T00:00:00+00:00")
            cp.copy(src, tree / "baselines" / "a.csv")
            self.assertEqual((tree / "baselines" / "a.csv").read_text(), "x,y\n1,2\n")
            self.assertEqual(cp.copied[0]["file"], "baselines/a.csv")
            self.assertEqual(cp.copied[0]["source"], str(src))
            self.assertEqual(cp.copied[0]["size"], 8)

    def test_repoint_rewrites_every_baseline_csv_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "run.meta.json"
            meta.write_text(json.dumps({"baseline_csv": "/old/b.csv", "provenance": {"baseline_csv": "/old/b.csv"},
                                        "other": {"baseline_csv": "/old/b.csv"}}))
            repoint_sidecar(meta, Path("/new/b.csv"), "provenance", source="/old/run.meta.json", now="now")
            out = json.loads(meta.read_text())
            self.assertEqual(out["baseline_csv"], "/new/b.csv")
            self.assertEqual(out["provenance"]["baseline_csv"], "/new/b.csv")
            self.assertEqual(out["other"]["baseline_csv"], "/old/b.csv")  # only the named blocks
            self.assertEqual(out["copied_into_tree"], {"utc": "now", "source": "/old/run.meta.json"})


if __name__ == "__main__":
    unittest.main()
