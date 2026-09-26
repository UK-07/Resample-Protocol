"""Tests for src/scripts/mirror_probe_store.py over a fake HF backend; nothing touches the network."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.lib import activation_store as st
from src.scripts import mirror_probe_store as mps
from tests.lib.activation_store_test import REPO, FakeApi, hf_backend_over, write_store


class MirrorProbeStoreScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.local, self.parts = write_store(self.tmp / "src")
        self.hf = hf_backend_over(self.local)
        self.api: FakeApi = self.hf.api
        self.dst = self.tmp / "mirror"
        self.addCleanup(st.reset_read_caches)

    def config(self, *, storage=None, store_folder="spec") -> Path:
        cfg = {"model_name": "fake/fake-model", "dataset": str(self.tmp / "ds.parquet"), "store_folder": store_folder,
               "storage": storage or {"backend": "hf", "hf_repo_id": REPO, "hf_private": False, "local_dir": None},
               "layers": [0, 1]}
        path = self.tmp / "collect.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def run_main(self, *args, api="fake"):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = mps.main(["--config", str(self.config()), "--dest", str(self.dst), *args],
                            api=self.api if api == "fake" else api)
        return code, buf.getvalue()

    def _part_downloads(self):
        return [f for f, _ in self.api.downloads if f.endswith(".safetensors")]

    def test_dry_run_plans_and_checks_disk(self):
        code, out = self.run_main("--layers", "0", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn(f"Source: hf:{REPO}/spec (store_folder)", out)
        self.assertIn("seq    layer   0:    2 file(s)", out)
        self.assertIn("points layer   1:    2 file(s)", out)
        self.assertNotIn("seq    layer   1", out)
        self.assertIn("total: 6 file(s)", out)
        self.assertIn("free disk at", out)
        self.assertIn("Dry run — nothing downloaded.", out)
        self.assertFalse(self.dst.exists())
        self.assertEqual(self._part_downloads(), [])
        with mock.patch.object(st, "_free_bytes", return_value=1):
            code, out = self.run_main("--layers", "0", "--dry-run")
            self.assertEqual(code, 1)
            self.assertIn("[refused]", out)
            code, out = self.run_main("--layers", "0")
            self.assertEqual(code, 1)
        self.assertFalse(self.dst.exists())
        self.assertEqual(self._part_downloads(), [])
        with self.assertRaisesRegex(ValueError, "nothing to mirror"):
            self.run_main("--no-points", "--dry-run")
        with self.assertRaisesRegex(ValueError, "--layers must be"):
            self.run_main("--layers", "a,b", "--dry-run")
        with self.assertRaisesRegex(st.LayerSelectionError, r"\[7\].*not stored"):
            self.run_main("--layers", "7", "--dry-run")
        self.assertEqual(mps.parse_layers("1, 0"), [0, 1])
        self.assertIsNone(mps.parse_layers(" "))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            mps.parse_layers("1,1")

    def test_mirror_end_to_end_and_idempotent(self):
        code, out = self.run_main("--layers", "0", "--workers", "2")
        self.assertEqual(code, 0)
        self.assertIn("Done: 6 downloaded", out)
        self.assertIn("mirrored seq layers [0], points layers [0, 1]", out)
        names = sorted(p.name for p in (self.dst / "spec").iterdir())
        expected = sorted(["manifest.csv", "manifest.meta.json"] + [p.seq_file(0) for p in self.parts]
                          + [p.points_file(l) for p in self.parts for l in (0, 1)])
        self.assertEqual(names, expected)
        meta = json.loads((self.dst / "spec" / "manifest.meta.json").read_text())
        self.assertEqual((meta["mirrored_layers"], meta["mirrored_points_layers"], meta["mirror_source"]["repo_id"]),
                         ([0], [0, 1], REPO))
        self.assertTrue(all(d is not None for f, d in self.api.downloads if f.endswith(".safetensors")))
        n = len(self.api.downloads)
        code, out = self.run_main("--layers", "0")
        self.assertEqual(code, 0)
        self.assertIn("Done: 0 downloaded", out)
        self.assertIn("6 skipped", out)
        self.assertEqual(self._part_downloads()[6:], [])
        self.assertEqual(len([f for f, _ in self.api.downloads[n:] if f.endswith(".safetensors")]), 0)
        victim = self.parts[0].seq_file(0)
        (self.dst / "spec" / victim).write_bytes(b"short")
        code, out = self.run_main("--layers", "0")
        self.assertIn("Done: 1 downloaded", out)
        self.assertEqual((self.dst / "spec" / victim).read_bytes(), (self.local.root / "spec" / victim).read_bytes())
        # The copy reads back through the trainer's backend block.
        manifest, meta = st.read_store_manifest(st.LocalBackend(self.dst), "spec", layers=[0], kinds=("seq",))
        self.assertEqual(len(manifest), 4)
        # Seq files only, into another destination, from a local source (no API).
        dst2 = self.tmp / "mirror2"
        cfg = self.config(storage={"backend": "local", "local_dir": str(self.local.root)})
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = mps.main(["--config", str(cfg), "--dest", str(dst2), "--layers", "1", "--no-points"])
        self.assertEqual(code, 0)
        self.assertIn("points: excluded", buf.getvalue())
        self.assertEqual(sorted(p.name for p in (dst2 / "spec").iterdir()),
                         sorted(["manifest.csv", "manifest.meta.json"] + [p.seq_file(1) for p in self.parts]))
        meta2 = json.loads((dst2 / "spec" / "manifest.meta.json").read_text())
        self.assertEqual((meta2["mirrored_layers"], meta2["mirrored_points_layers"], meta2["mirror_source"]["backend"]),
                         ([1], [], "local"))

    def test_seq_layers_source(self):
        local, parts = write_store(self.tmp / "src_seq", seq_layers=[1])
        self.hf = hf_backend_over(local)
        self.api = self.hf.api
        with self.assertRaisesRegex(st.LayerSelectionError, r"layers \[0\] have no seq files.*seq_layers: \[1\]"):
            self.run_main("--layers", "0", "--dry-run")
        code, out = self.run_main("--layers", "1", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("seq files for [1] (seq_layers)", out)
        self.assertFalse(self.dst.exists())
        code, out = self.run_main("--layers", "1")
        self.assertEqual(code, 0)
        self.assertIn("mirrored seq layers [1], points layers [0, 1]", out)
        self.assertEqual(sorted(p.name for p in (self.dst / "spec").iterdir()),
                         sorted(["manifest.csv", "manifest.meta.json"] + [p.seq_file(1) for p in parts]
                                + [p.points_file(l) for p in parts for l in (0, 1)]))
        meta = json.loads((self.dst / "spec" / "manifest.meta.json").read_text())
        self.assertEqual((meta["layers"], meta["seq_layers"], meta["mirrored_layers"], meta["mirrored_points_layers"]),
                         ([0, 1], [1], [1], [0, 1]))


if __name__ == "__main__":
    unittest.main()
