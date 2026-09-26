"""Tests for src/lib/paths.py."""

import os
import unittest
from pathlib import Path
from unittest import mock

from src.lib import paths


class DataRootTest(unittest.TestCase):
    def test_defaults_to_repo_data_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATA_ROOT", None)
            self.assertEqual(paths.REPO_ROOT / "data", paths.data_root())

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "/mnt/share/cot"}):
            self.assertEqual(Path("/mnt/share/cot"), paths.data_root())

    def test_relative_data_root_rejected(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "relative/dir"}):
            with self.assertRaises(ValueError):
                paths.data_root()


class ResolveDataPathTest(unittest.TestCase):
    def test_data_root_prefix_uses_env(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "/mnt/share/cot"}):
            self.assertEqual(
                Path("/mnt/share/cot/baselines/m.csv"),
                paths.resolve_data_path("${DATA_ROOT}/baselines/m.csv"),
            )

    def test_data_root_prefix_uses_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATA_ROOT", None)
            self.assertEqual(
                paths.REPO_ROOT / "data" / "baselines" / "m.csv",
                paths.resolve_data_path("${DATA_ROOT}/baselines/m.csv"),
            )

    def test_braced_form(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "/mnt/share/cot"}):
            self.assertEqual(
                Path("/mnt/share/cot/x"),
                paths.resolve_data_path("${DATA_ROOT}/x"),
            )

    def test_additional_env_var_expands_when_config_uses_it(self):
        with mock.patch.dict(
            os.environ, {"DATA_ROOT": "/mnt/share/cot", "SUBDIR": "alice"}
        ):
            self.assertEqual(
                Path("/mnt/share/cot/alice/activations"),
                paths.resolve_data_path("${DATA_ROOT}/$SUBDIR/activations"),
            )

    def test_absolute_path_passthrough(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "/mnt/share/cot"}):
            self.assertEqual(
                Path("/tmp/custom.csv"),
                paths.resolve_data_path("/tmp/custom.csv"),
            )

    def test_relative_without_data_root_rejected(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "/mnt/share/cot"}):
            with self.assertRaises(ValueError):
                paths.resolve_data_path("data/baselines/m.csv")

    def test_unresolved_variable_rejected(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "/mnt/share/cot"}, clear=False):
            os.environ.pop("NOPE", None)
            with self.assertRaises(ValueError):
                paths.resolve_data_path("${DATA_ROOT}/$NOPE/x")


if __name__ == "__main__":
    unittest.main()
