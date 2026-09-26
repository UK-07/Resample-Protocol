"""Tests for src/lib/env.py — the once-only repo-root .env loader."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.lib import env


class TestLoadEnv(unittest.TestCase):
    """load_env: missing file no-op, set variables win, once-only unless forced."""

    def setUp(self):
        for name in ("ENV_TEST_X", "ENV_TEST_Y", "ENV_TEST_Z"):
            os.environ.pop(name, None)
        self.addCleanup(lambda: [os.environ.pop(n, None) for n in ("ENV_TEST_X", "ENV_TEST_Y", "ENV_TEST_Z")])

    def _write_env(self, text: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / ".env"
        path.write_text(text)
        return path

    def test_missing_file_is_noop(self):
        missing = Path(tempfile.gettempdir()) / "env_test_does_not_exist" / ".env"
        with mock.patch.object(env, "ENV_FILE", missing), mock.patch.object(env, "_loaded", False):
            self.assertIsNone(env.load_env())
            self.assertTrue(env._loaded)
        self.assertNotIn("ENV_TEST_X", os.environ)

    def test_present_file_fills_unset_variables_only(self):
        path = self._write_env("ENV_TEST_X=from_file\nENV_TEST_Y=from_file\n")
        os.environ["ENV_TEST_Y"] = "from_shell"
        with mock.patch.object(env, "ENV_FILE", path), mock.patch.object(env, "_loaded", False):
            self.assertEqual(env.load_env(), path)
        self.assertEqual(os.environ["ENV_TEST_X"], "from_file")
        self.assertEqual(os.environ["ENV_TEST_Y"], "from_shell")

    def test_second_load_is_noop_and_force_reloads(self):
        path = self._write_env("ENV_TEST_X=first\n")
        with mock.patch.object(env, "ENV_FILE", path), mock.patch.object(env, "_loaded", False):
            self.assertEqual(env.load_env(), path)
            path.write_text("ENV_TEST_Z=second\n")
            self.assertIsNone(env.load_env())
            self.assertNotIn("ENV_TEST_Z", os.environ)
            self.assertEqual(env.load_env(force=True), path)
            self.assertEqual(os.environ["ENV_TEST_Z"], "second")
            self.assertEqual(os.environ["ENV_TEST_X"], "first")


if __name__ == "__main__":
    unittest.main()
