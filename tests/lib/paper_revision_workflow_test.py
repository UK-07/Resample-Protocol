"""Input-integrity and dry-run safeguards for the paper revision workflow."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from src.lib.paper_revision import workflow


class PaperRevisionWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.payload = b"question,value\nq1,1\n"
        self.relative = "results/example.csv"
        target = self.bundle / self.relative
        target.parent.mkdir(parents=True)
        target.write_bytes(self.payload)
        self.manifest = {
            "n_files": 1,
            "files": [{"path": self.relative, "bytes": len(self.payload),
                       "sha256": hashlib.sha256(self.payload).hexdigest()}],
        }
        self.write_manifest()

    def write_manifest(self):
        self.manifest_bytes = (json.dumps(self.manifest) + "\n").encode()
        (self.bundle / "MANIFEST.json").write_bytes(self.manifest_bytes)

    def edition_pin(self):
        # Small valid fixtures exercise the workflow without copying the released
        # data bundle. Production keeps this digest fixed to the paper edition.
        return patch.object(workflow, "MANIFEST_SHA256",
                            hashlib.sha256(self.manifest_bytes).hexdigest(), create=True)

    def make_archive(self, *, extra_members=(), payload=None):
        path = self.root / "fixture.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, content in [
                ("revision_analysis/MANIFEST.json", self.manifest_bytes),
                ("revision_analysis/" + self.relative,
                 self.payload if payload is None else payload),
            ]:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            for member, content in extra_members:
                archive.addfile(member, None if content is None else io.BytesIO(content))
        return path

    def archive_pin(self, path):
        # Permit the synthetic tar to reach structural/member verification. This
        # does not mock checksum computation or any of the archive safety checks.
        return patch.object(workflow, "ARCHIVE_SHA256", workflow.sha256(path))

    def test_directory_dry_run_checks_inputs_but_creates_no_output(self):
        output = self.root / "new" / "output"
        before = {p.relative_to(self.bundle): p.read_bytes()
                  for p in self.bundle.rglob("*") if p.is_file()}
        with self.edition_pin(), patch.object(workflow.subprocess, "run") as launch:
            result = workflow.reproduce(self.bundle, output, dry_run=True)
        self.assertEqual(result["verified_files"], 1)
        self.assertEqual(result["pdf_count"], 33)
        self.assertEqual(result["latex_table_count"], 3)
        self.assertEqual(result["phases"][0], "recompute")
        launch.assert_not_called()
        self.assertFalse(output.parent.exists())
        after = {p.relative_to(self.bundle): p.read_bytes()
                 for p in self.bundle.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_archive_dry_run_does_not_extract(self):
        archive = self.make_archive()
        output = self.root / "output"
        with self.archive_pin(archive), self.edition_pin(), patch.object(workflow.subprocess, "run") as launch:
            result = workflow.reproduce(archive, output, dry_run=True)
        self.assertEqual(result["archive_sha256"], workflow.sha256(archive))
        self.assertFalse(output.exists())
        launch.assert_not_called()

    def test_same_size_input_tamper_is_rejected(self):
        (self.bundle / self.relative).write_bytes(self.payload.replace(b"q1,1", b"q1,9"))
        with self.assertRaisesRegex(ValueError, "does not match manifest"):
            workflow.verify_directory(self.bundle)

    def test_missing_input_is_rejected(self):
        (self.bundle / self.relative).unlink()
        with self.assertRaisesRegex(ValueError, "does not match manifest"):
            workflow.verify_directory(self.bundle)

    def test_manifest_paths_cannot_escape_bundle(self):
        for name in ("../outside.csv", "/tmp/outside.csv", "results/../../outside.csv",
                     "results\\outside.csv", "C:/outside.csv"):
            with self.subTest(name=name):
                self.manifest["files"][0]["path"] = name
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
                    workflow.verify_directory(self.bundle)

    def test_empty_normalized_paths_are_rejected(self):
        # PurePosixPath normalizes both to an empty parts tuple; accepting them
        # previously caused archive verification to fail with an IndexError.
        for name in (".", "./"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
                workflow.relative_file(name)
        member = tarfile.TarInfo(".")
        member.size = 1
        archive = self.make_archive(extra_members=[(member, b"x")])
        with self.archive_pin(archive), self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
            workflow.verify_archive(archive)

    def test_manifest_duplicate_normalized_paths_are_rejected(self):
        duplicate = dict(self.manifest["files"][0], path="results/./example.csv")
        self.manifest["files"].append(duplicate)
        self.manifest["n_files"] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate paths"):
            workflow.verify_directory(self.bundle)

    def test_manifest_count_mismatch_is_rejected(self):
        self.manifest["n_files"] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "file count"):
            workflow.verify_directory(self.bundle)

    def test_input_symlink_is_rejected_even_if_target_hash_matches(self):
        outside = self.root / "outside.csv"
        outside.write_bytes(self.payload)
        target = self.bundle / self.relative
        target.unlink()
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            workflow.verify_directory(self.bundle)

    def test_archive_hash_gate_is_enforced(self):
        archive = self.make_archive()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            workflow.verify_archive(archive)

    def test_archive_rejects_traversal_and_unexpected_root(self):
        for name, expected in (("revision_analysis/../../escape.txt", "Unsafe bundle path"),
                               ("/tmp/escape.txt", "Unsafe bundle path"),
                               ("different_root/extra.txt", "Unexpected archive root")):
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.size = 1
                archive = self.make_archive(extra_members=[(member, b"x")])
                with self.archive_pin(archive), self.assertRaisesRegex(ValueError, expected):
                    workflow.verify_archive(archive)

    def test_archive_rejects_links_and_duplicate_members(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with self.subTest(kind=kind):
                member = tarfile.TarInfo("revision_analysis/link")
                member.type = kind
                member.linkname = "../../outside"
                archive = self.make_archive(extra_members=[(member, None)])
                with self.archive_pin(archive), self.assertRaisesRegex(ValueError, "Non-regular"):
                    workflow.verify_archive(archive)
        duplicate = tarfile.TarInfo("revision_analysis/" + self.relative)
        duplicate.size = len(self.payload)
        archive = self.make_archive(extra_members=[(duplicate, self.payload)])
        with self.archive_pin(archive), self.assertRaisesRegex(ValueError, "Duplicate archive member"):
            workflow.verify_archive(archive)

    def test_archive_rejects_unlisted_file(self):
        extra = tarfile.TarInfo("revision_analysis/unlisted.py")
        extra.size = 1
        archive = self.make_archive(extra_members=[(extra, b"x")])
        with self.archive_pin(archive), self.assertRaisesRegex(ValueError, "do not match manifest"):
            workflow.verify_archive(archive)

    def test_archive_member_hash_is_checked_after_archive_hash(self):
        archive = self.make_archive(payload=self.payload.replace(b"q1,1", b"q1,9"))
        with self.archive_pin(archive), self.assertRaisesRegex(ValueError, "member hash mismatch"):
            workflow.verify_archive(archive)

    def test_output_inside_bundle_is_rejected_without_writing(self):
        for output in (self.bundle, self.bundle / "new" / "output"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside the read-only bundle"):
                workflow.reproduce(self.bundle, output, dry_run=True)
        self.assertFalse((self.bundle / "new").exists())

    def test_nonempty_output_is_preserved_and_rejected(self):
        output = self.root / "output"
        output.mkdir()
        sentinel = output / "existing-result.pdf"
        sentinel.write_bytes(b"preserve me")
        with self.assertRaisesRegex(ValueError, "not empty"):
            workflow.reproduce(self.bundle, output, dry_run=True)
        self.assertEqual(sentinel.read_bytes(), b"preserve me")

    def test_valid_archive_extracts_only_to_scratch_and_preserves_source(self):
        archive = self.make_archive()
        before = archive.read_bytes()
        output = self.root / "output"
        with self.archive_pin(archive):
            extracted, report = workflow.prepare_bundle(archive, output)
        self.assertEqual(extracted, output / "inputs" / "revision_analysis")
        self.assertEqual(report["verified_files"], 1)
        self.assertEqual((extracted / self.relative).read_bytes(), self.payload)
        self.assertEqual(archive.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
