import contextlib
import os
import threading
import io
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import RepositoryNotFoundError
from safetensors import safe_open
from safetensors.torch import load as safetensors_load

from src.lib import activation_store as st
from src.lib.parsing import DEFAULT_DELIMITERS

CLOSE = DEFAULT_DELIMITERS.close
OPEN = DEFAULT_DELIMITERS.open
REPO = "example-org/probe-activations-fake"
HIDDEN = 8


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Character-level tokenizer with the delimiters as single ids, so tokenizing
    the pieces separately equals tokenizing the joined text (like the real one)."""

    SPECIAL = {OPEN: 1001, CLOSE: 1002}

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.template_calls.append({"messages": messages, "kwargs": kwargs})
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + f"<assistant>{OPEN}\n"

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for s, sid in self.SPECIAL.items():
                if text.startswith(s, i):
                    ids.append(sid)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}


class FakeLayer(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, x):
        out = x + 1.0
        return (out, "extra") if self.tuple_output else out


class FakeModel(torch.nn.Module):
    """Layer i (0-based) outputs position + i + 1 in every hidden dim."""

    def __init__(self, n_layers=4, hidden=HIDDEN, accept_logits_to_keep=True, tuple_output=False):
        super().__init__()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList(FakeLayer(tuple_output) for _ in range(n_layers))
        self.model = inner
        self.config = SimpleNamespace(num_hidden_layers=n_layers, hidden_size=hidden)
        self.hidden = hidden
        self.accept_logits_to_keep = accept_logits_to_keep
        self.device = torch.device("cpu")
        self.calls = []

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        if "logits_to_keep" in kwargs and not self.accept_logits_to_keep:
            raise TypeError("forward() got an unexpected keyword argument 'logits_to_keep'")
        self.calls.append((input_ids.clone(), {"use_cache": use_cache, **kwargs}))
        seq = input_ids.shape[1]
        h = torch.arange(seq, dtype=torch.float32).view(1, seq, 1).expand(1, seq, self.hidden).clone()
        for layer in self.model.layers:
            h = layer(h)
            h = h[0] if isinstance(h, tuple) else h
        return h


class FakeApi:
    """In-memory HfApi covering what HfBackend calls; ``short_once`` / ``short_always``
    name files whose download is truncated (the first time / every time)."""

    def __init__(self, *, files=None, user="alice", orgs=("example-org",),
                 repo_exists=True, fail_uploads=(), short_once=(), short_always=()):
        self.files: dict[str, bytes] = dict(files or {})
        self.user, self.orgs = user, orgs
        self.repo_exists = repo_exists
        self.fail_uploads = set(fail_uploads)
        self.short_once, self.short_always = set(short_once), set(short_always)
        self.created, self.commits, self.whoami_calls = [], [], 0
        self.downloads: list[tuple[str, str | None]] = []
        self._tmp = Path(tempfile.mkdtemp())

    def whoami(self):
        self.whoami_calls += 1
        return {"name": self.user, "orgs": [{"name": o} for o in self.orgs]}

    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
        self.created.append((repo_id, repo_type, private))
        self.repo_exists = True

    def list_repo_files(self, repo_id, repo_type=None):
        if not self.repo_exists:
            response = requests.Response()
            response.status_code = 404
            response.url = f"https://huggingface.co/api/datasets/{repo_id}"
            response._content = b""
            raise RepositoryNotFoundError("no such repo", response=response)
        return list(self.files)

    def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, expand=False, revision=None, repo_type=None):
        prefix = path_in_repo.rstrip("/") + "/" if path_in_repo else ""
        if not self.repo_exists:
            raise RepositoryNotFoundError("no such repo")
        return [SimpleNamespace(path=p, size=len(b)) for p, b in sorted(self.files.items()) if p.startswith(prefix)]

    def hf_hub_download(self, repo_id, filename, repo_type=None, force_download=False, local_dir=None, revision=None):
        """With ``local_dir`` the file lands at ``<local_dir>/<filename>``, else in the fake's cache dir."""
        self.downloads.append((filename, local_dir))
        path = (Path(local_dir) if local_dir else self._tmp) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.files[filename]
        if filename in self.short_always or filename in self.short_once:
            self.short_once.discard(filename)
            data = data[: len(data) // 2]
        path.write_bytes(data)
        return str(path)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
        if path_in_repo in self.fail_uploads:
            raise RuntimeError(f"simulated upload failure for {path_in_repo}")
        if isinstance(path_or_fileobj, (str, Path)):
            path_or_fileobj = Path(path_or_fileobj).read_bytes()
        self.files[path_in_repo] = bytes(path_or_fileobj)
        self.commits.append(("upload", path_in_repo, commit_message))

    def create_commit(self, *, repo_id, repo_type=None, operations, commit_message=None):
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                self.files[op.path_in_repo] = bytes(op.path_or_fileobj)
            elif isinstance(op, CommitOperationDelete):
                self.files.pop(op.path_in_repo, None)
        self.commits.append(("commit", [type(op).__name__ + ":" + op.path_in_repo for op in operations], commit_message))


class FailingLocalBackend(st.LocalBackend):
    """A local backend whose upload of the named files fails."""

    def __init__(self, root, fail=()):
        super().__init__(root)
        self.fail = set(fail)

    def upload_file(self, local_or_bytes, path, *, commit_message=None):
        if Path(path).name in self.fail:
            raise RuntimeError(f"simulated failure for {path}")
        super().upload_file(local_or_bytes, path, commit_message=commit_message)


@dataclass
class FakeItem:
    rollout_id: str
    n_tokens: int


def items(n_tokens_list, prefix="m:run:h"):
    return [FakeItem(rollout_id=f"{prefix}:{i}", n_tokens=n) for i, n in enumerate(n_tokens_list)]


def rollout(cot: str, final: str) -> str:
    return f"{cot}{CLOSE}\n\n<answer>{final}</answer>"


def write_part(writer: st.StoreWriter, part: st.PartPlan, layers, *, group_rows=True, meta_update=None):
    """Write every seq/points file of ``part`` and finalize it: seq of item i, layer L
    = (i + 1) * 10 + L everywhere; points = -(i + 1) * 10 - L."""
    rows = []
    for layer in layers:
        for kind, fname in (("seq", part.seq_file(layer)), ("points", part.points_file(layer))):
            pw = writer.new_part_writer(fname, part.entries(kind, hidden=HIDDEN), {"layer": str(layer), "kind": kind})
            for i, it in enumerate(part.items):
                if kind == "seq":
                    t = torch.full((it.n_tokens, HIDDEN), float((i + 1) * 10 + layer), dtype=torch.bfloat16)
                else:
                    t = torch.full((len(st.POINTS), HIDDEN), float(-(i + 1) * 10 - layer), dtype=torch.bfloat16)
                pw.add(it.rollout_id, t)
            writer.submit_file(f"part{part.index:02d}", fname, pw.finish())
    for it in part.items:
        rows.append({**{c: "" for c in st.STORE_MANIFEST_COLUMNS}, "rollout_id": it.rollout_id,
                     "n_cot_tokens": it.n_tokens, "seq_file": f"seq_layer_{{L}}_part{part.index:02d}",
                     "points_file": f"points_layer_{{L}}_part{part.index:02d}"})
    writer.finalize_group(f"part{part.index:02d}", rows=rows if group_rows else (), keep=part.files(layers),
                          stale_filter=None, meta_update=meta_update)
    return rows


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


class TestStoreFilenames(unittest.TestCase):
    """seq_file / points_file / is_store_file / POINTS / stable_seed."""

    def test_store_filenames(self):
        self.assertEqual(st.seq_file(3, 1), "seq_layer_3_part01.safetensors")
        self.assertEqual(st.points_file(47, 12), "points_layer_47_part12.safetensors")
        self.assertTrue(st.is_store_file("spec/points_layer_5_part03.safetensors"))
        for bad in ("manifest.csv", "mmlu_pro_grader_hacking.safetensors", "seq_layer_3.safetensors",
                    "seq_layer_x_part01.safetensors", "seqs_layer_3_part01.safetensors", "seq_layer_3_part01.pt"):
            self.assertFalse(st.is_store_file(bad), bad)

    def test_points_constant(self):
        self.assertEqual(st.POINTS, ("pre_cot", "last_cot", "pre_answer", "mean_cot"))

    def test_stable_seed_is_deterministic_and_31_bit(self):
        self.assertEqual(st.stable_seed(7, "d", "h"), st.stable_seed(7, "d", "h"))
        self.assertNotEqual(st.stable_seed(7, "d", "h"), st.stable_seed(8, "d", "h"))
        self.assertLess(st.stable_seed("x"), 2 ** 31)


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


class TestLayerStack(unittest.TestCase):
    """resolve_layer_stack / assert_layer_depth / resolve_hidden_size."""

    def test_resolves_registered_path(self):
        model = FakeModel(n_layers=4)
        layers, path = st.resolve_layer_stack(model)
        self.assertEqual((len(layers), path), (4, "model.layers"))
        layers, path = st.resolve_layer_stack(model, path="model.layers")
        self.assertEqual(path, "model.layers")

    def test_explicit_path_must_resolve(self):
        with self.assertRaisesRegex(RuntimeError, "does not resolve"):
            st.resolve_layer_stack(FakeModel(), path="model.vision")
        with self.assertRaisesRegex(RuntimeError, "Could not locate"):
            st.resolve_layer_stack(torch.nn.Linear(2, 2))

    def test_depth_mismatch_raises(self):
        model = FakeModel(n_layers=4)
        model.config.num_hidden_layers = 6
        with mock.patch.object(st, "get_model_config", lambda name: {"n_layers": 4, "hidden_size": 8}):
            with self.assertRaisesRegex(RuntimeError, "declares num_hidden_layers=6"):
                st.assert_layer_depth(model, "fake", model.model.layers, "model.layers")
            model.config.num_hidden_layers = 4
            st.assert_layer_depth(model, "fake", model.model.layers, "model.layers")
            self.assertEqual(st.resolve_hidden_size(model, "fake"), 8)


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------


class TestSequenceAssembly(unittest.TestCase):
    """build_messages / render_prompt_text."""

    def test_single_turn_messages(self):
        self.assertEqual(st.build_messages("Q?", "sys"),
                         [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}])

    def test_multi_turn_messages_from_json(self):
        turns = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A"},
                 {"role": "user", "content": "Sure?"}]
        self.assertEqual(st.build_messages(json.dumps(turns), "sys"),
                         [{"role": "system", "content": "sys"}] + turns)
        # A user prompt that merely starts with "[" is not a message list.
        self.assertEqual(st.build_messages("[note] Q?", "sys")[1]["content"], "[note] Q?")

    def test_enable_thinking_forwarded_or_omitted(self):
        tok = FakeTokenizer()
        st.render_prompt_text(tok, [{"role": "user", "content": "Q"}], True)
        self.assertTrue(tok.template_calls[-1]["kwargs"]["enable_thinking"])
        st.render_prompt_text(tok, [{"role": "user", "content": "Q"}], None)
        self.assertNotIn("enable_thinking", tok.template_calls[-1]["kwargs"])


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class TestCapture(unittest.TestCase):
    """ActivationCapturer and the Mamba chunk-size helpers."""

    def test_captures_cot_span_per_layer(self):
        model = FakeModel(n_layers=4, hidden=8)
        cap = st.ActivationCapturer(model, model.model.layers, [0, 2], expected_hidden=8)
        acts, _ = cap.capture_points(np.arange(10), cot_span=(3, 7), point_idx=(2, 6, 9))
        self.assertEqual(set(acts), {0, 2})
        for idx, t in acts.items():
            self.assertEqual(tuple(t.shape), (4, 8))
            self.assertEqual(t.dtype, torch.bfloat16)
            self.assertEqual(t[:, 0].float().tolist(), [3 + idx + 1, 4 + idx + 1, 5 + idx + 1, 6 + idx + 1])
        self.assertTrue(model.calls[0][1].get("logits_to_keep") == 1)
        self.assertFalse(model.calls[0][1].get("use_cache", True))

    def test_tuple_layer_outputs_supported(self):
        model = FakeModel(n_layers=2, hidden=4, tuple_output=True)
        cap = st.ActivationCapturer(model, model.model.layers, [1], expected_hidden=4)
        acts, _ = cap.capture_points(np.arange(5), cot_span=(0, 5), point_idx=(0, 4, 4))
        self.assertEqual(tuple(acts[1].shape), (5, 4))

    def test_falls_back_without_logits_to_keep(self):
        model = FakeModel(accept_logits_to_keep=False)
        cap = st.ActivationCapturer(model, model.model.layers, [0], expected_hidden=8)
        cap.capture_points(np.arange(4), cot_span=(0, 2), point_idx=(0, 1, 3))
        cap.capture_points(np.arange(4), cot_span=(0, 2), point_idx=(0, 1, 3))
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all("logits_to_keep" not in kw for _, kw in model.calls))

    def test_capture_points_returns_span_and_points(self):
        model = FakeModel(n_layers=3, hidden=HIDDEN)
        cap = st.ActivationCapturer(model, model.model.layers, [0, 2], expected_hidden=HIDDEN)
        ids = np.arange(12)
        seq, pts = cap.capture_points(ids, cot_span=(4, 9), point_idx=(3, 8, 11))
        for layer in (0, 2):
            self.assertEqual(tuple(seq[layer].shape), (5, HIDDEN))
            self.assertEqual(tuple(pts[layer].shape), (len(st.POINTS), HIDDEN))
            self.assertEqual((seq[layer].dtype, pts[layer].dtype), (torch.bfloat16, torch.bfloat16))
            # layer L outputs position + L + 1
            torch.testing.assert_close(seq[layer].float()[:, 0], torch.arange(4, 9, dtype=torch.float32) + layer + 1)
            torch.testing.assert_close(pts[layer].float()[:, 0],
                                       torch.tensor([3.0, 8.0, 11.0, 6.0]) + layer + 1)
            self.assertTrue(torch.equal(pts[layer][st.POINTS.index("last_cot")], seq[layer][-1]))
        with self.assertRaisesRegex(ValueError, "last_cot"):
            cap.capture_points(ids, cot_span=(4, 9), point_idx=(3, 7, 11))
        with self.assertRaisesRegex(ValueError, "outside"):
            cap.capture_points(ids, cot_span=(4, 9), point_idx=(3, 8, 12))
        with self.assertRaisesRegex(ValueError, "cot_span"):
            cap.capture_points(ids, cot_span=(9, 9), point_idx=(3, 8, 11))

    def test_width_mismatch_raises(self):
        model = FakeModel(hidden=8)
        cap = st.ActivationCapturer(model, model.model.layers, [0], expected_hidden=16)
        with self.assertRaisesRegex(RuntimeError, "hidden width 8, expected 16"):
            cap.capture_points(np.arange(4), cot_span=(0, 2), point_idx=(0, 1, 3))

    def test_mamba_chunk_size_helpers(self):
        model = FakeModel(n_layers=3)
        self.assertEqual(st.mamba_chunk_sizes(model), {})
        self.assertEqual(st.set_mamba_chunk_size(model, 32), 0)
        model.model.layers[0].mixer = torch.nn.Module()
        model.model.layers[0].mixer.chunk_size = 256
        model.model.layers[2].mixer = torch.nn.Module()
        model.model.layers[2].mixer.chunk_size = 256
        model.model.layers[1].chunk_size = None       # not an int: ignored
        model.model.layers[1].flag = True
        self.assertEqual(st.mamba_chunk_sizes(model), {"model.layers.0.mixer": 256, "model.layers.2.mixer": 256})
        self.assertEqual(st.set_mamba_chunk_size(model, 64), 2)
        self.assertEqual(st.mamba_chunk_sizes(model), {"model.layers.0.mixer": 64, "model.layers.2.mixer": 64})
        for bad in (0, -8, 2.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                st.set_mamba_chunk_size(model, bad)


# ---------------------------------------------------------------------------
# PartWriter
# ---------------------------------------------------------------------------


class TestPartWriter(unittest.TestCase):
    """build_safetensors_header / PartWriter."""

    def test_part_writer_memory_round_trip(self):
        entries = [("positive_1_layer_3", (4, 8)), ("negative_2_layer_3", (2, 8)), ("positive_1_layer_7", (4, 8))]
        w = st.PartWriter(entries, {"dataset": "d", "layers": "[3, 7]"}, staging="memory")
        self.assertEqual(len(w.header) % 8, 0)
        a = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
        b = torch.full((2, 8), -1.5, dtype=torch.bfloat16)
        c = (torch.arange(32, dtype=torch.float32).reshape(4, 8) * 0.25).to(torch.bfloat16)
        w.add("positive_1_layer_3", a)
        w.add("negative_2_layer_3", b)
        w.add("positive_1_layer_7", c)
        data = w.finish()
        self.assertIsInstance(data, bytes)
        self.assertEqual(len(data), w.total)
        loaded = safetensors_load(data)
        self.assertEqual(set(loaded), {k for k, _ in entries})
        self.assertTrue(torch.equal(loaded["positive_1_layer_3"], a))
        self.assertTrue(torch.equal(loaded["negative_2_layer_3"], b))
        self.assertTrue(torch.equal(loaded["positive_1_layer_7"], c))
        self.assertEqual(loaded["positive_1_layer_3"].dtype, torch.bfloat16)

    def test_part_writer_disk_round_trip_and_metadata(self):
        path = Path(tempfile.mkdtemp()) / "p.safetensors"
        w = st.PartWriter([("k_layer_0", (3, 4))], {"dataset": "d"}, staging="disk", staging_path=path)
        t = torch.randn(3, 4).to(torch.bfloat16)
        w.add("k_layer_0", t)
        out = w.finish()
        self.assertEqual(out, str(path))
        with safe_open(str(path), framework="pt") as f:
            self.assertEqual(f.metadata(), {"dataset": "d"})
            self.assertTrue(torch.equal(f.get_tensor("k_layer_0"), t))

    def test_part_writer_enforces_plan(self):
        w = st.PartWriter([("a", (2, 4)), ("b", (2, 4))], {}, staging="memory")
        with self.assertRaisesRegex(RuntimeError, "out of plan order"):
            w.add("b", torch.zeros(2, 4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "captured shape"):
            w.add("a", torch.zeros(3, 4, dtype=torch.bfloat16))
        w.add("a", torch.zeros(2, 4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "part incomplete"):
            w.finish()
        # a float32 input is stored as bf16
        w.add("b", torch.ones(2, 4))
        loaded = safetensors_load(w.finish())
        self.assertEqual(loaded["b"].dtype, torch.bfloat16)

    def test_part_writer_skip_zero_fills_slot(self):
        entries = [("a", (2, 4)), ("b", (3, 4)), ("c", (1, 4))]
        for staging in ("memory", "disk"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "p.safetensors"
                w = st.PartWriter(entries, {"k": "v"}, staging=staging, staging_path=path if staging == "disk" else None)
                with self.assertRaisesRegex(RuntimeError, "out of plan order"):
                    w.skip("b")
                w.add("a", torch.ones(2, 4, dtype=torch.bfloat16))
                w.skip("b")
                w.add("c", torch.full((1, 4), 3.0, dtype=torch.bfloat16))
                out = w.finish()
                data = Path(out).read_bytes() if staging == "disk" else out
                self.assertEqual(len(data), w.total)
                loaded = safetensors_load(data)
                self.assertEqual(sorted(loaded), ["a", "b", "c"])
                self.assertTrue(torch.equal(loaded["a"], torch.ones(2, 4, dtype=torch.bfloat16)))
                self.assertTrue(torch.equal(loaded["b"], torch.zeros(3, 4, dtype=torch.bfloat16)))
                self.assertTrue(torch.equal(loaded["c"], torch.full((1, 4), 3.0, dtype=torch.bfloat16)))

    def test_header_rejects_duplicate_or_reserved_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicate or reserved"):
            st.build_safetensors_header([("a", (1, 2)), ("a", (1, 2))], {})
        with self.assertRaisesRegex(ValueError, "duplicate or reserved"):
            st.build_safetensors_header([("__metadata__", (1, 2))], {})


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class TestPlanParts(unittest.TestCase):
    """plan_parts / PartPlan."""

    def test_plan_parts_pairs_seq_and_points_files(self):
        # 3 items of 10 tokens = 160 B each per layer; cap 400 B → 2 + 1
        parts = st.plan_parts(items([10, 10, 10]), hidden=HIDDEN, n_layers=2, max_bytes=400)
        self.assertEqual([p.index for p in parts], [0, 1])
        self.assertEqual([[it.rollout_id for it in p.items] for p in parts],
                         [["m:run:h:0", "m:run:h:1"], ["m:run:h:2"]])
        self.assertEqual(parts[0].files([3, 7]), [
            "seq_layer_3_part00.safetensors", "points_layer_3_part00.safetensors",
            "seq_layer_7_part00.safetensors", "points_layer_7_part00.safetensors",
        ])
        self.assertEqual(parts[1].seq_file(3), "seq_layer_3_part01.safetensors")
        self.assertEqual(parts[1].points_file(3), "points_layer_3_part01.safetensors")
        seq_entries = parts[0].entries("seq", hidden=HIDDEN)
        pts_entries = parts[0].entries("points", hidden=HIDDEN)
        self.assertEqual([k for k, _ in seq_entries], [k for k, _ in pts_entries])
        self.assertEqual([shape for _, shape in seq_entries], [(10, HIDDEN), (10, HIDDEN)])
        self.assertEqual([shape for _, shape in pts_entries], [(len(st.POINTS), HIDDEN)] * 2)
        self.assertEqual(parts[0].seq_bytes, 20 * HIDDEN * 2)
        self.assertEqual(parts[0].points_bytes, 2 * len(st.POINTS) * HIDDEN * 2)
        self.assertEqual(parts[0].total_bytes(2), 2 * (parts[0].seq_bytes + parts[0].points_bytes))
        # A seq subset: the points file of every layer, the seq file of the subset only, layer order kept.
        self.assertEqual(parts[0].files([3, 7], seq_layers=[7]), [
            "points_layer_3_part00.safetensors",
            "seq_layer_7_part00.safetensors", "points_layer_7_part00.safetensors",
        ])
        self.assertEqual(parts[0].files([3, 7], seq_layers=[]), ["points_layer_3_part00.safetensors",
                                                                 "points_layer_7_part00.safetensors"])
        self.assertEqual(parts[0].total_bytes(2, 1), parts[0].seq_bytes + 2 * parts[0].points_bytes)
        with self.assertRaises(ValueError):
            parts[0].entries("weights", hidden=HIDDEN)

    def test_cap_applies_to_one_layer_file(self):
        # 5 × 10 tokens × 16 B = 160 B per item per layer; cap 500 → 3 + 2 whatever n_layers says
        for n_layers in (1, 2, 7):
            parts = st.plan_parts(items([10] * 5), hidden=HIDDEN, n_layers=n_layers, max_bytes=500)
            self.assertEqual([len(p.items) for p in parts], [3, 2], n_layers)
        self.assertEqual([it.rollout_id for p in parts for it in p.items], [f"m:run:h:{i}" for i in range(5)])
        # one part when everything fits
        self.assertEqual(len(st.plan_parts(items([10, 10]), hidden=HIDDEN, n_layers=2, max_bytes=10_000)), 1)
        self.assertEqual(st.plan_parts([], hidden=HIDDEN, n_layers=2, max_bytes=10), [])

    def test_oversized_single_item_gets_own_part(self):
        parts = st.plan_parts(items([1, 100, 1]), hidden=HIDDEN, n_layers=2, max_bytes=100)
        self.assertEqual([len(p.items) for p in parts], [1, 1, 1])

    def test_item_above_hard_limit_raises(self):
        with self.assertRaisesRegex(ValueError, "hard limit"):
            st.plan_parts(items([100]), hidden=HIDDEN, n_layers=2, max_bytes=10, hard_limit_bytes=1000)

    def test_rejects_zero_layers(self):
        with self.assertRaises(ValueError):
            st.plan_parts(items([1]), hidden=HIDDEN, n_layers=0, max_bytes=10)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class TestLocalBackend(unittest.TestCase):
    """LocalBackend (+ StoreWriter over it)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_local_backend_roundtrip_and_atomic_write(self):
        backend = st.LocalBackend(self.tmp / "store")
        backend.preflight()
        layers = [3, 7]
        parts = st.plan_parts(items([10, 4, 6]), hidden=HIDDEN, n_layers=len(layers), max_bytes=14 * HIDDEN * 2)
        self.assertEqual(len(parts), 2)
        writer = st.StoreWriter(backend, "spec_a", meta={"layers": layers}, upload_workers=2)
        with contextlib.redirect_stdout(io.StringIO()):
            for part in parts:
                write_part(writer, part, layers,
                           meta_update=lambda meta, res: meta.setdefault("files", []).extend(res.ok_files))
            writer.close()
        self.assertEqual(writer.failed_parts, [])
        root = self.tmp / "store" / "spec_a"
        expected = {f for p in parts for f in p.files(layers)} | {"manifest.csv", "manifest.meta.json"}
        self.assertEqual({p.name for p in root.iterdir()}, expected)
        self.assertEqual(backend.list_files("spec_a/"), sorted(f"spec_a/{f}" for f in expected))
        # every tensor reads back exactly, under its rollout_id, in its part
        for part in parts:
            for layer in layers:
                with safe_open(str(root / part.seq_file(layer)), framework="pt") as f:
                    self.assertEqual(f.metadata(), {"layer": str(layer), "kind": "seq"})
                    for i, it in enumerate(part.items):
                        t = f.get_tensor(it.rollout_id)
                        self.assertEqual((tuple(t.shape), t.dtype), ((it.n_tokens, HIDDEN), torch.bfloat16))
                        self.assertEqual(t[0, 0].item(), (i + 1) * 10 + layer)
                with safe_open(str(root / part.points_file(layer)), framework="pt") as f:
                    for i, it in enumerate(part.items):
                        t = f.get_tensor(it.rollout_id)
                        self.assertEqual(tuple(t.shape), (len(st.POINTS), HIDDEN))
                        self.assertEqual(t[0, 0].item(), -(i + 1) * 10 - layer)
        manifest = pd.read_csv(root / "manifest.csv")
        self.assertEqual(list(manifest.columns), st.STORE_MANIFEST_COLUMNS)
        self.assertEqual(manifest["rollout_id"].tolist(), [f"m:run:h:{i}" for i in range(3)])
        self.assertEqual(manifest["n_cot_tokens"].tolist(), [10, 4, 6])
        meta = json.loads((root / "manifest.meta.json").read_text())
        self.assertEqual(meta["layers"], layers)
        self.assertEqual(sorted(meta["files"]), sorted(f for p in parts for f in p.files(layers)))
        files, manifest2, meta2 = st.load_store_state(backend, "spec_a")
        self.assertEqual(files, expected)
        self.assertEqual(manifest2["rollout_id"].tolist(), manifest["rollout_id"].tolist())
        self.assertEqual(meta2, meta)

        # Atomic: a copy that fails midway leaves neither the target nor a temp file.
        staged = self.tmp / "staged.bin"
        staged.write_bytes(b"x" * 64)
        with mock.patch.object(st.shutil, "copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                backend.upload_file(staged, "spec_a/seq_layer_9_part00.safetensors")
        self.assertFalse((root / "seq_layer_9_part00.safetensors").exists())
        self.assertEqual([p.name for p in root.iterdir() if p.name.startswith(".tmp-")], [])
        # A successful staged-file upload copies (the staged file is the writer's to delete).
        backend.upload_file(staged, "spec_a/blob.bin")
        self.assertEqual((root / "blob.bin").read_bytes(), b"x" * 64)
        self.assertTrue(staged.exists())

    def test_paths_are_relative_to_root(self):
        backend = st.LocalBackend(self.tmp)
        with self.assertRaises(ValueError):
            backend.upload_file(b"x", "/etc/passwd")
        with self.assertRaises(ValueError):
            backend.read_text("../outside.txt")

    def test_list_exists_delete_and_text(self):
        backend = st.LocalBackend(self.tmp / "s")
        self.assertEqual(backend.list_files(), [])
        self.assertFalse(backend.exists("a/x.txt"))
        backend.write_text("a/x.txt", "hello")
        backend.commit({"a/y.bin": b"\x00\x01", "z.txt": b"z"}, commit_message="ignored on disk")
        self.assertEqual(backend.list_files(), ["a/x.txt", "a/y.bin", "z.txt"])
        self.assertEqual(backend.list_files("a/"), ["a/x.txt", "a/y.bin"])
        self.assertEqual(backend.read_text("a/x.txt"), "hello")
        self.assertEqual(backend.download_file("a/y.bin").read_bytes(), b"\x00\x01")
        backend.delete(["a/x.txt", "missing.txt"])
        self.assertEqual(backend.list_files(), ["a/y.bin", "z.txt"])
        with self.assertRaises(FileNotFoundError):
            backend.download_file("a/x.txt")

    def test_preflight_creates_root_and_dry_run_touches_nothing(self):
        backend = st.LocalBackend(self.tmp / "new" / "store")
        self.assertEqual(backend.preflight(dry_run=True), str(self.tmp / "new" / "store"))
        self.assertFalse((self.tmp / "new").exists())
        backend.preflight()
        self.assertTrue((self.tmp / "new" / "store").is_dir())


class TestHfBackend(unittest.TestCase):
    """HfBackend over the in-memory FakeApi."""

    def test_hf_backend_checks_namespace_before_reading_data(self):
        api = FakeApi(user="someone", orgs=(), repo_exists=False)
        backend = st.HfBackend(REPO, api=api)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "cannot write to 'example-org'"):
                backend.preflight()
        self.assertEqual(api.created, [])
        self.assertEqual(api.commits, [])

        api = FakeApi(repo_exists=False)
        backend = st.HfBackend(REPO, private=True, api=api)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(backend.preflight(dry_run=True), "alice")
            self.assertEqual(api.created, [])
            backend.preflight()
        self.assertEqual(api.created, [(REPO, "dataset", True)])

        api = FakeApi()
        api.whoami = mock.Mock(side_effect=OSError("offline"))
        with self.assertRaisesRegex(RuntimeError, "token check failed"):
            st.HfBackend(REPO, api=api).preflight()

    def test_file_operations_through_api(self):
        api = FakeApi(repo_exists=False)
        backend = st.HfBackend(REPO, api=api)
        self.assertEqual(backend.list_files(), [])          # repo does not exist yet
        api.repo_exists = True
        backend.upload_file(b"abc", "spec/seq_layer_0_part00.safetensors")
        staged = Path(tempfile.mkdtemp()) / "staged.safetensors"
        staged.write_bytes(b"def")
        backend.upload_file(staged, "spec/points_layer_0_part00.safetensors", commit_message="Add points")
        self.assertTrue(staged.exists())
        backend.write_text("README.md", "card")
        self.assertEqual(api.files["spec/seq_layer_0_part00.safetensors"], b"abc")
        self.assertEqual(api.files["spec/points_layer_0_part00.safetensors"], b"def")
        self.assertEqual(api.commits[1][2], "Add points")
        self.assertEqual(backend.list_files("spec/"),
                         ["spec/points_layer_0_part00.safetensors", "spec/seq_layer_0_part00.safetensors"])
        self.assertTrue(backend.exists("README.md"))
        self.assertFalse(backend.exists("nope"))
        self.assertEqual(backend.read_text("README.md"), "card")
        backend.commit({"spec/manifest.csv": b"rollout_id\n"}, ["spec/seq_layer_0_part00.safetensors"],
                       commit_message="manifest")
        self.assertNotIn("spec/seq_layer_0_part00.safetensors", api.files)
        self.assertEqual(api.files["spec/manifest.csv"], b"rollout_id\n")
        self.assertEqual(api.commits[-1][1], ["CommitOperationAdd:spec/manifest.csv",
                                              "CommitOperationDelete:spec/seq_layer_0_part00.safetensors"])
        backend.delete(["README.md"])
        self.assertNotIn("README.md", api.files)
        backend.delete([])
        self.assertEqual(api.commits[-1][2], "Delete README.md")

    def test_rejects_bad_repo_id(self):
        with self.assertRaises(ValueError):
            st.HfBackend("no-namespace", api=FakeApi())


class TestMakeBackend(unittest.TestCase):
    """make_backend."""

    def test_make_backend_validates_keys(self):
        with self.assertRaisesRegex(ValueError, "Unknown storage keys"):
            st.make_backend({"backend": "local", "local_dir": "/tmp/x", "bucket": "y"})
        with self.assertRaisesRegex(ValueError, "storage.backend"):
            st.make_backend({"backend": "s3", "local_dir": "/tmp/x"})
        with self.assertRaisesRegex(ValueError, "hf_repo_id"):
            st.make_backend({"backend": "hf", "local_dir": "/tmp/x"}, api=FakeApi())
        with self.assertRaisesRegex(ValueError, "hf_repo_id"):
            st.make_backend({"backend": "hf", "hf_repo_id": "no-slash"}, api=FakeApi())
        with self.assertRaisesRegex(ValueError, "local_dir"):
            st.make_backend({"backend": "local", "hf_repo_id": REPO})
        with self.assertRaisesRegex(ValueError, "revision"):
            st.make_backend({"backend": "local", "local_dir": "/tmp/x", "revision": "abc"})
        with self.assertRaisesRegex(ValueError, "hf_repo_id"):
            st.make_backend({"backend": "local", "local_dir": "/tmp/x", "hf_repo_id": REPO, "hf_private": True})
        with self.assertRaisesRegex(ValueError, "local_dir"):
            st.make_backend({"backend": "hf", "hf_repo_id": REPO, "local_dir": "/tmp/x"}, api=FakeApi())
        with self.assertRaises(ValueError):
            st.make_backend(["backend"])
        backend = st.make_backend({"backend": "local", "local_dir": "/tmp/x", "hf_repo_id": None,
                                   "hf_private": None, "revision": None})
        self.assertIsInstance(backend, st.LocalBackend)
        self.assertEqual(backend.root, Path("/tmp/x"))

    def test_local_dir_resolves_data_root(self):
        with mock.patch.dict("os.environ", {"DATA_ROOT": "/data/root"}):
            backend = st.make_backend({"backend": "local", "local_dir": "${DATA_ROOT}/probe_activations/m"})
        self.assertEqual(backend.root, Path("/data/root/probe_activations/m"))
        with self.assertRaises(ValueError):
            st.make_backend({"backend": "local", "local_dir": "relative/dir"})

    def test_hf_builds_hf_backend(self):
        backend = st.make_backend({"backend": "hf", "hf_repo_id": REPO, "hf_private": True,
                                   "local_dir": None, "revision": "abc123"}, api=FakeApi())
        self.assertIsInstance(backend, st.HfBackend)
        self.assertEqual((backend.repo_id, backend.private, backend.revision), (REPO, True, "abc123"))
        self.assertEqual(backend.describe(), f"hf:{REPO}@abc123")


# ---------------------------------------------------------------------------
# StoreWriter
# ---------------------------------------------------------------------------


class TestStoreWriter(unittest.TestCase):
    """StoreWriter."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.layers = [0, 1]

    def _parts(self):
        # cap = two points rows (2 × 4 × 8 × 2 B): [3, 2] then [4, 1]
        return st.plan_parts(items([3, 2, 4, 1]), hidden=HIDDEN, n_layers=2, max_bytes=2 * len(st.POINTS) * HIDDEN * 2)

    def test_store_writer_reports_failed_part_and_keeps_manifest_consistent(self):
        parts = self._parts()
        self.assertEqual(len(parts), 2)
        backend = FailingLocalBackend(self.tmp, fail={parts[0].points_file(1)})
        seen = {}
        writer = st.StoreWriter(backend, "spec", meta={"n": 0}, upload_workers=2)

        def update(meta, res):
            seen[res.key] = res
            meta["n"] += res.n_rows

        with contextlib.redirect_stdout(io.StringIO()):
            for part in parts:
                write_part(writer, part, self.layers, meta_update=update)
            writer.close()
        self.assertEqual([f for f, _ in writer.failed_parts], [parts[0].points_file(1)])
        self.assertEqual(seen["part00"].failed_files, [parts[0].points_file(1)])
        self.assertEqual(len(seen["part00"].ok_files), 3)
        self.assertEqual(seen["part00"].n_rows, 0)
        self.assertEqual(seen["part01"].n_rows, len(parts[1].items))
        files, manifest, meta = st.load_store_state(backend, "spec")
        self.assertEqual(manifest["rollout_id"].tolist(), [it.rollout_id for it in parts[1].items])
        self.assertEqual(meta["n"], len(parts[1].items))
        self.assertNotIn(parts[0].points_file(1), files)
        self.assertIn(parts[0].seq_file(1), files)
        self.assertEqual(writer.results["part00"].n_rows, 0)

        # Rerun of the failed part only: its rows are added, the others untouched.
        ok = st.LocalBackend(self.tmp)
        writer = st.StoreWriter(ok, "spec", manifest=manifest, meta=meta, upload_workers=1)
        with contextlib.redirect_stdout(io.StringIO()):
            write_part(writer, parts[0], self.layers, meta_update=update)
            writer.close()
        self.assertEqual(writer.failed_parts, [])
        _, manifest, meta = st.load_store_state(ok, "spec")
        self.assertEqual(sorted(manifest["rollout_id"]), sorted(it.rollout_id for p in parts for it in p.items))
        self.assertEqual(meta["n"], 4)

    def test_per_file_rows_follow_their_file(self):
        backend = FailingLocalBackend(self.tmp, fail={"b.safetensors"})
        existing = pd.DataFrame([{"rollout_id": "r1", "seq_file": "old"}, {"rollout_id": "r9", "seq_file": "keep"}])
        writer = st.StoreWriter(backend, "", manifest=existing, columns=["rollout_id", "seq_file"], upload_workers=1)
        with contextlib.redirect_stdout(io.StringIO()):
            writer.submit_file("g", "a.safetensors", b"a", rows=[{"rollout_id": "r1", "seq_file": "a"}])
            writer.submit_file("g", "b.safetensors", b"b", rows=[{"rollout_id": "r2", "seq_file": "b"}])
            writer.finalize_group("g")
            writer.close()
        manifest = pd.read_csv(self.tmp / "manifest.csv")
        self.assertEqual(manifest.set_index("rollout_id")["seq_file"].to_dict(), {"r1": "a", "r9": "keep"})
        self.assertEqual(writer.results["g"].failed_files, ["b.safetensors"])
        # r2's file failed: neither its new row nor an older r2 row would survive
        backend = FailingLocalBackend(self.tmp, fail={"b.safetensors"})
        writer = st.StoreWriter(backend, "", manifest=pd.DataFrame([{"rollout_id": "r2", "seq_file": "old"}]),
                                columns=["rollout_id", "seq_file"], upload_workers=1)
        with contextlib.redirect_stdout(io.StringIO()):
            writer.submit_file("g", "b.safetensors", b"b", rows=[{"rollout_id": "r2", "seq_file": "b"}])
            writer.finalize_group("g")
            writer.close()
        self.assertTrue(pd.read_csv(self.tmp / "manifest.csv").empty)

    def test_stale_files_deleted_in_the_manifest_commit(self):
        backend = st.LocalBackend(self.tmp)
        for name in ("seq_layer_0_part00.safetensors", "seq_layer_0_part01.safetensors",
                     "points_layer_0_part01.safetensors", "README.md"):
            backend.upload_file(b"old", f"spec/{name}")
        writer = st.StoreWriter(backend, "spec", upload_workers=1)
        self.assertEqual(writer.stale_files(keep=["seq_layer_0_part00.safetensors"]),
                         ["points_layer_0_part01.safetensors", "seq_layer_0_part01.safetensors"])
        with contextlib.redirect_stdout(io.StringIO()):
            writer.submit_file("part00", "seq_layer_0_part00.safetensors", b"new")
            writer.finalize_group("part00", rows=[{"rollout_id": "r1"}], keep=["seq_layer_0_part00.safetensors"],
                                  stale_filter=st.is_store_file)
            writer.close()
        files, manifest, _ = st.load_store_state(backend, "spec")
        self.assertEqual(files, {"seq_layer_0_part00.safetensors", "README.md", "manifest.csv", "manifest.meta.json"})
        self.assertEqual((self.tmp / "spec" / "seq_layer_0_part00.safetensors").read_bytes(), b"new")
        self.assertEqual(writer.results["part00"].stale_deleted,
                         ["points_layer_0_part01.safetensors", "seq_layer_0_part01.safetensors"])
        self.assertEqual(manifest["rollout_id"].tolist(), ["r1"])

    def test_disk_staging_deletes_staged_files_after_upload(self):
        stage = self.tmp / "stage"
        writer = st.StoreWriter(st.LocalBackend(self.tmp / "store"), "spec", upload_workers=1,
                                staging="disk", staging_dir=stage, max_file_gb=1)
        self.assertEqual(writer.prepare_staging(), [])
        pw = writer.new_part_writer("seq_layer_0_part00.safetensors", [("r1", (2, HIDDEN))], {})
        self.assertTrue(str(pw.path).startswith(str(stage)))
        pw.add("r1", torch.ones(2, HIDDEN, dtype=torch.bfloat16))
        with contextlib.redirect_stdout(io.StringIO()):
            writer.submit_file("part00", "seq_layer_0_part00.safetensors", pw.finish())
            writer.finalize_group("part00", rows=[{"rollout_id": "r1"}])
            writer.close()
        self.assertEqual(list(stage.iterdir()), [])
        loaded = safetensors_load((self.tmp / "store" / "spec" / "seq_layer_0_part00.safetensors").read_bytes())
        self.assertTrue(torch.equal(loaded["r1"], torch.ones(2, HIDDEN, dtype=torch.bfloat16)))
        self.assertIn("RAM stays small", writer.staging_note(4))
        self.assertIn("peak RAM", st.StoreWriter(st.LocalBackend(self.tmp), upload_workers=2).staging_note(2))

    def test_resubmitted_filename_judged_on_its_own_upload(self):
        # part00's seq file uploads in group A; a retry of the same filename in
        # group B fails and must not inherit A's success.
        name = "seq_layer_0_part00.safetensors"

        class FailOnRetry(st.LocalBackend):
            def upload_file(self, local_or_bytes, path, *, commit_message=None):
                if local_or_bytes == b"retry":
                    raise RuntimeError("simulated failure on the retry")
                super().upload_file(local_or_bytes, path, commit_message=commit_message)

        writer = st.StoreWriter(FailOnRetry(self.tmp), "spec", upload_workers=1)
        with contextlib.redirect_stdout(io.StringIO()):
            writer.submit_file("A", name, b"first", rows=[{"rollout_id": "r1"}])
            writer.finalize_group("A")
            writer.submit_file("B", name, b"retry", rows=[{"rollout_id": "r1"}])
            writer.finalize_group("B")
            writer.close()
        self.assertEqual(writer.results["A"].failed_files, [])
        self.assertEqual(writer.results["B"].failed_files, [name])
        self.assertTrue(pd.read_csv(self.tmp / "spec" / "manifest.csv").empty)

    def test_rejects_bad_staging_config(self):
        with self.assertRaises(ValueError):
            st.StoreWriter(st.LocalBackend(self.tmp), staging="ram")
        with self.assertRaises(ValueError):
            st.StoreWriter(st.LocalBackend(self.tmp), staging="disk")
        with self.assertRaises(ValueError):
            st.StoreWriter(st.LocalBackend(self.tmp), upload_workers=0)

    def test_close_reraises_unexpected_errors(self):
        backend = st.LocalBackend(self.tmp)
        writer = st.StoreWriter(backend, "spec", upload_workers=1)
        with contextlib.redirect_stdout(io.StringIO()):
            writer.submit_file("g", "seq_layer_0_part00.safetensors", b"x")
            writer.finalize_group("g", meta_update=lambda meta, res: 1 / 0)
            with self.assertRaises(ZeroDivisionError):
                writer.close()


# ---------------------------------------------------------------------------
# Reader side
# ---------------------------------------------------------------------------


LAST = st.POINTS.index("last_cot")


def seq_value(i: int, layer: int, t: int) -> float:
    """Distinct, bf16-exact value per (item, layer, position)."""
    return float(i * 32 + layer * 16 + t)


def point_value(i: int, layer: int, k: int) -> float:
    return float(i * 32 + layer * 16 + 8 + k)


def write_store(root: Path, folder="spec", layers=(0, 1), n_tokens=(3, 2, 4, 1), *, meta_extra=None, seq_layers=None):
    """A two-part local store ([3, 2] then [4, 1]): seq row t = seq_value(i, L, t), points
    row k = point_value(i, L, k) except ``last_cot`` = the seq tensor's last row;
    ``seq_layers`` restricts the seq files written and is recorded in the meta."""
    layers = list(layers)
    seq_layers = layers if seq_layers is None else [int(l) for l in seq_layers]
    parts = st.plan_parts(items(list(n_tokens)), hidden=HIDDEN, n_layers=len(layers),
                          max_bytes=2 * len(st.POINTS) * HIDDEN * 2)
    backend = st.LocalBackend(root)
    meta = {"points": list(st.POINTS), "layers": layers, "hidden_size": HIDDEN, "dtype": st.STORE_DTYPE_NAME}
    if seq_layers != layers:
        meta["seq_layers"] = seq_layers
    meta.update(meta_extra or {})
    writer = st.StoreWriter(backend, folder, meta=meta, upload_workers=1)
    with contextlib.redirect_stdout(io.StringIO()):
        for part in parts:
            rows = []
            for layer in layers:
                for kind, fname in (("seq", part.seq_file(layer)), ("points", part.points_file(layer))):
                    if kind == "seq" and layer not in seq_layers:
                        continue
                    pw = writer.new_part_writer(fname, part.entries(kind, hidden=HIDDEN), {"layer": str(layer)})
                    for it in part.items:
                        i = int(it.rollout_id.rsplit(":", 1)[1])
                        seq = torch.stack([torch.full((HIDDEN,), seq_value(i, layer, t)) for t in range(it.n_tokens)])
                        if kind == "seq":
                            t = seq
                        else:
                            t = torch.stack([torch.full((HIDDEN,), point_value(i, layer, k)) for k in range(len(st.POINTS))])
                            t[LAST] = seq[-1]
                        pw.add(it.rollout_id, t.to(torch.bfloat16))
                    writer.submit_file(f"part{part.index:02d}", fname, pw.finish())
            for it in part.items:
                rows.append({**{c: "" for c in st.STORE_MANIFEST_COLUMNS}, "rollout_id": it.rollout_id,
                             "n_cot_tokens": it.n_tokens, "seq_file": st.part_template("seq", part.index),
                             "points_file": st.part_template("points", part.index)})
            writer.finalize_group(f"part{part.index:02d}", rows=rows)
        writer.close()
    return backend, parts


def expected_seq(i: int, layer: int, n: int) -> torch.Tensor:
    return torch.stack([torch.full((HIDDEN,), seq_value(i, layer, t)) for t in range(n)]).to(torch.bfloat16)


def hf_backend_over(local: st.LocalBackend) -> st.HfBackend:
    """An HfBackend whose FakeApi holds every file of the local store."""
    files = {f: (local.root / f).read_bytes() for f in local.list_files()}
    return st.HfBackend(REPO, api=FakeApi(files=files))


def stream_from_local(local: st.LocalBackend, calls: list):
    """A stand-in for hf_streaming.stream_tensor reading the local store's files."""
    def fake(repo_id, filename, tensor_key, dtype=None, revision=None, **kw):
        calls.append((repo_id, filename, tensor_key, dtype, revision, threading.get_ident(), kw.get("token")))
        with safe_open(local.root / filename, framework="pt", device="cpu") as f:
            t = f.get_tensor(tensor_key)
        return t if dtype is None else t.to(dtype)
    return fake


class TestPartTemplate(unittest.TestCase):
    """part_template / part_index."""

    def test_part_index_accepts_every_form(self):
        self.assertEqual(st.part_template("seq", 3), "seq_layer_{L}_part03")
        self.assertEqual(st.part_template("points", 12), "points_layer_{L}_part12")
        for v in ("seq_layer_{L}_part03", "points_layer_7_part03.safetensors", "part3", 3, np.int64(3), 3.0):
            self.assertEqual(st.part_index(v), 3, v)

    def test_part_index_rejects_garbage(self):
        for v in ("seq_layer_3", "", float("nan"), True, "manifest.csv"):
            with self.assertRaises(ValueError):
                st.part_index(v)
        with self.assertRaises(ValueError):
            st.part_template("other", 0)


class TestReadStoreManifest(unittest.TestCase):
    """read_store_manifest and the layer helpers (mirrored / seq_layers metas included)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.backend, self.parts = write_store(self.tmp)
        _, self.pristine, self.pristine_meta = st.load_store_state(self.backend, "spec")

    def _rewrite(self, edit_manifest=None, edit_meta=None):
        """Rewrite manifest + meta from the pristine copies through the edits."""
        manifest, meta = self.pristine.copy(), dict(self.pristine_meta)
        if edit_manifest is not None:
            manifest = edit_manifest(manifest)
        if edit_meta is not None:
            meta = edit_meta(meta)
        self.backend.write_text("spec/manifest.csv", manifest.to_csv(index=False))
        self.backend.write_text("spec/manifest.meta.json", json.dumps(meta))

    def test_read_store_manifest_normalises_part_columns(self):
        def loosen(m):
            m = m.copy()
            m.loc[0, "seq_file"] = "seq_layer_1_part00.safetensors"
            m.loc[1, "points_file"] = "part0"
            m.loc[2, "seq_file"] = "1"
            return m
        self._rewrite(edit_manifest=loosen)
        manifest, meta = st.read_store_manifest(self.backend, "spec")
        self.assertEqual(list(manifest.columns), st.STORE_MANIFEST_COLUMNS)
        self.assertEqual(manifest["seq_file"].tolist(), ["seq_layer_{L}_part00"] * 2 + ["seq_layer_{L}_part01"] * 2)
        self.assertEqual(manifest["points_file"].tolist(),
                         ["points_layer_{L}_part00"] * 2 + ["points_layer_{L}_part01"] * 2)
        self.assertEqual(meta["layers"], [0, 1])
        self.assertTrue(all(isinstance(l, int) for l in meta["layers"]))

    def test_read_store_manifest_validates_points_order(self):
        for bad in (list(reversed(st.POINTS)), list(st.POINTS[:3]), None, "pre_cot"):
            self._rewrite(edit_meta=lambda m, bad=bad: {**m, "points": bad})
            with self.assertRaisesRegex(ValueError, "points"):
                st.read_store_manifest(self.backend, "spec")
        self._rewrite(edit_meta=lambda m: {**m, "points": list(st.POINTS)})
        st.read_store_manifest(self.backend, "spec")

    def test_read_store_manifest_checks_files_present(self):
        victim = self.parts[1].points_file(1)
        self.backend.delete([f"spec/{victim}"])
        with self.assertRaisesRegex(ValueError, victim):
            st.read_store_manifest(self.backend, "spec")
        self.backend, _ = write_store(self.tmp / "again")
        self._rewrite(edit_manifest=lambda m: m.assign(seq_file=["seq_layer_{L}_part07"] * len(m)))
        with self.assertRaisesRegex(ValueError, r"seq_layer_0_part07\.safetensors"):
            st.read_store_manifest(self.backend, "spec")

    def test_read_store_manifest_rejects_bad_columns_duplicates_and_layers(self):
        with self.assertRaises(FileNotFoundError):
            st.read_store_manifest(self.backend, "nowhere")
        self._rewrite(edit_manifest=lambda m: m.drop(columns=["split"]))
        with self.assertRaisesRegex(ValueError, "columns"):
            st.read_store_manifest(self.backend, "spec")
        self._rewrite(edit_manifest=lambda m: m[list(reversed(m.columns))])
        with self.assertRaisesRegex(ValueError, "columns"):
            st.read_store_manifest(self.backend, "spec")
        self._rewrite(edit_manifest=lambda m: pd.concat([m, m.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "duplicate rollout_id"):
            st.read_store_manifest(self.backend, "spec")
        self._rewrite(edit_manifest=lambda m: m.assign(rollout_id=[""] + m["rollout_id"].tolist()[1:]))
        with self.assertRaisesRegex(ValueError, "blank rollout_id"):
            st.read_store_manifest(self.backend, "spec")
        for layers in ([1, 0], [0, 0, 1], ["0", "1"], [], None):
            self._rewrite(edit_meta=lambda m, layers=layers: {**m, "layers": layers})
            with self.assertRaisesRegex(ValueError, "layers"):
                st.read_store_manifest(self.backend, "spec")
        for hidden in (None, 0, "8", 8.0):
            self._rewrite(edit_meta=lambda m, hidden=hidden: {**m, "hidden_size": hidden})
            with self.assertRaisesRegex(ValueError, "hidden_size"):
                st.read_store_manifest(self.backend, "spec")
        self._rewrite(edit_manifest=lambda m: m.assign(rollout_id=[" " + m["rollout_id"].iloc[0]] + m["rollout_id"].tolist()[1:]))
        self.assertEqual(st.read_store_manifest(self.backend, "spec")[0]["rollout_id"].tolist(), self.pristine["rollout_id"].tolist())
        self._rewrite()
        st.read_store_manifest(self.backend, "spec")
        (self.tmp / "spec" / st.STORE_MANIFEST_META_NAME).unlink()
        with self.assertRaises(FileNotFoundError):
            st.read_store_manifest(self.backend, "spec")

    def test_read_store_manifest_layer_subset_and_mirrored_layers(self):
        # A store missing every seq file of layer 1 validates for layers=[0] (or for the points alone).
        for part in self.parts:
            self.backend.delete([f"spec/{part.seq_file(1)}"])
        with self.assertRaisesRegex(ValueError, r"seq_layer_1_part00"):
            st.read_store_manifest(self.backend, "spec")
        manifest, meta = st.read_store_manifest(self.backend, "spec", layers=[0])
        self.assertEqual(meta["layers"], [0, 1])
        self.assertEqual(len(manifest), 4)
        st.read_store_manifest(self.backend, "spec", layers=[1], kinds=("points",))
        with self.assertRaisesRegex(ValueError, r"seq_layer_1_part00"):
            st.read_store_manifest(self.backend, "spec", layers=[1], kinds=("seq",))
        with self.assertRaisesRegex(st.LayerSelectionError, r"\[7\].*not stored"):
            st.read_store_manifest(self.backend, "spec", layers=[0, 7])
        with self.assertRaisesRegex(ValueError, "kinds"):
            st.read_store_manifest(self.backend, "spec", kinds=("seq", "foo"))
        # A mirror's meta: the reader defaults to the mirrored layers per kind and refuses the rest.
        self._rewrite(edit_meta=lambda m: {**m, "mirrored_layers": [0], "mirrored_points_layers": [0, 1]})
        manifest, meta = st.read_store_manifest(self.backend, "spec")
        self.assertEqual((meta["layers"], meta["mirrored_layers"], meta["mirrored_points_layers"]), ([0, 1], [0], [0, 1]))
        self.assertTrue(st.is_mirror(meta))
        self.assertEqual((st.available_layers(meta, "seq"), st.available_layers(meta, "points")), ([0], [0, 1]))
        self.assertEqual(st.resolve_kind_layers(None, meta, "seq"), [0])
        with self.assertRaisesRegex(st.LayerSelectionError, r"\[1\] are not in this mirror.*seq.*--layers 1"):
            st.read_store_manifest(self.backend, "spec", layers=[1])
        with self.assertRaisesRegex(st.LayerSelectionError, r"\[1\] are not in this mirror"):
            st.read_store_manifest(self.backend, "spec", layers=[0, 1], kinds=("seq",))
        st.read_store_manifest(self.backend, "spec", layers=[0, 1], kinds=("points",))
        with self.assertRaisesRegex(st.LayerSelectionError, r"\[7\].*not stored"):
            st.resolve_kind_layers([7], meta, "seq")
        seq = st.SequenceStore(self.backend, "spec", manifest, meta)
        self.assertEqual(seq.layers, [0])
        self.assertEqual(st.PointStore(self.backend, "spec", manifest, meta).layers, [0, 1])
        with self.assertRaisesRegex(st.LayerSelectionError, "not in this mirror"):
            st.SequenceStore(self.backend, "spec", manifest, meta, layers=[1])
        torch.testing.assert_close(seq.get("m:run:h:2")[0], expected_seq(2, 0, 4))
        # A points-only mirror ([] seq layers) reads; a mirrored layer the store lacks is refused.
        self._rewrite(edit_meta=lambda m: {**m, "mirrored_layers": [], "mirrored_points_layers": [0, 1]})
        _, meta = st.read_store_manifest(self.backend, "spec")
        self.assertEqual(meta["mirrored_layers"], [])
        self.assertEqual(st.PointStore(self.backend, "spec", manifest, meta).layers, [0, 1])
        with self.assertRaisesRegex(st.LayerSelectionError, "not in this mirror"):
            st.read_store_manifest(self.backend, "spec", layers=[0], kinds=("seq",))
        self._rewrite(edit_meta=lambda m: {**m, "mirrored_layers": [5]})
        with self.assertRaisesRegex(ValueError, r"mirrored_layers.*\[5\].*does not hold"):
            st.read_store_manifest(self.backend, "spec")
        self._rewrite(edit_meta=lambda m: {**m, "mirrored_layers": [1, 0]})
        with self.assertRaisesRegex(ValueError, "mirrored_layers.*sorted"):
            st.read_store_manifest(self.backend, "spec")
        with self.assertRaisesRegex(ValueError, "kind must be"):
            st.available_layers(meta, "foo")

    def test_seq_layers_subset(self):
        root = Path(tempfile.mkdtemp())
        backend, parts = write_store(root, seq_layers=[1])
        names = sorted(f for f in backend.list_files("spec/") if st.is_store_file(f))
        self.assertEqual(names, sorted([f"spec/{p.seq_file(1)}" for p in parts]
                                       + [f"spec/{p.points_file(l)}" for p in parts for l in (0, 1)]))
        manifest, meta = st.read_store_manifest(backend, "spec")
        self.assertEqual((meta["layers"], meta["seq_layers"]), ([0, 1], [1]))
        self.assertFalse(st.is_mirror(meta))
        self.assertEqual((st.stored_layers(meta, "seq"), st.stored_layers(meta, "points")), ([1], [0, 1]))
        self.assertEqual((st.available_layers(meta, "seq"), st.available_layers(meta, "points")), ([1], [0, 1]))
        self.assertEqual(st.resolve_kind_layers(None, meta, "seq"), [1])
        self.assertEqual(st.resolve_kind_layers(None, meta, "points"), [0, 1])
        self.assertEqual(st.resolve_kind_layers([0, 1], meta, "points"), [0, 1])
        with self.assertRaisesRegex(st.LayerSelectionError, r"layers \[0\] have no seq files.*seq_layers: \[1\]"):
            st.resolve_kind_layers([0], meta, "seq")
        with self.assertRaisesRegex(ValueError, "kind must be"):
            st.stored_layers(meta, "foo")
        # The presence check per kind uses the kind's list: points for both layers, seq for [1] only.
        st.read_store_manifest(backend, "spec", layers=[0, 1], kinds=("points",))
        st.read_store_manifest(backend, "spec", layers=[1], kinds=("seq",))
        with self.assertRaisesRegex(st.LayerSelectionError, r"\[0\] have no seq files"):
            st.read_store_manifest(backend, "spec", layers=[0], kinds=("seq",))
        with self.assertRaisesRegex(st.LayerSelectionError, r"\[0\] have no seq files"):
            st.read_store_manifest(backend, "spec", layers=[0, 1])
        seq = st.SequenceStore(backend, "spec", manifest, meta)
        self.assertEqual(seq.layers, [1])
        torch.testing.assert_close(seq.get("m:run:h:2")[1], expected_seq(2, 1, 4))
        with self.assertRaisesRegex(st.LayerSelectionError, "have no seq files"):
            st.SequenceStore(backend, "spec", manifest, meta, layers=[0])
        points = st.PointStore(backend, "spec", manifest, meta)
        self.assertEqual(points.layers, [0, 1])
        self.assertEqual(sorted(points.get("m:run:h:2", "pre_cot")), [0, 1])
        # The meta key is validated: a non-empty sorted subset of `layers`; a mirror's seq layers ⊆ it.
        meta_path = root / "spec" / st.STORE_MANIFEST_META_NAME
        pristine = json.loads(meta_path.read_text())
        for bad, pattern in (([5], r"seq_layers.*unknown layers \[5\]"), ([], "seq_layers.*non-empty"),
                             ([1, 0], "seq_layers.*sorted"), ("1", "list of ints")):
            meta_path.write_text(json.dumps({**pristine, "seq_layers": bad}))
            with self.assertRaisesRegex(ValueError, pattern, msg=repr(bad)):
                st.read_store_manifest(backend, "spec")
        meta_path.write_text(json.dumps({**pristine, "mirrored_layers": [0], "mirrored_points_layers": [0, 1]}))
        with self.assertRaisesRegex(ValueError, r"mirrored_layers.*\[0\].*does not hold seq files.*seq_layers: \[1\]"):
            st.read_store_manifest(backend, "spec")
        meta_path.write_text(json.dumps({**pristine, "mirrored_layers": [1], "mirrored_points_layers": [0]}))
        _, mirrored = st.read_store_manifest(backend, "spec")
        self.assertEqual((st.available_layers(mirrored, "seq"), st.available_layers(mirrored, "points")), ([1], [0]))
        # A meta without the key reads every layer for both kinds.
        meta_path.write_text(json.dumps({k: v for k, v in pristine.items() if k != "seq_layers"}))
        _, old = st.read_store_manifest(backend, "spec", layers=[0, 1], kinds=("points",))
        self.assertEqual((st.available_layers(old, "seq"), st.stored_layers(old, "seq")), ([0, 1], [0, 1]))
        with self.assertRaisesRegex(ValueError, "seq_layer_0_part00"):   # every layer's seq file is expected then
            st.read_store_manifest(backend, "spec")


class TestResolveStoreLayers(unittest.TestCase):
    """resolve_store_layers."""

    def test_none_means_all_stored_sorted(self):
        self.assertEqual(st.resolve_store_layers(None, [7, 3, 11]), [3, 7, 11])

    def test_subset_kept_sorted(self):
        self.assertEqual(st.resolve_store_layers([11, 3], [3, 7, 11]), [3, 11])

    def test_unknown_layer_raises(self):
        with self.assertRaisesRegex(ValueError, r"\[5\].*stored layers: \[3, 7, 11\]"):
            st.resolve_store_layers([3, 5], [3, 7, 11])

    def test_empty_or_duplicate_raises(self):
        with self.assertRaises(ValueError):
            st.resolve_store_layers([], [3])
        with self.assertRaises(ValueError):
            st.resolve_store_layers([3, 3], [3])


class TestSequenceStore(unittest.TestCase):
    """SequenceStore over local and (fake) HF backends."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.backend, self.parts = write_store(self.tmp)
        self.manifest, self.meta = st.read_store_manifest(self.backend, "spec")
        self.n_tokens = {it.rollout_id: it.n_tokens for p in self.parts for it in p.items}

    def test_sequence_store_local_exact(self):
        store = st.SequenceStore(self.backend, "spec", self.manifest, self.meta, layers=None)
        self.assertEqual(store.layers, [0, 1])
        self.assertEqual(len(store), 4)
        for rid, n in self.n_tokens.items():
            i = int(rid.rsplit(":", 1)[1])
            H = store.get(rid)
            self.assertEqual(sorted(H), [0, 1])
            for layer in (0, 1):
                self.assertEqual(H[layer].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(H[layer], expected_seq(i, layer, n)))
        sub = st.SequenceStore(self.backend, "spec", self.manifest, self.meta, layers=[1], dtype=torch.float32)
        H = sub.get("m:run:h:2")
        self.assertEqual(list(H), [1])
        self.assertEqual(H[1].dtype, torch.float32)
        self.assertTrue(torch.equal(H[1], expected_seq(2, 1, 4).float()))
        self.assertEqual(list(store.get("m:run:h:0", layers=[0])), [0])

    def test_unknown_layer_lists_stored(self):
        with self.assertRaisesRegex(ValueError, r"\[5\].*stored layers: \[0, 1\]"):
            st.SequenceStore(self.backend, "spec", self.manifest, self.meta, layers=[0, 5])
        store = st.SequenceStore(self.backend, "spec", self.manifest, self.meta, layers=[1])
        with self.assertRaisesRegex(ValueError, r"\[0\].*stored layers: \[1\]"):
            store.get("m:run:h:0", layers=[0])

    def test_unknown_rollout_raises(self):
        store = st.SequenceStore(self.backend, "spec", self.manifest, self.meta)
        with self.assertRaisesRegex(KeyError, "m:run:h:9"):
            store.get("m:run:h:9")
        self.assertIn("m:run:h:1", store)
        self.assertNotIn("m:run:h:9", store)

    def test_hf_sequence_store_uses_stream_tensor_with_rollout_id_key(self):
        hf = hf_backend_over(self.backend)
        manifest, meta = st.read_store_manifest(hf, "spec")
        self.assertEqual(manifest["rollout_id"].tolist(), self.manifest["rollout_id"].tolist())
        calls = []
        with mock.patch.object(st.hf_streaming, "stream_tensor", side_effect=stream_from_local(self.backend, calls)):
            store = st.SequenceStore(hf, "spec", manifest, meta, layers=None)
            self.assertTrue(store.streaming)
            self.assertEqual(store.fetch_threads, 2)
            H = store.get("m:run:h:2")
            for layer in (0, 1):
                self.assertTrue(torch.equal(H[layer], expected_seq(2, layer, 4)))
            self.assertEqual(sorted(c[:5] for c in calls), [
                (REPO, "spec/seq_layer_0_part01.safetensors", "m:run:h:2", None, None),
                (REPO, "spec/seq_layer_1_part01.safetensors", "m:run:h:2", None, None),
            ])
            # Layers are fetched in threads, not one after the other on the caller.
            self.assertTrue(all(c[5] != threading.get_ident() for c in calls))
            calls.clear()
            hf.revision = "abc123"
            st.SequenceStore(hf, "spec", manifest, meta, layers=[0], dtype=torch.float32).get("m:run:h:0")
            self.assertEqual(calls[0][:5], (REPO, "spec/seq_layer_0_part00.safetensors", "m:run:h:0", torch.float32, "abc123"))
            calls.clear()
            st.SequenceStore(hf, "spec", manifest, meta, layers=[0], revision="pinned").get("m:run:h:0")
            self.assertEqual(calls[0][4], "pinned")
            # The backend's explicit token signs the streamed URLs too (None = the cached login).
            self.assertIsNone(calls[0][6])
            calls.clear()
            hf.api.token = "tok-x"
            st.SequenceStore(hf, "spec", manifest, meta, layers=[0]).get("m:run:h:0")
            self.assertEqual(calls[0][6], "tok-x")

    def test_local_handles_cached_per_process(self):
        st._open_local.cache_clear()
        self.addCleanup(st._open_local.cache_clear)
        path = self.tmp / "spec" / self.parts[0].seq_file(0)
        h1 = st.local_handle(path)
        self.assertIs(h1, st.local_handle(str(path)))
        with mock.patch.object(st.os, "getpid", return_value=os.getpid() + 1):
            self.assertIsNot(h1, st.local_handle(path))


class TestPointStore(unittest.TestCase):
    """PointStore over local and (fake) HF backends."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.backend, self.parts = write_store(self.tmp)
        self.manifest, self.meta = st.read_store_manifest(self.backend, "spec")
        self.ids = self.manifest["rollout_id"].tolist()

    def _row(self, i, layer, point):
        """The stored point row for a point other than last_cot (that one is seq[-1])."""
        assert point != "last_cot"
        return torch.full((HIDDEN,), point_value(i, layer, st.POINTS.index(point))).to(torch.bfloat16)

    def test_point_store_slices_by_point_name(self):
        store = st.PointStore(self.backend, "spec", self.manifest, self.meta)
        seq = st.SequenceStore(self.backend, "spec", self.manifest, self.meta)
        self.assertEqual(store.fetch_threads, st.DEFAULT_FETCH_THREADS)
        for rid in self.ids:
            i = int(rid.rsplit(":", 1)[1])
            for point in st.POINTS:
                H = store.get(rid, point)
                self.assertEqual(sorted(H), [0, 1])
                for layer in (0, 1):
                    self.assertEqual(tuple(H[layer].shape), (HIDDEN,))
                    self.assertEqual(H[layer].dtype, torch.bfloat16)
                    want = seq.get(rid)[layer][-1] if point == "last_cot" else self._row(i, layer, point)
                    self.assertTrue(torch.equal(H[layer], want), (rid, point, layer))
        with self.assertRaisesRegex(ValueError, "point must be one of"):
            store.get(self.ids[0], "final")
        self.assertEqual(list(store.get(self.ids[0], "mean_cot", layers=[1])), [1])

    def test_get_many_stacks_in_request_order(self):
        store = st.PointStore(self.backend, "spec", self.manifest, self.meta)
        order = ["m:run:h:3", "m:run:h:0", "m:run:h:2", "m:run:h:0", "m:run:h:1"]
        out = store.get_many(order, "pre_answer")
        self.assertEqual(sorted(out), [0, 1])
        for layer in (0, 1):
            self.assertEqual(tuple(out[layer].shape), (5, HIDDEN))
            self.assertEqual(out[layer].dtype, torch.bfloat16)
            for r, rid in enumerate(order):
                i = int(rid.rsplit(":", 1)[1])
                self.assertTrue(torch.equal(out[layer][r], self._row(i, layer, "pre_answer")))
        out = store.get_many(order[:2], "last_cot", layers=[1])
        self.assertEqual(list(out), [1])
        self.assertTrue(torch.equal(out[1][0], expected_seq(3, 1, 1)[-1]))
        self.assertTrue(torch.equal(out[1][1], expected_seq(0, 1, 3)[-1]))
        wide = st.PointStore(self.backend, "spec", self.manifest, self.meta, dtype=torch.float32)
        self.assertEqual(wide.get_many(order, "pre_cot")[0].dtype, torch.float32)
        empty = store.get_many([], "pre_cot")
        self.assertEqual({l: tuple(t.shape) for l, t in empty.items()}, {0: (0, HIDDEN), 1: (0, HIDDEN)})
        with self.assertRaisesRegex(KeyError, "m:run:h:9"):
            store.get_many(["m:run:h:0", "m:run:h:9"], "pre_cot")
        with self.assertRaisesRegex(ValueError, r"\[4\].*stored layers"):
            store.get_many(order, "pre_cot", layers=[4])

    def test_hf_point_store_get_many_uses_thread_pool(self):
        hf = hf_backend_over(self.backend)
        manifest, meta = st.read_store_manifest(hf, "spec")
        local = st.PointStore(self.backend, "spec", self.manifest, self.meta)
        calls, pools = [], []
        real_pool = st.ThreadPoolExecutor

        def spy_pool(max_workers=None, **kw):
            pools.append(max_workers)
            return real_pool(max_workers=max_workers, **kw)

        with mock.patch.object(st.hf_streaming, "stream_tensor", side_effect=stream_from_local(self.backend, calls)), \
                mock.patch.object(st, "ThreadPoolExecutor", side_effect=spy_pool):
            store = st.PointStore(hf, "spec", manifest, meta, fetch_threads=3)
            order = ["m:run:h:2", "m:run:h:0", "m:run:h:3", "m:run:h:2"]
            out = store.get_many(order, "mean_cot")
        want = local.get_many(order, "mean_cot")
        for layer in (0, 1):
            self.assertTrue(torch.equal(out[layer], want[layer]))
            self.assertTrue(torch.equal(out[layer][0], out[layer][3]))
        fetched = sorted((c[1], c[2]) for c in calls)
        self.assertEqual(fetched, sorted(
            (f"spec/points_layer_{l}_part{store.part_of(r):02d}.safetensors", r) for l in (0, 1) for r in set(order)
        ))
        self.assertEqual(len(fetched), len(set(fetched)))
        self.assertEqual(pools, [3])
        self.assertTrue(all(c[5] != threading.get_ident() for c in calls))
        # Chunked submission: with one thread the chunk is 4 tasks, so a failure in the
        # first chunk surfaces before the second chunk's tasks are ever requested.
        calls.clear()

        def failing(repo_id, filename, tensor_key, **kw):
            calls.append(tensor_key)
            raise RuntimeError("boom")
        with mock.patch.object(st.hf_streaming, "stream_tensor", side_effect=failing):
            store = st.PointStore(hf, "spec", manifest, meta, fetch_threads=1)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                store.get_many(order, "mean_cot")
        self.assertLessEqual(len(calls), 4)

    def test_point_dataset_and_sequence_dataset_share_ids(self):
        seq = st.SequenceStore(self.backend, "spec", self.manifest, self.meta)
        pts = st.PointStore(self.backend, "spec", self.manifest, self.meta)
        self.assertEqual(seq.rollout_ids, pts.rollout_ids)
        self.assertEqual([seq.part_of(r) for r in self.ids], [pts.part_of(r) for r in self.ids])
        self.assertEqual(seq.file_of("m:run:h:3", 1), "seq_layer_1_part01.safetensors")
        self.assertEqual(pts.file_of("m:run:h:3", 1), "points_layer_1_part01.safetensors")


class TestRolloutDatasets(unittest.TestCase):
    """RolloutSequenceDataset."""

    def setUp(self):
        from src.lib import probe_training
        self.collate = probe_training.collate_multi_layer
        self.tmp = Path(tempfile.mkdtemp())
        self.backend, self.parts = write_store(self.tmp)
        self.manifest, self.meta = st.read_store_manifest(self.backend, "spec")

    def test_datasets_yield_collate_compatible_items(self):
        store = st.SequenceStore(self.backend, "spec", self.manifest, self.meta)
        ids = ["m:run:h:2", "m:run:h:1", "m:run:h:3"]
        ds = st.RolloutSequenceDataset(ids, [0, 1, np.int64(0)], store, weights=[0.5, 1.0, 2.0])
        self.assertEqual(len(ds), 3)
        H, y, w = ds[0]
        self.assertTrue(torch.equal(H[1], expected_seq(2, 1, 4)))
        self.assertEqual((y.dtype, y.shape, int(y)), (torch.long, torch.Size([]), 0))
        self.assertEqual((w.dtype, w.shape, float(w)), (torch.float32, torch.Size([]), 0.5))
        padded, labels, lengths = self.collate([ds[i] for i in range(3)])
        self.assertEqual(sorted(padded), [0, 1])
        self.assertEqual(tuple(padded[0].shape), (3, 4, HIDDEN))
        self.assertEqual(padded[0].dtype, torch.bfloat16)
        self.assertEqual(labels.tolist(), [0, 1, 0])
        self.assertEqual(lengths.tolist(), [4, 2, 1])
        self.assertTrue(torch.equal(padded[0][1, :2], expected_seq(1, 0, 2)))
        self.assertEqual(float(padded[0][1, 2:].abs().sum()), 0.0)

    def test_lengths_and_defaults(self):
        store = st.SequenceStore(self.backend, "spec", self.manifest, self.meta)
        ds = st.RolloutSequenceDataset(["m:run:h:0"], [np.int64(1)], store)
        self.assertEqual((ds.labels, ds.weights), ([1], [1.0]))
        self.assertIs(type(ds.labels[0]), int)
        with self.assertRaisesRegex(ValueError, "labels has 2 entries"):
            st.RolloutSequenceDataset(["m:run:h:0"], [1, 0], store)
        with self.assertRaisesRegex(ValueError, "weights has 1 entries"):
            st.RolloutSequenceDataset(["m:run:h:0", "m:run:h:1"], [1, 0], store, weights=[1.0])
        with self.assertRaisesRegex(KeyError, "m:run:h:9"):
            st.RolloutSequenceDataset(["m:run:h:9"], [1], store)


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------


class TestMirrorStore(unittest.TestCase):
    """plan_mirror / mirror_store (a fake HF backend over a local two-part store)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.local, self.parts = write_store(self.tmp / "src")
        self.hf = hf_backend_over(self.local)
        self.api = self.hf.api
        self.dst = self.tmp / "mirror"
        self.addCleanup(st.reset_read_caches)

    def _names(self, kind, layer):
        return sorted(p.seq_file(layer) if kind == "seq" else p.points_file(layer) for p in self.parts)

    def _local_files(self):
        return sorted(p.name for p in (self.dst / "spec").iterdir())

    def test_mirror_store_copies_points_and_selected_layers(self):
        log = []
        report = st.mirror_store(self.hf, "spec", self.dst, layers=[0], log=log.append)
        expected = ["manifest.csv", "manifest.meta.json", *self._names("seq", 0),
                    *self._names("points", 0), *self._names("points", 1)]
        self.assertEqual(self._local_files(), sorted(expected))
        self.assertEqual((report["n_downloaded"], report["n_skipped"], report["mirrored_layers"],
                          report["mirrored_points_layers"], report["dropped"]), (6, 0, [0], [0, 1], {}))
        self.assertEqual(report["bytes_downloaded"], sum((self.local.root / "spec" / n).stat().st_size for n in expected[2:]))
        self.assertTrue(any("6 file(s) to download" in l for l in log))
        meta = json.loads((self.dst / "spec" / "manifest.meta.json").read_text())
        self.assertEqual((meta["layers"], meta["mirrored_layers"], meta["mirrored_points_layers"]), ([0, 1], [0], [0, 1]))
        self.assertEqual((meta["mirror_source"]["backend"], meta["mirror_source"]["repo_id"], meta["mirror_source"]["revision"]),
                         ("hf", REPO, None))
        self.assertIn("mirrored_utc", meta["mirror_source"])
        self.assertEqual((self.dst / "spec" / "manifest.csv").read_bytes(), (self.local.root / "spec" / "manifest.csv").read_bytes())
        for name in expected[2:]:
            self.assertEqual((self.dst / "spec" / name).read_bytes(), (self.local.root / "spec" / name).read_bytes(), name)
        # Every part file went straight to the destination volume (local_dir set), none via the fake's cache.
        big = [(f, d) for f, d in self.api.downloads if f.endswith(".safetensors")]
        self.assertEqual(len(big), 6)
        self.assertTrue(all(d is not None and str(self.dst / "spec") in d for _, d in big))
        self.assertFalse([p for p in (self.dst / "spec").iterdir() if p.name.startswith(".tmp-")])
        self.assertFalse((self.dst / ".cache").exists())
        # Idempotent: nothing downloaded again.
        n_calls = len(self.api.downloads)
        report = st.mirror_store(self.hf, "spec", self.dst, layers=[0])
        self.assertEqual((report["n_downloaded"], report["n_skipped"], report["bytes_downloaded"]), (0, 6, 0))
        self.assertEqual([f for f, _ in self.api.downloads[n_calls:] if f.endswith(".safetensors")], [])
        # A truncated file is fetched again — only that one.
        victim = self.parts[1].points_file(1)
        data = (self.dst / "spec" / victim).read_bytes()
        (self.dst / "spec" / victim).write_bytes(data[:-3])
        n_calls = len(self.api.downloads)
        report = st.mirror_store(self.hf, "spec", self.dst, layers=[0])
        self.assertEqual((report["n_downloaded"], report["n_skipped"]), (1, 5))
        self.assertEqual([f for f, _ in self.api.downloads[n_calls:] if f.endswith(".safetensors")], [f"spec/{victim}"])
        self.assertEqual((self.dst / "spec" / victim).read_bytes(), data)
        # The copy reads back through the local backend, restricted to the mirrored layers.
        mirror = st.LocalBackend(self.dst)
        manifest, meta = st.read_store_manifest(mirror, "spec")
        seq = st.SequenceStore(mirror, "spec", manifest, meta)
        self.assertEqual(seq.layers, [0])
        torch.testing.assert_close(seq.get("m:run:h:2")[0], expected_seq(2, 0, 4))
        points = st.PointStore(mirror, "spec", manifest, meta)
        self.assertEqual(points.layers, [0, 1])
        got = points.get_many(["m:run:h:3", "m:run:h:0"], "pre_answer")
        self.assertEqual(got[1].shape, (2, HIDDEN))
        self.assertEqual(float(got[1][0, 0]), point_value(3, 1, st.POINTS.index("pre_answer")))
        with self.assertRaisesRegex(st.LayerSelectionError, "not in this mirror"):
            st.read_store_manifest(mirror, "spec", layers=[1], kinds=("seq",))
        # Adding layer 1 later: only its seq files are new.
        n_calls = len(self.api.downloads)
        report = st.mirror_store(self.hf, "spec", self.dst, layers=[1])
        self.assertEqual((report["n_downloaded"], report["n_skipped"], report["mirrored_layers"]), (2, 4, [0, 1]))
        self.assertEqual(sorted(f for f, _ in self.api.downloads[n_calls:] if f.endswith(".safetensors")),
                         [f"spec/{n}" for n in self._names("seq", 1)])
        st.reset_read_caches()
        _, meta = st.read_store_manifest(mirror, "spec", layers=[0, 1])
        self.assertEqual(meta["mirrored_layers"], [0, 1])

    def test_mirror_refuses_unrequested_seq_download(self):
        plan = st.plan_mirror(self.hf, "spec", self.dst, layers=[0])
        self.assertEqual(sorted(f.name for f in plan.files if f.kind == "seq"), self._names("seq", 0))
        self.assertEqual({f.action for f in plan.files}, {"download"})
        st.mirror_store(self.hf, "spec", self.dst, layers=[0])
        fetched = [f for f, _ in self.api.downloads if "seq_layer_1" in f]
        self.assertEqual(fetched, [])
        self.assertFalse([n for n in self._local_files() if n.startswith("seq_layer_1")])
        # A local source behaves the same and is recorded as such.
        dst2 = self.tmp / "mirror2"
        report = st.mirror_store(self.local, "spec", dst2, layers=[0], include_points=False)
        self.assertEqual(sorted(p.name for p in (dst2 / "spec").iterdir()),
                         sorted(["manifest.csv", "manifest.meta.json", *self._names("seq", 0)]))
        self.assertEqual((report["mirrored_layers"], report["mirrored_points_layers"]), ([0], []))
        meta = json.loads((dst2 / "spec" / "manifest.meta.json").read_text())
        self.assertEqual((meta["mirror_source"]["backend"], meta["mirror_source"]["root"]), ("local", str(self.local.root)))
        st.read_store_manifest(st.LocalBackend(dst2), "spec")
        with self.assertRaisesRegex(st.LayerSelectionError, "not in this mirror.*points"):
            st.read_store_manifest(st.LocalBackend(dst2), "spec", layers=[0], kinds=("points",))
        # A hand-built plan with a seq file of an unrequested layer is refused before any download.
        bad = st.plan_mirror(self.hf, "spec", self.tmp / "mirror3", layers=[0, 1])
        bad.layers = [0]
        with self.assertRaisesRegex(ValueError, r"not requested: \[1\]"):
            st.mirror_store(self.hf, "spec", self.tmp / "mirror3", plan=bad)
        self.assertFalse((self.tmp / "mirror3").exists())

    def test_mirror_verifies_sizes_and_resumes(self):
        victim = f"spec/{self.parts[0].seq_file(0)}"
        self.api.short_once.add(victim)
        report = st.mirror_store(self.hf, "spec", self.dst, layers=[0])
        self.assertEqual(report["n_downloaded"], 6)
        self.assertEqual([f for f, _ in self.api.downloads].count(victim), 2)
        self.assertEqual((self.dst / self.parts[0].seq_file(0).join(["spec/", ""])).read_bytes(), self.api.files[victim])
        # Always short: reported after the others finished, manifest untouched, rerun resumes.
        dst2 = self.tmp / "mirror2"
        self.api.short_always.add(victim)
        log = []
        with self.assertRaisesRegex(RuntimeError, r"1 of 6 file\(s\) failed.*seq_layer_0_part00.*downloaded \d+ bytes"):
            st.mirror_store(self.hf, "spec", dst2, layers=[0], log=log.append)
        present = sorted(p.name for p in (dst2 / "spec").iterdir())
        self.assertNotIn("manifest.csv", present)
        self.assertNotIn(self.parts[0].seq_file(0), present)
        self.assertEqual(len(present), 5)
        self.assertFalse([n for n in present if n.startswith(".tmp-")])
        self.api.short_always.clear()
        report = st.mirror_store(self.hf, "spec", dst2, layers=[0])
        self.assertEqual((report["n_downloaded"], report["n_skipped"]), (1, 5))
        st.read_store_manifest(st.LocalBackend(dst2), "spec", layers=[0])

    def test_mirror_plan_validation_and_disk_check(self):
        with self.assertRaisesRegex(ValueError, "nothing to mirror"):
            st.plan_mirror(self.hf, "spec", self.dst, layers=None, include_points=False)
        with self.assertRaisesRegex(st.LayerSelectionError, r"\[7\].*not stored"):
            st.plan_mirror(self.hf, "spec", self.dst, layers=[7])
        with self.assertRaisesRegex(FileNotFoundError, "nowhere"):
            st.plan_mirror(self.hf, "nowhere", self.dst, layers=[0])
        plan = st.plan_mirror(self.hf, "spec", self.dst, layers=None)
        self.assertEqual([f.kind for f in plan.files], ["points"] * 4)
        self.assertEqual(plan.layers, [])
        summary = plan.summary()
        self.assertEqual((summary["n_files"], summary["n_download"], summary["n_skip"]), (4, 4, 0))
        self.assertEqual(set(summary["points_bytes_by_layer"]), {0, 1})
        self.assertEqual(summary["seq_bytes_by_layer"], {})
        self.assertEqual(summary["bytes_total"], sum(f.size for f in plan.files))
        self.assertIsInstance(plan.free_bytes, int)
        # The manifest names a file the listing lacks.
        del self.api.files[f"spec/{self.parts[1].points_file(1)}"]
        with self.assertRaisesRegex(ValueError, r"names 1 file\(s\) the listing lacks.*points_layer_1_part01"):
            st.plan_mirror(self.hf, "spec", self.dst, layers=[0])
        st.plan_mirror(self.hf, "spec", self.dst, layers=[0], include_points=False)
        # Above the free disk: refused before anything is written; check_disk=False proceeds.
        with mock.patch.object(st, "_free_bytes", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "GB free"):
                st.mirror_store(self.hf, "spec", self.dst, layers=[0], include_points=False)
            self.assertFalse(self.dst.exists())
            report = st.mirror_store(self.hf, "spec", self.dst, layers=[0], include_points=False, check_disk=False)
        self.assertEqual(report["n_downloaded"], 2)
        # A mirror as the source: a layer it lacks cannot be mirrored on.
        with self.assertRaisesRegex(st.LayerSelectionError, "not in this mirror"):
            st.plan_mirror(st.LocalBackend(self.dst), "spec", self.tmp / "m2", layers=[1])

    def test_mirror_tracks_source_growth(self):
        st.mirror_store(self.hf, "spec", self.dst, layers=[0])
        # The source gains a part (fake bytes: the reader checks presence, the mirror copies bytes).
        manifest = pd.read_csv(self.local.root / "spec" / "manifest.csv")
        row = manifest.iloc[[0]].copy()
        row["rollout_id"] = "m:run:h:9"
        row["seq_file"], row["points_file"] = st.part_template("seq", 2), st.part_template("points", 2)
        self.api.files["spec/manifest.csv"] = pd.concat([manifest, row]).to_csv(index=False).encode()
        for layer in (0, 1):
            self.api.files[f"spec/{st.seq_file(layer, 2)}"] = b"S" * (10 + layer)
            self.api.files[f"spec/{st.points_file(layer, 2)}"] = b"P" * (20 + layer)
        n_calls = len(self.api.downloads)
        log = []
        report = st.mirror_store(self.hf, "spec", self.dst, layers=[1], log=log.append)
        new = sorted(f for f, _ in self.api.downloads[n_calls:] if f.endswith(".safetensors"))
        self.assertEqual(new, sorted([f"spec/{st.seq_file(1, p)}" for p in (0, 1, 2)]
                                     + [f"spec/{st.points_file(l, 2)}" for l in (0, 1)]))
        self.assertEqual((report["mirrored_layers"], report["mirrored_points_layers"], report["dropped"]),
                         ([1], [0, 1], {"seq": [0]}))
        self.assertTrue(any("[warn] seq layers [0]" in l for l in log))
        meta = json.loads((self.dst / "spec" / "manifest.meta.json").read_text())
        self.assertEqual(meta["mirrored_layers"], [1])
        self.assertIn("m:run:h:9", (self.dst / "spec" / "manifest.csv").read_text())
        report = st.mirror_store(self.hf, "spec", self.dst, layers=[0])
        self.assertEqual((report["n_downloaded"], report["mirrored_layers"], report["dropped"]), (1, [0, 1], {}))

    def test_mirror_respects_seq_layers(self):
        local, parts = write_store(self.tmp / "src_seq", seq_layers=[1])
        hf = hf_backend_over(local)
        dst = self.tmp / "m_seq"
        with self.assertRaisesRegex(st.LayerSelectionError, r"layers \[0\] have no seq files.*seq_layers: \[1\]"):
            st.plan_mirror(hf, "spec", dst, layers=[0])
        with self.assertRaisesRegex(st.LayerSelectionError, "have no seq files"):
            st.plan_mirror(hf, "spec", dst, layers=[0, 1])
        plan = st.plan_mirror(hf, "spec", dst, layers=None)
        self.assertEqual(sorted({(f.kind, f.layer) for f in plan.files}), [("points", 0), ("points", 1)])
        report = st.mirror_store(hf, "spec", dst, layers=[1])
        self.assertEqual((report["mirrored_layers"], report["mirrored_points_layers"]), ([1], [0, 1]))
        self.assertEqual(sorted(p.name for p in (dst / "spec").iterdir()),
                         sorted(["manifest.csv", "manifest.meta.json"] + [p.seq_file(1) for p in parts]
                                + [p.points_file(l) for p in parts for l in (0, 1)]))
        meta = json.loads((dst / "spec" / "manifest.meta.json").read_text())
        self.assertEqual((meta["layers"], meta["seq_layers"], meta["mirrored_layers"]), ([0, 1], [1], [1]))
        copy = st.LocalBackend(dst)
        manifest, meta = st.read_store_manifest(copy, "spec")
        self.assertEqual(st.SequenceStore(copy, "spec", manifest, meta).layers, [1])
        with self.assertRaisesRegex(st.LayerSelectionError, "have no seq files"):
            st.plan_mirror(copy, "spec", self.tmp / "m2", layers=[0])


if __name__ == "__main__":
    unittest.main()
