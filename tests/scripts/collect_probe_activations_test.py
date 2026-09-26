"""Tests for src/scripts/collect_probe_activations.py — GPU, network and model faked."""

import contextlib
import importlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from safetensors.torch import load as safetensors_load

from src.lib import activation_store as st
from src.lib.config import load_config
from src.lib.parsing import DEFAULT_DELIMITERS
from src.lib.paths import REPO_ROOT
from src.lib.probe_datasets import DATASET_COLUMNS, dataset_fingerprint

MODULE_NAME = "src.scripts.collect_probe_activations"
mod = importlib.import_module(MODULE_NAME)

CLOSE = DEFAULT_DELIMITERS.close
OPEN = DEFAULT_DELIMITERS.open
MODEL = "fake/fake-model"
SHORT = "fake-model"
RUN = "gpqa-extended_0909"
SPEC_NAME = "fake-model_test_spec"
HIDDEN = 8

FAKE_MODEL_CONFIG = {
    "short_name": SHORT,
    "thinking_mode": "prompted",
    "thinking_default": True,
    "delimiters": DEFAULT_DELIMITERS,
    "vllm_extra_kwargs": {},
    "hf_auto_class": None,
    "layer_stack_path": None,
    "n_layers": 4,
    "hidden_size": HIDDEN,
    "supports_thinking_off": True,
}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Character-level tokenizer with the delimiters as single ids; ``preinject``
    appends the open delimiter to the generation prompt."""

    SPECIAL = {OPEN: 1001, CLOSE: 1002}

    def __init__(self, preinject=True):
        self.preinject = preinject
        self.template_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.template_calls.append({"messages": messages, "kwargs": kwargs})
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + "<assistant>" + (f"{OPEN}\n" if self.preinject else "")

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
    def forward(self, x):
        return x + 1.0


class FakeModel(torch.nn.Module):
    """Layer i (0-based) outputs position + i + 1 in every hidden dim."""

    def __init__(self, n_layers=4, hidden=HIDDEN):
        super().__init__()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList(FakeLayer() for _ in range(n_layers))
        self.model = inner
        self.config = SimpleNamespace(num_hidden_layers=n_layers, hidden_size=hidden)
        self.hidden = hidden
        self.device = torch.device("cpu")
        self.calls = []

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        self.calls.append((input_ids.clone(), {"use_cache": use_cache, **kwargs}))
        seq = input_ids.shape[1]
        h = torch.arange(seq, dtype=torch.float32).view(1, seq, 1).expand(1, seq, self.hidden).clone()
        for layer in self.model.layers:
            h = layer(h)
        return h


class FakeOomModel(FakeModel):
    """A FakeModel with Mamba-like ``chunk_size`` modules whose forward OOMs on its first ``fail_first`` calls."""

    def __init__(self, fail_first: int, **kw):
        super().__init__(**kw)
        self.mixer = torch.nn.Module()
        self.mixer.chunk_size = 64
        self.model.layers[1].mixer2 = torch.nn.Module()
        self.model.layers[1].mixer2.chunk_size = 64
        self.fail_first = fail_first
        self.chunk_calls: list[int] = []

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        self.chunk_calls.append(self.mixer.chunk_size)
        if len(self.chunk_calls) <= self.fail_first:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 35.00 GiB")
        return super().forward(input_ids=input_ids, use_cache=use_cache, **kwargs)


# ---------------------------------------------------------------------------
# Test data: a tiny probe dataset parquet
# ---------------------------------------------------------------------------

STYLES = ("grader_hacking", "metadata")


def cot_text(i: int) -> str:
    return f"cot {i} abc"


def rollout_text(i: int, *, tail="\n\nAnswer: ", commit="<answer>B</answer>", close=True, cot=None) -> str:
    cot = cot_text(i) if cot is None else cot
    return f"{cot}{CLOSE if close else ''}{tail}{commit}\n"


def dataset_frame(n: int, *, model=SHORT, run=RUN, rollouts=None) -> pd.DataFrame:
    rows = []
    for i in range(n):
        style = STYLES[i % len(STYLES)]
        row = {c: None for c in DATASET_COLUMNS}
        row.update({
            "rollout_id": f"{model}:{run}:{style}:{i}#rs43", "question_id": f"gpqa:train:{i}",
            "dataset": "gpqa", "dataset_split": "train", "original_index": i, "subject_model": model,
            "subject_model_id": MODEL, "run": run, "hint_style": style, "case": ("positive", "negative")[i % 2],
            "target_option": "B", "groundtruth": "A", "split": ("train", "val", "test")[i % 3],
            "source_csv": "x.csv", "provenance": "resample_k4", "is_resample": True, "sample_seed": 43,
            "label": i % 2, "balance_kept": True, "prompt": f"Q{i}?", "hinted_prompt": f"Q{i}? [hint B]",
            "reasoning": cot_text(i), "rollout": (rollouts or {}).get(i, rollout_text(i)),
            "choices": json.dumps(["a", "b", "c", "d"]), "option_letters": json.dumps(list("ABCD")),
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=list(DATASET_COLUMNS))


def write_dataset_files(tmp: Path, df: pd.DataFrame, *, name=SPEC_NAME) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    parquet = tmp / f"{name}.parquet"
    df.to_parquet(parquet, index=False)
    meta = {"name": name, "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": dataset_fingerprint(df), "n_rows": int(len(df))}
    (tmp / f"{name}.meta.json").write_text(json.dumps(meta))
    return parquet


def base_cfg(tmp: Path, parquet: Path, **overrides) -> dict:
    cfg = {
        "model_name": MODEL, "seed": 7, "thinking": True,
        "dataset": str(parquet),
        "storage": {"backend": "local", "local_dir": str(tmp / "store")},
        "layers": [0, 2],
        # ~1000 bytes per file: 3 rollouts of ~11 CoT chars × 8 × 2 bytes per part
        "max_file_gb": 1000 / st.GB,
        "upload_workers": 2,
    }
    cfg.update(overrides)
    return cfg


def patched_registry():
    return mock.patch.object(mod, "get_model_config", lambda name: dict(FAKE_MODEL_CONFIG))


def thinking_args(tokenizer=None):
    return dict(tokenizer=tokenizer or FakeTokenizer(), system_prompt="SYS", enable_thinking=None,
                delimiters=DEFAULT_DELIMITERS)


# ---------------------------------------------------------------------------
# build_item
# ---------------------------------------------------------------------------


class TestBuildItem(unittest.TestCase):
    """build_item: point indices, commit forms, the head rule and the skip reasons."""

    def row(self, i=0, **overrides):
        r = dataset_frame(1).iloc[0].to_dict()
        r.update(overrides)
        return r

    def test_build_item_positions(self):
        tok = FakeTokenizer()
        prompt = tok.apply_chat_template([{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q0? [hint B]"}])
        n_prompt = len(tok(prompt)["input_ids"])
        cot = cot_text(0)
        cases = [
            ("<answer>B</answer>", "\n\nAnswer: ", "answer_tag"),
            ("\\boxed{\\text{B}}", " so ", "boxed"),
            ("<answer>B</answer>", "", "answer_tag"),
            ('{"answer": "B"}', "\n", "json"),
            ("<answer>B) b</answer>", " ", "labelled_tag"),
        ]
        for commit, tail, form in cases:
            with self.subTest(commit=commit, tail=repr(tail)):
                item = mod.build_item(self.row(rollout=rollout_text(0, tail=tail, commit=commit)), **thinking_args(tok))
                self.assertIsInstance(item, mod.Item)
                n_cot = len(cot)
                n_seq = n_prompt + n_cot + 1 + len(tail)
                self.assertEqual(item.cot_span, (n_prompt, n_prompt + n_cot))
                self.assertEqual(item.point_idx, (n_prompt - 1, n_prompt + n_cot - 1, n_seq - 1))
                self.assertEqual((item.n_seq_tokens, item.n_cot_tokens, item.commit_form), (n_seq, n_cot, form))
                ids = item.input_ids.tolist()
                self.assertEqual(ids[n_prompt + n_cot], FakeTokenizer.SPECIAL[CLOSE])
                self.assertEqual("".join(chr(i) for i in ids[n_prompt + n_cot + 1:]), tail)
                self.assertEqual(ids[-1], FakeTokenizer.SPECIAL[CLOSE] if not tail else ord(tail[-1]))
                self.assertEqual(item.input_ids.dtype, np.int32)
                self.assertEqual((item.hint_style, item.case, item.split, item.subject_model),
                                 ("grader_hacking", "positive", "train", SHORT))

    def test_head_kept_for_preinjected_open_delimiter(self):
        # Pre-injected open delimiter: the head is empty, pre_cot is the template's last token.
        tok = FakeTokenizer(preinject=True)
        item = mod.build_item(self.row(), **thinking_args(tok))
        start = item.cot_span[0]
        self.assertEqual(item.input_ids[start - 2:start].tolist(), [FakeTokenizer.SPECIAL[OPEN], ord("\n")])
        self.assertEqual(item.point_idx[0], start - 1)
        # Model-emitted open delimiter: the rollout's head is kept in the prompt piece.
        tok = FakeTokenizer(preinject=False)
        rollout = f"{OPEN}\n" + rollout_text(0)
        item2 = mod.build_item(self.row(rollout=rollout), **thinking_args(tok))
        self.assertIsInstance(item2, mod.Item)
        start2 = item2.cot_span[0]
        self.assertEqual(item2.input_ids[start2 - 2:start2].tolist(), [FakeTokenizer.SPECIAL[OPEN], ord("\n")])
        self.assertEqual(item2.point_idx[0], start2 - 1)
        self.assertEqual(item2.n_cot_tokens, item.n_cot_tokens)
        # Leading text without the open delimiter is not a head (build_item's rule).
        item3 = mod.build_item(self.row(rollout="\n\n" + rollout_text(0)), **thinking_args(tok))
        self.assertEqual(item3.cot_span[0], len(tok(tok.apply_chat_template(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q0? [hint B]"}]))["input_ids"]))

    def test_skip_without_commit_or_close(self):
        cases = {
            "no close": (rollout_text(0, close=False), mod.SKIP_NO_CLOSE),
            "commit only inside the CoT": (f"{cot_text(0)} <answer>B</answer> more", mod.SKIP_NO_CLOSE),
            "close without a commit after it": (f"{cot_text(0)}{CLOSE} nothing here", mod.SKIP_NO_CLOSE),
            "quoted close inside the cot, real one later": (
                f"{cot_text(0)} quoting {CLOSE} <answer>A</answer> then{CLOSE}<answer>B</answer>", None),
        }
        for name, (rollout, reason) in cases.items():
            with self.subTest(name):
                out = mod.build_item(self.row(rollout=rollout), **thinking_args())
                if reason is None:
                    self.assertIsInstance(out, mod.Item)
                else:
                    self.assertIsInstance(out, mod.Skip)
                    self.assertEqual(out.reason, reason)
                    self.assertEqual(out.rollout_id, self.row()["rollout_id"])
        # The quoted case: the CoT of the dataset must be the text before the REAL close.
        real_cot = f"{cot_text(0)} quoting {CLOSE} <answer>A</answer> then"
        out = mod.build_item(self.row(rollout=f"{real_cot}{CLOSE}<answer>B</answer>", reasoning=real_cot), **thinking_args())
        self.assertIsInstance(out, mod.Item)
        self.assertEqual(out.n_cot_tokens, len(tok_ids(real_cot)))
        out = mod.build_item(self.row(reasoning=""), **thinking_args())
        self.assertEqual(out.reason, mod.SKIP_EMPTY_COT)
        out = mod.build_item(self.row(reasoning=None), **thinking_args())
        self.assertEqual(out.reason, mod.SKIP_EMPTY_COT)
        out = mod.build_item(self.row(reasoning="something else entirely"), **thinking_args())
        self.assertEqual(out.reason, mod.SKIP_COT_NOT_IN_ROLLOUT)
        # A reasoning that only appears AFTER the close is not the CoT either.
        out = mod.build_item(self.row(rollout=f"x{CLOSE} {cot_text(0)} <answer>B</answer>"), **thinking_args())
        self.assertEqual(out.reason, mod.SKIP_COT_NOT_IN_ROLLOUT)

    def test_skip_too_long_sequence(self):
        row = self.row()
        ok = mod.build_item(row, **thinking_args())
        out = mod.build_item(row, **thinking_args(), max_positions=ok.n_seq_tokens - 1)
        self.assertIsInstance(out, mod.Skip)
        self.assertTrue(out.reason.startswith(mod.SKIP_TOO_LONG))
        self.assertIn(str(ok.n_seq_tokens), out.reason)
        self.assertIsInstance(mod.build_item(row, **thinking_args(), max_positions=ok.n_seq_tokens), mod.Item)


def tok_ids(text):
    return FakeTokenizer()(text)["input_ids"]


# ---------------------------------------------------------------------------
# capture_points
# ---------------------------------------------------------------------------


class TestCapturePoints(unittest.TestCase):
    """ActivationCapturer.capture_points through a built item."""

    def test_capture_points_and_sequence_consistent(self):
        item = mod.build_item(dataset_frame(1).iloc[0].to_dict(), **thinking_args())
        model = FakeModel()
        cap = st.ActivationCapturer(model, model.model.layers, [0, 2], expected_hidden=HIDDEN)
        seq, pts = cap.capture_points(item.input_ids, cot_span=item.cot_span, point_idx=item.point_idx)
        start, end = item.cot_span
        pre_cot, last_cot, pre_answer = item.point_idx
        for layer in (0, 2):
            self.assertEqual(tuple(seq[layer].shape), (end - start, HIDDEN))
            self.assertEqual(tuple(pts[layer].shape), (len(st.POINTS), HIDDEN))
            self.assertEqual(seq[layer].dtype, torch.bfloat16)
            self.assertEqual(pts[layer].dtype, torch.bfloat16)
            expected_seq = torch.arange(start, end, dtype=torch.float32).view(-1, 1).expand(-1, HIDDEN) + layer + 1
            torch.testing.assert_close(seq[layer].float(), expected_seq)
            for k, idx in zip(range(3), (pre_cot, last_cot, pre_answer)):
                torch.testing.assert_close(pts[layer][k].float(), torch.full((HIDDEN,), float(idx + layer + 1)))
            torch.testing.assert_close(pts[layer][3].float(), expected_seq.mean(dim=0), rtol=1e-2, atol=1e-2)
            self.assertTrue(torch.equal(pts[layer][st.POINTS.index("last_cot")], seq[layer][-1]))
        self.assertEqual(model.calls[0][1].get("logits_to_keep"), 1)
        with self.assertRaisesRegex(ValueError, "last_cot"):
            cap.capture_points(item.input_ids, cot_span=item.cot_span, point_idx=(pre_cot, last_cot - 1, pre_answer))
        with self.assertRaisesRegex(ValueError, "outside"):
            cap.capture_points(item.input_ids, cot_span=item.cot_span, point_idx=(pre_cot, last_cot, item.n_seq_tokens))


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun(unittest.TestCase):
    """run() / verify_store / parse_config on a local store."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.df = dataset_frame(7)
        self.parquet = write_dataset_files(self.tmp / "ds", self.df)
        self.cfg = base_cfg(self.tmp, self.parquet)
        self.root = self.tmp / "store"

    def run_script(self, cfg=None, model=None, tokenizer=None, **kw):
        cc = mod.parse_config(cfg or self.cfg)
        model = model or FakeModel()
        tokenizer = tokenizer or FakeTokenizer()
        kw.setdefault("max_positions", 100_000)
        kw.setdefault("verify_sample", 5)
        with patched_registry(), contextlib.redirect_stdout(io.StringIO()) as buf:
            summary = mod.run(cc, tokenizer=tokenizer, model_loader=lambda name, **kw: (model, None), **kw)
        return summary, buf.getvalue(), model

    def backend(self):
        return st.LocalBackend(self.root)

    def test_end_to_end_local_two_parts_and_manifest(self):
        summary, out, model = self.run_script()
        self.assertEqual(summary["folder"], SPEC_NAME)
        self.assertEqual((summary["n_items"], summary["n_skipped"], summary["failed_parts"]), (7, 0, []))
        self.assertGreaterEqual(len(summary["parts"]), 2)
        self.assertTrue(summary["verify"]["ok"])
        self.assertIn("staging: memory", out)
        manifest, meta = st.read_store_manifest(self.backend(), SPEC_NAME)
        self.assertEqual(list(manifest.columns), st.STORE_MANIFEST_COLUMNS)
        self.assertEqual(sorted(manifest["rollout_id"]), sorted(self.df["rollout_id"]))
        self.assertEqual(sorted(manifest["seq_file"].unique()), [f"seq_layer_{{L}}_part{p['index']:02d}" for p in summary["parts"]])
        files = set(self.backend().list_files(f"{SPEC_NAME}/"))
        for p in summary["parts"]:
            for layer in (0, 2):
                self.assertIn(f"{SPEC_NAME}/seq_layer_{layer}_part{p['index']:02d}.safetensors", files)
                self.assertIn(f"{SPEC_NAME}/points_layer_{layer}_part{p['index']:02d}.safetensors", files)
        self.assertEqual(meta["model_name"], MODEL)
        self.assertEqual((meta["short_name"], meta["hidden_size"], meta["layers"], meta["dtype"]),
                         (SHORT, HIDDEN, [0, 2], "bfloat16"))
        self.assertEqual(meta["points"], list(st.POINTS))
        self.assertEqual(meta["span"], "chain_of_thought")
        ds_meta = json.loads((self.tmp / "ds" / f"{SPEC_NAME}.meta.json").read_text())
        self.assertEqual(meta["dataset_spec"], {"name": SPEC_NAME, "built_utc": ds_meta["built_utc"],
                                                "fingerprint": ds_meta["fingerprint"], "n_rows": 7})
        self.assertEqual((meta["n_rollouts"], meta["n_skipped"], meta["skipped"]), (7, 0, {}))
        self.assertEqual(sorted(meta["files"]), sorted(f[len(SPEC_NAME) + 1:] for f in files if st.is_store_file(f)))
        self.assertEqual(meta["thinking_mode"], "prompted")
        self.assertEqual(meta["reasoning_delimiters"], [OPEN, CLOSE])
        self.assertEqual(meta["layer_stack_path"], "model.layers")
        self.assertIn("collected_utc", meta)
        # Tensors read back through the reader side equal a fresh capture.
        seq_store = st.SequenceStore(self.backend(), SPEC_NAME, manifest, meta)
        point_store = st.PointStore(self.backend(), SPEC_NAME, manifest, meta)
        cap = st.ActivationCapturer(model, model.model.layers, [0, 2], expected_hidden=HIDDEN)
        for row in self.df.to_dict("records"):
            item = mod.build_item(row, tokenizer=FakeTokenizer(), system_prompt=mod.get_thinking_config(MODEL, True).system_prompt,
                                  enable_thinking=None, delimiters=DEFAULT_DELIMITERS)
            seq, pts = cap.capture_points(item.input_ids, cot_span=item.cot_span, point_idx=item.point_idx)
            got_seq = seq_store.get(item.rollout_id)
            m = manifest.set_index("rollout_id").loc[item.rollout_id]
            self.assertEqual((int(m["n_cot_tokens"]), int(m["n_seq_tokens"])), (item.n_cot_tokens, item.n_seq_tokens))
            self.assertEqual((int(m["pre_cot_idx"]), int(m["last_cot_idx"]), int(m["pre_answer_idx"])), item.point_idx)
            self.assertEqual((m["commit_form"], m["hint_style"], m["case"], m["split"], m["subject_model"]),
                             ("answer_tag", row["hint_style"], row["case"], row["split"], SHORT))
            for layer in (0, 2):
                self.assertTrue(torch.equal(got_seq[layer], seq[layer]))
                for point in st.POINTS:
                    self.assertTrue(torch.equal(point_store.get(item.rollout_id, point)[layer],
                                                pts[layer][st.POINTS.index(point)]))
                self.assertTrue(torch.equal(point_store.get(item.rollout_id, "last_cot")[layer], got_seq[layer][-1]))
        report = mod.verify_store(self.backend(), SPEC_NAME, self.parquet, sample=20, tokenizer=FakeTokenizer())
        self.assertEqual(report["n_checked"], 7)

    def test_resume_requires_same_fingerprint(self):
        self.run_script()
        # No flag: an existing store is refused.
        with self.assertRaisesRegex(ValueError, "already holds 7 rollouts"):
            self.run_script()
        # A rewritten dataset (one label flipped) has a new fingerprint: --resume refuses.
        df2 = self.df.copy()
        df2.loc[0, "label"] = 1 - int(df2.loc[0, "label"])
        write_dataset_files(self.tmp / "ds", df2)
        with self.assertRaisesRegex(ValueError, "dataset_spec.fingerprint"):
            self.run_script(resume=True)
        # --force rewrites the folder against the new dataset: same rows, stale parts gone.
        cfg = dict(self.cfg, max_file_gb=10.0)   # one part now: the old part01 files are stale
        summary, _, _ = self.run_script(cfg, force=True)
        self.assertEqual(len(summary["parts"]), 1)
        files = [f for f in self.backend().list_files(f"{SPEC_NAME}/") if st.is_store_file(f)]
        self.assertEqual(sorted(files), sorted(f"{SPEC_NAME}/{k}_layer_{l}_part00.safetensors" for k in ("seq", "points") for l in (0, 2)))
        manifest, meta = st.read_store_manifest(self.backend(), SPEC_NAME)
        self.assertEqual(len(manifest), 7)
        self.assertEqual(meta["dataset_spec"]["fingerprint"], dataset_fingerprint(df2))
        with self.assertRaisesRegex(ValueError, "exclusive"):
            self.run_script(cfg, force=True, resume=True)

    def test_resume_skips_stored_rollouts(self):
        first, _, _ = self.run_script(limit=3)
        self.assertEqual(first["n_items"], 3)
        manifest1, meta1 = st.read_store_manifest(self.backend(), SPEC_NAME)
        files1 = {f for f in self.backend().list_files(f"{SPEC_NAME}/") if st.is_store_file(f)}
        # A file an earlier run failed to upload leaves failed_files once a rerun writes it.
        stale_failed = st.seq_file(0, mod.next_part_index(manifest1))
        meta_path = self.root / SPEC_NAME / st.STORE_MANIFEST_META_NAME
        meta_path.write_text(json.dumps({**meta1, "failed_files": [stale_failed, "seq_layer_0_part99.safetensors"]}))
        # Another thinking mode or delimiter pair is an identity mismatch for --resume.
        identity = mod.store_identity(mod.parse_config(self.cfg), hidden=HIDDEN, layer_stack_path=None,
                                      thinking=mod.get_thinking_config(MODEL, True), ds_meta={"fingerprint": dataset_fingerprint(self.df)},
                                      collected_utc="now")
        with patched_registry():
            self.assertEqual(mod.resume_mismatches(meta1, identity), [])
            self.assertEqual(mod.resume_mismatches({**meta1, "thinking_mode": "native"}, identity), ["thinking_mode"])
            self.assertEqual(mod.resume_mismatches({**meta1, "reasoning_delimiters": ["<a>", "</a>"]}, identity),
                             ["reasoning_delimiters"])
        second, out, _ = self.run_script(resume=True)
        self.assertEqual((second["n_resumed"], second["n_items"]), (3, 4))
        self.assertIn("3 already stored", out)
        manifest2, meta2 = st.read_store_manifest(self.backend(), SPEC_NAME)
        self.assertEqual(len(manifest2), 7)
        self.assertEqual(sorted(manifest2["rollout_id"]), sorted(self.df["rollout_id"]))
        first_parts = {st.part_index(v) for v in manifest1["seq_file"]}
        new_parts = {p["index"] for p in second["parts"]}
        self.assertTrue(min(new_parts) > max(first_parts))
        files2 = {f for f in self.backend().list_files(f"{SPEC_NAME}/") if st.is_store_file(f)}
        self.assertTrue(files1 <= files2)
        self.assertEqual(meta2["n_rollouts"], 7)
        self.assertIn(stale_failed, meta2["files"])
        self.assertEqual(meta2["failed_files"], ["seq_layer_0_part99.safetensors"])
        self.assertTrue(second["verify"]["ok"])
        # Everything stored: --resume has nothing left to do and writes nothing.
        third, out, _ = self.run_script(resume=True)
        self.assertEqual((third["n_items"], third["n_resumed"]), (0, 7))
        self.assertIn("Nothing to collect", out)

    def test_rejects_other_subject_model_rows(self):
        df = pd.concat([self.df, dataset_frame(2, model="other-model")], ignore_index=True)
        parquet = write_dataset_files(self.tmp / "ds2", df)
        with self.assertRaisesRegex(ValueError, "other-model"):
            self.run_script(base_cfg(self.tmp, parquet), dry_run=True)

    def test_dry_run_loads_no_model(self):
        cc = mod.parse_config(self.cfg)

        def no_model(name):
            raise AssertionError("model loaded on a dry run")

        with patched_registry(), contextlib.redirect_stdout(io.StringIO()) as buf:
            summary = mod.run(cc, dry_run=True, tokenizer=FakeTokenizer(), model_loader=no_model, max_positions=10_000)
        out = buf.getvalue()
        self.assertTrue(summary["dry_run"])
        self.assertEqual((summary["n_items"], summary["n_skipped"]), (7, 0))
        self.assertGreaterEqual(len(summary["parts"]), 2)
        self.assertIn("Dry run", out)
        self.assertIn("grader_hacking", out)
        self.assertIn("0 rows skipped", out)
        self.assertFalse(self.root.exists())
        # A skipped row is listed with its reason.
        df = self.df.copy()
        df.loc[1, "rollout"] = rollout_text(1, close=False)
        parquet = write_dataset_files(self.tmp / "ds3", df)
        with patched_registry(), contextlib.redirect_stdout(io.StringIO()) as buf:
            summary = mod.run(mod.parse_config(base_cfg(self.tmp, parquet)), dry_run=True,
                              tokenizer=FakeTokenizer(), model_loader=no_model, max_positions=10_000)
        self.assertEqual(summary["skipped"], {df.loc[1, "rollout_id"]: mod.SKIP_NO_CLOSE})
        self.assertIn(f"{df.loc[1, 'rollout_id']}: {mod.SKIP_NO_CLOSE}", buf.getvalue())
        # The skipped row never reaches the manifest, but the meta records it.
        summary, _, _ = self.run_script(base_cfg(self.tmp, parquet))
        manifest, meta = st.read_store_manifest(self.backend(), SPEC_NAME)
        self.assertEqual(len(manifest), 6)
        self.assertNotIn(df.loc[1, "rollout_id"], set(manifest["rollout_id"]))
        self.assertEqual((meta["n_skipped"], meta["skipped"]), (1, {df.loc[1, "rollout_id"]: mod.SKIP_NO_CLOSE}))

    def test_verify_store_detects_corrupted_index(self):
        self.run_script()
        path = self.root / SPEC_NAME / st.STORE_MANIFEST_NAME
        manifest = pd.read_csv(path)
        manifest.loc[0, "pre_answer_idx"] = int(manifest.loc[0, "pre_answer_idx"]) + 1
        manifest.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "pre_answer_idx"):
            mod.verify_store(self.backend(), SPEC_NAME, self.parquet, sample=20, tokenizer=FakeTokenizer())
        manifest.loc[0, "pre_answer_idx"] -= 1
        manifest.loc[1, "n_cot_tokens"] = int(manifest.loc[1, "n_cot_tokens"]) - 1
        manifest.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "n_cot_tokens"):
            mod.verify_store(self.backend(), SPEC_NAME, self.parquet, sample=20, tokenizer=FakeTokenizer())
        manifest.loc[1, "n_cot_tokens"] += 1
        manifest.to_csv(path, index=False)
        # A dataset whose fingerprint differs from the store's is reported too.
        df2 = self.df.copy()
        df2.loc[0, "label"] = 1 - int(df2.loc[0, "label"])
        other = write_dataset_files(self.tmp / "ds_other", df2)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            mod.verify_store(self.backend(), SPEC_NAME, other, sample=2, tokenizer=FakeTokenizer())
        self.assertTrue(mod.verify_store(self.backend(), SPEC_NAME, self.parquet, sample=20, tokenizer=FakeTokenizer())["ok"])

    def test_only_and_limit_filters(self):
        summary, _, _ = self.run_script(dry_run=True, only=["metadata"])
        self.assertEqual(summary["n_items"], sum(self.df["hint_style"] == "metadata"))
        summary, _, _ = self.run_script(dry_run=True, only=["gpqa"])
        self.assertEqual(summary["n_items"], 7)
        summary, _, _ = self.run_script(dry_run=True, only=["nothing_like_this"])
        self.assertEqual(summary["n_items"], 0)
        summary, _, _ = self.run_script(dry_run=True, limit=2)
        self.assertEqual(summary["n_items"], 2)
        rows = mod.select_rows(self.df, short_name=SHORT, limit=2)
        self.assertEqual(list(rows["rollout_id"]), sorted(self.df["rollout_id"])[:2])
        with self.assertRaisesRegex(ValueError, "limit"):
            mod.select_rows(self.df, short_name=SHORT, limit=0)

    def test_config_rejects_unknown_keys_and_thinking_off(self):
        with self.assertRaisesRegex(ValueError, "Unknown config keys.*rollouts"):
            mod.parse_config(dict(self.cfg, rollouts=[]))
        with self.assertRaisesRegex(ValueError, "Unknown config keys.*hf_repo_id"):
            mod.parse_config(dict(self.cfg, hf_repo_id="a/b"))
        for bad in ([], [-1], [1.5], [3, 3], None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                mod.parse_config(dict(self.cfg, layers=bad))
        with self.assertRaisesRegex(ValueError, "storage"):
            mod.parse_config(dict(self.cfg, storage={"backend": "ftp"}))
        with self.assertRaisesRegex(ValueError, "staging"):
            mod.parse_config(dict(self.cfg, staging="ram"))
        with self.assertRaisesRegex(ValueError, "hard per-file limit"):
            mod.parse_config(dict(self.cfg, max_file_gb=501))
        cc = mod.parse_config(dict(self.cfg, staging="disk", staging_dir=str(self.tmp / "stage"), layers=[2, 0],
                                   inference={"max_tokens": 8192}))
        self.assertEqual((cc.staging, cc.staging_dir, cc.layers), ("disk", self.tmp / "stage", [0, 2]))
        self.assertEqual(cc.dataset, self.parquet)
        self.assertIsNone(cc.store_folder)
        self.assertEqual(mod.parse_config(dict(self.cfg, store_folder=" shared/ ")).store_folder, "shared")
        self.assertIsNone(mod.parse_config(dict(self.cfg, store_folder=None)).store_folder)
        for bad in ("", "  ", "/", ".", "..", 3, "a/b", ["x"]):
            with self.assertRaisesRegex(ValueError, "store_folder", msg=repr(bad)):
                mod.parse_config(dict(self.cfg, store_folder=bad))
        cc = mod.parse_config(dict(self.cfg, thinking="off"))
        with patched_registry(), self.assertRaisesRegex(ValueError, "requires thinking"):
            mod.run(cc, dry_run=True, tokenizer=FakeTokenizer(), model_loader=lambda n, **kw: (FakeModel(), None))

    def test_hf_model_kwargs_config(self):
        self.assertEqual(mod.parse_config(self.cfg).hf_model_kwargs, {})
        self.assertEqual(mod.parse_config(dict(self.cfg, hf_model_kwargs=None)).hf_model_kwargs, {})
        cc = mod.parse_config(dict(self.cfg, hf_model_kwargs={"chunk_size": 64}))
        self.assertEqual(cc.hf_model_kwargs, {"chunk_size": 64})
        for bad in ([64], "chunk_size=64", {64: 1}, 5):
            with self.assertRaisesRegex(ValueError, "hf_model_kwargs", msg=repr(bad)):
                mod.parse_config(dict(self.cfg, hf_model_kwargs=bad))
        seen = {}

        def loader(name, **kw):
            seen.update(kw)
            return FakeModel(), None

        with patched_registry(), contextlib.redirect_stdout(io.StringIO()) as buf:
            mod.run(cc, tokenizer=FakeTokenizer(), model_loader=loader, max_positions=10_000, verify_sample=2)
        self.assertEqual(seen, {"model_kwargs": {"chunk_size": 64}})
        self.assertIn("hf_model_kwargs={'chunk_size': 64}", buf.getvalue())
        _, meta = st.read_store_manifest(self.backend(), SPEC_NAME)
        self.assertEqual(meta["hf_model_kwargs"], {"chunk_size": 64})
        real = mod.parse_config(load_config(REPO_ROOT / "configs" / "collect_probe_activations_nemotron.yaml"))
        self.assertEqual(real.hf_model_kwargs, {})

    def test_seq_layers_subset(self):
        cfg = dict(self.cfg, layers=[0, 1, 2], seq_layers=[1])
        # Validation: null = every layer; else a non-empty subset of `layers`, ints, no duplicates, sorted.
        self.assertEqual(mod.parse_config(dict(cfg, seq_layers=None)).seq_layers, [0, 1, 2])
        self.assertEqual(mod.parse_config(self.cfg).seq_layers, [0, 2])
        self.assertEqual(mod.parse_config(dict(cfg, seq_layers=[2, 0])).seq_layers, [0, 2])
        for bad, pattern in (([], "non-empty"), ([3], r"\[3\] are not in `layers`"), ([1, 1], "duplicates"),
                             ([True], "integers"), ("1", "non-empty list"), ([1.0], "integers")):
            with self.assertRaisesRegex(ValueError, pattern, msg=repr(bad)):
                mod.parse_config(dict(cfg, seq_layers=bad))
        self.assertEqual(mod.parse_seq_layers(None, [2, 0]), [0, 2])
        # Unit level: the plan's files and the memory plan follow the two counts.
        part = st.PartPlan(index=0, items=[], n_tokens=0, seq_bytes=st.GB, points_bytes=1)
        cc = mod.parse_config(cfg)
        plan = mod.memory_plan(cc, [part], n_layers=3, n_seq_layers=1)
        self.assertEqual(plan["peak_ram_bytes"], (st.GB + 3) + 3 * st.GB)   # 1 seq + 3 points open, 2 uploads + 1 copy
        self.assertEqual(part.files([0, 1, 2], seq_layers=[1]), [
            "points_layer_0_part00.safetensors", "seq_layer_1_part00.safetensors",
            "points_layer_1_part00.safetensors", "points_layer_2_part00.safetensors"])
        frame = pd.DataFrame({"seq_file": ["seq_layer_{L}_part00"], "points_file": ["points_layer_{L}_part00"]})
        self.assertEqual(mod.existing_part_files(frame, [0, 1, 2], [1]), set(part.files([0, 1, 2], seq_layers=[1])))
        self.assertEqual(mod.existing_part_files(frame, [0, 1]), set(part.files([0, 1])))
        # End to end: seq files for layer 1 only, points for every layer.
        summary, out, model = self.run_script(cfg)
        self.assertIn("seq_layers=[1] (seq files for these only)", out)
        self.assertIn("1 seq + 3 points buffers open per part", out)
        self.assertIn("→ 4 files", out)
        self.assertIn("(seq files for [1])", out)
        self.assertTrue(summary["verify"]["ok"])
        self.assertEqual((summary["verify"]["layers"], summary["verify"]["points_layers"]), ([1], [0, 1, 2]))
        self.assertEqual(summary["identity"]["seq_layers"], [1])
        files = sorted(f for f in self.backend().list_files(f"{SPEC_NAME}/") if st.is_store_file(f))
        expected = sorted([f"{SPEC_NAME}/seq_layer_1_part{p['index']:02d}.safetensors" for p in summary["parts"]]
                          + [f"{SPEC_NAME}/points_layer_{l}_part{p['index']:02d}.safetensors"
                             for p in summary["parts"] for l in (0, 1, 2)])
        self.assertEqual(files, expected)
        manifest, meta = st.read_store_manifest(self.backend(), SPEC_NAME)
        self.assertEqual((meta["layers"], meta["seq_layers"]), ([0, 1, 2], [1]))
        self.assertEqual(sorted(meta["files"]), [f[len(SPEC_NAME) + 1:] for f in files])
        self.assertEqual(sorted(manifest["seq_file"].unique()),   # the cell names the part, not a layer
                         [f"seq_layer_{{L}}_part{p['index']:02d}" for p in summary["parts"]])
        seq_store = st.SequenceStore(self.backend(), SPEC_NAME, manifest, meta)
        self.assertEqual(seq_store.layers, [1])
        with self.assertRaisesRegex(st.LayerSelectionError, r"layers \[0\] have no seq files.*seq_layers: \[1\]"):
            st.SequenceStore(self.backend(), SPEC_NAME, manifest, meta, layers=[0])
        point_store = st.PointStore(self.backend(), SPEC_NAME, manifest, meta)
        self.assertEqual(point_store.layers, [0, 1, 2])
        cap = st.ActivationCapturer(model, model.model.layers, [0, 1, 2], expected_hidden=HIDDEN)
        for row in self.df.to_dict("records")[:3]:
            item = mod.build_item(row, tokenizer=FakeTokenizer(), system_prompt=mod.get_thinking_config(MODEL, True).system_prompt,
                                  enable_thinking=None, delimiters=DEFAULT_DELIMITERS)
            seq, pts = cap.capture_points(item.input_ids, cot_span=item.cot_span, point_idx=item.point_idx)
            self.assertTrue(torch.equal(seq_store.get(item.rollout_id)[1], seq[1]))
            for layer in (0, 1, 2):
                self.assertTrue(torch.equal(point_store.get(item.rollout_id, "mean_cot")[layer],
                                            pts[layer][st.POINTS.index("mean_cot")]))
        from safetensors import safe_open
        with safe_open(self.root / SPEC_NAME / "points_layer_0_part00.safetensors", "pt") as f:
            header = f.metadata()
        self.assertEqual((header["layers"], header["seq_layers"]), ("[0, 1, 2]", "[1]"))
        # Resume identity: another seq subset refuses to extend the store; the same one resumes.
        identity = summary["identity"]
        self.assertEqual(mod.resume_mismatches(meta, identity), [])
        self.assertEqual(mod.resume_mismatches({**meta, "seq_layers": [0, 1, 2]}, identity), ["seq_layers"])
        legacy = {k: v for k, v in meta.items() if k != "seq_layers"}   # no key = seq files for every layer
        self.assertEqual(mod.resume_mismatches(legacy, identity), ["seq_layers"])
        self.assertEqual(mod.resume_mismatches(legacy, {**identity, "seq_layers": [0, 1, 2]}), [])
        with self.assertRaisesRegex(ValueError, r"--resume refused.*seq_layers"):
            self.run_script(dict(cfg, seq_layers=[0, 1]), resume=True)
        resumed, _, _ = self.run_script(cfg, resume=True)
        self.assertEqual((resumed["n_items"], resumed["n_resumed"]), (0, 7))
        report = mod.verify_store(self.backend(), SPEC_NAME, self.parquet, sample=7, tokenizer=FakeTokenizer())
        self.assertEqual((report["layers"], report["points_layers"], report["n_checked"]), ([1], [0, 1, 2], 7))

    def test_configs_parse(self):
        import yaml
        fractions = (0.27, 0.41, 0.48, 0.55, 0.63, 0.70, 0.84)   # Nemotron's [15, 23, 27, 31, 35, 39, 47] of 56
        expected = {
            "nemotron": ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nemotron-nano-9b-v2",
                         [15, 23, 27, 31, 35, 39, 47], None, 3),
            "olmo": ("allenai/Olmo-3-7B-Think", "olmo3-7b-think", [9, 13, 15, 18, 20, 22, 27], [9, 18, 27], 6),
            "qwen3-8b": ("Qwen/Qwen3-8B", "qwen3-8b", [10, 15, 17, 20, 23, 25, 30], [10, 20, 30], 6),
            "gemma4-12b": ("google/gemma-4-12B-it", "gemma4-12b-it", [13, 20, 23, 26, 30, 34, 40], [13, 26, 40], 6),
        }
        for stem, (model, short, layers, seq_layers, workers) in expected.items():
            with self.subTest(stem=stem):
                path = REPO_ROOT / "configs" / f"collect_probe_activations_{stem}.yaml"
                self.assertEqual(yaml.safe_load(path.read_text())["seq_layers"], seq_layers)
                cc = mod.parse_config(load_config(path))
                self.assertEqual((cc.model_name, mod.get_model_short_name(cc.model_name)), (model, short))
                self.assertEqual((cc.layers, cc.seq_layers), (layers, seq_layers or layers))
                self.assertEqual(cc.store_folder, f"{short}_shared")
                self.assertEqual(cc.storage, {"backend": "local", "local_dir": f"${{DATA_ROOT}}/probe_activations/{short}"})
                self.assertEqual(cc.dataset.name, f"{short}_used_vs_ignored.parquet")
                self.assertEqual((cc.max_file_gb, cc.upload_workers, cc.staging, cc.hf_model_kwargs),
                                 (8.0, workers, "memory", {}))
                registry = mod.get_model_config(cc.model_name)
                self.assertLess(max(cc.layers), registry["n_layers"])
                self.assertTrue(registry["hidden_size"])
                if stem != "nemotron":
                    self.assertEqual(cc.layers, [round(f * registry["n_layers"]) for f in fractions])
                    self.assertEqual(cc.seq_layers, [cc.layers[0], cc.layers[3], cc.layers[6]])
                self.assertTrue(mod.get_thinking_config(cc.model_name, cc.thinking).enabled)

    def test_cuda_oom_retries_with_smaller_chunk_then_restores(self):
        model = FakeOomModel(fail_first=2)
        with self.assertLogs("collect_probe_activations", level="WARNING") as logs:
            summary, out, _ = self.run_script(model=model)
        self.assertEqual((summary["n_items"], summary["n_skipped"], summary["n_cuda_oom"], summary["failed_parts"]),
                         (7, 0, 0, []))
        self.assertTrue(summary["verify"]["ok"])
        self.assertEqual(model.chunk_calls[:4], [64, 32, 16, 64])   # halved twice, restored after the item
        self.assertEqual(len(model.chunk_calls), 7 + 2)
        self.assertEqual((model.mixer.chunk_size, model.model.layers[1].mixer2.chunk_size), (64, 64))
        manifest, meta = st.read_store_manifest(self.backend(), SPEC_NAME)
        self.assertEqual(sorted(manifest["rollout_id"]), sorted(self.df["rollout_id"]))
        self.assertEqual((meta["n_skipped"], meta["skipped"]), (0, {}))
        first = self.df.loc[0, "rollout_id"]
        retries = [m for m in logs.output if "retrying" in m]
        self.assertEqual(len(retries), 2)
        self.assertIn(first, retries[0])
        self.assertIn("n_seq_tokens=", retries[0])
        self.assertIn("chunk_size 32 on 2 module(s)", retries[0])
        self.assertIn("chunk_size 16 on 2 module(s)", retries[1])
        self.assertIn("cuda_oom=0", out)

    def test_cuda_oom_at_floor_skips_row_and_run_completes(self):
        model = FakeOomModel(fail_first=4)   # 64, 32, 16, 8 all fail → skipped
        with self.assertLogs("collect_probe_activations", level="WARNING") as logs:
            summary, out, _ = self.run_script(model=model)
        first = self.df.loc[0, "rollout_id"]
        run_args = dict(thinking_args(), system_prompt=mod.get_thinking_config(MODEL, True).system_prompt)
        n_seq = mod.build_item(self.df.iloc[0].to_dict(), **run_args).n_seq_tokens
        reason = f"{mod.SKIP_CUDA_OOM} (n_seq_tokens={n_seq})"
        self.assertEqual((summary["n_items"], summary["n_skipped"], summary["n_cuda_oom"], summary["failed_parts"]),
                         (7, 1, 1, []))
        self.assertEqual(summary["skipped"], {first: reason})
        self.assertTrue(summary["verify"]["ok"])
        self.assertEqual(model.chunk_calls[:5], [64, 32, 16, 8, 64])
        self.assertEqual(len(model.chunk_calls), 6 + 4)
        self.assertEqual(model.mixer.chunk_size, 64)
        manifest, meta = st.read_store_manifest(self.backend(), SPEC_NAME)
        self.assertEqual(sorted(manifest["rollout_id"]), sorted(self.df["rollout_id"][1:]))
        self.assertEqual((meta["n_rollouts"], meta["n_skipped"], meta["skipped"]), (6, 1, {first: reason}))
        skips = [m for m in logs.output if "skipped" in m]
        self.assertEqual(len(skips), 1)
        self.assertIn(first, skips[0])
        self.assertIn(f"n_seq_tokens={n_seq}", skips[0])
        self.assertIn("cuda_oom=1", out)
        self.assertIn(f"{first}: {reason}", out)
        # The skipped rollout's planned slot is zero-filled in part 0 of every file family.
        part0 = f"part{summary['parts'][0]['index']:02d}"
        for kind in ("seq", "points"):
            path = self.root / SPEC_NAME / f"{kind}_layer_0_{part0}.safetensors"
            loaded = safetensors_load(path.read_bytes())
            self.assertIn(first, loaded)
            self.assertFalse(loaded[first].any())
            self.assertTrue(loaded[self.df.loc[1, "rollout_id"]].any())
        # The other rows read back exactly.
        seq_store = st.SequenceStore(self.backend(), SPEC_NAME, manifest, meta)
        for row in self.df.iloc[1:].to_dict("records"):
            item = mod.build_item(row, **run_args)
            expected = (torch.arange(*item.cot_span, dtype=torch.float32) + 1).view(-1, 1).expand(-1, HIDDEN)
            self.assertTrue(torch.equal(seq_store.get(item.rollout_id)[0], expected.to(torch.bfloat16)))
        # No chunk_size module at all: the OOM is not retried.
        plain = FakeModel()
        plain.forward = lambda *a, **k: (_ for _ in ()).throw(torch.OutOfMemoryError("oom"))
        item = mod.build_item(self.df.iloc[0].to_dict(), **run_args)
        cap = st.ActivationCapturer(plain, plain.model.layers, [0], expected_hidden=HIDDEN)
        with self.assertLogs("collect_probe_activations", level="WARNING"):
            got = mod.capture_item(cap, item, model=plain)
        self.assertEqual(got, mod.Skip(first, reason))

    def test_hf_preflight_rejects_foreign_token_before_reading(self):
        cfg = dict(self.cfg, storage={"backend": "hf", "hf_repo_id": "example-org/probe-activations-fake"},
                   dataset=str(self.tmp / "missing.parquet"))
        api = SimpleNamespace(whoami=lambda: {"name": "stranger", "orgs": []})
        with patched_registry(), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "cannot write to 'example-org'"):
                mod.run(mod.parse_config(cfg), dry_run=True, api=api, tokenizer=FakeTokenizer(),
                        model_loader=lambda n, **kw: (FakeModel(), None))

    def test_memory_plan_refused_above_cgroup_limit(self):
        cc = mod.parse_config(self.cfg)
        plan = mod.memory_plan(cc, [st.PartPlan(index=0, items=[], n_tokens=0, seq_bytes=st.GB, points_bytes=1)], n_layers=2)
        self.assertEqual(plan["peak_ram_bytes"], 2 * (st.GB + 1) + 3 * st.GB)
        with self.assertRaisesRegex(ValueError, "staging: memory would peak"):
            mod.check_memory_plan(plan, limit=4 * st.GB)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            mod.check_memory_plan(plan, limit=4 * st.GB, dry_run=True)   # a dry run only warns
        self.assertIn("[warn] staging: memory would peak", buf.getvalue())
        self.assertIn("would be refused", buf.getvalue())
        mod.check_memory_plan(plan, limit=6 * st.GB)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            mod.check_memory_plan(plan, limit=6 * st.GB, dry_run=True)
        self.assertNotIn("[warn]", buf.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            mod.check_memory_plan(plan, limit=None)
        self.assertIn("unreadable", buf.getvalue())
        cc_disk = mod.parse_config(dict(self.cfg, staging="disk", staging_dir=str(self.tmp / "stage")))
        disk = mod.memory_plan(cc_disk, [st.PartPlan(index=0, items=[], n_tokens=0, seq_bytes=st.GB, points_bytes=1)], n_layers=2)
        self.assertEqual(disk["peak_ram_bytes"], 0)
        mod.check_memory_plan(disk, limit=1)
        self.assertIsNone(mod.cgroup_memory_limit_bytes(self.tmp / "nope"))
        (self.tmp / "max").write_text("max\n")
        self.assertIsNone(mod.cgroup_memory_limit_bytes(self.tmp / "max"))
        (self.tmp / "num").write_text("8000000000\n")
        self.assertEqual(mod.cgroup_memory_limit_bytes(self.tmp / "num"), 8_000_000_000)
        # Through run(): the dry run warns, the real run refuses before the model loads.
        with mock.patch.object(mod, "cgroup_memory_limit_bytes", lambda *a, **k: 1):
            summary, out, _ = self.run_script(dry_run=True)
            self.assertTrue(summary["dry_run"])
            self.assertIn("[warn] staging: memory would peak", out)
            self.assertIn("Dry run", out)
            self.assertFalse(self.root.exists())
            with self.assertRaisesRegex(ValueError, "staging: memory would peak"):
                self.run_script()
            self.assertEqual(list(self.root.rglob("*")), [])   # refused before the model loads and anything is written
            cfg_disk = dict(self.cfg, staging="disk", staging_dir=str(self.tmp / "stage"))
            summary, out, _ = self.run_script(cfg_disk, dry_run=True)
            self.assertNotIn("[warn]", out)

    def test_store_folder_names_the_store(self):
        cfg = dict(self.cfg, store_folder="shared")
        summary, out, _ = self.run_script(cfg)
        self.assertEqual(summary["folder"], "shared")
        self.assertIn("store folder 'shared' (store_folder)", out)
        self.assertIn(f"collected from dataset {SPEC_NAME!r} (7 rows", out)
        self.assertTrue((self.root / "shared" / st.STORE_MANIFEST_NAME).exists())
        self.assertFalse((self.root / SPEC_NAME).exists())
        manifest, meta = st.read_store_manifest(self.backend(), "shared")
        self.assertEqual(len(manifest), 7)
        ds_meta = json.loads((self.tmp / "ds" / f"{SPEC_NAME}.meta.json").read_text())
        self.assertEqual(meta["dataset_spec"], {"name": SPEC_NAME, "built_utc": ds_meta["built_utc"],
                                                "fingerprint": ds_meta["fingerprint"], "n_rows": 7})
        self.assertEqual(summary["identity"]["dataset_spec"], meta["dataset_spec"])
        self.assertEqual(summary["verify"]["folder"], "shared")
        # The safetensors header names both the collection dataset and the folder.
        from safetensors import safe_open
        with safe_open(self.root / "shared" / "seq_layer_0_part00.safetensors", "pt") as f:
            header = f.metadata()
        self.assertEqual((header["dataset_spec"], header["store_folder"]), (SPEC_NAME, "shared"))
        # verify-only, the no-flag refusal and --resume all address that folder.
        self.assertEqual(mod.dataset_folder(self.parquet, "shared"), "shared")
        self.assertEqual(mod.dataset_folder(self.parquet), SPEC_NAME)
        with self.assertRaisesRegex(ValueError, "under 'shared'"):
            self.run_script(cfg)
        resumed, out, _ = self.run_script(cfg, resume=True)
        self.assertEqual((resumed["n_rows"], resumed["n_resumed"]), (0, 7))
        # A sidecar without n_rows falls back to the frame's length.
        meta_file = self.tmp / "ds" / f"{SPEC_NAME}.meta.json"
        meta_file.write_text(json.dumps({k: v for k, v in ds_meta.items() if k != "n_rows"}))
        summary, _, _ = self.run_script(dict(cfg, store_folder="shared2"), dry_run=True)
        self.assertEqual(summary["identity"]["dataset_spec"]["n_rows"], 7)
        self.assertEqual(mod.dataset_spec({"name": "x"}, n_rows=None)["n_rows"], None)
        self.assertEqual(mod.dataset_spec({"name": "x", "n_rows": "3"})["n_rows"], 3)


class TestDiskStagingAndMain(unittest.TestCase):
    """Disk staging and main()."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.df = dataset_frame(5)
        self.parquet = write_dataset_files(self.tmp / "ds", self.df)

    def test_disk_staging_writes_through_temp_files_and_cleans_up(self):
        cfg = base_cfg(self.tmp, self.parquet, staging="disk", staging_dir=str(self.tmp / "stage"))
        cc = mod.parse_config(cfg)
        with patched_registry(), contextlib.redirect_stdout(io.StringIO()) as buf:
            summary = mod.run(cc, tokenizer=FakeTokenizer(), model_loader=lambda n, **kw: (FakeModel(), None),
                              max_positions=10_000, verify_sample=5)
        self.assertIn("staging: disk", buf.getvalue())
        self.assertEqual(summary["failed_parts"], [])
        self.assertEqual(list((self.tmp / "stage").glob("*")), [])
        manifest, _ = st.read_store_manifest(st.LocalBackend(self.tmp / "store"), SPEC_NAME)
        self.assertEqual(len(manifest), 5)

    def test_main_verify_only_and_flags(self):
        import yaml
        cfg_path = self.tmp / "collect.yaml"
        cfg_path.write_text(yaml.safe_dump(base_cfg(self.tmp, self.parquet)))
        with patched_registry(), contextlib.redirect_stdout(io.StringIO()) as buf, \
                mock.patch.object(mod, "load_model_hf", lambda name, **kw: (FakeModel(), None)), \
                mock.patch.object(mod, "resolve_max_positions", lambda name: 10_000), \
                mock.patch("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer()):
            self.assertEqual(mod.main(["--config", str(cfg_path), "--dry-run", "--limit", "2"]), 0)
            self.assertFalse((self.tmp / "store").exists())
            self.assertEqual(mod.main(["--config", str(cfg_path), "--sample", "3"]), 0)
            self.assertEqual(mod.main(["--config", str(cfg_path), "--verify-only", "--sample", "2"]), 0)
        out = buf.getvalue()
        self.assertIn("Dry run", out)
        self.assertIn('"n_checked": 2', out)


if __name__ == "__main__":
    unittest.main()
