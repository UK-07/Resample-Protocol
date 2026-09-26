import json
import os
import re
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

import src.lib.hf_streaming as hs
import requests

from src.lib.hf_streaming import stream_tensor


class RangeSession:
    def __init__(self, data):
        self.data = data
        self.requests = []

    def get(self, url, headers=None, timeout=None, stream=False):
        if not headers or "Range" not in headers:
            raise AssertionError("request without a Range header")
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
        if match is None:
            raise AssertionError(f"malformed Range header: {headers['Range']}")
        start, end = int(match.group(1)), int(match.group(2))
        if not (0 <= start <= end < len(self.data)):
            raise AssertionError(
                f"Range {start}-{end} outside file of {len(self.data)} bytes"
            )
        self.requests.append((start, end))
        return self.respond(start, end)

    def respond(self, start, end):
        return FakeResponse(self.data[start : end + 1])


class FakeResponse:
    """What ``_range_get`` consumes; ``content`` is exposed so tests can truncate the body."""

    status_code = 206

    def __init__(self, body):
        self.content = body
        self._pos = 0

    def raise_for_status(self):
        pass

    def close(self):
        pass

    @property
    def raw(self):
        return self

    def read(self, amt, decode_content=True):
        chunk = self.content[self._pos : self._pos + amt]
        self._pos += len(chunk)
        return chunk


TENSORS = {
    "layer_0": torch.tensor([[1.0, -2.5, 3.25], [0.5, 4.0, -8.0]], dtype=torch.float32),
    "layer_1": torch.tensor([[7, -3], [12, 0]], dtype=torch.int64),
    "layer_2": torch.tensor([[0.5, 1.5], [-2.0, 3.0]], dtype=torch.float16),
    "layer_3": torch.tensor([[1.0, -2.5], [0.5, 4.0]], dtype=torch.bfloat16),
}


def write_shard(dirpath, tensors, name="shard.safetensors"):
    path = os.path.join(dirpath, name)
    save_file(tensors, path)
    with open(path, "rb") as f:
        return path, f.read()


class StreamingBase(unittest.TestCase):
    def setUp(self):
        hs._fetch_header.cache_clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path, self.raw = write_shard(self.tmp.name, TENSORS)
        self.cdn_url = f"https://cdn.test/{self.id()}/shard.safetensors"
        self.session = RangeSession(self.raw)
        self.resolve = MagicMock(side_effect=lambda url, tok: self.cdn_url)
        self.resolve.cache_clear = lambda: None
        patches = [
            patch.object(hs, "_session", self.session),
            patch.object(hs, "_session_pid", os.getpid()),
            patch.object(hs, "_resolve_url", self.resolve),
            patch.object(hs, "_get_hf_token", lambda: None),
            patch.object(hs, "STREAM_BACKOFF_S", 0.0),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def stream(self, key, filename="shard.safetensors"):
        return stream_tensor("org/repo", filename, key)


class TestFetchHeader(StreamingBase):
    """Safetensors header parsing over ranged HTTP."""

    def test_data_offset_is_8_plus_header_len(self):
        header_len = struct.unpack("<Q", self.raw[:8])[0]
        data_offset, _ = hs._fetch_header(self.cdn_url)
        self.assertEqual(data_offset, 8 + header_len)

    def test_header_lists_all_tensors(self):
        _, header = hs._fetch_header(self.cdn_url)
        for key in TENSORS:
            self.assertIn(key, header)
            self.assertIn("dtype", header[key])
            self.assertIn("shape", header[key])
            self.assertIn("data_offsets", header[key])

    def test_header_requests_use_exact_ranges(self):
        header_len = struct.unpack("<Q", self.raw[:8])[0]
        hs._fetch_header(self.cdn_url)
        self.assertEqual(self.session.requests[0], (0, 7))
        self.assertEqual(self.session.requests[1], (8, 8 + header_len - 1))


class TestStreamTensor(StreamingBase):
    """Single-tensor streaming, offset math and dtype handling."""

    def test_float32_roundtrip(self):
        out = self.stream("layer_0")
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out, TENSORS["layer_0"]))

    def test_int64_roundtrip(self):
        out = self.stream("layer_1")
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out, TENSORS["layer_1"].float()))

    def test_float16_roundtrip(self):
        out = self.stream("layer_2")
        self.assertTrue(torch.equal(out, TENSORS["layer_2"].float()))

    def test_bfloat16_roundtrip(self):
        out = self.stream("layer_3")
        self.assertTrue(
            torch.equal(out, TENSORS["layer_3"].float()),
            f"BF16 tensor corrupted in streaming: expected "
            f"{TENSORS['layer_3'].float().tolist()}, got {out.tolist()}",
        )

    def test_fetches_only_tensor_bytes(self):
        header_len = struct.unpack("<Q", self.raw[:8])[0]
        header = json.loads(self.raw[8 : 8 + header_len])
        rel_start, rel_end = header["layer_0"]["data_offsets"]
        self.stream("layer_0")
        data_request = self.session.requests[-1]
        self.assertEqual(
            data_request, (8 + header_len + rel_start, 8 + header_len + rel_end - 1)
        )

    def test_header_cached_across_tensors(self):
        self.stream("layer_0")
        self.stream("layer_1")
        self.assertEqual(len(self.session.requests), 4)

    def test_missing_key_raises_keyerror(self):
        with self.assertRaises(KeyError) as ctx:
            self.stream("layer_99")
        self.assertIn("layer_0", str(ctx.exception))


class TestStreamDtypeAndRetry(StreamingBase):
    """The dtype switch and the retry loop of stream_tensor."""

    def test_dtype_none_keeps_bf16(self):
        out = stream_tensor("org/repo", "shard.safetensors", "layer_3", dtype=None)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, TENSORS["layer_3"]))

    def test_expired_url_is_reresolved_and_retried(self):
        session = self.session
        state = {"failed": False}
        original = session.respond

        def respond(start, end):
            if start > 8 and not state["failed"]:       # first data-range request
                state["failed"] = True
                r = SimpleNamespace(status_code=403, content=b"", close=lambda: None)
                def boom():
                    raise requests.HTTPError("403 expired", response=r)
                r.raise_for_status = boom
                return r
            return original(start, end)

        session.respond = respond
        out = self.stream("layer_0")
        self.assertTrue(torch.equal(out, TENSORS["layer_0"]))
        self.assertTrue(state["failed"])
        self.assertGreaterEqual(self.resolve.call_count, 2)

    def test_non_partial_response_never_reads_body(self):
        session = self.session
        original = session.respond

        class Poison:
            status_code = 200
            def raise_for_status(self): pass
            def close(self): pass
            @property
            def content(self):
                raise AssertionError("body of a non-206 response was read")

        session.respond = lambda start, end: Poison() if start > 8 else original(start, end)
        with self.assertRaisesRegex(RuntimeError, "expected 206"):
            self.stream("layer_0")

    def test_short_body_is_retried_then_raises(self):
        session = self.session
        original = session.respond

        def respond(start, end):
            r = original(start, end)
            if start > 8:
                r.content = r.content[:-1]
            return r

        session.respond = respond
        with self.assertRaisesRegex(RuntimeError, "attempts"):
            self.stream("layer_0")
        data_requests = [r for r in session.requests if r[0] > 8]
        self.assertEqual(len(data_requests), hs.STREAM_RETRIES)

    def test_missing_key_is_not_retried(self):
        with self.assertRaises(KeyError):
            self.stream("layer_99")
        self.assertEqual(len(self.session.requests), 2)   # header length + header only


class TestSessionPerProcess(unittest.TestCase):
    """_get_session: one Session per process."""

    def test_same_process_reuses_session(self):
        with patch.object(hs, "_session_pid", os.getpid()):
            self.assertIs(hs._get_session(), hs._session)

    def test_forked_process_gets_new_session(self):
        before = hs._session
        with patch.object(hs, "_session_pid", -1), patch.object(hs, "_session", before):
            after = hs._get_session()
            self.assertIsNot(after, before)
            self.assertEqual(hs._session_pid, os.getpid())


class TestRevisionPinning(StreamingBase):
    """The revision argument reaches hf_hub_url."""

    def test_stream_tensor_passes_revision_to_hub_url(self):
        with patch.object(hs, "hf_hub_url", wraps=hs.hf_hub_url) as url:
            stream_tensor("org/repo", "shard.safetensors", "layer_0", revision="abc123")
            self.assertEqual(url.call_args.kwargs.get("revision"), "abc123")
            stream_tensor("org/repo", "shard.safetensors", "layer_0")
            self.assertIsNone(url.call_args.kwargs.get("revision"))


if __name__ == "__main__":
    unittest.main()
