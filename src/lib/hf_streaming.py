"""
Zero-storage streaming of single safetensors tensors from the Hugging Face Hub.

Safetensors wire format: ``[8 bytes header_len (uint64 LE)] [JSON header]
[tensor data]``; the header declares every tensor's byte range, so a tensor
costs two small HTTP Range requests (header + data) and nothing is written to
disk. ``stream_tensor`` is what the activation store's readers use on an
``HfBackend``.
"""

import json
import os
import struct
import time
from functools import lru_cache

import numpy as np
import requests
import torch
from huggingface_hub import get_token, hf_hub_url
from huggingface_hub.utils import HfHubHTTPError

def _get_hf_token() -> str | None:
    """The Hub token: ``huggingface_hub.get_token`` (``HF_TOKEN`` in the
    environment, else the stored login); the raw environment variable if that
    lookup raises."""
    try:
        return get_token()
    except Exception:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# One session per process: a forked DataLoader worker must not share the
# parent's pooled sockets (the module-level attribute is kept patchable).
_session = requests.Session()
_session_pid = os.getpid()


def _get_session() -> requests.Session:
    global _session, _session_pid
    if _session_pid != os.getpid():
        _session = requests.Session()
        _session_pid = os.getpid()
    return _session


# Retries for a streamed range: transient network errors and expired signed
# CDN URLs (403 once the ``Expires`` timestamp passes).
STREAM_RETRIES = 4
STREAM_BACKOFF_S = 2.0

_DTYPE_MAP: dict[str, np.dtype] = {
    "F64":  np.float64,
    "F32":  np.float32,
    "F16":  np.float16,
    "I64":  np.int64,
    "I32":  np.int32,
    "I16":  np.int16,
    "I8":   np.int8,
    "U8":   np.uint8,
    "BOOL": np.bool_,
}


@lru_cache(maxsize=4096)
def _resolve_url(hf_url: str, token: str | None) -> str:
    """Follow the Hub's auth redirects and return the CDN URL that accepts
    unauthenticated Range requests; cached per (url, token) per process."""
    from huggingface_hub import get_hf_file_metadata
    meta = get_hf_file_metadata(hf_url, token=token)
    return meta.location


@lru_cache(maxsize=4096)
def _fetch_header(cdn_url: str) -> tuple[int, dict]:
    """Parse the safetensors header at ``cdn_url``; cached per URL per process.

    Returns ``(data_offset, header)``: the byte position where tensor data
    starts and ``{tensor_key: {dtype, shape, data_offsets}, ...}``."""
    head = _range_get(cdn_url, 0, 7, timeout=30)
    header_len = struct.unpack("<Q", head)[0]
    body = _range_get(cdn_url, 8, 8 + header_len - 1, timeout=30)
    return 8 + header_len, json.loads(body)


def _range_get(cdn_url: str, start: int, end: int, timeout: int) -> bytearray:
    """GET ``bytes=start-end`` (inclusive) and return exactly those bytes as a
    writable buffer. A non-206 response is refused before its body is read (a
    server ignoring ``Range`` would stream the whole file); a short or long
    body raises ``requests.HTTPError`` so the caller's retry loop re-resolves."""
    r = _get_session().get(cdn_url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout, stream=True)
    r.raise_for_status()
    status = getattr(r, "status_code", 206)
    if status != 206:
        r.close()
        raise requests.HTTPError(f"expected 206 Partial Content for bytes={start}-{end}, got {status}", response=r)
    expected = end - start + 1
    # Large reads from the raw stream: ``r.content`` iterates 10 KB chunks,
    # ~30% slower on a 100 MB tensor.
    buf = bytearray(expected)
    view = memoryview(buf)
    pos = 0
    while pos < expected:
        chunk = r.raw.read(min(expected - pos, 8 << 20), decode_content=True)
        if not chunk:
            break
        view[pos:pos + len(chunk)] = chunk
        pos += len(chunk)
    extra = r.raw.read(1, decode_content=True) if pos == expected else b""
    r.close()
    if pos != expected or extra:
        raise requests.HTTPError(
            f"range bytes={start}-{end}: expected {expected} bytes, got {pos + len(extra)}{'+' if extra else ''}",
            response=r,
        )
    return buf


def stream_tensor(
    repo_id: str,
    filename: str,
    tensor_key: str,
    repo_type: str = "dataset",
    token: str | None = None,
    dtype: torch.dtype | None = torch.float32,
    revision: str | None = None,
) -> torch.Tensor:
    """Fetch one tensor from a safetensors file on the Hub via HTTP Range
    requests, writing nothing to disk.

    ``token`` falls back to ``_get_hf_token``; ``dtype=None`` keeps the stored
    dtype (bf16 in the activation store); ``revision`` pins a commit (``None``
    = the default branch). Transient failures and expired signed URLs are
    retried ``STREAM_RETRIES`` times, re-resolving the URL and header each
    time; an unknown ``tensor_key`` raises ``KeyError`` without retrying."""
    tok     = token or _get_hf_token()
    hf_url  = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)

    for attempt in range(1, STREAM_RETRIES + 1):
        try:
            cdn_url = _resolve_url(hf_url, tok)
            data_offset, header = _fetch_header(cdn_url)

            if tensor_key not in header:
                available = [k for k in header if not k.startswith("__")]
                raise KeyError(
                    f"Tensor '{tensor_key}' not in {filename}. Available: {available}"
                )

            info               = header[tensor_key]
            dtype_str          = info["dtype"]
            shape              = info["shape"]
            rel_start, rel_end = info["data_offsets"]   # relative to data region

            byte_start = data_offset + rel_start
            byte_end   = data_offset + rel_end - 1      # inclusive for HTTP Range

            content = _range_get(cdn_url, byte_start, byte_end, timeout=120)
            break
        except (requests.RequestException, HfHubHTTPError) as e:
            _resolve_url.cache_clear()
            _fetch_header.cache_clear()
            if attempt == STREAM_RETRIES:
                raise RuntimeError(
                    f"streaming {tensor_key} from {filename} failed after {attempt} attempts: {e}"
                ) from e
            time.sleep(STREAM_BACKOFF_S * attempt)

    # The tensor views the downloaded buffer directly (no copy of the bytes).
    if dtype_str == "BF16":
        # bfloat16 has no numpy dtype; decode via torch's native bfloat16
        t = torch.frombuffer(content, dtype=torch.bfloat16).reshape(shape)
    else:
        np_dtype = _DTYPE_MAP.get(dtype_str, np.float32)
        t = torch.from_numpy(np.frombuffer(content, dtype=np_dtype).reshape(shape))
    return t if dtype is None else t.to(dtype)
