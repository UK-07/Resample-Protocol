"""Memory-efficient, corrected ``torch_forward`` for transformers' Nemotron-H Mamba-2 mixer.

The installed pure-torch chunked SSD scan materialises every contraction as a
rank-6 broadcast product (O(seq · h · p · n) scratch per contraction) and reduces
its inter-chunk decay product over the wrong axis, so its output is not
chunk-size invariant. :func:`memory_efficient_torch_forward` is the same
function with the five contractions as ``torch.einsum`` calls and the correct
recurrence ``new_states[z] = Σ_{c ≤ z} decay_chunk[z, c] · states[c]``;
:func:`patch_nemotron_h` swaps it onto the class. The imports are guarded so
this module imports on a transformers without ``nemotron_h``.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHMamba2Mixer,
        pad_tensor_by_size,
        reshape_into_chunks,
        segment_sum,
    )
except ImportError:  # pragma: no cover - depends on the installed transformers
    NemotronHMamba2Mixer = None  # type: ignore[assignment]
    pad_tensor_by_size = reshape_into_chunks = segment_sum = None  # type: ignore[assignment]

PATCH_NAME = "nemotron_h_memory_efficient"
_ORIG_ATTR = "_orig_torch_forward"


# fmt: off
def memory_efficient_torch_forward(self, input_states, cache_params=None, attention_mask=None):
    """``NemotronHMamba2Mixer.torch_forward`` with einsum contractions and the corrected inter-chunk recurrence."""
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
        projected_states = self.in_proj(input_states)
    else:
        if attention_mask is not None:
            # zero pad tokens, see https://github.com/state-spaces/mamba/issues/66
            input_states = (input_states * attention_mask[:, :, None]).to(dtype)
        projected_states = self.in_proj(input_states)
    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size- self.num_heads) // 2
    _, _, gate, hidden_states, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
    )
    hidden_states = hidden_states.transpose(1, 2)

    use_precomputed_state = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
    if use_precomputed_state:
        conv_state = cache_params.layers[self.layer_idx].conv_states[0]

    if use_precomputed_state and seq_len == 1:
        conv_states = cache_params.update_conv_state(hidden_states, self.layer_idx)[..., -self.conv_kernel_size:]
        hidden_states = torch.sum(conv_states * self.conv1d.weight[:, 0, :], dim=-1)
        if self.use_conv_bias:
            hidden_states += self.conv1d.bias
        hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]         # [batch, 1, intermediate_size] : decoding
    else:
        if use_precomputed_state:
            hidden_states = torch.cat([conv_state, hidden_states], dim=-1)
        if cache_params is not None:
            conv_states = nn.functional.pad(
                hidden_states,
                (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            conv_states = cache_params.update_conv_state(conv_states, self.layer_idx)[..., -self.conv_kernel_size:]

        hidden_states = self.act(self.conv1d(hidden_states)[..., :hidden_states.shape[-1]].transpose(1, 2))
        if use_precomputed_state:
            hidden_states = hidden_states[:, -seq_len:, :]
        if attention_mask is not None:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)
    A = -torch.exp(self.A_log.float())                            # [num_heads]
    if use_precomputed_state and seq_len == 1:
        dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
        dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
        dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

        dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
        dt = torch.clamp(dt, self.time_step_min) #, self.time_step_max)
        A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
        dA = torch.exp(dt[..., None] * A)                         # [bsz, num_heads, head_dim, state_size]

        B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
        B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
        B = B.reshape(batch_size, -1, B.shape[-1])
        dB = dt[..., None] * B[..., None, :]                      # [bsz, num_heads, head_dim, state_size]

        hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
        dBx = dB * hidden_states[..., None]

        ssm_states = cache_params.layers[self.layer_idx].recurrent_states[0].clone()
        ssm_states = ssm_states * dA + dBx
        ssm_states = cache_params.update_recurrent_state(ssm_states, self.layer_idx)

        C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
        C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
        C = C.reshape(batch_size, -1, C.shape[-1])

        ssm_states = ssm_states.to(C.dtype)  # Shape: [b, h, d, n]
        ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
        C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
        y = torch.bmm(ssm_states_reshaped, C_reshaped)
        y = y.view(batch_size, self.num_heads, self.head_dim)

        D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
        y = (y + hidden_states * D).to(y.dtype)

        y = y.reshape(batch_size, -1)[:, None, ...]              # [bsz, 1, intermediate_size]
    else:
        dt = nn.functional.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_min)
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len,  -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

        hidden_states = hidden_states * dt[..., None]
        A = A.to(hidden_states.dtype) * dt

        hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]

        A = A.permute(0, 3, 1, 2)                                 # [bsz, num_heads, -1, chunk_size]
        A_cumsum = torch.cumsum(A, dim=-1)

        # intra-chunk (diagonal blocks)
        L = torch.exp(segment_sum(A))
        G = torch.einsum("bclhn,bcshn->bclsh", C, B)  # shape: (b, c, l, s, h)
        M = G * L.permute(0, 2, 3, 4, 1)
        Y_diag = torch.einsum("bclsh,bcshp->bclhp", M, hidden_states)

        # inter-chunk (off-diagonal blocks): per-chunk states, then the decay recurrence
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
        states = torch.einsum("bclhn,bclhp->bchpn", B_decay_contraction, hidden_states)
        previous_states = (
            cache_params.layers[self.layer_idx].recurrent_states[0][:, None].to(dtype=states.dtype, device=states.device)
            if use_precomputed_state
            else torch.zeros_like(states[:, :1])
        )
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

        states_permuted = states.permute(0, 2, 1, 3, 4)
        # decay_chunk is (b, h, z, c), z = target chunk, c = source chunk: reduce over c
        # (transformers' installed torch_forward reduces over z, which is wrong).
        result = torch.einsum("bhzc,bhcpn->bhzpn", decay_chunk, states_permuted)
        new_states = result.permute(0, 2, 1, 3, 4)
        states, ssm_state = new_states[:, :-1], new_states[:, -1]

        state_decay_out = torch.exp(A_cumsum)
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = torch.einsum("bclhn,bchpn->bclhp", C, states) * state_decay_out_permuted[..., None]

        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

        y = y + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)
        if ssm_state is not None and cache_params is not None:
            cache_params.update_recurrent_state(ssm_state, self.layer_idx)

    scan_output = self.norm(y, gate)

    contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
    return contextualized_states
# fmt: on


def _require_mixer():
    if NemotronHMamba2Mixer is None:
        raise ImportError(
            "transformers.models.nemotron_h is not available in the installed "
            "transformers; the Nemotron-H forward patch cannot be applied."
        )
    return NemotronHMamba2Mixer


def _count_mixers(model_or_module) -> int:
    cls = _require_mixer()
    if isinstance(model_or_module, cls):
        return 1
    modules = getattr(model_or_module, "modules", None)
    if modules is None:
        return 0
    return sum(1 for m in modules() if isinstance(m, cls))


def patch_nemotron_h(model_or_module=None) -> int:
    """Swap the einsum forward onto the class (idempotent; the original is kept as ``_orig_torch_forward``).

    Returns the number of mixer modules in ``model_or_module``, or 1 when none is given.
    """
    cls = _require_mixer()
    if not hasattr(cls, _ORIG_ATTR):
        setattr(cls, _ORIG_ATTR, cls.torch_forward)
    cls.torch_forward = memory_efficient_torch_forward
    if model_or_module is None:
        return 1
    return _count_mixers(model_or_module)


def unpatch_nemotron_h() -> None:
    """Restore the original ``torch_forward`` (no-op when never patched)."""
    cls = _require_mixer()
    orig = getattr(cls, _ORIG_ATTR, None)
    if orig is not None:
        cls.torch_forward = orig
        delattr(cls, _ORIG_ATTR)


def is_patched() -> bool:
    """True while the class carries the memory-efficient forward."""
    return NemotronHMamba2Mixer is not None and hasattr(NemotronHMamba2Mixer, _ORIG_ATTR)


HF_FORWARD_PATCHES = {PATCH_NAME: patch_nemotron_h}
