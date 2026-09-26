import unittest
from unittest.mock import MagicMock, patch

import torch

import src.lib.nemotron_h_patch as nhp
from src.lib.nemotron_h_patch import (
    HF_FORWARD_PATCHES,
    PATCH_NAME,
    is_patched,
    memory_efficient_torch_forward,
    patch_nemotron_h,
    unpatch_nemotron_h,
)

try:
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer
except ImportError:  # pragma: no cover
    NemotronHConfig = NemotronHMamba2Mixer = None

HIDDEN = 32


def make_mixer(chunk_size: int, seed: int = 0) -> "NemotronHMamba2Mixer":
    """A tiny CPU float32 Mamba-2 mixer: 4 heads × 8 head dim, 2 groups, state 16."""
    cfg = NemotronHConfig(
        hidden_size=HIDDEN,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=16,
        n_groups=2,
        chunk_size=chunk_size,
        conv_kernel=4,
        layer_types=["linear_attention", "full_attention"],
        use_mamba_kernels=False,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=100,
    )
    torch.manual_seed(seed)
    mixer = NemotronHMamba2Mixer(cfg, layer_idx=0).float().eval()
    with torch.no_grad():
        # spread the parameters so every SSD term is non-trivial
        for p in mixer.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return mixer


INTER_CHUNK_EQ = "bhzc,bhcpn->bhzpn"
EXPECTED_EINSUMS = [
    "bclhn,bcshn->bclsh",
    "bclsh,bcshp->bclhp",
    "bclhn,bclhp->bchpn",
    INTER_CHUNK_EQ,
    "bclhn,bchpn->bclhp",
]


def installed_torch_forward():
    """transformers' own torch_forward, whatever the current patch state."""
    orig = getattr(NemotronHMamba2Mixer, "_orig_torch_forward", None)
    if orig is None and NemotronHMamba2Mixer.torch_forward is not memory_efficient_torch_forward:
        orig = NemotronHMamba2Mixer.torch_forward
    return orig


def inter_chunk_reference(decay_chunk, states_permuted):
    """Loop reference: new_states[z] = sum_{c <= z} decay_chunk[z, c] * states[c]."""
    b, h, n_chunks, _ = decay_chunk.shape
    out = torch.zeros_like(states_permuted)
    for bi in range(b):
        for hi in range(h):
            for z in range(n_chunks):
                for c in range(z + 1):
                    out[bi, hi, z] += decay_chunk[bi, hi, z, c] * states_permuted[bi, hi, c]
    return out


@unittest.skipIf(NemotronHMamba2Mixer is None, "transformers has no nemotron_h")
class TestMemoryEfficientForwardCorrectness(unittest.TestCase):
    """The einsum forward is chunk-size invariant and matches the loop reference of the inter-chunk recurrence."""

    def _run(self, fn, chunk_size, seq_len, batch=1, mask=None):
        mixer = make_mixer(chunk_size)
        torch.manual_seed(123)
        x = torch.randn(batch, seq_len, HIDDEN)
        with torch.no_grad():
            return fn(mixer, x, None, mask)

    def _assert_close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.isfinite(expected).all())
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-5, rtol=1e-4),
            f"max abs diff {(actual - expected).abs().max().item():.3e}",
        )

    def test_chunk_size_invariance(self):
        chunked = self._run(memory_efficient_torch_forward, 4, 37)
        single = self._run(memory_efficient_torch_forward, 64, 37)
        self.assertGreater(single.abs().max().item(), 1e-3)  # not degenerate
        self._assert_close(chunked, single)

    def test_chunk_size_invariance_batched_exact_multiple(self):
        chunked = self._run(memory_efficient_torch_forward, 8, 16, batch=2)
        single = self._run(memory_efficient_torch_forward, 64, 16, batch=2)
        self._assert_close(chunked, single)

    def test_inter_chunk_term_matches_loop_reference(self):
        captured = []
        real_einsum = torch.einsum

        def capturing_einsum(eq, *ops):
            out = real_einsum(eq, *ops)
            if eq == INTER_CHUNK_EQ:
                captured.append((ops, out))
            return out

        with patch.object(torch, "einsum", capturing_einsum):
            self._run(memory_efficient_torch_forward, 4, 37)
        self.assertEqual(len(captured), 1)
        (decay_chunk, states_permuted), result = captured[0]
        self.assertEqual(decay_chunk.shape, (1, 4, 11, 11))       # (b, h, c+1, c+1)
        self.assertEqual(states_permuted.shape, (1, 4, 11, 8, 16))  # (b, h, c+1, p, n)
        reference = inter_chunk_reference(decay_chunk, states_permuted)
        self.assertGreater(reference.abs().max().item(), 1e-3)
        self._assert_close(result, reference)

    def test_single_chunk_matches_installed_forward(self):
        installed = installed_torch_forward()
        self.assertIsNotNone(installed)
        for seq_len in (8, 5):
            with self.subTest(seq_len=seq_len):
                self._assert_close(
                    self._run(memory_efficient_torch_forward, 8, seq_len),
                    self._run(installed, 8, seq_len),
                )

    def test_multi_chunk_differs_from_installed_forward(self):
        installed = installed_torch_forward()
        diff = self._run(memory_efficient_torch_forward, 4, 37) - self._run(installed, 4, 37)
        self.assertGreater(diff.abs().max().item(), 1e-3)

    def test_uses_einsum_for_every_contraction(self):
        mixer = make_mixer(4)
        x = torch.randn(1, 37, HIDDEN)
        calls = []
        real_einsum = torch.einsum

        def counting_einsum(eq, *ops):
            calls.append(eq)
            return real_einsum(eq, *ops)

        with patch.object(torch, "einsum", counting_einsum), torch.no_grad():
            memory_efficient_torch_forward(mixer, x)
        self.assertGreaterEqual(len(calls), 5)
        self.assertEqual(calls, EXPECTED_EINSUMS)


@unittest.skipIf(NemotronHMamba2Mixer is None, "transformers has no nemotron_h")
class TestPatchNemotronH(unittest.TestCase):
    """patch_nemotron_h / unpatch_nemotron_h swap the class method in place."""

    def setUp(self):
        self.addCleanup(unpatch_nemotron_h)
        unpatch_nemotron_h()

    def test_patch_is_idempotent_and_reversible(self):
        original = NemotronHMamba2Mixer.torch_forward
        self.assertFalse(is_patched())
        self.assertEqual(patch_nemotron_h(), 1)
        self.assertTrue(is_patched())
        self.assertIs(NemotronHMamba2Mixer.torch_forward, memory_efficient_torch_forward)
        self.assertIs(NemotronHMamba2Mixer._orig_torch_forward, original)
        patch_nemotron_h()  # idempotent: the record still points at the real original
        self.assertIs(NemotronHMamba2Mixer._orig_torch_forward, original)
        self.assertIs(NemotronHMamba2Mixer.torch_forward, memory_efficient_torch_forward)
        unpatch_nemotron_h()
        self.assertFalse(is_patched())
        self.assertIs(NemotronHMamba2Mixer.torch_forward, original)
        self.assertFalse(hasattr(NemotronHMamba2Mixer, "_orig_torch_forward"))
        unpatch_nemotron_h()  # no-op when not patched
        self.assertIs(NemotronHMamba2Mixer.torch_forward, original)

    def test_patch_counts_mixer_modules(self):
        model = torch.nn.Sequential(make_mixer(4), torch.nn.Linear(HIDDEN, HIDDEN), make_mixer(4))
        self.assertEqual(patch_nemotron_h(model), 2)
        self.assertEqual(patch_nemotron_h(make_mixer(4)), 1)
        self.assertEqual(patch_nemotron_h(torch.nn.Linear(2, 2)), 0)

    def test_patched_module_dispatches_through_forward(self):
        mixer = make_mixer(4)
        x = torch.randn(1, 37, HIDDEN)
        patch_nemotron_h()
        # the class holds the function object itself, so count einsum calls (5 per forward)
        calls = []
        real_einsum = torch.einsum
        with patch.object(torch, "einsum", lambda eq, *ops: (calls.append(eq), real_einsum(eq, *ops))[1]), torch.no_grad():
            out = mixer(x)
            direct = memory_efficient_torch_forward(mixer, x)
        self.assertEqual(len(calls), 10)
        self.assertTrue(torch.equal(out, direct))

    def test_registered_under_patch_name(self):
        self.assertEqual(PATCH_NAME, "nemotron_h_memory_efficient")
        self.assertIs(HF_FORWARD_PATCHES[PATCH_NAME], patch_nemotron_h)


class TestGuardedImport(unittest.TestCase):
    """The module degrades gracefully without transformers' nemotron_h."""

    def test_patch_raises_clear_import_error_without_mixer(self):
        with patch.object(nhp, "NemotronHMamba2Mixer", None):
            self.assertFalse(is_patched())
            with self.assertRaisesRegex(ImportError, "nemotron_h"):
                patch_nemotron_h(MagicMock())
            with self.assertRaisesRegex(ImportError, "nemotron_h"):
                unpatch_nemotron_h()


if __name__ == "__main__":
    unittest.main()
