import unittest

import torch
import torch.nn.functional as F

from src.lib.attention_probe import AttentionProbe


def make_probe(hidden_dim=8, num_classes=3, seed=0):
    torch.manual_seed(seed)
    return AttentionProbe(hidden_dim, num_classes)


class TestForward(unittest.TestCase):
    """Forward pass."""

    def test_output_shape(self):
        probe = make_probe(hidden_dim=8, num_classes=3)
        H = torch.randn(4, 7, 8)
        out = probe(H)
        self.assertEqual(out.shape, (4, 3))

    def test_matches_manual_computation(self):
        probe = make_probe()
        H = torch.randn(2, 5, 8)
        with torch.no_grad():
            out = probe(H)
            weights = F.softmax(probe.Wq(H), dim=1)
            expected = (probe.Wv(H) * weights).sum(dim=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_single_token_sequence(self):
        probe = make_probe()
        H = torch.randn(3, 1, 8)
        with torch.no_grad():
            out = probe(H)
            expected = probe.Wv(H).squeeze(1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_zero_query_gives_mean_pooling(self):
        probe = make_probe()
        with torch.no_grad():
            probe.Wq.weight.zero_()
        H = torch.randn(2, 6, 8)
        with torch.no_grad():
            out = probe(H)
            expected = probe.Wv(H).mean(dim=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_no_lengths_equals_full_lengths(self):
        probe = make_probe()
        H = torch.randn(3, 5, 8)
        with torch.no_grad():
            out_plain = probe(H)
            out_full = probe(H, torch.tensor([5, 5, 5]))
        self.assertTrue(torch.allclose(out_plain, out_full, atol=1e-6))

    def test_batch_independence(self):
        probe = make_probe()
        H = torch.randn(4, 6, 8)
        with torch.no_grad():
            batched = probe(H)
            single = torch.cat([probe(H[i : i + 1]) for i in range(4)], dim=0)
        self.assertTrue(torch.allclose(batched, single, atol=1e-5))


class TestMasking(unittest.TestCase):
    """Padding mask driven by ``lengths``."""

    def test_masked_sample_equals_trimmed_sample(self):
        probe = make_probe()
        H = torch.randn(2, 7, 8)
        lengths = torch.tensor([4, 7])
        with torch.no_grad():
            out = probe(H, lengths)
            trimmed = probe(H[0:1, :4])
        self.assertTrue(torch.allclose(out[0], trimmed[0], atol=1e-5))

    def test_padding_content_is_ignored(self):
        probe = make_probe()
        H = torch.randn(2, 7, 8)
        lengths = torch.tensor([3, 5])
        H_corrupted = H.clone()
        H_corrupted[0, 3:] = 1e4
        H_corrupted[1, 5:] = -1e4
        with torch.no_grad():
            out = probe(H, lengths)
            out_corrupted = probe(H_corrupted, lengths)
        self.assertTrue(torch.allclose(out, out_corrupted, atol=1e-5))

    def test_mixed_lengths_batch(self):
        probe = make_probe()
        H = torch.randn(3, 9, 8)
        lengths = torch.tensor([2, 6, 9])
        with torch.no_grad():
            out = probe(H, lengths)
            for i, L in enumerate(lengths.tolist()):
                trimmed = probe(H[i : i + 1, :L])
                self.assertTrue(torch.allclose(out[i], trimmed[0], atol=1e-5))


class TestForwardAllPrefixes(unittest.TestCase):
    """Prefix-trajectory evaluation."""

    def test_output_shape(self):
        probe = make_probe(hidden_dim=8, num_classes=4)
        H = torch.randn(6, 8)
        with torch.no_grad():
            traj = probe.forward_all_prefixes(H)
        self.assertEqual(traj.shape, (6, 4))

    def test_each_prefix_matches_forward(self):
        probe = make_probe()
        H = torch.randn(5, 8)
        with torch.no_grad():
            traj = probe.forward_all_prefixes(H)
            for t in range(5):
                expected = probe(H[: t + 1].unsqueeze(0))[0]
                self.assertTrue(torch.allclose(traj[t], expected, atol=1e-6))

    def test_last_prefix_equals_full_forward(self):
        probe = make_probe()
        H = torch.randn(8, 8)
        with torch.no_grad():
            traj = probe.forward_all_prefixes(H)
            full = probe(H.unsqueeze(0))[0]
        self.assertTrue(torch.allclose(traj[-1], full, atol=1e-6))


class TestMultiHead(unittest.TestCase):
    """Multi-head extension."""

    def test_output_shape(self):
        probe = AttentionProbe(8, 2, heads=3)
        out = probe(torch.randn(4, 7, 8))
        self.assertEqual(out.shape, (4, 2))

    def test_zero_query_is_sum_of_head_means(self):
        torch.manual_seed(0)
        probe = AttentionProbe(8, 2, heads=3)
        with torch.no_grad():
            probe.bias.copy_(torch.tensor([0.3, -0.2]))
        H = torch.randn(2, 6, 8)
        with torch.no_grad():
            out = probe(H)
            vals = probe.Wv(H).view(2, 6, 3, 2).mean(dim=1).sum(dim=1) + probe.bias
        self.assertTrue(torch.allclose(out, vals, atol=1e-6))

    def test_padded_equals_trimmed_with_position_bias(self):
        torch.manual_seed(1)
        probe = AttentionProbe(8, 2, heads=2, position_bias=True)
        with torch.no_grad():
            probe.Wq.weight.normal_()
            probe.pos_slope.copy_(torch.tensor([1.5, -2.0]))
        H = torch.randn(3, 9, 8)
        lengths = torch.tensor([2, 6, 9])
        with torch.no_grad():
            out = probe(H, lengths)
            for i, L in enumerate(lengths.tolist()):
                trimmed = probe(H[i : i + 1, :L])
                self.assertTrue(torch.allclose(out[i], trimmed[0], atol=1e-5))

    def test_single_head_is_base_probe(self):
        torch.manual_seed(2)
        probe = AttentionProbe(8, 3, heads=1)
        with torch.no_grad():
            probe.Wq.weight.normal_()
        H = torch.randn(2, 5, 8)
        with torch.no_grad():
            out = probe(H)
            weights = F.softmax(probe.Wq(H), dim=1)
            expected = (probe.Wv(H) * weights).sum(dim=1) + probe.bias
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


class TestPositionBias(unittest.TestCase):
    """Learned relative-position slope."""

    def test_positive_slope_prefers_late_tokens(self):
        probe = AttentionProbe(4, 2, heads=1, position_bias=True)
        with torch.no_grad():
            probe.pos_slope.fill_(2.0)
        H = torch.randn(1, 6, 4)
        with torch.no_grad():
            attn = probe.attention(H)[0, :, 0]
        self.assertTrue(bool((attn[1:] > attn[:-1]).all()))

    def test_attention_sums_to_one_and_ignores_padding(self):
        torch.manual_seed(3)
        probe = AttentionProbe(4, 2, heads=2, position_bias=True)
        with torch.no_grad():
            probe.Wq.weight.normal_()
        H = torch.randn(2, 7, 4)
        lengths = torch.tensor([3, 7])
        with torch.no_grad():
            attn = probe.attention(H, lengths)
        self.assertTrue(torch.allclose(attn.sum(dim=1), torch.ones(2, 2), atol=1e-6))
        self.assertTrue(bool((attn[0, 3:] == 0).all()))

    def test_slope_absent_without_flag(self):
        self.assertIsNone(AttentionProbe(4, 2).pos_slope)
        self.assertIsNotNone(AttentionProbe(4, 2, position_bias=True).pos_slope)


class TestStandardization(unittest.TestCase):
    """Input standardisation buffers."""

    def test_forward_matches_manual_standardisation(self):
        torch.manual_seed(4)
        probe = AttentionProbe(6, 2, heads=2)
        with torch.no_grad():
            probe.Wq.weight.normal_()
        plain = AttentionProbe(6, 2, heads=2)
        plain.load_state_dict(probe.state_dict())
        mean, std = torch.randn(6), torch.rand(6) + 0.5
        probe.set_standardization(mean, std)
        H = torch.randn(3, 5, 6) * 4 + 2
        with torch.no_grad():
            out = probe(H)
            expected = plain((H - mean) / std)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_shape_and_positivity_checks(self):
        probe = AttentionProbe(6, 2)
        with self.assertRaises(ValueError):
            probe.set_standardization(torch.zeros(5), torch.ones(5))
        with self.assertRaises(ValueError):
            probe.set_standardization(torch.zeros(6), torch.zeros(6))

    def test_buffers_round_trip_through_state(self):
        torch.manual_seed(5)
        probe = AttentionProbe(6, 2, heads=2, position_bias=True, aggregation="rolling_max", window=3)
        with torch.no_grad():
            probe.Wq.weight.normal_()
        probe.set_standardization(torch.randn(6), torch.rand(6) + 0.1)
        rebuilt = AttentionProbe.from_state(probe.probe_config, probe.state_dict())
        self.assertEqual(rebuilt.probe_config, probe.probe_config)
        self.assertTrue(torch.equal(rebuilt.mean, probe.mean))
        H = torch.randn(2, 7, 6)
        with torch.no_grad():
            self.assertTrue(torch.allclose(rebuilt(H), probe(H), atol=1e-6))


class TestRollingMax(unittest.TestCase):
    """Rolling-window max aggregation."""

    def _pair(self, window, T=7):
        torch.manual_seed(6)
        roll = AttentionProbe(5, 2, heads=2, aggregation="rolling_max", window=window)
        with torch.no_grad():
            roll.Wq.weight.normal_()
        soft = AttentionProbe(5, 2, heads=2, aggregation="softmax")
        soft.load_state_dict(roll.state_dict())
        return roll, soft

    def test_output_shape(self):
        roll, _ = self._pair(3)
        self.assertEqual(roll(torch.randn(4, 9, 5)).shape, (4, 2))

    def test_window_covering_sequence_equals_softmax(self):
        roll, soft = self._pair(window=16)
        H = torch.randn(3, 7, 5)
        lengths = torch.tensor([7, 4, 2])
        with torch.no_grad():
            self.assertTrue(torch.allclose(roll(H, lengths), soft(H, lengths), atol=1e-5))

    def test_padded_equals_trimmed(self):
        roll, _ = self._pair(window=3)
        H = torch.randn(3, 9, 5)
        lengths = torch.tensor([2, 5, 9])
        with torch.no_grad():
            out = roll(H, lengths)
            for i, L in enumerate(lengths.tolist()):
                trimmed = roll(H[i : i + 1, :L])
                self.assertTrue(torch.allclose(out[i], trimmed[0], atol=1e-5))

    def test_gradients_finite_with_padding(self):
        roll, _ = self._pair(window=4)
        H = torch.randn(2, 10, 5)
        lengths = torch.tensor([1, 10])
        loss = roll(H, lengths).sum()
        loss.backward()
        for p in roll.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            AttentionProbe(4, 2, aggregation="max")
        with self.assertRaises(ValueError):
            AttentionProbe(4, 2, window=0)
        with self.assertRaises(ValueError):
            AttentionProbe(4, 2, heads=0)
        with self.assertRaises(ValueError):
            AttentionProbe(4, 2, aggregation="rolling_max").attention(torch.randn(1, 2, 4))


if __name__ == "__main__":
    unittest.main()
