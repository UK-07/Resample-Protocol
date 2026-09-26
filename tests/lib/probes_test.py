import unittest

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from src.lib import attention_probe as base_module
from src.lib import probes
from src.lib.probes import (
    PROBE_TYPES,
    AttentionProbe,
    LinearProbe,
    TfidfLogReg,
    build_probe,
    load_probe,
    p_detection_target,
    to_checkpoint,
)


def csr_to_torch(X: sp.csr_matrix) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(X.indptr).long(), torch.from_numpy(X.indices).long(),
        torch.from_numpy(X.data).float(), size=X.shape,
    )


class TestLinearProbe(unittest.TestCase):
    """Tests LinearProbe."""

    def test_linear_probe_standardisation_and_roundtrip(self):
        torch.manual_seed(0)
        probe = LinearProbe(6, 2)
        self.assertEqual(probe.input_kind, "vector")
        mean, std = torch.randn(6), torch.rand(6) + 0.5
        probe.set_standardization(mean, std)
        H = torch.randn(4, 6)
        with torch.no_grad():
            out = probe(H, lengths=torch.tensor([3, 3, 3, 3]))   # lengths accepted, ignored
            expected = probe.linear((H - mean) / std)
        self.assertEqual(out.shape, (4, 2))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

        cfg = probe.probe_config
        self.assertEqual(cfg, {"type": "linear", "hidden_dim": 6, "num_classes": 2, "standardize": True})
        rebuilt = LinearProbe.from_state(cfg, probe.state_dict())
        self.assertTrue(torch.equal(rebuilt.mean, mean))
        self.assertTrue(torch.equal(rebuilt.std, std))
        with torch.no_grad():
            self.assertTrue(torch.allclose(rebuilt(H), out, atol=1e-6))
        # from_state also accepts the config without the type key
        no_type = {k: v for k, v in cfg.items() if k != "type"}
        rebuilt2 = LinearProbe.from_state(no_type, probe.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.allclose(rebuilt2(H), out, atol=1e-6))
        with self.assertRaises(ValueError):
            LinearProbe.from_state({**cfg, "type": "tfidf"}, probe.state_dict())

    def test_linear_probe_without_standardize_warns_and_ignores(self):
        torch.manual_seed(1)
        probe = LinearProbe(5, 2, standardize=False)
        with self.assertWarnsRegex(UserWarning, "standardize=False"):
            probe.set_standardization(torch.randn(5), torch.rand(5) + 1)
        self.assertTrue(torch.equal(probe.mean, torch.zeros(5)))
        self.assertTrue(torch.equal(probe.std, torch.ones(5)))
        H = torch.randn(3, 5)
        with torch.no_grad():
            self.assertTrue(torch.allclose(probe(H), probe.linear(H), atol=1e-6))
        self.assertFalse(probe.probe_config["standardize"])

    def test_linear_probe_rejects_bad_shapes_and_stats(self):
        probe = LinearProbe(4, 2)
        with self.assertRaises(ValueError):
            probe(torch.randn(2, 5))
        with self.assertRaises(ValueError):
            probe(torch.randn(2, 3, 4))
        with self.assertRaises(ValueError):
            probe.set_standardization(torch.zeros(3), torch.ones(4))
        with self.assertRaises(ValueError):
            probe.set_standardization(torch.zeros(4), torch.tensor([1.0, 0.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            probe.set_standardization(torch.zeros(4), torch.tensor([1.0, float("nan"), 1.0, 1.0]))
        with self.assertRaises(ValueError):
            probe.set_standardization(torch.tensor([0.0, float("inf"), 0.0, 0.0]), torch.ones(4))


class TestTfidfLogReg(unittest.TestCase):
    """Tests TfidfLogReg."""

    def test_tfidf_logreg_sparse_forward_equals_dense(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        X = sp.random(5, 12, density=0.3, format="csr", dtype=np.float32, random_state=rng)
        probe = TfidfLogReg(12, 2)
        self.assertEqual(probe.input_kind, "sparse")
        Xt = csr_to_torch(X)
        self.assertEqual(Xt.layout, torch.sparse_csr)
        out = probe(Xt)
        dense = torch.from_numpy(X.toarray())
        expected = probe(dense)
        self.assertEqual(out.shape, (5, 2))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))
        self.assertTrue(torch.allclose(out, dense @ probe.linear.weight.t() + probe.linear.bias, atol=1e-6))
        # trainable through the sparse path
        F.cross_entropy(out, torch.tensor([0, 1, 0, 1, 0])).backward()
        self.assertIsNotNone(probe.linear.weight.grad)
        self.assertGreater(probe.linear.weight.grad.abs().sum().item(), 0.0)
        self.assertEqual(probe.probe_config, {"type": "tfidf", "n_features": 12, "num_classes": 2})
        with self.assertRaises(ValueError):
            probe(csr_to_torch(sp.random(2, 7, density=0.5, format="csr", dtype=np.float32)))

    def test_tfidf_logreg_roundtrip_and_standardization_noop(self):
        torch.manual_seed(2)
        probe = TfidfLogReg(8, 2)
        X = torch.rand(3, 8)
        with torch.no_grad():
            before = probe(X)
        with self.assertWarnsRegex(UserWarning, "does not standardise"):
            probe.set_standardization(torch.zeros(8), torch.ones(8))
        with torch.no_grad():
            self.assertTrue(torch.equal(probe(X), before))
        rebuilt = TfidfLogReg.from_state(probe.probe_config, probe.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.allclose(rebuilt(X), before, atol=1e-6))


class TestAttentionProbeWrapped(unittest.TestCase):
    """Tests the attention probe under the checkpoint contract."""

    def test_attention_probe_wrapped_config_has_type(self):
        torch.manual_seed(0)
        probe = AttentionProbe(8, 2, heads=2, position_bias=True)
        self.assertEqual(probe.input_kind, "sequence")
        self.assertIsInstance(probe, base_module.AttentionProbe)
        cfg = probe.probe_config
        self.assertEqual(cfg["type"], "attention")
        base_cfg = base_module.AttentionProbe.probe_config.fget(probe)
        self.assertEqual({k: v for k, v in cfg.items() if k != "type"}, base_cfg)

        H = torch.randn(2, 5, 8)
        lengths = torch.tensor([5, 3])
        with torch.no_grad():
            out = probe(H, lengths)
            ref = base_module.AttentionProbe.from_state(base_cfg, probe.state_dict())(H, lengths)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

        for config in (cfg, base_cfg):
            rebuilt = AttentionProbe.from_state(config, probe.state_dict())
            self.assertIsInstance(rebuilt, AttentionProbe)
            self.assertEqual(rebuilt.probe_config, cfg)
            with torch.no_grad():
                self.assertTrue(torch.allclose(rebuilt(H, lengths), out, atol=1e-6))
        with self.assertRaises(ValueError):
            AttentionProbe.from_state({**cfg, "type": "linear"}, probe.state_dict())

    def test_base_module_untouched(self):
        base = base_module.AttentionProbe(4, 2)
        self.assertNotIn("type", base.probe_config)
        self.assertFalse(hasattr(base, "input_kind"))


class TestBuildProbe(unittest.TestCase):
    """Tests build_probe."""

    def test_build_probe_per_type(self):
        attn = build_probe({"type": "attention", "heads": 3, "aggregation": "rolling_max", "window": 4,
                            "position_bias": True, "standardize": True, "norm_tokens": 1000},
                           hidden_dim=16, n_features=99)
        self.assertIsInstance(attn, AttentionProbe)
        self.assertEqual(attn.probe_config, {"type": "attention", "hidden_dim": 16, "num_classes": 2, "heads": 3,
                                             "aggregation": "rolling_max", "window": 4, "position_bias": True})
        lin = build_probe({"type": "linear", "standardize": False, "points": ["pre_cot", "mean_cot"],
                           "num_classes": 3}, hidden_dim=16)
        self.assertIsInstance(lin, LinearProbe)
        self.assertEqual(lin.probe_config, {"type": "linear", "hidden_dim": 16, "num_classes": 3, "standardize": False})
        tfidf = build_probe({"type": "tfidf", "text": {"lemmatize": True}, "vectorizer": {"min_df": 5}},
                            n_features=250, hidden_dim=16)
        self.assertIsInstance(tfidf, TfidfLogReg)
        self.assertEqual(tfidf.probe_config, {"type": "tfidf", "n_features": 250, "num_classes": 2})
        # defaults when only the type is given
        self.assertEqual(build_probe({"type": "attention"}, hidden_dim=8).probe_config["heads"], 1)
        self.assertTrue(build_probe({"type": "linear"}, hidden_dim=8).standardize)
        self.assertEqual(build_probe({"type": "linear", "num_classes": None}, hidden_dim=8).num_classes, 2)
        self.assertEqual(set(PROBE_TYPES), set(probes.PROBE_CLASSES))

    def test_build_probe_rejects_unknown_keys_per_type(self):
        bad = {
            "attention": [{"points": ["pre_cot"]}, {"text": {}}, {"n_features": 3}, {"bogus": 1}],
            "linear": [{"heads": 2}, {"norm_tokens": 10}, {"vectorizer": {}}, {"bogus": 1}],
            "tfidf": [{"standardize": True}, {"heads": 2}, {"points": []}, {"bogus": 1}],
        }
        for probe_type, extras in bad.items():
            for extra in extras:
                with self.subTest(type=probe_type, extra=extra):
                    with self.assertRaises(ValueError) as ctx:
                        build_probe({"type": probe_type, **extra}, hidden_dim=8, n_features=8)
                    msg = str(ctx.exception)
                    self.assertIn(next(iter(extra)), msg)
                    self.assertIn("allowed", msg)
        with self.assertRaises(ValueError):
            build_probe({"heads": 2}, hidden_dim=8)
        with self.assertRaises(ValueError):
            build_probe({"type": "mlp"}, hidden_dim=8)
        with self.assertRaises(ValueError):
            build_probe({"type": "attention"}, n_features=8)
        with self.assertRaises(ValueError):
            build_probe({"type": "linear"})
        with self.assertRaises(ValueError):
            build_probe({"type": "tfidf"}, hidden_dim=8)
        with self.assertRaises(TypeError):
            build_probe("attention", hidden_dim=8)
        self.assertEqual(probes.allowed_probe_keys("linear"), ("type", "num_classes", "standardize", "points"))


class TestLoadProbe(unittest.TestCase):
    """Tests to_checkpoint / load_probe."""

    def test_load_probe_dispatches(self):
        torch.manual_seed(0)
        cases = [
            (AttentionProbe(6, 2, heads=2), torch.randn(2, 4, 6), torch.tensor([4, 2])),
            (LinearProbe(6, 2), torch.randn(3, 6), None),
            (TfidfLogReg(9, 2), csr_to_torch(sp.random(3, 9, density=0.4, format="csr", dtype=np.float32)), None),
        ]
        cases[1][0].set_standardization(torch.randn(6), torch.rand(6) + 0.5)
        for probe, X, lengths in cases:
            with self.subTest(type=probe.probe_config["type"]):
                ckpt = {"layer": 3, **to_checkpoint(probe), "history": []}
                self.assertEqual(set(to_checkpoint(probe)), {"probe_config", "model_state_dict"})
                rebuilt = load_probe(ckpt)
                self.assertIs(type(rebuilt), type(probe))
                self.assertEqual(rebuilt.probe_config, probe.probe_config)
                with torch.no_grad():
                    self.assertTrue(torch.allclose(rebuilt(X, lengths), probe(X, lengths), atol=1e-6))
                # a snapshot, not a view: mutating the live probe leaves it unchanged
                key, before = next((k, v.clone()) for k, v in ckpt["model_state_dict"].items() if v.numel())
                with torch.no_grad():
                    probe.state_dict()[key].add_(1.0)
                self.assertTrue(torch.equal(ckpt["model_state_dict"][key], before))

    def test_load_probe_rejects_bad_checkpoints(self):
        probe = LinearProbe(4, 2)
        ckpt = to_checkpoint(probe)
        with self.assertRaises(ValueError):
            load_probe({"probe_config": ckpt["probe_config"]})
        no_type = {k: v for k, v in ckpt["probe_config"].items() if k != "type"}
        with self.assertRaises(ValueError):
            load_probe({"probe_config": no_type, "model_state_dict": ckpt["model_state_dict"]})
        with self.assertRaises(ValueError):
            load_probe({"probe_config": {**ckpt["probe_config"], "type": "mlp"},
                        "model_state_dict": ckpt["model_state_dict"]})


class TestScore(unittest.TestCase):
    """Tests p_detection_target."""

    def test_score_is_class_zero_probability(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        p = p_detection_target(logits)
        self.assertEqual(p.shape, (3,))
        self.assertTrue(torch.allclose(p, F.softmax(logits, dim=-1)[:, 0]))
        self.assertGreater(p[0].item(), 0.5)
        self.assertLess(p[1].item(), 0.5)
        self.assertAlmostEqual(p[2].item(), 0.5, places=6)
        self.assertEqual(p_detection_target(logits.to(torch.bfloat16)).dtype, torch.float32)
        three = torch.tensor([[3.0, 0.0, 0.0]])
        self.assertAlmostEqual(p_detection_target(three).item(), F.softmax(three, dim=-1)[0, 0].item(), places=6)
        with self.assertRaises(ValueError):
            p_detection_target(torch.zeros(4))


if __name__ == "__main__":
    unittest.main()
