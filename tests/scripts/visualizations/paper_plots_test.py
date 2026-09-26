"""Tests for src/scripts/visualizations/paper_plots.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.visualizations import paper_plots as mod


def synthetic_tree(root: Path, seed: int = 0) -> None:
    """Two manifests + question_reliance.csv covering every model, dataset, cue and case."""
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for model in mod.MODEL_ORDER:
        for dataset in mod.DATASET_ORDER:
            for cue in mod.CUE_ORDER:
                for case in ("positive", "negative"):
                    for _ in range(3):
                        switched = bool(rng.random() < 0.6)
                        label = int(rng.random() < 0.8) if switched else None
                        rows.append({
                            "rollout_id": f"{model}:{dataset}:{cue}:{i}", "question_id": f"{dataset}:test:{i}",
                            "subject_model": model, "dataset": dataset, "hint_style": cue, "case": case,
                            "original_index": i, "run": f"{dataset}_cueball", "to_hint": switched,
                            "changed": switched or bool(rng.random() < 0.1),
                            "model_answer": "A" if rng.random() > 0.05 else None,
                            "judge_label": label, "judge_label_final": label,
                            "judge_confidence": float(rng.choice([0.6, 0.9, 0.95])) if switched else np.nan,
                            "judge_role": rng.choice(mod.ROLE_ORDER) if label is not None else None,
                            "baseline_stability": int(rng.choice([5, 6, 7, 8])), "baseline_n_samples": 8,
                            "trace_token_len": int(rng.integers(200, 9000)), "truncated": bool(rng.random() < 0.05),
                            "exclude_reason": None if rng.random() < 0.9 else "noise_flip",
                        })
                        i += 1
    once = pd.DataFrame(rows)
    (root / "hinted_rollouts").mkdir(parents=True)
    once.to_parquet(root / "hinted_rollouts" / "rollout_manifest.parquet")

    sel = once[once.to_hint].copy()
    rerolls = []
    for _, r in sel.iterrows():
        for s in (43, 44, 45, 46):
            rr = r.to_dict(); rr["rollout_id"] = f"{r.rollout_id}#rs{s}"; rr["sample_seed"] = s
            rr["to_hint"] = bool(rng.random() < 0.7); rr["judge_label_final"] = int(rng.random() < 0.8) if rr["to_hint"] else None
            rr["judge_label"] = rr["judge_label_final"]; rr["contrast_b"] = rr["to_hint"] and rr["judge_label_final"] is not None
            rerolls.append(rr)
    res = pd.concat([sel.assign(provenance="hinted_once", contrast_b=False), pd.DataFrame(rerolls).assign(provenance="resample_k4")])
    (root / "resample").mkdir()
    res.to_parquet(root / "resample" / "resample_manifest.parquet")

    q = sel[["rollout_id", "subject_model", "hint_style", "case", "dataset"]].copy()
    q["role"] = rng.choice(["used_candidate", "control"], size=len(q))
    q["k_to_hint_count"] = rng.integers(0, 5, size=len(q)); q["k_judged"] = q.k_to_hint_count
    q["k_unfaithful"] = (q.k_judged * rng.random(len(q))).astype(int); q["k_faithful"] = q.k_judged - q.k_unfaithful
    q["reliance_label"] = np.where(q.k_to_hint_count >= 3, "robust_used", np.where(q.k_to_hint_count >= 1, "weak_used", "mixed"))
    q.to_csv(root / "resample" / "question_reliance.csv", index=False)


class PaperPlotsTest(unittest.TestCase):
    def test_every_variant_draws_and_is_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cueball"; synthetic_tree(root)
            out = Path(tmp) / "figs"
            code = mod.main(["--cueball-dir", str(root), "--out", str(out), "--formats", "png"])
            self.assertEqual(code, 0)
            pngs = sorted(p.name for p in out.glob("*.png"))
            self.assertEqual(len(pngs), 29, pngs)   # F8d/e need baseline sidecars
            for key in mod.REGISTRY:
                self.assertTrue(any(p.startswith(key) for p in pngs), key)
            index = (out / "figures.md").read_text()
            for p in pngs:
                self.assertIn(f"**{p.removesuffix('.png')}**", index)

    def test_only_filters_figures_and_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cueball"; synthetic_tree(root)
            out = Path(tmp) / "figs"
            mod.main(["--cueball-dir", str(root), "--out", str(out), "--formats", "png", "--only", "F1,F3c"])
            self.assertEqual(sorted(p.name for p in out.glob("*.png")), ["F1a.png", "F1b.png", "F1c.png", "F1d.png", "F1e.png", "F1f.png", "F1g.png", "F3c.png"])

    def test_rate_table_wilson_bounds(self):
        df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 4, "unfaithful": [True] * 3 + [False] * 7 + [False] * 4,
                           "judged": [True] * 10 + [True] * 4})
        t = mod.rate_table(df, ["g"]).set_index("g")
        self.assertAlmostEqual(t.loc["a", "rate"], 0.3)
        self.assertLess(t.loc["a", "lo"], 0.3); self.assertGreater(t.loc["a", "hi"], 0.3)
        self.assertEqual(t.loc["b", "rate"], 0.0); self.assertEqual(t.loc["b", "lo"], 0.0)


if __name__ == "__main__":
    unittest.main()
