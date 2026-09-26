"""Shared fixtures for the re-sampling scripts' tests: a tiny DATA_ROOT tree."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.lib.rollout_manifest import write_manifest
from tests.lib.resample_test import manifest_rows

MODEL = "qwen3-8b"
MODEL_ID = "Qwen/Qwen3-8B"
RUN = "medqa-test_0911"
STEM = f"{MODEL}_{RUN}_baseline_hinted_rollouts"


def source_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """A judged hinted-rollouts CSV matching a manifest slice row for row."""
    df = manifest[manifest["run"] == RUN]
    rows = []
    for _, r in df.iterrows():
        ans = r["model_answer"]
        rows.append({
            "original_index": r["original_index"], "sample_type": r["case"], "hint_name": r["hint_style"],
            "prompt": f"Q{r['original_index']}?", "hinted_prompt": f"hinted Q{r['original_index']} ({r['hint_style']})",
            "hinted_answer": r["target_option"], "baseline_answer": r["baseline_modal_answer"],
            "groundtruth": r["groundtruth"],
            "rollout": f"<think>cot</think><answer>{ans}</answer>" if pd.notna(ans) else "<think>cut",
            "reasoning": "cot", "final_answer": ans if pd.notna(ans) else "", "additional_fields": "{}",
            "judge_model": r["judge_model"] if pd.notna(r["judge_model"]) else "",
            "judge_label": r["judge_label"] if pd.notna(r["judge_label"]) else "",
            "judge_confidence": "", "judge_reasoning": "", "judge_role": "", "judge_hint_quote": "",
        })
    return pd.DataFrame(rows)


def make_data_root(tmp: Path, *, manifest: pd.DataFrame | None = None) -> Path:
    """``tmp/data`` with a manifest, its source judged CSV (+ recipe sidecar) and a baseline."""
    root = tmp / "data"
    rollouts = root / "hinted_rollouts"
    baselines = root / "baselines"
    rollouts.mkdir(parents=True)
    baselines.mkdir()
    if manifest is None:
        manifest = pd.concat([
            manifest_rows(4, 1, 6, model=MODEL, run=RUN, stability=8),
            manifest_rows(2, 1, 2, model=MODEL, run=RUN, stability=6, start=100),
            manifest_rows(2, 0, 2, model=MODEL, run=RUN, hint="consensus", start=200),
            manifest_rows(1, 0, 1, model=MODEL, run="medqa-test_smoke", start=300),
        ], ignore_index=True)
        manifest["source_csv"] = f"{STEM}_judged.csv"
        manifest["subject_model_id"] = MODEL_ID
    baseline_csv = baselines / f"{MODEL}_{RUN}_baseline.csv"
    base = manifest.drop_duplicates("original_index")
    pd.DataFrame({
        "original_index": base["original_index"], "baseline_answer": base["baseline_modal_answer"],
        "n_top_votes": base["baseline_stability"], "sample_answers": '["A","A","A","A","A","A","A","A"]',
        "choices": '["a","b","c","d"]',
    }).to_csv(baseline_csv, index=False)
    source_rows(manifest).to_csv(rollouts / f"{STEM}_judged.csv", index=False)
    (rollouts / f"{STEM}.meta.json").write_text(json.dumps({
        "model_name": MODEL_ID, "baseline_csv": str(baseline_csv), "thinking": True,
        "temperature": 0.7, "seed": 42, "max_tokens": 16384, "max_model_len": 24576, "cases": "both",
    }))
    write_manifest(manifest, rollouts / "rollout_manifest.parquet", {"sources": [
        {"subject_model": MODEL, "subject_model_id": MODEL_ID, "run": RUN, "dataset": "medqa",
         "dataset_split": "test", "judge_prompt": "faithfulness_default.txt", "max_tokens": 16384,
         "source_csv": f"{STEM}_judged.csv", "baseline_n_samples": 8, "judge_model": "j",
         "baseline_csv": str(baseline_csv)},
        {"subject_model": MODEL, "subject_model_id": MODEL_ID, "run": "medqa-test_smoke", "dataset": "medqa",
         "dataset_split": "test", "judge_prompt": None, "max_tokens": 16384,
         "source_csv": f"{MODEL}_medqa-test_smoke_baseline_hinted_rollouts_judged.csv",
         "baseline_n_samples": 8, "judge_model": "j", "baseline_csv": None},
    ]})
    return root


def write_rerolls(root: Path, seeds=(43, 44, 45, 46), *, answer_for=None, judged: bool = True) -> None:
    """Fabricate ``resample/rollouts/<STEM>_rs<seed>[_judged].csv`` from the items file.

    ``answer_for(row, seed) -> letter | None`` picks each re-roll's answer
    (default: used candidates re-flip to the target on seeds 43–45 and keep
    the baseline on 46; controls keep the baseline).
    """
    items = pd.read_csv(root / "resample" / "items" / f"{STEM}_resample_items.csv", dtype=str, keep_default_na=False)
    out_dir = root / "resample" / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)

    def default_answer(row, seed):
        if row["role"] == "used_candidate":
            return row["hinted_answer"] if seed != 46 else row["baseline_answer"]
        return row["baseline_answer"]

    pick = answer_for or default_answer
    for seed in seeds:
        rows = []
        for _, it in items.iterrows():
            ans = pick(it, seed)
            to_hint = ans == it["hinted_answer"]
            rows.append({
                **{c: it[c] for c in ("original_index", "sample_type", "hint_name", "prompt", "hinted_prompt",
                                      "hinted_answer", "baseline_answer", "groundtruth")},
                "rollout": f"<think>cot</think><answer>{ans}</answer>" if ans else "<think>cut",
                "reasoning": "cot", "final_answer": ans or "", "additional_fields": it["additional_fields"],
                **({"judge_model": "j" if to_hint else "", "judge_label": (seed % 2) if to_hint else "",
                    "judge_confidence": "", "judge_reasoning": "", "judge_role": "", "judge_hint_quote": ""}
                   if judged else {}),
            })
        name = f"{STEM}_rs{seed}{'_judged' if judged else ''}.csv"
        pd.DataFrame(rows).to_csv(out_dir / name, index=False)
        (out_dir / f"{STEM}_rs{seed}.meta.json").write_text(json.dumps({"max_tokens": 24576, "seed": seed}))


# ---------------------------------------------------------------------------
# A synthetic resample manifest (MANIFEST_COLUMNS + resample_relabel's EXTRA_COLUMNS)
# ---------------------------------------------------------------------------

SEEDS = (43, 44, 45, 46)
PROBE_MODELS = ("m1", "m2")
PROBE_RUNS = (("medqa-test_0911", "medqa"), ("gpqa-diamond_0912", "gpqa"))
PROBE_STYLES = ("metadata", "consensus", "post_hoc", "tool_output")


def probe_manifest(*, models=PROBE_MODELS, runs=PROBE_RUNS, styles=PROBE_STYLES, n_used: int = 6,
                   n_ignored: int = 4, split_seed: int | None = 7) -> pd.DataFrame:
    """A 46-column resample manifest: per (model, run, style, case) cell ``n_used`` robust_used
    questions (re-rolls to the target, judged 0/1 with one label-0 re-roll in three questions of four,
    else two) and ``n_ignored`` robust_ignored controls (re-rolls on the modal answer), each question
    = the hinted-once original + 4 re-rolls; questions are shared across models (same ``question_id``).
    ``split_seed`` assigns the per-question split (None leaves it null)."""
    from src.lib.rollout_manifest import MANIFEST_COLUMNS, make_question_id, make_rollout_id
    from src.lib.splits import assign_splits
    from src.scripts.resample_relabel import EXTRA_COLUMNS

    rows = []
    for model in models:
        for run, dataset in runs:
            for s, style in enumerate(styles):
                for c, case in enumerate(("positive", "negative")):
                    kinds = ["used"] * n_used + ["ignored"] * n_ignored
                    for k, kind in enumerate(kinds):
                        idx = 1000 * s + 100 * c + k
                        sid = make_rollout_id(model, run, style, idx)
                        used = kind == "used"
                        base = {
                            "question_id": make_question_id(dataset, "test", idx), "dataset": dataset,
                            "dataset_split": "test", "original_index": idx, "subject_model": model,
                            "subject_model_id": f"org/{model}", "run": run, "hint_style": style, "case": case,
                            "target_option": "B", "groundtruth": "A" if case == "positive" else "B",
                            "baseline_modal_answer": "A", "baseline_stability": 8, "baseline_n_samples": 8,
                            "baseline_hint_votes": 0, "judge_prompt": "faithfulness_default.txt",
                            "judge_confidence": None, "judge_role": None, "hint_quote": None,
                            "trace_token_len": 100, "truncated": False, "max_tokens_used": 24576,
                            "exclude_reason": None, "split": None, "source_csv": f"{model}_{run}_judged.csv",
                            "source_rollout_id": sid, "role": "used_candidate" if used else "control",
                            "reliance_label": "robust_used" if used else "robust_ignored",
                            "contrast_a_set": "train", "paired_a": None, "paired_b": None,
                        }
                        rows.append({**base, "rollout_id": sid, "provenance": "hinted_once", "is_resample": False,
                                     "sample_seed": None, "model_answer": "B" if used else "A", "changed": used,
                                     "to_hint": used, "judge_model": "j" if used else None,
                                     "judge_label": 0 if used else None, "judge_label_final": 0 if used else None,
                                     "judge_label_final_model": "j" if used else None, "contrast_a": None,
                                     "contrast_b": None, "label_verbalised_rest": None, "label_used_ignored": None})
                        for j, seed in enumerate(SEEDS):
                            n_zero = 1 if k % 4 else 2
                            lab = (0 if j < n_zero else 1) if used else None
                            rows.append({
                                **base, "rollout_id": f"{sid}#rs{seed}", "provenance": "resample_k4",
                                "is_resample": True, "sample_seed": seed, "model_answer": "B" if used else "A",
                                "changed": used, "to_hint": used, "judge_model": "j" if used else None,
                                "judge_label": lab, "judge_label_final": lab,
                                "judge_label_final_model": "j" if used else None,
                                "contrast_a": "used" if used else "ignored", "contrast_b": lab,
                                "label_verbalised_rest": ("verbalised" if lab == 1 else "rest") if used else "rest",
                                "label_used_ignored": "used" if used else "ignored",
                            })
    df = pd.DataFrame(rows)[list(MANIFEST_COLUMNS) + EXTRA_COLUMNS]
    for col, dtype in MANIFEST_COLUMNS.items():
        df[col] = df[col].astype(dtype)
    if split_seed is not None:
        df = assign_splits(df, split_seed)
    return df
