#!/usr/bin/env python3
"""Steps 3–5 of the re-sampling design: relabel the re-rolled questions and report.

GPU-free. Reads the selection (``select_resample_set.py``), the judged re-rolls
(``<source stem>_rs<seed>_judged.csv``; an unjudged ``_rs<seed>.csv`` is read when
no judged file exists, its switched rows counted as unjudged) and the per-rollout
manifest, and writes under ``--resample-dir``: ``resample_manifest.parquet``
(+ ``.meta.json``; the original row and its ``k`` re-rolls per selected
(question, hint) in the manifest's columns plus ``EXTRA_COLUMNS``; re-roll ids
are ``<source id>#rs<seed>`` and a re-roll is never ``noise_flip``),
``question_reliance.csv`` (re-roll outcome counts and ``reliance_label`` per
selected pair), ``survival_table.md`` (+ ``survival_*.csv``) and
``resample_relabel_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.lib.paths import resolve_data_path
from src.lib.resample import (
    ROBUST_MIN,
    assign_contrasts,
    filter_manifest,
    noise_model,
    reliance_labels,
    survival_table,
)
from src.lib.rollout_manifest import (
    DEFAULT_MANIFEST_PATH,
    JUDGED_READ_COLUMNS,
    MANIFEST_COLUMNS,
    PairInfo,
    derive_rollout_rows,
    load_baseline_votes,
    read_manifest,
)
from src.lib.selection import HINTED_ONCE, resample_provenance
from src.scripts.build_rollout_manifest import make_token_counter
from src.scripts.unfaithfulness_metrics_summary import delimiters_for

DEFAULT_RESAMPLE_DIR = "${DATA_ROOT}/resample"
EXTRA_COLUMNS = [
    "provenance", "is_resample", "sample_seed", "source_rollout_id", "role", "reliance_label",
    "contrast_a", "contrast_a_set", "paired_a", "contrast_b", "paired_b",
    "label_verbalised_rest", "label_used_ignored",
]


def resampled_csv(rollouts_dir: Path, source_csv: str, seed: int) -> Path | None:
    """The judged re-roll file for (source, seed), else the raw one, else None."""
    stem = Path(source_csv).stem.removesuffix("_judged")
    judged = rollouts_dir / f"{stem}_rs{seed}_judged.csv"
    raw = rollouts_dir / f"{stem}_rs{seed}.csv"
    return judged if judged.exists() else raw if raw.exists() else None


def reroll_recipe(path: Path) -> dict:
    """The re-roll file's ``.meta.json`` recipe sidecar (``{}`` when absent)."""
    sidecar = path.with_name(f"{path.stem.removesuffix('_judged')}.meta.json")
    return json.loads(sidecar.read_text()) if sidecar.exists() else {}


def derive_resampled(path: Path, meta: dict, seed: int, selection: pd.DataFrame, *, token_len, chunk_rows: int) -> pd.DataFrame:
    """Manifest-shaped rows of one re-roll file, restricted to the selected pairs."""
    pair = PairInfo(
        subject_model=meta["subject_model"], subject_model_id=meta.get("subject_model_id"),
        run=meta["run"], dataset=meta["dataset"], dataset_split=meta.get("dataset_split") or "",
        judge_prompt=meta.get("judge_prompt"),
        max_tokens=reroll_recipe(path).get("max_tokens"), source_csv=path.name,
        baseline_n_samples=meta.get("baseline_n_samples"),
    )
    baseline_csv = meta.get("baseline_csv")
    baseline = load_baseline_votes(Path(baseline_csv)) if baseline_csv and Path(baseline_csv).exists() else None
    delimiters = delimiters_for(meta.get("subject_model_id"))
    keys = set(zip(selection["original_index"].astype(int), selection["hint_style"].astype(str)))
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in JUDGED_READ_COLUMNS if c in header]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_rows, dtype=str):
        hit = [(int(i), str(h)) in keys for i, h in zip(chunk["original_index"], chunk["hint_name"])]
        chunk = chunk[hit]
        if len(chunk):
            # Never noise_flip on a re-roll: the k re-rolls are the test that replaces that rule.
            parts.append(derive_rollout_rows(chunk, pair, delimiters=delimiters, baseline=baseline,
                                             token_len=token_len, noise_flip_min_votes=None))
    if not parts:
        return pd.DataFrame(columns=list(MANIFEST_COLUMNS))
    rows = pd.concat(parts, ignore_index=True)
    rows["source_rollout_id"] = rows["rollout_id"]
    rows["rollout_id"] = rows["rollout_id"] + f"#rs{seed}"
    rows["sample_seed"] = seed
    rows["is_resample"] = True
    return rows


def _seed_range(seeds: list[int]) -> str:
    seeds = sorted(set(seeds))
    if len(seeds) > 1 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}–{seeds[-1]}"
    return ", ".join(str(s) for s in seeds)


def render_survival(tables: dict[str, pd.DataFrame], noise: pd.DataFrame, k: int, *,
                    seeds: list[int], temperatures: list[float]) -> str:
    def fmt(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False, floatfmt=".3f")

    temperature = "/".join(str(t) for t in sorted(set(temperatures))) if temperatures else "unrecorded"
    lines = [
        "# Survival of single-rollout used labels under re-sampling",
        "",
        f"Every `to_hint` rollout (a single flip to the hint's target) was re-rolled k={k} times with its "
        f"stored hinted prompt (seeds {_seed_range(seeds)}, temperature {temperature}). "
        f"`robust_used` = ≥{ROBUST_MIN}/{k} re-rolls to "
        f"target, `weak_used` = 1–{ROBUST_MIN - 1}/{k}, `n_zero` = 0/{k} (the single flip was noise-shaped). "
        "`survival_rate` = robust_used / relabeled (95 % Wilson interval). Controls (matched on model, run, "
        f"hint, case, stability bin among rollouts that kept the baseline answer) are `robust_ignored` at "
        f"0/{k} to target and ≥{ROBUST_MIN}/{k} on the baseline modal answer.",
        "",
        "## Per hint style (pooled over models and datasets)",
        "",
        fmt(tables["by_hint"]),
        "",
        "## Per hint style × case",
        "",
        fmt(tables["by_hint_case"]),
        "",
        "## Per baseline-stability bin",
        "",
        fmt(tables["by_stability"]),
        "",
        "## Per model × run × hint style",
        "",
        fmt(tables["by_model_hint"]),
        "",
        "## Noise model vs. observed",
        "",
        "`noise_share_of_to_hint` is the hinted-once noise model's predicted share of noise-shaped switches "
        "(off-target flips / (n_options − 2), over the to-target switches); `zero_rate` is the observed share "
        f"of used candidates with 0/{k} re-rolls to target, `non_robust_rate` the share below robust.",
        "",
        fmt(noise),
        "",
    ]
    return "\n".join(lines)


def noise_vs_observed(questions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Per hint style × case: the noise model's prediction next to the re-roll outcome.

    The noise model is computed over the (subject_model, run) pairs the selection covers only.
    """
    pairs = set(zip(questions["subject_model"].astype(str), questions["run"].astype(str)))
    in_scope = [(m, r) in pairs for m, r in zip(manifest["subject_model"].astype(str), manifest["run"].astype(str))]
    manifest = manifest[in_scope]
    used = questions[(questions["role"] == "used_candidate") & questions["reliance_label"].notna()]
    obs = used.groupby(["hint_style", "case"]).agg(
        n_relabeled=("rollout_id", "size"),
        zero_rate=("k_to_hint_count", lambda s: float((s == 0).mean())),
        non_robust_rate=("reliance_label", lambda s: float((s != "robust_used").mean())),
    )
    nm = noise_model(manifest).reset_index().groupby(["hint_style", "case"]).agg(
        n_to_hint=("n_to_hint", "sum"), noise_flip_estimate=("noise_flip_estimate", "sum"),
        prior_to_hint_estimate=("prior_to_hint_estimate", "sum"))
    nm["noise_share_of_to_hint"] = nm["noise_flip_estimate"] / nm["n_to_hint"]
    nm["prior_share_of_to_hint"] = nm["prior_to_hint_estimate"] / nm["n_to_hint"]
    return obs.join(nm[["noise_share_of_to_hint", "prior_share_of_to_hint"]], how="left").reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resample-dir", default=DEFAULT_RESAMPLE_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--no-token-len", action="store_true", help="skip the tokenizer pass on the re-rolls")
    parser.add_argument("--chunk-rows", type=int, default=20000)
    args = parser.parse_args(argv)

    root = resolve_data_path(args.resample_dir)
    selection = pd.read_csv(root / "resample_selection.csv", dtype={"stability_bin": str})
    sel_meta = json.loads((root / "resample_selection.meta.json").read_text())
    k = int(sel_meta["k"])
    seeds = [int(s) for s in sel_meta["sample_seeds"]]
    manifest = filter_manifest(read_manifest(args.manifest))
    rollouts_dir = root / "rollouts"

    frames, sources, missing, temperatures = [], [], [], []
    source_seed: dict[str, int | None] = {}
    for source_csv, group in selection.groupby("source_csv", sort=True):
        items_meta_path = root / "items" / f"{Path(source_csv).stem.removesuffix('_judged')}_resample_items.meta.json"
        if not items_meta_path.exists():
            missing.append(f"{source_csv}: no items sidecar")
            continue
        meta = json.loads(items_meta_path.read_text())
        source_seed[source_csv] = (meta.get("source_recipe") or {}).get("seed")
        counter = None if args.no_token_len else make_token_counter(meta.get("subject_model_id"))
        for seed in seeds:
            path = resampled_csv(rollouts_dir, source_csv, seed)
            if path is None:
                missing.append(f"{source_csv} seed {seed}: no re-roll file")
                continue
            rows = derive_resampled(path, meta, seed, group, token_len=counter, chunk_rows=args.chunk_rows)
            frames.append(rows)
            temperature = reroll_recipe(path).get("temperature")
            if temperature is not None:
                temperatures.append(float(temperature))
            sources.append({"source_csv": source_csv, "seed": seed, "file": path.name,
                            "judged": path.name.endswith("_judged.csv"), "n_rows": int(len(rows)),
                            "n_switched": int(rows["to_hint"].fillna(False).sum()),
                            "n_unjudged_switched": int((rows["to_hint"].fillna(False) & rows["judge_label"].isna()).sum())})
            print(f"{path.name}: {len(rows)} re-rolls ({sources[-1]['n_switched']} to target, "
                  f"{sources[-1]['n_unjudged_switched']} of them unjudged)", flush=True)
    for line in missing:
        print(f"  missing: {line}", file=sys.stderr)
    if not frames:
        print("no re-roll files found — nothing to relabel", file=sys.stderr)
        return 1
    resampled = pd.concat(frames, ignore_index=True)

    questions = reliance_labels(selection, resampled, k=k)
    original = manifest[manifest["rollout_id"].isin(selection["rollout_id"])].copy()
    original["source_rollout_id"] = original["rollout_id"]
    original["sample_seed"] = original["source_csv"].map(source_seed)
    original["is_resample"] = False
    original["provenance"] = HINTED_ONCE
    resampled["provenance"] = resample_provenance(k)
    rollouts = pd.concat([original, resampled], ignore_index=True)
    rollouts["role"] = rollouts["source_rollout_id"].map(questions.set_index("rollout_id")["role"])
    rollouts = assign_contrasts(rollouts, questions)
    rollouts = rollouts[list(MANIFEST_COLUMNS) + EXTRA_COLUMNS]
    rollouts = rollouts.sort_values(["subject_model", "run", "hint_style", "case", "original_index", "rollout_id"],
                                    kind="stable").reset_index(drop=True)

    tables = {
        "by_hint": survival_table(questions),
        "by_hint_case": survival_table(questions, ["hint_style", "case"]),
        "by_stability": survival_table(questions, ["stability_bin"]),
        "by_model_hint": survival_table(questions, ["subject_model", "run", "hint_style"]),
        "pooled": survival_table(questions, []),
    }
    noise = noise_vs_observed(questions, manifest)

    rollouts.to_parquet(root / "resample_manifest.parquet", index=False)
    questions.to_csv(root / "question_reliance.csv", index=False)
    for name, table in tables.items():
        table.to_csv(root / f"survival_{name}.csv", index=False)
    noise.to_csv(root / "survival_noise_vs_observed.csv", index=False)
    (root / "survival_table.md").write_text(
        render_survival(tables, noise, k, seeds=seeds, temperatures=temperatures))

    labels = questions["reliance_label"].value_counts(dropna=False).to_dict()
    report = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k, "sample_seeds": seeds, "robust_min": ROBUST_MIN, "rerolls_noise_flip": "disabled",
        "n_selected": int(len(selection)), "n_rerolls": int(len(resampled)),
        "n_relabeled": int(questions["reliance_label"].notna().sum()),
        "reliance_labels": {str(key): int(v) for key, v in labels.items()},
        "contrast_a": {f"{s}/{c}": int(n) for (s, c), n in
                       rollouts.groupby(["contrast_a_set", "contrast_a"]).size().items()},
        "paired_a_questions": int(rollouts.loc[rollouts["paired_a"].fillna(False), "source_rollout_id"].nunique()),
        "contrast_b_rollouts": int(rollouts["contrast_b"].fillna(False).sum()),
        "contrast_b_unfaithful": int((rollouts["contrast_b"].fillna(False)
                                      & (rollouts["judge_label_final"] == 0)).sum()),
        "paired_b_questions": int(rollouts.loc[rollouts["paired_b"].fillna(False), "source_rollout_id"].nunique()),
        "pooled_survival": tables["pooled"].iloc[0].to_dict(),
        "missing": missing,
        "sources": sources,
    }
    (root / "resample_relabel_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (root / "resample_manifest.meta.json").write_text(json.dumps({
        "written_utc": report["written_utc"], "n_rows": int(len(rollouts)),
        "columns": list(rollouts.columns), "manifest": str(resolve_data_path(args.manifest)),
        "selection_meta": sel_meta, "sources": sources,
    }, indent=2) + "\n")

    print(f"\n{len(rollouts)} rollouts ({len(original)} original + {len(resampled)} re-rolls) → {root / 'resample_manifest.parquet'}")
    print(f"reliance labels: {report['reliance_labels']}")
    print(f"contrast A: {report['contrast_a']}; paired-A questions: {report['paired_a_questions']}")
    print(f"contrast B: {report['contrast_b_rollouts']} rollouts ({report['contrast_b_unfaithful']} unfaithful); "
          f"paired-B questions: {report['paired_b_questions']}")
    print(tables["by_hint"].to_string(index=False, float_format="{:.3f}".format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
