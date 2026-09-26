#!/usr/bin/env python3
"""Backfill the shared per-rollout manifest from the judged hinted-rollouts CSVs.

Walks every ``*_judged.csv`` in a directory (pair identity and judge provenance
resolved as in ``unfaithfulness_metrics_summary.py``) and writes one row per
rollout, as defined in :mod:`src.lib.rollout_manifest`, to a parquet file plus a
``.meta.json`` sidecar listing the sources. CSVs are streamed in chunks; the
baseline CSV named in the recipe sidecar (fallback
``${DATA_ROOT}/baselines/<model>_<dataset>_baseline.csv``) supplies the vote
stability and the current ``baseline_status``; ``*_smoke`` runs are skipped
unless ``--include-smoke``; ``judge_label_overrides.csv`` beside the manifest
sets ``judge_label_final`` on the re-judged rows.

    uv run python -m src.scripts.build_rollout_manifest
    uv run python -m src.scripts.build_rollout_manifest --only medqa --no-token-len
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.lib.paths import resolve_data_path
from src.lib.rollout_manifest import (
    DEFAULT_LABEL_OVERRIDES_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_NOISE_FLIP_MIN_VOTES,
    JUDGED_READ_COLUMNS,
    PairInfo,
    apply_label_overrides,
    dataset_identity,
    derive_rollout_rows,
    empty_manifest,
    is_smoke_run,
    judge_input_degraded,
    load_baseline_votes,
    pair_info_dict,
    read_label_overrides,
    read_manifest,
    read_manifest_meta,
    write_manifest,
)
from src.lib.splits import carry_over_splits
from src.scripts.unfaithfulness_metrics_summary import (
    delimiters_for,
    judge_provenance,
    pair_identity,
    read_json,
)


def model_id_for(short_name: str, sidecar: dict) -> str | None:
    """HF id of the rollout model: the recipe sidecar's, else the registry's."""
    if sidecar.get("model_name"):
        return str(sidecar["model_name"])
    from src.lib.model_utils import MODEL_CONFIGS

    for model_id, cfg in MODEL_CONFIGS.items():
        if cfg.get("short_name") == short_name:
            return model_id
    return None


def baseline_path_for(judged_csv: Path, sidecar: dict, model: str, dataset: str) -> Path | None:
    ref = sidecar.get("baseline_csv")
    candidates = [Path(str(ref))] if ref else []
    candidates.append(resolve_data_path(f"${{DATA_ROOT}}/baselines/{model}_{dataset}_baseline.csv"))
    for path in candidates:
        if path.exists():
            return path
    return None


# Traces per tokenizer call: the tokenizer materialises every id even when only lengths are wanted.
TOKENIZE_BATCH = 256


def make_token_counter(model_id: str | None, batch: int = TOKENIZE_BATCH):
    """A ``list[str] -> list[int]`` over the model's tokenizer, or None."""
    if not model_id:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def count(texts: list[str]) -> list[int]:
        lengths: list[int] = []
        for start in range(0, len(texts), batch):
            enc = tokenizer(texts[start:start + batch], add_special_tokens=False,
                            return_attention_mask=False, return_length=True)
            lengths.extend(int(n) for n in enc["length"])
        return lengths

    return count


def judge_models_in(chunk: pd.DataFrame) -> set[str]:
    """The judge slugs a chunk's ``judge_model`` column names (blank = none)."""
    if "judge_model" not in chunk.columns:
        return set()
    return {m for m in chunk["judge_model"].dropna().unique() if m.strip()}


def build_one(
    judged_csv: Path,
    *,
    chunk_rows: int,
    token_len: bool,
    noise_flip_min_votes: int,
) -> tuple[pd.DataFrame, dict]:
    """Manifest rows + a provenance record for one judged CSV."""
    sidecar = read_json(judged_csv.with_name(f"{judged_csv.stem.removesuffix('_judged')}.meta.json"))
    model, dataset = pair_identity(judged_csv, sidecar)
    # The judge model is resolved after the pass over the CSV has collected its column.
    _, judge_prompt = judge_provenance(judged_csv, set())
    model_id = model_id_for(model, sidecar)

    baseline_csv = baseline_path_for(judged_csv, sidecar, model, dataset)
    baseline_meta = read_json(baseline_csv.with_suffix(".meta.json")) if baseline_csv else {}
    ds_name, ds_split = dataset_identity(baseline_meta)
    n_samples = baseline_meta.get("consistency_samples")
    pair = PairInfo(
        subject_model=model,
        subject_model_id=model_id,
        run=dataset,
        dataset=ds_name or dataset,
        dataset_split=ds_split,
        judge_prompt=judge_prompt,
        max_tokens=int(sidecar["max_tokens"]) if sidecar.get("max_tokens") else None,
        source_csv=judged_csv.name,
        baseline_n_samples=int(n_samples) if n_samples and int(n_samples) > 1 else None,
    )
    baseline = load_baseline_votes(baseline_csv) if baseline_csv else None
    counter = make_token_counter(model_id) if token_len else None
    delimiters = delimiters_for(model_id)

    header = pd.read_csv(judged_csv, nrows=0).columns
    usecols = [c for c in JUDGED_READ_COLUMNS if c in header]
    degraded = judge_input_degraded(header)
    parts = []
    judges: set[str] = set()
    for chunk in pd.read_csv(judged_csv, usecols=usecols, chunksize=chunk_rows, dtype=str):
        judges |= judge_models_in(chunk)
        parts.append(derive_rollout_rows(
            chunk, pair, delimiters=delimiters, baseline=baseline,
            token_len=counter, noise_flip_min_votes=noise_flip_min_votes, degraded=degraded,
        ))
    rows = pd.concat(parts, ignore_index=True) if parts else empty_manifest()
    judge_model, _ = judge_provenance(judged_csv, judges)

    n_mismatch = 0
    if baseline is not None:
        stored = baseline["baseline_answer"].reindex(rows["original_index"].to_numpy())
        stored = stored.fillna("").astype(str).str.strip().str.upper().to_numpy()
        modal = rows["baseline_modal_answer"].fillna("").astype(str).to_numpy()
        n_mismatch = int((stored != modal).sum())
    stat = judged_csv.stat()
    record = {
        **pair_info_dict(pair),
        "judge_model": judge_model,
        "baseline_csv": str(baseline_csv) if baseline_csv else None,
        "source_size": stat.st_size,
        "source_mtime_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(timespec="seconds"),
        "n_rows": int(len(rows)),
        "n_switched": int(rows["to_hint"].fillna(False).sum()),
        "n_baseline_answer_mismatch": n_mismatch,
        "n_baseline_relabelled": int((rows["exclude_reason"] == "baseline_relabelled").sum()),
        "judge_input_degraded": degraded,
        "trace_token_len": bool(counter is not None),
    }
    return rows, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="${DATA_ROOT}/hinted_rollouts",
                        help="directory holding the *_judged.csv files")
    parser.add_argument("--output", default=DEFAULT_MANIFEST_PATH,
                        help=f"parquet output path (default {DEFAULT_MANIFEST_PATH})")
    parser.add_argument("--only", default=None,
                        help="only judged CSVs whose file name contains this substring")
    parser.add_argument("--chunk-rows", type=int, default=20000,
                        help="rows per read chunk; bounds peak memory on multi-GB CSVs")
    parser.add_argument("--no-token-len", action="store_true",
                        help="skip the tokenizer pass (trace_token_len stays null)")
    parser.add_argument("--include-smoke", action="store_true",
                        help="also build the *_smoke runs (skipped by default; the summary "
                             "must be built with the same flag)")
    parser.add_argument("--noise-flip-min-votes", type=int, default=DEFAULT_NOISE_FLIP_MIN_VOTES,
                        help="no-hint votes for the target option at which a switch is "
                             f"flagged noise_flip (default {DEFAULT_NOISE_FLIP_MIN_VOTES})")
    parser.add_argument("--label-overrides", default=DEFAULT_LABEL_OVERRIDES_PATH,
                        help="label-overrides CSV applied to judge_label_final "
                             f"(default {DEFAULT_LABEL_OVERRIDES_PATH}; absent = no overrides)")
    parser.add_argument("--no-label-overrides", action="store_true",
                        help="leave judge_label_final = judge_label on every row")
    args = parser.parse_args(argv)
    if args.noise_flip_min_votes < 1:
        # At 0 every switched row would be flagged noise_flip.
        parser.error("--noise-flip-min-votes must be >= 1")

    root = resolve_data_path(args.dir)
    output = resolve_data_path(args.output)
    paths = sorted(p for p in root.glob("*_judged.csv") if not args.only or args.only in p.name)
    skipped = []
    if not args.include_smoke:
        skipped = [{"source_csv": p.name, "reason": "smoke"} for p in paths if is_smoke_run(p.stem)]
        paths = [p for p in paths if not is_smoke_run(p.stem)]
    if not paths:
        print(f"no *_judged.csv under {root}", file=sys.stderr)
        return 1
    if skipped:
        print(f"skipping {len(skipped)} smoke run(s) (--include-smoke keeps them)")

    frames, sources = [], []
    for path in paths:
        print(f"reading {path.name} …", flush=True)
        rows, record = build_one(
            path, chunk_rows=args.chunk_rows, token_len=not args.no_token_len,
            noise_flip_min_votes=args.noise_flip_min_votes,
        )
        note = f"  {record['n_rows']} rows, {record['n_switched']} switched"
        if record["n_baseline_answer_mismatch"]:
            note += (f"; WARNING {record['n_baseline_answer_mismatch']} rows whose baseline "
                     "answer differs from the baseline CSV's")
        if record["n_baseline_relabelled"]:
            note += f"; {record['n_baseline_relabelled']} rows excluded as baseline_relabelled"
        if record["judge_input_degraded"]:
            note += "; WARNING judged without prompt/reasoning — every row judge_input_degraded"
        if record["baseline_csv"] is None:
            note += "; no baseline CSV (stability/votes null)"
        print(note, flush=True)
        frames.append(rows)
        sources.append(record)

    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest.sort_values(["subject_model", "run", "hint_style", "case", "original_index"],
                                    kind="stable").reset_index(drop=True)
    overrides_meta = None
    if not args.no_label_overrides:
        overrides_path = resolve_data_path(args.label_overrides)
        if not overrides_path.exists() and output.exists():
            previous_overrides = read_manifest_meta(output).get("label_overrides") or {}
            if previous_overrides.get("n_applied"):
                raise SystemExit(
                    f"{output} was built with {previous_overrides['n_applied']} label override(s) from "
                    f"{previous_overrides.get('path')}, but {overrides_path} does not exist: copy the "
                    "overrides file there, or pass --no-label-overrides to drop them deliberately."
                )
        overrides = read_label_overrides(overrides_path)
        manifest, report = apply_label_overrides(manifest, overrides, noise_flip_min_votes=args.noise_flip_min_votes)
        overrides_meta = {"path": str(overrides_path), "present": overrides_path.exists(), **report}
        if overrides_path.exists():
            print(f"label overrides: {report['n_applied']} of {report['n_overrides']} applied "
                  f"({report['n_unknown']} unknown rollout_id) from {overrides_path.name}; "
                  f"models {report['models']}")
    # A rebuild keeps the existing split assignment; new questions stay null for `assign_splits --extend`.
    split_meta = None
    if output.exists():
        previous = read_manifest(output)
        manifest, carried = carry_over_splits(manifest, previous)
        if carried["previous_had_split"]:
            split_meta = {**(read_manifest_meta(output).get("split") or {}), "carried_over": carried}
            print(f"split carried over from the existing manifest: {carried['n_carried']} rows "
                  f"({carried['n_questions_carried']} questions); {carried['n_unassigned']} rows unassigned"
                  + (" — run assign_splits --extend" if carried["n_unassigned"] else ""))
    write_manifest(manifest, output, {
        "rollouts_dir": str(root),
        "noise_flip_min_votes": args.noise_flip_min_votes,
        "label_overrides": overrides_meta,
        "split": split_meta,
        "sources": sources,
        "skipped_sources": skipped,
    })
    excluded = manifest["exclude_reason"].value_counts(dropna=False).to_dict()
    print(f"{len(manifest)} rollouts from {len(paths)} judged CSV(s) → {output}")
    print(f"exclude_reason: {json.dumps({str(k): int(v) for k, v in excluded.items()})}")
    print(f"judge_role filled: {int(manifest['judge_role'].notna().sum())} "
          f"of {int(manifest['judge_label'].notna().sum())} judged rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
