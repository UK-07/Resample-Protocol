#!/usr/bin/env python3
"""Canonical baseline no-hint generation (vLLM, batched).

Runs the model without a hint over the configured dataset and writes one CSV
row per question (``original_index, question, choices, prompt, groundtruth,
baseline_answer, correct, baseline_status, additional_fields``; plus the
``sample_*`` / vote columns under consistency sampling) and a ``.meta.json``
sidecar with the effective settings. The CSV is appended batch by batch and
resumable: an existing CSV is extended only when its sidecar matches the
current config's resume identity.

Usage:
    python -m src.scripts.compute_baseline --config <flat config: the `compute_baseline` keys of
        configs/pipeline/run_pipeline.yaml laid over the model base; run_pipeline writes one per stage>
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.lib.baseline import (
    BASELINE_STATUSES,
    BASELINE_STATUS_CORRECT,
    baseline_status_from_vote,
    encode_additional_fields,
    threshold_vote,
)
from src.lib.config import (
    LEGACY_TOP_K,
    LEGACY_TOP_P,
    get_inference_config,
    get_sampling_config,
    load_config,
)
from src.lib.dataset import (
    DATASET_REGISTRY,
    MMLU_SPLIT_ALIASES,
    CommonsenseQADataset,
    load_data,
)
from src.lib.paths import resolve_data_path
from src.lib.model_utils import (
    build_chat_messages,
    generate_rollouts_batch,
    get_model_short_name,
    get_thinking_config,
    load_model_vLLM,
)
from src.lib.parsing import parse_answer_from_response

# Temperature used for consistency sampling when the config/CLI leave it unset.
DEFAULT_SAMPLING_TEMPERATURE = 0.7


def resolve_sampling(
    consistency_samples: int | None,
    temperature: float | None,
) -> tuple[int, float, bool]:
    """Reconcile sampling settings into ``(n, effective_temperature, enabled)``.

    ``consistency_samples`` in ``{None, 1}`` is a single greedy pass and a
    ``temperature`` is then an error; k > 1 samples at ``temperature`` or
    :data:`DEFAULT_SAMPLING_TEMPERATURE`. Raises ``ValueError`` on k < 1.
    """
    if consistency_samples is not None and consistency_samples < 1:
        raise ValueError(
            f"consistency_samples must be >= 1 (got {consistency_samples})."
        )
    enabled = consistency_samples is not None and consistency_samples > 1
    if not enabled:
        if temperature is not None:
            raise ValueError(
                "temperature may only be set when consistency_samples > 1 "
                "(greedy baseline always decodes at temperature 0.0)."
            )
        return 1, 0.0, False
    eff_temp = temperature if temperature is not None else DEFAULT_SAMPLING_TEMPERATURE
    return int(consistency_samples), float(eff_temp), True


def resolve_threshold(
    vote_threshold: int | None,
    n_rollouts: int,
    sampling_enabled: bool,
) -> int | None:
    """Vote count the top answer must reach: ``None`` when greedy (a threshold
    is then an error), else the given value in ``[1, n_rollouts]`` or the strict
    majority ``n_rollouts // 2 + 1``."""
    if not sampling_enabled:
        if vote_threshold is not None:
            raise ValueError(
                "vote_threshold may only be set when consistency_samples > 1 "
                "(the greedy baseline always accepts its single answer)."
            )
        return None
    if vote_threshold is None:
        return n_rollouts // 2 + 1
    if not (1 <= vote_threshold <= n_rollouts):
        raise ValueError(
            f"vote_threshold must be in [1, {n_rollouts}] (got {vote_threshold})."
        )
    return int(vote_threshold)


def build_loader_kwargs(
    dataset_cfg: dict, seed: int, random_answers_order: bool = False
) -> tuple[str, dict]:
    """Normalize a config ``dataset:`` block (``{"name", "params"}``) into
    ``(name, loader_kwargs)`` for ``load_data``: friendly param keys are mapped
    to the loader's kwargs, ``seed`` and ``random_answers_order`` injected.
    Raises ``ValueError`` on a dataset without a branch here."""
    name = str(dataset_cfg["name"]).strip().lower()
    params = dict(dataset_cfg.get("params") or {})
    if name == "mmlu":
        # Resolve split aliases here so the canonical split name is recorded in the sidecar.
        split = params.get("split", "test")
        return name, {
            "config": "all",
            "split": MMLU_SPLIT_ALIASES.get(split, split),
            "subject_filter": params.get("subjects"),
            "exclude_subjects": params.get("exclude_subjects"),
            "samples_per_subject": params.get("samples_per_subject"),
            "seed": seed,
            "random_answers_order": random_answers_order,
        }
    if name == "mmlu_pro":
        return name, {
            "split": params.get("split", "test"),
            "category_filter": params.get("categories"),
            "exclude_categories": params.get("exclude_categories"),
            "samples_per_category": params.get("samples_per_category"),
            "require_n_options": params.get("require_n_options"),
            "seed": seed,
            "random_answers_order": random_answers_order,
        }
    if name == "gpqa":
        return name, {
            "config": params.get("config", "gpqa_diamond"),
            "split": params.get("split", "train"),
            "seed": seed,
            "random_answers_order": random_answers_order,
        }
    if name == "medqa":
        return name, {
            "split": params.get("split", "test"),
            "seed": seed,
            "random_answers_order": random_answers_order,
        }
    if name == "commonsense_qa":
        split = params.get("split", "validation")
        # Rejected here so the misconfig fails before the model load, not in load_data after it.
        if str(split).strip().lower() == CommonsenseQADataset.UNLABELLED_SPLIT:
            raise ValueError(
                f"commonsense_qa split '{split}' has no gold labels (answerKey "
                "is empty on the Hub); use 'validation' or 'train'."
            )
        return name, {
            "split": split,
            "seed": seed,
            "random_answers_order": random_answers_order,
        }
    if name == "aqua":
        return name, {
            "split": params.get("split", "train"),
            "seed": seed,
            "random_answers_order": random_answers_order,
        }
    raise ValueError(
        f"Unsupported dataset '{name}'. Supported: "
        f"{', '.join(repr(k) for k in sorted(DATASET_REGISTRY))}."
    )


def derive_output_path(
    output_csv: str | None, model_name: str, dataset_name: str
) -> Path:
    """Explicit ``output_csv`` verbatim, else ``${DATA_ROOT}/baselines/<model>_<dataset>_baseline.csv``."""
    if output_csv:
        return resolve_data_path(output_csv)
    short = get_model_short_name(model_name)
    return resolve_data_path(
        f"${{DATA_ROOT}}/baselines/{short}_{dataset_name}_baseline.csv"
    )


# Sidecar fields that must match for a resume; knobs that don't affect the rows
# (batch_size, gpu_memory_utilization, tensor_parallel_size) are excluded.
RESUME_IDENTITY_FIELDS = [
    "model_name", "dataset", "consistency_samples", "temperature",
    "sampling_enabled", "vote_threshold", "max_samples", "split", "seed",
    "max_tokens", "random_answers_order", "thinking", "top_p", "top_k",
]
# Only compared while sampling: greedy decoding ignores the nucleus / top-k cut.
SAMPLING_ONLY_IDENTITY_FIELDS = ("top_p", "top_k")
# What a sidecar written before a field existed was actually produced with.
LEGACY_IDENTITY_DEFAULTS = {
    "random_answers_order": False,
    "thinking": True,
    "top_p": LEGACY_TOP_P,
    "top_k": LEGACY_TOP_K,
}


def apply_legacy_identity_defaults(existing_meta: dict) -> dict:
    """Fill :data:`LEGACY_IDENTITY_DEFAULTS` into ``existing_meta`` for the
    identity fields it predates (in place; returned for convenience)."""
    for key, value in LEGACY_IDENTITY_DEFAULTS.items():
        existing_meta.setdefault(key, value)
    return existing_meta


def assert_resume_compatible(existing_meta: dict, current_identity: dict) -> None:
    """Raise if any :data:`RESUME_IDENTITY_FIELDS` value differs between the
    existing ``.meta.json`` and the current config; the
    :data:`SAMPLING_ONLY_IDENTITY_FIELDS` are skipped when neither side samples."""
    greedy = not (existing_meta.get("sampling_enabled") or current_identity.get("sampling_enabled"))
    fields = [
        k for k in RESUME_IDENTITY_FIELDS
        if not (greedy and k in SAMPLING_ONLY_IDENTITY_FIELDS)
    ]
    mismatches = [
        (k, existing_meta.get(k), current_identity.get(k))
        for k in fields
        if existing_meta.get(k) != current_identity.get(k)
    ]
    if mismatches:
        lines = "\n".join(
            f"  - {k}: existing={e!r} current={c!r}" for k, e, c in mismatches
        )
        hint = ""
        if any(k in SAMPLING_ONLY_IDENTITY_FIELDS for k, _, _ in mismatches):
            legacy = {k: existing_meta.get(k) for k in SAMPLING_ONLY_IDENTITY_FIELDS}
            hint = (
                "\nThe existing rows were sampled with a different top_p/top_k "
                f"(a sidecar without the fields means vLLM's defaults, top_p {LEGACY_TOP_P} / "
                f"top_k {LEGACY_TOP_K}). To extend this CSV with the same sampling, set "
                f"{legacy} in the config; the shared defaults are what the hinted "
                "rollouts use."
            )
        raise ValueError(
            "Output CSV already exists but was produced by a different config:\n"
            f"{lines}\n"
            "Refusing to append incompatible rows. Delete the CSV (+ .meta.json) "
            f"to regenerate, or point output_csv at a new path.{hint}"
        )


def load_done_indices(csv_path: Path) -> set[int]:
    """Set of ``original_index`` values already recorded in an existing CSV."""
    try:
        existing = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return set()
    if "original_index" not in existing.columns:
        return set()
    return set(existing["original_index"].dropna().astype(int).tolist())


# How many existing rows the delimiter guard inspects on resume.
DELIMITER_GUARD_ROWS = 50


def assert_stored_samples_have_delimiter(
    csv_path: Path, delimiters, *, thinking_enabled: bool, n_rows: int = DELIMITER_GUARD_ROWS
) -> None:
    """Refuse to resume a thinking-on CSV none of whose first ``n_rows``
    non-empty ``sample_rollouts`` cells contains the close delimiter (its
    reasoning blocks are unrecoverable); an empty or column-less CSV passes."""
    if not thinking_enabled:
        return
    try:
        head = pd.read_csv(csv_path, nrows=n_rows, usecols=lambda c: c == "sample_rollouts")
    except (pd.errors.EmptyDataError, ValueError):
        return
    if "sample_rollouts" not in head.columns:
        return
    cells = [c for c in head["sample_rollouts"].dropna().astype(str) if c.strip()]
    if not cells:
        return
    close = delimiters.close.lower()
    if not any(close in c.lower() for c in cells):
        raise ValueError(
            f"{csv_path}: none of the first {len(cells)} stored samples contains the close "
            f"delimiter {delimiters.close!r} although thinking is on — the run that wrote "
            "them decoded with special tokens stripped, so its reasoning blocks are "
            "unrecoverable. Regenerate into a fresh CSV instead of resuming."
        )


def assert_columns_compatible(csv_path: Path) -> None:
    """Raise if an existing CSV lacks the ``baseline_status`` column: appending
    header-less rows would shift every column right of it."""
    try:
        header = pd.read_csv(csv_path, nrows=0).columns
    except pd.errors.EmptyDataError:
        return  # No header row yet → the header we write next is authoritative.
    if "baseline_status" not in header:
        raise ValueError(
            f"{csv_path} was written before the 'baseline_status' column existed; "
            "appending to it would shift columns and corrupt the file. Delete the "
            "CSV to regenerate from scratch."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--consistency-samples", type=int, default=None,
        help="Override consistency_samples: k>1 samples k rollouts/question.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override sampling temperature (only valid with consistency_samples > 1).",
    )
    parser.add_argument(
        "--vote-threshold", type=int, default=None,
        help="Override vote_threshold: min votes for the top answer to be accepted "
        "(only valid with consistency_samples > 1; default majority k//2 + 1).",
    )
    parser.add_argument(
        "--random_answers_order", action="store_true", default=None,
        help="Shuffle each question's answer options into a random per-question "
        "(seed-deterministic) order before prompting, remapping the groundtruth "
        "letter — guards against memorized answer letters. Overrides the YAML "
        "random_answers_order key. The permutation is recorded per row in "
        "additional_fields and the flag in the .meta.json sidecar.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_name: str = cfg["model_name"]
    dataset_cfg: dict = cfg["dataset"]
    random_answers_order: bool = bool(
        args.random_answers_order if args.random_answers_order is not None
        else cfg.get("random_answers_order", False)
    )
    dataset_name, loader_kwargs = build_loader_kwargs(
        dataset_cfg, cfg.get("seed", 42), random_answers_order
    )

    max_samples = cfg.get("max_samples")
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be >= 1 (got {max_samples}).")
    seed: int = cfg.get("seed", 42)
    batch_size: int = cfg.get("batch_size", 32)
    inference = get_inference_config(cfg)
    max_tokens: int = inference.max_tokens
    gpu_memory_utilization: float = inference.gpu_memory_utilization
    tensor_parallel_size: int = inference.tensor_parallel_size

    consistency_samples = (
        args.consistency_samples if args.consistency_samples is not None
        else cfg.get("consistency_samples")
    )
    temperature = (
        args.temperature if args.temperature is not None
        else cfg.get("temperature")
    )
    n_rollouts, eff_temperature, sampling_enabled = resolve_sampling(
        consistency_samples, temperature
    )
    vote_threshold_cfg = (
        args.vote_threshold if args.vote_threshold is not None
        else cfg.get("vote_threshold")
    )
    vote_threshold = resolve_threshold(vote_threshold_cfg, n_rollouts, sampling_enabled)
    sampling = get_sampling_config(cfg)
    # Greedy (k=1) has no threshold and always accepts its single answer → 1.
    eff_threshold = vote_threshold if vote_threshold is not None else 1

    output_csv = derive_output_path(cfg.get("output_csv"), model_name, dataset_name)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_csv.with_suffix(".meta.json")
    random.seed(seed)

    # Resolved before the resume check: `thinking` is an identity field.
    thinking = get_thinking_config(model_name, cfg.get("thinking"))
    system_prompt = thinking.system_prompt

    current_identity = {
        "model_name": model_name,
        "dataset": {"name": dataset_name, "params": dataset_cfg.get("params") or {}},
        "consistency_samples": n_rollouts,
        "temperature": eff_temperature,
        "sampling_enabled": sampling_enabled,
        "vote_threshold": vote_threshold,
        "max_samples": max_samples,
        "split": loader_kwargs.get("split"),
        "seed": seed,
        "max_tokens": max_tokens,
        "random_answers_order": random_answers_order,
        "thinking": thinking.enabled,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
    }
    done_indices: set[int] = set()
    if output_csv.exists():
        if not meta_path.exists():
            raise ValueError(
                f"{output_csv} exists but its sidecar {meta_path.name} is missing — "
                "cannot verify the config matches. Delete the CSV to regenerate."
            )
        with open(meta_path) as f:
            existing_meta = json.load(f)
        apply_legacy_identity_defaults(existing_meta)
        assert_resume_compatible(existing_meta, current_identity)
        assert_columns_compatible(output_csv)
        assert_stored_samples_have_delimiter(
            output_csv, thinking.delimiters, thinking_enabled=thinking.enabled
        )
        done_indices = load_done_indices(output_csv)
        print(f"Resuming: {len(done_indices)} rows already present in {output_csv}.")

    print(
        f"Thinking: {'on' if thinking.enabled else 'off'} "
        f"(mode={thinking.mode}, delimiters={thinking.delimiters.open!r}"
        f"/{thinking.delimiters.close!r})"
    )
    print(f"Loading {model_name} via vLLM …")
    model, tokenizer = load_model_vLLM(
        model_name,
        max_model_len=inference.max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=inference.max_num_seqs,
    )

    print(f"Loading dataset '{dataset_name}' ({loader_kwargs}) …")
    df, possible_options = load_data(dataset_name, **loader_kwargs)

    if df.empty:
        raise ValueError(
            f"Loaded 0 rows for dataset '{dataset_name}' ({loader_kwargs}). "
            "Check the split and dataset params — a typo'd filter matches nothing."
        )

    df_sampled = df
    if max_samples is not None:
        df_sampled = df_sampled.sample(
            n=min(int(max_samples), len(df_sampled)), random_state=seed
        )
    df_sampled = df_sampled.reset_index(drop=True)
    scope = "full selection" if max_samples is None else f"≤{max_samples} total"
    if sampling_enabled:
        mode_str = (
            f"consistency sampling k={n_rollouts} @ temp={eff_temperature} "
            f"top_p={sampling.top_p} top_k={sampling.top_k} seed={seed} "
            f"(accept top answer with ≥{vote_threshold} votes)"
        )
    else:
        mode_str = "greedy (temp=0.0)"
    print(f"Selected {len(df_sampled)} questions ({scope}); {mode_str}.")

    # Computed from the full selection so it is stable across resumes.
    additional_field_keys = sorted(
        {k for af in df_sampled["additional_fields"] for k in af}
    )

    def write_meta(*, complete: bool, n_rows: int, accuracy: float) -> None:
        """Write the run-level sidecar: up-front with ``complete=False`` so a
        crashed run stays resumable, then again with the final stats."""
        meta = {
            **current_identity,
            # Informational, not resume identity: `thinking` above is the identity field.
            "thinking_mode": thinking.mode,
            "reasoning_delimiters": [
                thinking.delimiters.open, thinking.delimiters.close
            ],
            "sampling_seed": seed if sampling_enabled else None,
            "additional_field_keys": additional_field_keys,
            "n_rows": int(n_rows),
            "accuracy": accuracy,
            "complete": complete,
            "output_csv": str(output_csv),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    write_meta(complete=False, n_rows=len(done_indices), accuracy=0.0)

    # Append per batch; the header is written only when the file is new/empty.
    header_written = output_csv.exists() and output_csv.stat().st_size > 0
    total = len(df_sampled)
    n_new = 0
    for start in range(0, total, batch_size):
        chunk = df_sampled.iloc[start : start + batch_size]
        chunk = chunk[~chunk["original_index"].astype(int).isin(done_indices)]
        if chunk.empty:
            continue
        batch_messages = [
            build_chat_messages(row["prompt"], system_prompt=system_prompt)
            for _, row in chunk.iterrows()
        ]
        rollouts = generate_rollouts_batch(
            model, tokenizer, batch_messages,
            n=n_rollouts, max_tokens=max_tokens, temperature=eff_temperature,
            top_p=sampling.top_p, top_k=sampling.top_k,
            seed=seed if sampling_enabled else None,
            enable_thinking=thinking.enable_thinking, delimiters=thinking.delimiters,
        )
        batch_records: list[dict] = []
        for (_, row), sample_texts in zip(chunk.iterrows(), rollouts):
            parsed = [
                (
                    parse_answer_from_response(
                        t,
                        option_letters=possible_options,
                        delimiters=thinking.delimiters,
                        thinking=thinking.enabled,
                        choices=list(row["choices"]),
                    )
                    or ""
                ).upper()
                for t in sample_texts
            ]
            answer, top_votes, self_consistency = threshold_vote(parsed, eff_threshold)
            groundtruth = str(row["groundtruth"]).upper()
            status = baseline_status_from_vote(answer, parsed, groundtruth)
            rec = {
                "original_index": int(row["original_index"]),
                "question": row["question"],
                "choices": json.dumps(list(row["choices"])),
                "prompt": row["prompt"],
                "groundtruth": groundtruth,
                "baseline_answer": answer,
                "correct": status == BASELINE_STATUS_CORRECT,
                "baseline_status": status,
                "additional_fields": encode_additional_fields(row["additional_fields"]),
            }
            if sampling_enabled:
                rec["sample_answers"] = json.dumps(parsed)
                rec["sample_rollouts"] = json.dumps(list(sample_texts))
                rec["n_top_votes"] = top_votes
                rec["self_consistency"] = self_consistency
                rec["n_correct_samples"] = sum(1 for a in parsed if a == groundtruth)
            batch_records.append(rec)
        pd.DataFrame(batch_records).to_csv(
            output_csv, mode="a", header=not header_written, index=False
        )
        header_written = True
        n_new += len(batch_records)
        done = min(start + batch_size, total)
        print(f"  {done}/{total} processed ({n_new} new rows written)")

    # Stats over the full CSV (old + new rows), not just this run's rows.
    results = pd.read_csv(output_csv)
    n_correct = int(results["correct"].astype(bool).sum())
    acc = n_correct / len(results) if len(results) else 0.0
    print(f"\nSaved {len(results)} rows ({n_new} new) → {output_csv}")
    print(f"Baseline accuracy: {n_correct}/{len(results)} ({100 * acc:.1f}%)")
    if "baseline_status" in results.columns:
        status_counts = results["baseline_status"].value_counts().to_dict()
        print(
            "Status breakdown: "
            + ", ".join(
                f"{s}={int(status_counts.get(s, 0))}" for s in BASELINE_STATUSES
            )
            + (
                f"  (inconsistent = top answer < {vote_threshold}/{n_rollouts} votes)"
                if sampling_enabled else ""
            )
        )

    write_meta(complete=True, n_rows=len(results), accuracy=acc)
    print(f"Wrote run metadata → {meta_path}")


if __name__ == "__main__":
    main()
