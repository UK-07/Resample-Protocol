#!/usr/bin/env python3
"""Batch-judge hinted-rollouts CSVs with the binary LLM judge (GPU-free).

Labels every ``rollouts:`` CSV under the ``judge_rollouts.py`` contract (same
``select_judgeable`` scope, same ``${DATA_ROOT}/judge_cache/<stem>/`` cache),
writing ``<stem>_judged.csv`` and ``<stem>_judged_summary.json`` per input.
A bare filename resolves under ``${DATA_ROOT}/hinted_rollouts/``; globs expand
with this script's own ``*_judged.csv`` outputs excluded. A ``JudgeSetupError``
aborts the batch; other per-CSV failures are reported and exit 1 at the end.

Usage:
    uv run python -m src.scripts.batch_judge_rollouts \\
        --config configs/cueball/batch_judge_resample.yaml
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from src.lib.config import load_config, utc_now
from src.lib.hinted_rollouts import select_judgeable
from src.lib.llm_judge_verb import (
    BATCH_JUDGE_MAX_TOKENS,
    JudgeSetupError,
    check_judge_model,
    load_prompt_template,
    prompt_template_hash,
    run_judge_batch,
)
from src.lib.paths import resolve_data_path
from src.scripts.judge_rollouts import attach_judgements, summarize

# Batch-only default; the single-CSV judge_rollouts.py keeps DEFAULT_JUDGE_MODEL.
DEFAULT_BATCH_JUDGE_MODEL = "z-ai/glm-5.3-flash"

DEFAULT_ROLLOUTS_DIR = "${DATA_ROOT}/hinted_rollouts"

KNOWN_CONFIG_KEYS = {
    "rollouts", "judge_model", "judge_prompt_file", "judge_workers", "retry_errors",
    "judge_max_tokens",
}

_GLOB_CHARS = set("*?[")


def resolve_rollouts_entry(raw) -> list[Path]:
    """Resolve one ``rollouts:`` entry to existing CSV paths.

    A bare filename (or bare glob) lands under ``${DATA_ROOT}/hinted_rollouts/``;
    glob patterns expand, excluding this script's own ``*_judged.csv`` outputs.
    Raises ValueError when the file is missing or the glob matches nothing.
    """
    text = str(raw).strip()
    if "/" not in text and "$" not in text:
        text = f"{DEFAULT_ROLLOUTS_DIR}/{text}"
    resolved = resolve_data_path(text)
    if _GLOB_CHARS & set(str(resolved)):
        matches = sorted(
            p for p in (Path(m) for m in globlib.glob(str(resolved)))
            if p.suffix == ".csv" and not p.name.endswith("_judged.csv")
        )
        if not matches:
            raise ValueError(f"{raw}: glob matched no rollouts CSVs at {resolved}")
        return matches
    if not resolved.exists():
        raise ValueError(f"{raw}: rollouts CSV not found at {resolved}")
    return [resolved]


def build_targets(entries: list) -> tuple[list[Path], list[str]]:
    """Resolve every ``rollouts:`` entry; unresolvable ones become problem strings."""
    if not entries:
        raise ValueError("Config has no `rollouts:` list — nothing to judge.")
    targets: list[Path] = []
    problems: list[str] = []
    seen: set[Path] = set()
    for entry in entries:
        try:
            paths = resolve_rollouts_entry(entry)
        except ValueError as e:
            problems.append(str(e))
            continue
        for path in paths:
            if path not in seen:  # a glob and an explicit entry may overlap
                seen.add(path)
                targets.append(path)
    return targets, problems


def load_rollouts_sidecar(rollouts_csv: Path) -> dict:
    """The generation-recipe ``.meta.json`` beside the CSV ({} when absent or unreadable)."""
    meta_path = rollouts_csv.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def aggregate_judged(labeled: pd.DataFrame) -> list[dict]:
    """Per-(case, hint) switch and verdict counts for one judged CSV.

    ``n_switched`` counts the ``select_judgeable`` rows; ``unfaithful_rate`` is
    ``unfaithful / (faithful + unfaithful)`` — incoherent (-1) verdicts and
    judge errors leave the denominator.
    """
    if labeled.empty:
        return []
    judgeable = select_judgeable(labeled)
    out = []
    for (case, hint), group in labeled.groupby(["sample_type", "hint_name"], sort=True):
        switched = judgeable[
            (judgeable["sample_type"] == case) & (judgeable["hint_name"] == hint)
        ]
        if "judge_label" in switched.columns:
            labels = pd.to_numeric(switched["judge_label"], errors="coerce")
        else:
            labels = pd.Series(dtype=float)
        n_unfaithful = int((labels == 0).sum())
        n_faithful = int((labels == 1).sum())
        n_incoherent = int((labels == -1).sum())
        n_judged = n_unfaithful + n_faithful + n_incoherent
        denom = n_unfaithful + n_faithful
        out.append({
            "case": str(case),
            "hint_name": str(hint),
            "n_rollouts": int(len(group)),
            "n_switched": int(len(switched)),
            "switch_rate": round(len(switched) / len(group), 4) if len(group) else None,
            "n_judged": n_judged,
            "n_unfaithful": n_unfaithful,
            "n_faithful": n_faithful,
            "n_incoherent": n_incoherent,
            "n_judge_errors": int(len(switched) - n_judged),
            "unfaithful_rate": round(n_unfaithful / denom, 4) if denom else None,
        })
    return out


def judge_one(
    rollouts_csv: Path,
    *,
    judge_model: str,
    prompt_template: str | None,
    judge_workers: int,
    retry_errors: bool = True,
    judge_max_tokens: int = BATCH_JUDGE_MAX_TOKENS,
) -> dict:
    """Judge one rollouts CSV; write the judged CSV + summary JSON."""
    df = pd.read_csv(rollouts_csv)
    to_judge = select_judgeable(df)
    print(
        f"  {len(df)} rollouts — judging {len(to_judge)} "
        f"(switched to the hinted option) with {judge_model}"
    )
    # Same default cache location as judge_rollouts.py, so the two share caches.
    cache_dir = resolve_data_path(f"${{DATA_ROOT}}/judge_cache/{rollouts_csv.stem}")
    records = run_judge_batch(
        to_judge, judge_model, cache_dir,
        prompt_template=prompt_template, max_workers=judge_workers,
        retry_errors=retry_errors, judge_max_tokens=judge_max_tokens,
    )
    labeled = attach_judgements(df, records, scope=to_judge.index)
    judged_csv = rollouts_csv.with_name(f"{rollouts_csv.stem}_judged.csv")
    labeled.to_csv(judged_csv, index=False)
    print(f"  Labeled CSV → {judged_csv}")
    if len(to_judge):
        summarize(to_judge, records)

    summary = {
        "rollouts_csv": str(rollouts_csv),
        "judged_csv": str(judged_csv),
        "provenance": load_rollouts_sidecar(rollouts_csv),
        "judge_model": judge_model,
        "judge_prompt_hash": prompt_template_hash(prompt_template),
        "created_utc": utc_now(),
        "results": aggregate_judged(labeled),
    }
    summary_path = judged_csv.with_name(f"{judged_csv.stem}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON → {summary_path}")

    n_errors = sum(
        1 for row in to_judge.itertuples(index=False)
        if (rec := records.get((int(row.original_index), str(row.hint_name)))) is None
        or rec.get("error") is not None
    )
    status = "ok" if n_errors == 0 else f"ok ({n_errors} judge errors — rerun retries them)"
    return {
        "csv": rollouts_csv.name,
        "status": status,
        "n_rollouts": len(df),
        "n_judged": len(to_judge) - n_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-judge hinted-rollouts CSVs with the binary LLM judge"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--rollouts", default=None,
        help="Comma-separated rollouts CSV paths/globs (overrides the YAML "
             "`rollouts` list).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Only judge CSVs whose path contains this substring.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help=f"OpenRouter slug of the judge (overrides the YAML `judge_model`; "
             f"default {DEFAULT_BATCH_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--no-retry-errors", action="store_false", dest="retry_errors", default=None,
        help="Keep cached judge errors instead of retrying them; only rows with "
             "no cached record are sent to the API (overrides the YAML "
             "`retry_errors`, default true).",
    )
    parser.add_argument(
        "--judge-max-tokens", type=int, default=None, dest="judge_max_tokens",
        help=f"Completion budget per judge call (overrides the YAML "
             f"`judge_max_tokens`, default {BATCH_JUDGE_MAX_TOKENS}; a `length` "
             "failure is retried at double the budget, up to 32768).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the plan (CSVs, judge identity), then exit "
             "without judging.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    unknown = sorted(set(cfg) - KNOWN_CONFIG_KEYS - {"extends"})
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {args.config}: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(KNOWN_CONFIG_KEYS))}"
        )
    if args.rollouts is not None:
        cfg["rollouts"] = [p.strip() for p in args.rollouts.split(",") if p.strip()]

    judge_model = args.judge_model or cfg.get("judge_model") or DEFAULT_BATCH_JUDGE_MODEL
    judge_workers = int(cfg.get("judge_workers", 4))
    retry_errors = bool(cfg.get("retry_errors", True))
    if args.retry_errors is not None:
        retry_errors = args.retry_errors
    judge_max_tokens = int(args.judge_max_tokens or cfg.get("judge_max_tokens") or BATCH_JUDGE_MAX_TOKENS)
    prompt_template = (
        load_prompt_template(cfg["judge_prompt_file"])
        if cfg.get("judge_prompt_file") else None
    )

    targets, problems = build_targets(cfg.get("rollouts") or [])
    if args.only:
        needle = args.only.lower()
        targets = [t for t in targets if needle in str(t).lower()]

    print(f"Judge: {judge_model} (prompt hash {prompt_template_hash(prompt_template)}, "
          f"max_tokens {judge_max_tokens})")
    print(f"Rollout CSVs: {len(targets)}")
    for target in targets:
        judged = target.with_name(f"{target.stem}_judged.csv")
        note = "  (judged CSV exists — will be rewritten)" if judged.exists() else ""
        print(f"  - {target}{note}")
    for problem in problems:
        print(f"  ! skipped: {problem}")
    if not targets:
        raise ValueError("No judgeable rollouts CSVs (see the skip reasons above).")

    if args.dry_run:
        print("\nDry run — nothing judged.")
        return

    # Preflight: a bad slug or missing OPENROUTER_API_KEY fails before any CSV is read.
    check_judge_model(judge_model)

    summaries = []
    for i, target in enumerate(targets, start=1):
        print(f"\n=== [{i}/{len(targets)}] {target.name} ===")
        try:
            summaries.append(judge_one(
                target, judge_model=judge_model,
                prompt_template=prompt_template, judge_workers=judge_workers,
                retry_errors=retry_errors, judge_max_tokens=judge_max_tokens,
            ))
        except JudgeSetupError:
            raise  # a misconfigured judge fails identically for every CSV
        except Exception as e:
            traceback.print_exc()
            summaries.append({"csv": target.name, "status": f"error: {type(e).__name__}: {e}"})

    print("\n=== Batch summary ===")
    failed = 0
    for s in summaries:
        counts = ""
        if "n_rollouts" in s:
            counts = f"  (rollouts: {s['n_rollouts']}, judged: {s['n_judged']})"
        print(f"  {s['csv']}: {s['status']}{counts}")
        failed += not str(s["status"]).startswith("ok")
    for problem in problems:
        print(f"  skipped: {problem}")
    if failed or problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
