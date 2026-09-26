#!/usr/bin/env python3
"""Label a hinted-rollouts CSV with the binary LLM faithfulness judge.

Judges the rows that switched to the hinted option (every non-blank rollout
under `--all`) and writes `<input stem>_judged.csv` with the `JUDGE_COLUMNS`
appended (`judge_label`: 0 unfaithful / 1 faithful / -1 incoherent; empty
outside the judged scope or on a judge error) plus a `.meta.json` sidecar.
Verdicts are cached per (judge model, prompt hash) under `--cache-dir`.

    uv run python -m src.scripts.judge_rollouts --config <flat config: the `judge_rollouts` keys of configs/pipeline/run_pipeline.yaml> [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.lib.config import load_config
from src.lib.hinted_rollouts import select_judgeable
from src.lib.llm_judge_verb import (
    BATCH_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    FAITHFULNESS_PROMPT,
    build_judge_prompt,
    judge_verdict,
    load_prompt_template,
    prompt_template_hash,
    run_judge_batch,
    template_style,
)
from src.lib.parsing import DEFAULT_DELIMITERS
from src.lib.paths import resolve_data_path

JUDGE_COLUMNS = [
    "judge_model", "judge_label", "judge_confidence", "judge_reasoning",
    "judge_role", "judge_hint_quote",
]


@dataclass(frozen=True)
class JudgeRolloutsConfig:
    """Every accepted config key with its default; the field names are the schema."""

    input_csv: str | None = None
    output_csv: str | None = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_prompt_file: str | None = None
    model_name: str | None = None
    cache_dir: str | None = None
    judge_all: bool = False
    workers: int = 4
    retry_errors: bool = True  # False keeps cached errors instead of retrying them
    judge_max_tokens: int = BATCH_JUDGE_MAX_TOKENS


# argparse dest -> the config field that flag overrides.
FLAG_TO_FIELD = {
    "input": "input_csv",
    "output": "output_csv",
    "judge_model": "judge_model",
    "judge_prompt_file": "judge_prompt_file",
    "model": "model_name",
    "cache_dir": "cache_dir",
    "judge_all": "judge_all",
    "workers": "workers",
    "retry_errors": "retry_errors",
    "judge_max_tokens": "judge_max_tokens",
}


def resolve_config(args: argparse.Namespace) -> JudgeRolloutsConfig:
    """Build the run config: flag > YAML > dataclass default; unknown YAML keys raise."""
    cfg = load_config(args.config) if args.config else {}
    known = {f.name for f in fields(JudgeRolloutsConfig)}
    unknown = sorted(set(cfg) - known - {"extends"})
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {args.config}: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(known))}"
        )

    values = {key: cfg[key] for key in known if cfg.get(key) is not None}
    for flag, field_name in FLAG_TO_FIELD.items():
        value = getattr(args, flag, None)
        # store_true flags are False (not None) when omitted; treat as unset.
        if value is None or (flag == "judge_all" and value is False):
            continue
        values[field_name] = value

    config = dataclasses.replace(JudgeRolloutsConfig(), **values)
    config = dataclasses.replace(
        config, workers=int(config.workers), judge_all=bool(config.judge_all),
        judge_max_tokens=int(config.judge_max_tokens),
    )
    if not config.input_csv:
        raise ValueError(
            "An input CSV is required: pass --input or set input_csv in --config."
        )
    return config


def attach_judgements(
    df: pd.DataFrame, records: dict, *, scope: pd.Index | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the judge columns filled from ``records``.

    Only rows in ``scope`` (default: ``select_judgeable(df).index``) get a verdict;
    every other row, and every errored record, keeps NA in every judge column.
    A record without ``hint_role`` / ``hint_quote`` leaves those cells NA.
    """
    out = df.copy()
    for col in JUDGE_COLUMNS:
        out[col] = pd.NA
    judge_idx = out.columns.get_indexer(JUDGE_COLUMNS)
    if scope is None:
        scope = select_judgeable(df).index
    in_scope = df.index.isin(scope)
    for pos, row in enumerate(out.itertuples(index=False)):
        if not in_scope[pos]:
            continue
        rec = records.get((int(row.original_index), str(row.hint_name)))
        if rec is None or rec.get("error") is not None:
            continue
        out.iloc[pos, judge_idx] = [
            rec["judge_model"], rec["label"], rec["confidence"], rec["reasoning"],
            rec.get("hint_role") if rec.get("hint_role") is not None else pd.NA,
            rec.get("hint_quote") if rec.get("hint_quote") is not None else pd.NA,
        ]
    return out


def verdict_counts(df: pd.DataFrame, records: dict) -> tuple[list[str], dict]:
    """Per-row verdicts for ``df``, and their counts."""
    verdicts = [
        judge_verdict(records.get((int(r.original_index), str(r.hint_name))))
        for r in df.itertuples(index=False)
    ]
    return verdicts, dict(Counter(verdicts))


def total_cost(records: dict) -> float:
    """Judge spend across every cached call, including retries."""
    return sum(rec.get("cost_usd") or 0.0 for rec in records.values())


def summarize(df: pd.DataFrame, records: dict) -> None:
    """Print verdict counts overall and per (sample_type, hint_name)."""
    verdicts, counts = verdict_counts(df, records)
    print(f"\nVerdicts: {counts}")
    by_group = (
        df.assign(verdict=verdicts)
        .groupby(["sample_type", "hint_name"])["verdict"]
        .value_counts()
        .unstack(fill_value=0)
    )
    print(by_group.to_string())
    cost = total_cost(records)
    if cost:
        print(f"\nTotal judge cost (all cached calls): ${cost:.4f}")


def write_meta(
    path: Path,
    *,
    config: JudgeRolloutsConfig,
    input_csv: Path,
    output_csv: Path,
    cache_dir: Path,
    prompt_template: str | None,
    to_judge: pd.DataFrame,
    records: dict,
    n_rows: int,
) -> None:
    """Write the run-provenance sidecar; the prompt is pinned by content hash
    (``cache_hash`` names the JSONL cache the labels came from)."""
    template = prompt_template if prompt_template is not None else FAITHFULNESS_PROMPT
    _, counts = verdict_counts(to_judge, records)
    meta = {
        "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "cache_dir": str(cache_dir),
        "judge_model": config.judge_model,
        "rollout_model": config.model_name,
        "judge_all": config.judge_all,
        "workers": config.workers,
        "retry_errors": bool(config.retry_errors),
        "judge_max_tokens": int(config.judge_max_tokens),
        "prompt": {
            "path": config.judge_prompt_file,
            "builtin": prompt_template is None,
            "style": template_style(template),
            "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "cache_hash": prompt_template_hash(prompt_template),
        },
        "counts": {
            "rows_in_csv": int(n_rows),
            "selected_for_judging": int(len(to_judge)),
            "verdicts": counts,
        },
        "cost_usd": round(total_cost(records), 6),
    }
    path.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label a hinted-rollouts CSV with the binary LLM faithfulness judge"
    )
    parser.add_argument(
        "--config", default=None,
        help="YAML config (keys: input_csv, output_csv, judge_model, "
             "judge_prompt_file, model_name, cache_dir, judge_all, workers, "
             "retry_errors, judge_max_tokens). "
             "Every CLI flag overrides its key.",
    )
    parser.add_argument(
        "--input", default=None,
        help="collect_hinted_rollouts output CSV (config key: input_csv)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Labeled output CSV (default: <input stem>_judged.csv beside the input)",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help=f"OpenRouter slug of the judge (default {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--judge-prompt-file", default=None,
        help="Custom judge prompt template file (default: the built-in FAITHFULNESS_PROMPT)",
    )
    parser.add_argument(
        "--model", default=None,
        help="HF id of the model that produced the rollouts, used to pick the "
             "reasoning delimiters (default: <think>/</think>). Only needed "
             "for old CSVs without a `reasoning` column.",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Judgement JSONL cache dir "
             "(default: ${DATA_ROOT}/judge_cache/<input stem>)",
    )
    parser.add_argument(
        "--all", action="store_true", dest="judge_all",
        help="Exploratory: judge every row with a non-blank rollout regardless "
             "of outcome — i.e. also rows whose answer did not change, drifted "
             "to a third option, or was unparseable (all outside the "
             "faithfulness definition). Default: only answers that switched to "
             "the hinted option.",
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent OpenRouter calls (default 4)")
    parser.add_argument(
        "--no-retry-errors", action="store_false", dest="retry_errors", default=None,
        help="Keep cached judge errors instead of retrying them; only rows with "
             "no cached record are sent to the API.",
    )
    parser.add_argument(
        "--judge-max-tokens", type=int, default=None, dest="judge_max_tokens",
        help=f"Completion budget per judge call (default {BATCH_JUDGE_MAX_TOKENS}; "
             "a `length` failure is retried at double the budget, up to 32768).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Render and print the prompt for the first judgeable row, then "
             "exit without calling the API or writing any file.",
    )
    args = parser.parse_args()
    config = resolve_config(args)

    # `model_utils` pulls in vLLM; import it only when a model was named.
    if config.model_name:
        from src.lib.model_utils import get_model_config

        delimiters = get_model_config(config.model_name)["delimiters"]
    else:
        delimiters = DEFAULT_DELIMITERS

    input_csv = resolve_data_path(config.input_csv)
    output_csv = (
        resolve_data_path(config.output_csv)
        if config.output_csv is not None
        else input_csv.with_name(f"{input_csv.stem}_judged.csv")
    )
    cache_dir = resolve_data_path(
        config.cache_dir
        if config.cache_dir is not None
        else f"${{DATA_ROOT}}/judge_cache/{input_csv.stem}"
    )
    prompt_template = (
        load_prompt_template(config.judge_prompt_file)
        if config.judge_prompt_file else None
    )

    df = pd.read_csv(input_csv)
    to_judge = select_judgeable(df, judge_all=config.judge_all)
    if config.judge_all:
        breakdown = to_judge["outcome"].value_counts().to_dict()
        scope = f"--all: every non-blank rollout regardless of outcome; {breakdown}"
    else:
        scope = "answer switched to the hinted option"
    print(f"{len(df)} rows in {input_csv.name} — judging {len(to_judge)} ({scope}).")

    if args.dry_run:
        if to_judge.empty:
            print("Nothing judgeable — no prompt to render.")
            return
        row = to_judge.iloc[0]
        print(
            f"\nPrompt for original_index={row['original_index']} "
            f"hint={row['hint_name']} "
            f"(template: {config.judge_prompt_file or 'built-in FAITHFULNESS_PROMPT'}, "
            f"hash {prompt_template_hash(prompt_template)})"
        )
        print("=" * 80)
        print(build_judge_prompt(
            row, delimiters=delimiters, prompt_template=prompt_template
        ))
        print("=" * 80)
        print(f"\nDry run: 1 of {len(to_judge)} judgeable row(s) rendered, none sent.")
        return

    records = run_judge_batch(
        to_judge, config.judge_model, cache_dir,
        delimiters=delimiters, max_workers=config.workers,
        prompt_template=prompt_template, retry_errors=bool(config.retry_errors),
        judge_max_tokens=config.judge_max_tokens,
    )

    labeled = attach_judgements(df, records, scope=to_judge.index)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_csv, index=False)
    print(f"\nLabeled CSV → {output_csv}")
    meta_path = output_csv.with_suffix(".meta.json")
    write_meta(
        meta_path,
        config=config,
        input_csv=input_csv,
        output_csv=output_csv,
        cache_dir=cache_dir,
        prompt_template=prompt_template,
        to_judge=to_judge,
        records=records,
        n_rows=len(df),
    )
    print(f"Run provenance → {meta_path}")
    summarize(to_judge, records)


if __name__ == "__main__":
    main()
