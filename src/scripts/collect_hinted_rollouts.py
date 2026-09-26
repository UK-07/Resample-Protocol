#!/usr/bin/env python3
"""Generate hinted rollouts with vLLM and append them to a CSV (no judge, no activations).

Selects baseline rows by case mode (`cases` / `--cases`: `positive_cases` — baseline-correct
rows, hint points at a wrong option; `negative_cases` — baseline-wrong rows with a parsed
answer, hint points at the correct option; `both`), builds one hinted prompt per
(question, hint style) and writes one row per rollout with columns OUTPUT_COLUMNS.
`prompt` is the no-hint baseline prompt, `hinted_prompt` the prompt sent (a JSON message
list for multi-turn hints), `groundtruth` is `HintResult.groundtruth`, `reasoning` the
extracted CoT and `final_answer` the parsed letter ("" when unparseable).

Resumable: (original_index, hint_name) pairs already in the output CSV are skipped and rows
are appended per chunk; hint randomness is seeded per (seed, original_index, hint_name).

Usage:
    python -m src.scripts.collect_hinted_rollouts --config <flat config: the `collect_hinted_rollouts` keys of configs/pipeline/run_pipeline.yaml>
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.distributed
from vllm import SamplingParams

from src.lib.baseline import (
    encode_additional_fields,
    load_baseline,
)
from src.lib.config import get_inference_config, get_sampling_config, load_config
from src.lib.hinted_rollouts import (
    CASE_MODES,
    DEFAULT_HINT_STYLES,
    OUTPUT_COLUMNS,
    apply_exclusion_list,
    assert_resumable_schema,
    build_hinted_items,
    load_done_pairs,
    load_example_pool,
    parse_rollout,
    resolve_hint_n_examples,
    resolve_option_letters,
    select_case_rows,
    validate_cases,
)
from src.lib.hints import HINTS
from src.lib.model_utils import (
    get_thinking_config,
    load_model_vLLM,
    load_tokenizer,
    sampling_kwargs_for,
    strip_terminal_special_tokens,
)
from src.lib.paths import resolve_data_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--hints", default=None,
        help="Comma-separated hint styles to run (overrides the YAML `hints` list).",
    )
    parser.add_argument(
        "--cases", default=None, choices=CASE_MODES,
        help="Which baseline rows to hint (overrides the YAML `cases` value).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_name: str = cfg["model_name"]
    baseline_csv = resolve_data_path(cfg["baseline_csv"])
    output_csv = resolve_data_path(cfg["output_csv"])
    inference = get_inference_config(cfg)
    max_tokens: int = inference.max_tokens
    temperature: float = cfg.get("temperature", 0.0)
    sampling = get_sampling_config(cfg)
    top_p: float = sampling.top_p
    top_k: int = sampling.top_k
    seed: int = cfg.get("seed", 42)
    chunk_size: int = cfg.get("chunk_size", 512)
    hint_n_examples: dict[str, int] = resolve_hint_n_examples(cfg)
    cases = validate_cases(args.cases or cfg.get("cases", "positive_cases"))

    hint_names = cfg.get("hints") or list(DEFAULT_HINT_STYLES)
    if args.hints is not None:
        hint_names = [h.strip() for h in args.hints.split(",") if h.strip()]
    unknown = [h for h in hint_names if h not in HINTS]
    if unknown:
        raise ValueError(f"Unknown hint style(s) {unknown}. Available: {list(HINTS)}")

    thinking = get_thinking_config(model_name, cfg.get("thinking"))
    system_prompt = thinking.system_prompt
    print(f"Thinking: {'on' if thinking.enabled else 'off'} (mode={thinking.mode})")

    print(f"Loading baseline CSV: {baseline_csv}")
    df_baseline = load_baseline(baseline_csv)
    df_baseline = apply_exclusion_list(df_baseline, cfg.get("exclusion_list"))
    df_selected = select_case_rows(df_baseline, cases)
    counts = df_selected["sample_type"].value_counts().to_dict()
    print(
        f"Baseline rows: {len(df_baseline)} — selected ({cases}): {len(df_selected)} "
        f"(positive: {counts.get('positive', 0)}, negative: {counts.get('negative', 0)})"
    )
    verified_pool = load_example_pool(cfg.get("sample_questions_csv"), df_baseline)
    # Mixed option widths -> per-question letters and the constant-width hints dropped.
    option_letters, hint_names = resolve_option_letters(
        df_selected, verified_pool, hint_names
    )
    print(f"Option letters: {option_letters or 'dynamic (per-question)'}")

    if output_csv.exists() and output_csv.stat().st_size > 0:
        assert_resumable_schema(output_csv)  # before the model load: fail fast
    done_pairs = load_done_pairs(output_csv)
    if done_pairs:
        print(f"Resuming: {len(done_pairs)} (question, hint) pairs already in {output_csv}.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_name)
    print(f"Building hinted prompts for hints: {hint_names} …")
    items, skipped = build_hinted_items(
        df_selected, verified_pool, hint_names, hint_n_examples,
        tokenizer, system_prompt,
        enable_thinking=thinking.enable_thinking, seed=seed, done_pairs=done_pairs,
        option_letters=option_letters,
    )
    print(f"Hinted rollouts queued: {len(items)} (skipped: {skipped})")
    if not items:
        print("Nothing to generate: every (question, hint) pair is already in the CSV.")
        return

    print(f"Loading model: {model_name} …")
    llm, _ = load_model_vLLM(
        model_name,
        max_model_len=inference.max_model_len,
        gpu_memory_utilization=inference.gpu_memory_utilization,
        tensor_parallel_size=inference.tensor_parallel_size,
        max_num_seqs=inference.max_num_seqs,
    )

    # Special-token delimiters must survive decoding; terminal tokens are stripped below.
    sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        top_k=top_k, seed=seed, **sampling_kwargs_for(tokenizer, thinking.delimiters),
    )

    header_written = output_csv.exists() and output_csv.stat().st_size > 0
    n_written = 0
    n_unparsed = 0
    try:
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            outputs = llm.generate([it["chat_text"] for it in chunk], sampling_params)
            rows = []
            for it, out in zip(chunk, outputs):
                rollout = strip_terminal_special_tokens(
                    out.outputs[0].text, tokenizer, thinking.delimiters
                )
                parsed = parse_rollout(
                    rollout,
                    option_letters=it["option_letters"],
                    delimiters=thinking.delimiters,
                    thinking=thinking.enabled,
                    choices=it["choices"],
                )
                if not parsed["final_answer"]:
                    n_unparsed += 1
                rows.append({
                    "original_index": it["original_index"],
                    "sample_type": it["sample_type"],
                    "hint_name": it["hint_name"],
                    "prompt": it["prompt"],
                    "hinted_prompt": it["hinted_prompt"],
                    "hinted_answer": it["hinted_answer"],
                    "baseline_answer": it["baseline_answer"],
                    "groundtruth": it["groundtruth"],
                    "rollout": rollout,
                    "reasoning": parsed["reasoning"],
                    "final_answer": parsed["final_answer"],
                    "additional_fields": encode_additional_fields(it["additional_fields"]),
                })
            pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
                output_csv, mode="a", header=not header_written, index=False
            )
            header_written = True
            n_written += len(rows)
            print(f"  {n_written}/{len(items)} rollouts written → {output_csv}")
    finally:
        del llm
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    print(
        f"\nDone. written={n_written}  unparsed_final_answer={n_unparsed}  "
        f"skipped={skipped}"
    )
    print(f"Output CSV → {output_csv}")


if __name__ == "__main__":
    main()
