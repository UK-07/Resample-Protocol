#!/usr/bin/env python3
"""Re-roll the selected hinted prompts ``k`` times with vLLM (GPU; no judging).

For every items file under ``items_dir`` (``<source stem>_resample_items.csv`` +
``.meta.json``, from ``select_resample_set.py``) of the one model this process
runs (``--model``), rebuilds the chat text from the stored ``hinted_prompt`` and
generates one rollout per (item, sampling seed). Writes one CSV per seed under
``output_dir``, ``<source stem>_rs<seed>.csv`` (+ ``.meta.json`` recipe sidecar),
in ``collect_hinted_rollouts.py``'s schema; resumable per seed file by
(original_index, hint_name), and ``--seeds`` restricts a process to a subset of
the config's ``sample_seeds`` so each seed's CSV has exactly one writer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.lib.baseline import letters_for_width
from src.lib.config import get_inference_config, get_sampling_config, load_config, utc_now
from src.lib.hinted_rollouts import (
    OUTPUT_COLUMNS,
    apply_chat_template,
    assert_resumable_schema,
    load_done_pairs,
    parse_rollout,
)
from src.lib.model_utils import (
    get_model_short_name,
    get_thinking_config,
    sampling_kwargs_for,
    strip_terminal_special_tokens,
)
from src.lib.paths import resolve_data_path
from src.lib.resample import DEFAULT_SAMPLE_SEEDS, ITEM_CHOICES_COLUMN, ITEM_COLUMNS

KNOWN_CONFIG_KEYS = {
    "items_dir", "output_dir", "model_bases", "sample_seeds", "temperature",
    "top_p", "top_k", "inference", "chunk_size",
}


@dataclass(frozen=True)
class ItemsJob:
    """One items file: its rows, sidecar and the output stem."""

    items_csv: Path
    meta: dict
    model_name: str

    @property
    def stem(self) -> str:
        return self.items_csv.stem.removesuffix("_resample_items")


def output_path(output_dir: Path, stem: str, seed: int) -> Path:
    return output_dir / f"{stem}_rs{seed}.csv"


def load_jobs(items_dir: Path, model: str | None, only: str | None) -> list[ItemsJob]:
    """Every items file (with a sidecar naming its model), filtered to ``model`` / ``only``."""
    jobs = []
    for csv in sorted(items_dir.glob("*_resample_items.csv")):
        meta_path = csv.with_suffix(".meta.json")
        if not meta_path.exists():
            raise ValueError(f"{csv.name}: missing sidecar {meta_path.name}")
        meta = json.loads(meta_path.read_text())
        model_name = meta.get("model_name")
        if not model_name:
            raise ValueError(f"{meta_path.name}: no model_name")
        if only and only not in csv.name:
            continue
        if model and model not in (model_name, get_model_short_name(model_name)):
            continue
        jobs.append(ItemsJob(csv, meta, model_name))
    return jobs


def resolve_base(cfg: dict, config_path: Path, model_name: str) -> dict:
    """The per-model base config named by ``model_bases`` (checked to pin the same model)."""
    bases = cfg.get("model_bases") or {}
    ref = bases.get(model_name)
    if ref is None:
        raise ValueError(f"model_bases has no entry for {model_name!r} (have {sorted(bases)})")
    base = load_config(str(config_path.parent / ref))
    if base.get("model_name") != model_name:
        raise ValueError(f"{ref} pins model_name {base.get('model_name')!r}, mapped from {model_name!r}")
    if cfg.get("inference"):
        base["inference"] = {**(base.get("inference") or {}), **cfg["inference"]}
    return base


def chat_messages(hinted_prompt: str, system_prompt: str) -> list[dict]:
    """System prompt + the stored user turn (or the stored multi-turn message list)."""
    text = str(hinted_prompt)
    if text.strip().startswith("["):
        try:
            msgs = json.loads(text)
            if isinstance(msgs, list) and all(isinstance(m, dict) and "role" in m for m in msgs):
                return [{"role": "system", "content": system_prompt}] + msgs
        except json.JSONDecodeError:
            pass
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]


def item_choices(row) -> list[str] | None:
    """The item's option texts (JSON list in :data:`ITEM_CHOICES_COLUMN`), None when absent or unparsable."""
    raw = row.get(ITEM_CHOICES_COLUMN) if hasattr(row, "get") else None
    if raw is None or (isinstance(raw, float) and math.isnan(raw)) or not str(raw).strip():
        return None
    try:
        choices = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return [str(c) for c in choices] if isinstance(choices, list) else None


def build_items(rows: pd.DataFrame, seeds: list[int], done: dict[int, set], tokenizer, thinking) -> list[dict]:
    """One work item per (row, seed) not yet in that seed's CSV, with its chat text."""
    items = []
    for _, row in rows.iterrows():
        idx = int(row["original_index"])
        hint = str(row["hint_name"])
        chat_text = None
        for seed in seeds:
            if (idx, hint) in done.get(seed, set()):
                continue
            if chat_text is None:
                chat_text = apply_chat_template(
                    tokenizer, chat_messages(row["hinted_prompt"], thinking.system_prompt),
                    enable_thinking=thinking.enable_thinking,
                )
            items.append({**{c: row[c] for c in ITEM_COLUMNS}, "seed": seed, "chat_text": chat_text,
                          "option_letters": letters_for_width(int(row["n_options"])),
                          "choices": item_choices(row)})
    return items


def write_sidecar(job: ItemsJob, seed: int, out_csv: Path, base: dict, inference, temperature: float, k: int,
                  *, top_p: float | None = None, top_k: int | None = None) -> None:
    recipe = dict(job.meta.get("source_recipe") or {})
    sidecar = {
        "model_name": job.model_name,
        "baseline_csv": job.meta.get("baseline_csv"),
        "cases": recipe.get("cases"),
        "hints": sorted(set(pd.read_csv(job.items_csv, usecols=["hint_name"])["hint_name"].astype(str))),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "thinking": base.get("thinking"),
        "max_tokens": inference.max_tokens,
        "max_model_len": inference.max_model_len,
        "updated_utc": utc_now(),
        "resample": {
            "source_csv": job.meta.get("source_csv"),
            "run": job.meta.get("run"),
            "subject_model": job.meta.get("subject_model"),
            "items_csv": job.items_csv.name,
            "sample_seed": seed,
            "k": k,
            "source_recipe": recipe,
        },
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="HF id or short name of the one model to run")
    parser.add_argument("--only", default=None, help="only items files whose name contains this")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated subset of the config's sample_seeds this process generates "
                             "(default: all). Lets one cell's seeds run in separate processes: every "
                             "seed's CSV is written by exactly one process, so the split is safe by construction")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; load nothing")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = load_config(str(config_path))
    unknown = set(cfg) - KNOWN_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config key(s): {sorted(unknown)}")
    items_dir = resolve_data_path(cfg["items_dir"])
    output_dir = resolve_data_path(cfg["output_dir"])
    all_seeds = [int(s) for s in (cfg.get("sample_seeds") or DEFAULT_SAMPLE_SEEDS)]
    seeds = all_seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        unknown_seeds = sorted(set(seeds) - set(all_seeds))
        if unknown_seeds:
            parser.error(f"--seeds {unknown_seeds} not in the config's sample_seeds {all_seeds}")
    temperature = float(cfg.get("temperature", 0.7))
    chunk_size = int(cfg.get("chunk_size", 512))

    jobs = load_jobs(items_dir, args.model, args.only)
    if not jobs:
        print(f"no items files under {items_dir} match", file=sys.stderr)
        return 1
    models = sorted({j.model_name for j in jobs})
    if len(models) > 1:
        print(f"items files span several models {models}; pass --model to pick one", file=sys.stderr)
        return 1
    model_name = models[0]
    base = resolve_base(cfg, config_path, model_name)
    inference = get_inference_config(base)
    thinking = get_thinking_config(model_name, base.get("thinking"))
    if not thinking.enabled:
        raise ValueError("resampling needs thinking on (the CoT is what gets judged)")
    for job in jobs:
        # The re-roll must rebuild the source's chat text byte for byte (same system prompt / enable_thinking).
        source_thinking = (job.meta.get("source_recipe") or {}).get("thinking")
        if source_thinking is not None and bool(source_thinking) != thinking.enabled:
            raise ValueError(f"{job.items_csv.name}: the source run was generated with thinking="
                             f"{source_thinking!r} but this run resolves to thinking={thinking.enabled}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for job in jobs:
        n_rows = int(len(pd.read_csv(job.items_csv, usecols=["original_index"])))
        pending = {}
        for seed in seeds:
            out_csv = output_path(output_dir, job.stem, seed)
            if out_csv.exists() and out_csv.stat().st_size > 0:
                assert_resumable_schema(out_csv)
            pending[seed] = n_rows - len(load_done_pairs(out_csv))
        plan.append((job, n_rows, pending))
        print(f"{job.stem}: {n_rows} items × {len(seeds)} seeds; pending per seed {pending}")
    total = sum(sum(p.values()) for _, _, p in plan)
    print(f"model {model_name}: {total} rollouts to generate, max_tokens={inference.max_tokens}, "
          f"max_model_len={inference.max_model_len}, temperature={temperature}, seeds={seeds}"
          + (f" (of {all_seeds})" if seeds != all_seeds else ""))
    if args.dry_run or total == 0:
        return 0

    import torch
    import torch.distributed
    from vllm import SamplingParams

    from src.lib.model_utils import load_model_vLLM

    print(f"Loading model: {model_name} …")
    llm, tokenizer = load_model_vLLM(
        model_name, max_model_len=inference.max_model_len,
        gpu_memory_utilization=inference.gpu_memory_utilization,
        tensor_parallel_size=inference.tensor_parallel_size, max_num_seqs=inference.max_num_seqs,
    )
    special = sampling_kwargs_for(tokenizer, thinking.delimiters)  # Gemma 4: keep <|channel>/<channel|>
    sampling = get_sampling_config(cfg)
    params = {seed: SamplingParams(max_tokens=inference.max_tokens, temperature=temperature,
                                   top_p=sampling.top_p, top_k=sampling.top_k,
                                   seed=seed, **special) for seed in seeds}
    try:
        for job, _, _ in plan:
            rows = pd.read_csv(job.items_csv, dtype=str, keep_default_na=False)
            rows["n_options"] = rows["n_options"].astype(int)
            done = {seed: load_done_pairs(output_path(output_dir, job.stem, seed)) for seed in seeds}
            items = build_items(rows, seeds, done, tokenizer, thinking)
            print(f"{job.stem}: {len(items)} rollouts queued")
            n_written = 0
            for start in range(0, len(items), chunk_size):
                chunk = items[start:start + chunk_size]
                outputs = llm.generate([it["chat_text"] for it in chunk], [params[it["seed"]] for it in chunk])
                by_seed: dict[int, list[dict]] = {}
                for it, out in zip(chunk, outputs):
                    rollout = strip_terminal_special_tokens(out.outputs[0].text, tokenizer, thinking.delimiters)
                    parsed = parse_rollout(rollout, option_letters=it["option_letters"],
                                           delimiters=thinking.delimiters, thinking=thinking.enabled,
                                           choices=it["choices"])
                    by_seed.setdefault(it["seed"], []).append({
                        **{c: it[c] for c in OUTPUT_COLUMNS if c in it},
                        "rollout": rollout, "reasoning": parsed["reasoning"],
                        "final_answer": parsed["final_answer"],
                    })
                for seed, out_rows in by_seed.items():
                    out_csv = output_path(output_dir, job.stem, seed)
                    header = not (out_csv.exists() and out_csv.stat().st_size > 0)
                    pd.DataFrame(out_rows, columns=OUTPUT_COLUMNS).to_csv(out_csv, mode="a", header=header, index=False)
                n_written += len(chunk)
                print(f"  {n_written}/{len(items)} rollouts written")
            for seed in seeds:
                out_csv = output_path(output_dir, job.stem, seed)
                if out_csv.exists():
                    write_sidecar(job, seed, out_csv, base, inference, temperature, len(all_seeds),
                                  top_p=sampling.top_p, top_k=sampling.top_k)
    finally:
        del llm
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
