"""Shared machinery for hinted-rollout generation: case selection, hinted-prompt
construction, rollout parsing, and the rollouts-CSV row contract the judges read.

Case modes (``CASE_MODES``): ``positive_cases`` — baseline-correct questions, the
hint points at a wrong option; ``negative_cases`` — baseline-wrong questions with
a parsed answer, the hint points at the correct option; ``both``.

Answer columns: ``hinted_answer`` — the letter the hint points at;
``final_answer`` — the letter parsed from the rollout; ``groundtruth`` —
``HintResult.groundtruth`` (always score against this letter).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

from src.lib.baseline import (
    build_verified_pool,
    collect_choice_widths,
    letters_for_width,
    load_exclusion_list,
    load_sample_questions,
)
from src.lib.hints import HINTS, HintInput, extract_hint_text, get_hinted_prompts
from src.lib.parsing import extract_cot, parse_answer_from_response
from src.lib.paths import resolve_data_path

CASE_MODES = ("positive_cases", "negative_cases", "both")

# Default hint set for rollout collection: every registered style except few_shot.
DEFAULT_HINT_STYLES = [name for name in HINTS if name != "few_shot"]

# Hints that need a non-empty verified-example pool to build their preamble.
POOL_DEPENDENT_HINTS = ("few_shot", "visual_pattern")

# Default preamble sizes of the pool-dependent hints; must match the
# ``n_examples`` defaults in ``src/lib/hints.py``.
DEFAULT_HINT_N_EXAMPLES: dict[str, int] = {
    "few_shot": 6,
    "visual_pattern": 6,
}


def resolve_hint_n_examples(cfg: dict | None) -> dict[str, int]:
    """Merge a config's ``hint_n_examples`` mapping over the shared defaults."""
    override = (cfg or {}).get("hint_n_examples") or {}
    return {
        **DEFAULT_HINT_N_EXAMPLES,
        **{str(k): int(v) for k, v in override.items()},
    }


def apply_exclusion_list(baseline_df: pd.DataFrame, exclusion_path) -> pd.DataFrame:
    """Drop rows whose ``original_index`` is in the exclusion list (``None`` = no-op)."""
    excluded = load_exclusion_list(
        resolve_data_path(exclusion_path) if exclusion_path else None
    )
    if not excluded:
        return baseline_df
    before = len(baseline_df)
    df = baseline_df[~baseline_df["original_index"].astype(int).isin(excluded)].copy()
    print(f"Exclusion list: dropped {before - len(df)} of {before} rows "
          f"({len(excluded)} indices reserved).")
    return df


def load_example_pool(sample_questions_path, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Example pool for pool-dependent hints: the curated CSV when a path is
    given, else the baseline's own verified pool."""
    if sample_questions_path:
        return load_sample_questions(resolve_data_path(sample_questions_path))
    return build_verified_pool(baseline_df)


def validate_cases(cases: str) -> str:
    if cases not in CASE_MODES:
        raise ValueError(f"Unknown cases mode '{cases}'. Valid modes: {list(CASE_MODES)}")
    return cases


def apply_chat_template(
    tokenizer, messages: list[dict], *, enable_thinking: bool | None
) -> str:
    """Render *messages* to prompt text; ``enable_thinking=None`` omits the kwarg."""
    # Lazy import: model_utils pulls in torch/vLLM, which GPU-free consumers never need.
    from src.lib.model_utils import template_kwargs

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs(enable_thinking),
    )


def select_case_rows(
    baseline_df: pd.DataFrame,
    cases: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Select baseline rows for the case mode, tagged with ``sample_type``.

    Negatives need a parsed baseline answer. ``limit`` caps each category
    independently, preserving baseline order; positives come first.
    """
    validate_cases(cases)
    correct = baseline_df["correct"].astype(bool)
    has_answer = baseline_df["baseline_answer"].apply(
        lambda v: isinstance(v, str) and bool(v.strip())
    )
    positive = baseline_df[correct].copy()
    positive["sample_type"] = "positive"
    negative = baseline_df[~correct & has_answer].copy()
    negative["sample_type"] = "negative"
    if limit is not None:
        positive = positive.head(limit)
        negative = negative.head(limit)
    if cases == "positive_cases":
        frames = [positive]
    elif cases == "negative_cases":
        frames = [negative]
    else:
        frames = [positive, negative]
    return pd.concat(frames, ignore_index=True)


def _parsed_baseline_answer(row: pd.Series) -> str:
    """The row's parsed baseline answer letter, "" when missing/unparsed."""
    raw = row.get("baseline_answer", "")
    return str(raw).upper() if isinstance(raw, str) and raw.strip() else ""


def resolve_option_letters(
    df_rows: pd.DataFrame,
    verified_pool: pd.DataFrame,
    hint_names: list[str],
    *,
    verbose: bool = True,
) -> tuple[list[str] | None, list[str]]:
    """Decide the run's answer-letter mode; returns ``(option_letters, active_hints)``.

    Constant choice width → static mode (one shared letter set, hints unchanged).
    Mixed widths → dynamic mode (``None``; each row derives its own letters in
    :func:`build_hinted_items`) and the constant-width hints are dropped. In
    static mode a pool whose width conflicts with the rows' drops the
    pool-dependent hints; an empty pool keeps them.

    Raises ``ValueError`` when no selected row carries a non-empty ``choices`` list.
    """
    def _drop_pool_hints(reason: str) -> list[str]:
        dropped = [h for h in hint_names if h in POOL_DEPENDENT_HINTS]
        if dropped and verbose:
            print(f"  Disabled constant-width hints {dropped}: {reason}.")
        return [h for h in hint_names if h not in POOL_DEPENDENT_HINTS]

    row_widths = collect_choice_widths(df_rows)
    if not row_widths:
        raise ValueError(
            "resolve_option_letters: no selected rows with a non-empty "
            "'choices' list."
        )
    if len(row_widths) > 1:
        if verbose:
            print(
                f"  Mixed option counts across rows {sorted(row_widths)} — "
                "using dynamic per-question option letters."
            )
        return None, _drop_pool_hints("mixed option counts in the selected rows")

    letters = letters_for_width(row_widths.pop())
    if any(h in POOL_DEPENDENT_HINTS for h in hint_names):
        pool_widths = collect_choice_widths(verified_pool)
        if pool_widths and pool_widths != {len(letters)}:
            return letters, _drop_pool_hints(
                f"example-pool option counts {sorted(pool_widths)} conflict "
                f"with the rows' ({len(letters)})"
            )
    return letters, list(hint_names)


def build_hinted_items(
    df_rows: pd.DataFrame,
    verified_pool: pd.DataFrame,
    hint_names: list[str],
    hint_n_examples: dict[str, int],
    tokenizer,
    system_prompt: str,
    *,
    enable_thinking: bool | None,
    seed: int,
    done_pairs: set[tuple[int, str]] | None = None,
    option_letters: list[str] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """One work item per not-yet-done (selected row, hint) pair.

    ``df_rows`` must carry ``sample_type``; negative rows get
    ``suggested_answer = groundtruth``. The example pool handed to a hint never
    contains the evaluated question. ``option_letters`` is the static letter
    set; ``None`` resolves the mode via :func:`resolve_option_letters` (quietly).

    Returns ``(items, skipped)``: item keys are original_index, sample_type,
    hint_name, question, prompt, chat_text, hinted_prompt (text, or a JSON
    message list for multi-turn hints), hinted_answer, baseline_answer,
    groundtruth, option_letters, choices, additional_fields; ``skipped`` counts
    build_failed / empty_pool / no_baseline_answer / bad_choices.
    """
    done_pairs = done_pairs or set()
    if option_letters is None:
        option_letters, hint_names = resolve_option_letters(
            df_rows, verified_pool, hint_names, verbose=False
        )
    dynamic = option_letters is None
    # Floor at the shared defaults so the pool slice covers what the hint will ask for.
    max_n = max(
        list(hint_n_examples.values()) + list(DEFAULT_HINT_N_EXAMPLES.values())
    )
    top_pool = verified_pool.head(max_n + 1)

    items: list[dict] = []
    skipped = {
        "build_failed": 0, "empty_pool": 0, "no_baseline_answer": 0,
        "bad_choices": 0,
    }
    for _, row in df_rows.iterrows():
        idx = int(row["original_index"])
        sample_type = str(row["sample_type"])
        question = str(row.get("question", "") or "")
        baseline_answer = _parsed_baseline_answer(row)
        groundtruth = str(row["groundtruth"]).upper()
        additional_fields = dict(row["additional_fields"] or {})
        choices = list(row["choices"] or [])
        if dynamic:
            try:
                row_letters = letters_for_width(len(choices))
            except ValueError:
                skipped["bad_choices"] += 1
                continue
        else:
            row_letters = option_letters
        row_pool = top_pool[top_pool["question"] != question]
        suggested_answer = groundtruth if sample_type == "negative" else None
        for hint_name in hint_names:
            if (idx, hint_name) in done_pairs:
                continue
            if hint_name in POOL_DEPENDENT_HINTS and len(row_pool) == 0:
                skipped["empty_pool"] += 1
                continue
            if hint_name == "pushback" and not baseline_answer:
                skipped["no_baseline_answer"] += 1
                continue
            hint_input = HintInput(
                prompt=row["prompt"],
                groundtruth=groundtruth,
                model_answer=baseline_answer,
                possible_options=row_letters,
                question=question,
                choices=choices,
                additional_fields=additional_fields,
                example_pool=row_pool,
                suggested_answer=suggested_answer,
                example_seed=seed,
            )
            # Per-(question, hint) seeding keeps the chosen option stable across chunking/resume.
            random.seed(f"{seed}:{idx}:{hint_name}")
            try:
                hint_res = get_hinted_prompts(
                    hint_input, [hint_name], hint_n_examples=hint_n_examples
                )[hint_name]
            except Exception as e:
                print(f"  [{idx}] {hint_name}: hint build failed: {e}")
                skipped["build_failed"] += 1
                continue
            if hint_res.messages is not None:
                messages = [{"role": "system", "content": system_prompt}] + hint_res.messages
                hinted_prompt_record = json.dumps(hint_res.messages, ensure_ascii=False)
            else:
                from src.lib.model_utils import build_chat_messages  # lazy: see apply_chat_template

                messages = build_chat_messages(hint_res.prompt, system_prompt)
                hinted_prompt_record = hint_res.prompt
            items.append({
                "original_index": idx,
                "sample_type": sample_type,
                "hint_name": hint_name,
                "question": question,
                "prompt": str(row["prompt"]),
                "chat_text": apply_chat_template(
                    tokenizer, messages, enable_thinking=enable_thinking
                ),
                "hinted_prompt": hinted_prompt_record,
                "hinted_answer": hint_res.hinted_answer,
                "baseline_answer": baseline_answer,
                "groundtruth": hint_res.groundtruth,
                "option_letters": row_letters,
                "choices": choices,
                "additional_fields": additional_fields,
            })
    return items, skipped


def parse_rollout(
    rollout: str,
    *,
    option_letters,
    delimiters,
    thinking: bool = True,
    choices=None,
) -> dict:
    """Parse one raw generation into ``{"reasoning", "final_answer"}``.

    ``reasoning`` is the thinking-block content (the whole stripped rollout when
    none is found); ``final_answer`` is the parsed letter, ``""`` when
    unparseable. ``option_letters`` must be the item's own set; ``choices`` arms
    the labelled-tag contradiction check.
    """
    final_answer = parse_answer_from_response(
        rollout, option_letters=option_letters, delimiters=delimiters, thinking=thinking,
        choices=choices,
    )
    reasoning = extract_cot(rollout, delimiters=delimiters) or str(rollout).strip()
    return {"reasoning": reasoning, "final_answer": final_answer or ""}


# ---------------------------------------------------------------------------
# Rollouts-CSV schema and resume (collect_hinted_rollouts / resample_hinted_rollouts)
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "original_index",
    "sample_type",
    "hint_name",
    "prompt",
    "hinted_prompt",
    "hinted_answer",
    "baseline_answer",
    "groundtruth",
    "rollout",
    "reasoning",
    "final_answer",
    "additional_fields",
]


def load_done_pairs(output_csv) -> set[tuple[int, str]]:
    """(original_index, hint_name) pairs already recorded — used to resume."""
    output_csv = Path(output_csv)
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()
    try:
        existing = pd.read_csv(output_csv, usecols=["original_index", "hint_name"])
    except (pd.errors.EmptyDataError, ValueError):
        return set()
    return {
        (int(idx), str(name))
        for idx, name in zip(existing["original_index"], existing["hint_name"])
        if pd.notna(idx)
    }


def assert_resumable_schema(output_csv) -> None:
    """Refuse to append to an existing CSV whose header differs from OUTPUT_COLUMNS."""
    output_csv = Path(output_csv)
    existing = list(pd.read_csv(output_csv, nrows=0).columns)
    if existing == OUTPUT_COLUMNS:
        return
    missing = [c for c in OUTPUT_COLUMNS if c not in existing]
    extra = [c for c in existing if c not in OUTPUT_COLUMNS]
    raise ValueError(
        f"Refusing to resume {output_csv}: its header does not match this "
        f"version's columns (missing: {missing or '-'}; unexpected: {extra or '-'}; "
        f"order-only mismatch: {not missing and not extra}). Point output_csv at a "
        "new path, or regenerate the file with this version."
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _norm_letter(value) -> str | None:
    return str(value).upper() if isinstance(value, str) and value.strip() else None


def compute_sensitivity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``response_changed`` / ``changed_in_direction`` columns (returns a copy).

    ``response_changed``: baseline and final answers both present and differ.
    ``changed_in_direction``: the final answer equals the hinted one (not gated
    on ``response_changed``, so a no-baseline row can still count).
    """
    out = df.copy()
    baseline = [_norm_letter(v) for v in out["baseline_answer"]]
    final = [_norm_letter(v) for v in out["final_answer"]]
    hinted = [_norm_letter(v) for v in out["hinted_answer"]]
    out["response_changed"] = [
        b is not None and f is not None and f != b for b, f in zip(baseline, final)
    ]
    out["changed_in_direction"] = [
        f is not None and h is not None and f == h for f, h in zip(final, hinted)
    ]
    return out


# ---------------------------------------------------------------------------
# Rollouts-CSV row contract (shared by every judge)
# ---------------------------------------------------------------------------


def render_hinted_prompt(hp) -> str:
    """Multi-turn hints store a JSON message list; render it readably for a judge."""
    s = str(hp).strip()
    if s.startswith("["):
        try:
            msgs = json.loads(s)
            return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in msgs)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    return s


def row_get(row, key):
    """``row[key]`` as a non-empty string, or None when absent/NaN/blank."""
    value = row.get(key) if hasattr(row, "get") else None
    # ``str(pd.NA)`` is "<NA>", which would leak into judge prompts and cache hashes.
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return text if text.strip() else None


def row_changed_to_hint(row) -> bool | None:
    """Whether the parsed final answer equals the hinted option (None if unknown)."""
    final, hinted = row_get(row, "final_answer"), row_get(row, "hinted_answer")
    if final is None or hinted is None:
        return None
    return final.strip().upper() == hinted.strip().upper()


def row_baseline_prompt(row) -> str:
    """Baseline (no-hint) prompt: the ``prompt`` column, else the rendered
    hinted prompt."""
    prompt = row_get(row, "prompt")
    if prompt is not None:
        return prompt
    return render_hinted_prompt(row["hinted_prompt"])


def row_hint_text(row) -> str:
    """Judge-facing hint excerpt, recomputed from ``hinted_prompt`` / ``prompt``
    (no CSV column stores it); "" when it cannot be recovered."""
    return extract_hint_text(
        row_get(row, "hint_name") or "", row_get(row, "hinted_prompt"), row_get(row, "prompt")
    )


def row_reasoning(row, delimiters) -> str:
    """Reasoning trace: the ``reasoning`` column, else the CoT extracted from
    ``rollout`` (whole stripped rollout when no thinking block); "" when blank."""
    reasoning = row_get(row, "reasoning")
    if reasoning is not None:
        return reasoning
    rollout = row_get(row, "rollout")
    if rollout is None:
        return ""
    return extract_cot(rollout, delimiters=delimiters) or rollout.strip()


def select_judgeable(df: pd.DataFrame, *, judge_all: bool = False) -> pd.DataFrame:
    """Rows a judge should score, with an ``outcome`` column
    (``switched_to_hint`` / ``unchanged`` / ``third_option`` / ``unparsed``).

    Default: rows whose final answer is the hinted option. ``judge_all`` keeps
    every row with a non-blank ``rollout`` regardless of outcome (exploratory).
    """
    flagged = compute_sensitivity_flags(df)
    has_rollout = flagged["rollout"].fillna("").astype(str).str.strip().astype(bool)
    final = flagged["final_answer"].fillna("").astype(str).str.strip()
    flagged["outcome"] = "third_option"
    flagged.loc[flagged["changed_in_direction"], "outcome"] = "switched_to_hint"
    flagged.loc[~flagged["response_changed"] & (final != ""), "outcome"] = "unchanged"
    flagged.loc[final == "", "outcome"] = "unparsed"
    if judge_all:
        return flagged[has_rollout]
    return flagged[flagged["changed_in_direction"] & has_rollout]
