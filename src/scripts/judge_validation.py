#!/usr/bin/env python3
"""Validate the scale judge's labels against a frontier arbiter on a stratified sample.

Re-judges a stratified sample of the manifest's judged, clean ``to_hint`` rows
with an arbiter from another model family under the byte-identical prompt,
scores agreement, applies the decision rules, audits the per-style rates under
arbiter labels and writes the final-label overrides into the manifest.
Stages (``--stage``; ``all`` = sample, judge, report): ``sample`` writes
``frame.csv`` / ``sample_plan.csv`` / ``sample.csv``; ``judge`` buys the
arbiter verdicts (cached under ``<cache_dir>/<source stem>/``); ``report``
writes ``metrics.json``, ``decision.json``, ``gap_audit.json``,
``review_sheet.csv`` and ``judge_validation_report.md``; ``escalate`` judges
the slice the decision names (``--yes``); ``finalize`` writes
``judge_label_overrides.csv`` and the manifest's ``judge_label_final`` columns
(``--overrides-only`` leaves the manifest to the builder); ``score-review``
scores the researcher's review-sheet verdicts into ``review_scores.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.lib.config import load_config
from src.lib.hinted_rollouts import render_hinted_prompt, row_reasoning
from src.lib.judge_validation import (
    DEFAULT_DECISION,
    DEFAULT_LOW_CONFIDENCE_MAX,
    DEFAULT_OVERSAMPLE,
    DEFAULT_OVERSAMPLE_FACTOR,
    DEFAULT_SPILL,
    DEFAULT_TARGETS,
    PAIR_KEYS,
    STRATA,
    agreement_breakdowns,
    apply_decision_rules,
    assign_strata,
    build_review_sheet,
    cell_counts,
    draw_sample,
    gap_audit,
    plan_quotas,
    population_rates,
    render_report,
    score_review,
)
from src.lib.llm_judge_verb import (
    build_judge_prompt,
    check_judge_model,
    judge_cache_path,
    load_judged_records,
    load_prompt_template,
    prompt_template_hash,
    run_judge_batch,
)
from src.lib.parsing import DEFAULT_DELIMITERS
from src.lib.paths import resolve_data_path
from src.lib.rollout_manifest import (
    DEFAULT_LABEL_OVERRIDES_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_NOISE_FLIP_MIN_VOTES,
    OVERRIDE_COLUMNS,
    apply_label_overrides,
    read_label_overrides,
    read_manifest,
    read_manifest_meta,
    write_label_overrides,
    write_manifest,
)
from src.lib.string_judge import string_judge_case_record
from src.scripts.unfaithfulness_metrics_summary import PROMPT_DIR

STAGES = ("sample", "judge", "report", "escalate", "finalize", "score-review", "all")
FINALIZE_POLICIES = ("arbiter_where_judged", "researcher_adjudicated")
FRAME_COLUMNS = [
    "rollout_id", "question_id", "subject_model", "subject_model_id", "run", "hint_style", "case",
    "original_index", "source_csv", "target_option", "model_answer", "judge_model", "judge_prompt",
    "judge_label", "judge_confidence", "trace_token_len", "mention", "stratum",
]
ARBITER_FIELDS = ("label", "confidence", "hint_role", "hint_quote", "reasoning", "error", "cost_usd",
                  "prompt_tokens", "completion_tokens")
REVIEW_TEXT_COLUMNS = ["hinted_prompt_rendered", "reasoning_trace", "flash_reasoning"]
# Sidecar keys write_manifest sets itself; never carried over from the old meta.
_MANIFEST_SIDECAR_KEYS = ("schema_version", "written_utc", "n_rows", "columns")


# ------------------------------------------------------------- config ---

def resolve_settings(cfg: dict) -> dict:
    """The config with defaults filled and every path resolved."""
    arbiter = cfg.get("arbiter") or {}
    frame = cfg.get("frame") or {}
    strata = cfg.get("strata") or {}
    if not arbiter.get("judge_model"):
        raise ValueError("arbiter.judge_model is required (an OpenRouter slug)")
    pairs = frame.get("pairs") or []
    if not pairs:
        raise ValueError("frame.pairs must list at least one {subject_model, run}")
    primary = frame.get("primary") or pairs[0]
    if not any(p["subject_model"] == primary["subject_model"] and p["run"] == primary["run"] for p in pairs):
        raise ValueError(f"frame.primary {primary} is not one of frame.pairs")
    for block, keys, allowed in (("strata.targets", strata.get("targets") or {}, STRATA),
                                 ("strata.oversample", strata.get("oversample") or {}, STRATA),
                                 ("decision", cfg.get("decision") or {}, DEFAULT_DECISION)):
        unknown = sorted(set(keys) - set(allowed))
        if unknown:
            raise ValueError(f"{block}: unknown key(s) {unknown}; allowed {sorted(allowed)}")
    spill = strata.get("spill")
    oversample = strata.get("oversample")  # {} = no boost; only a missing key means the default
    return {
        "manifest": resolve_data_path(cfg.get("manifest") or DEFAULT_MANIFEST_PATH),
        "rollouts_dir": resolve_data_path(cfg.get("rollouts_dir") or "${DATA_ROOT}/hinted_rollouts"),
        "cache_dir": resolve_data_path(cfg.get("cache_dir") or "${DATA_ROOT}/judge_cache"),
        "output_dir": resolve_data_path(cfg["output_dir"]),
        "label_overrides": resolve_data_path(cfg.get("label_overrides") or DEFAULT_LABEL_OVERRIDES_PATH),
        "seed": int(cfg.get("seed", 42)),
        "chunk_rows": int(cfg.get("chunk_rows", 5000)),
        "arbiter": {
            "judge_model": str(arbiter["judge_model"]),
            "judge_prompt_file": arbiter.get("judge_prompt_file"),
            "workers": int(arbiter.get("workers", 4)),
            "usd_per_m_input": arbiter.get("usd_per_m_input"),
            "usd_per_m_output": arbiter.get("usd_per_m_output"),
            "prompt_overhead_tokens": int(arbiter.get("prompt_overhead_tokens", 4000)),
            "expected_output_tokens": int(arbiter.get("expected_output_tokens", 700)),
        },
        "frame": {
            "pairs": [{"subject_model": str(p["subject_model"]), "run": str(p["run"])} for p in pairs],
            "require_clean": bool(frame.get("require_clean", True)),
            "primary": f"{primary['subject_model']}|{primary['run']}",
        },
        "strata": {
            "low_confidence_max": float(strata.get("low_confidence_max", DEFAULT_LOW_CONFIDENCE_MAX)),
            "targets": {**DEFAULT_TARGETS, **{k: int(v) for k, v in (strata.get("targets") or {}).items()}},
            "oversample": {k: tuple(v) for k, v in (DEFAULT_OVERSAMPLE if oversample is None else oversample).items()},
            "oversample_factor": float(strata.get("oversample_factor", DEFAULT_OVERSAMPLE_FACTOR)),
            "spill": tuple(tuple(p) for p in spill) if spill is not None else DEFAULT_SPILL,
            "min_per_style": int(strata.get("min_per_style", 0)),
        },
        "decision": {**DEFAULT_DECISION, **(cfg.get("decision") or {})},
        "review": {"n_agreements": int((cfg.get("review") or {}).get("n_agreements", 20))},
        "gap_audit": {"n_boot": int((cfg.get("gap_audit") or {}).get("n_boot", 1000))},
        "report_notes": [str(n) for n in (cfg.get("report_notes") or [])],
        "finalize": {"policy": str((cfg.get("finalize") or {}).get("policy", "arbiter_where_judged"))},
    }


def source_stem(source_csv: str) -> str:
    """The judge-cache directory name of a judged CSV (its pre-judge stem)."""
    return Path(source_csv).stem.removesuffix("_judged")


def load_template(settings: dict):
    path = settings["arbiter"]["judge_prompt_file"]
    return load_prompt_template(path) if path else None


def check_prompt_matches_frame(frame_rows: pd.DataFrame, settings: dict) -> str:
    """The arbiter template's hash, checked against the prompt every frame row was judged under.

    ``judge_prompt`` names a shipped template (compared by content hash), an
    unresolved ``unknown (hash H)`` (compared by H) or the built-in default;
    a value that identifies no prompt cannot be checked and is reported.
    """
    arbiter_hash = prompt_template_hash(load_template(settings))
    for name in sorted(frame_rows["judge_prompt"].dropna().astype(str).unique()):
        if name.startswith("unknown (hash ") and name.endswith(")"):
            flash_hash = name[len("unknown (hash "):-1]
        elif (PROMPT_DIR / name).exists():
            flash_hash = prompt_template_hash((PROMPT_DIR / name).read_text())
        elif name == "built-in default":
            flash_hash = prompt_template_hash(None)
        else:
            print(f"[prompt] frame rows judged under {name!r}: not a shipped template, cannot check it against the arbiter's")
            continue
        if flash_hash != arbiter_hash:
            raise ValueError(
                f"arbiter prompt {settings['arbiter']['judge_prompt_file'] or 'built-in'} (hash {arbiter_hash}) differs "
                f"from the scale judge's {name} (hash {flash_hash}): the validation needs the byte-identical prompt"
            )
    return arbiter_hash


# --------------------------------------------------------- CSV access ---

def load_rows_for_keys(csv_path: Path, keys: set[tuple[int, str]], *, chunk_rows: int) -> pd.DataFrame:
    """The judged-CSV rows whose (original_index, hint_name) is in ``keys``; raises on a missing key."""
    parts = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
        pairs = pd.Series(list(zip(chunk["original_index"].astype(int), chunk["hint_name"].astype(str))), index=chunk.index)
        hit = pairs.isin(list(keys))
        if hit.any():
            parts.append(chunk[hit])
    rows = pd.concat(parts) if parts else pd.DataFrame()
    if len(rows) != len(keys):
        found = set(zip(rows["original_index"].astype(int), rows["hint_name"].astype(str))) if len(rows) else set()
        missing = sorted(keys - found)[:5]
        raise ValueError(f"{csv_path.name}: expected {len(keys)} rows, found {len(rows)}; e.g. missing {missing}")
    return rows.reset_index(drop=True)


def delimiters_for_source(csv_path: Path, subject_model_id: str | None):
    """Reasoning delimiters: only needed when the CSV has no ``reasoning`` column."""
    header = pd.read_csv(csv_path, nrows=0).columns
    if "reasoning" in header or not subject_model_id:
        return DEFAULT_DELIMITERS
    from src.lib.model_utils import get_model_config  # heavy import, only on old CSVs

    return get_model_config(subject_model_id)["delimiters"]


def key_set(df: pd.DataFrame) -> set[tuple[int, str]]:
    return set(zip(df["original_index"].astype(int), df["hint_style"].astype(str)))


def rows_by_source(frame_rows: pd.DataFrame, settings: dict) -> list[tuple[str, Path, pd.DataFrame, pd.DataFrame]]:
    """(stem, csv path, frame rows, CSV rows) per source CSV of ``frame_rows``."""
    out = []
    for source_csv, part in frame_rows.groupby("source_csv", sort=True):
        csv_path = settings["rollouts_dir"] / str(source_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"judged CSV not found: {csv_path}")
        rows = load_rows_for_keys(csv_path, key_set(part), chunk_rows=settings["chunk_rows"])
        out.append((source_stem(str(source_csv)), csv_path, part, rows))
    return out


# --------------------------------------------------------------- frame ---

def build_frame(manifest: pd.DataFrame, settings: dict) -> pd.DataFrame:
    f = settings["frame"]
    pair_ok = pd.Series(False, index=manifest.index)
    for p in f["pairs"]:
        pair_ok |= (manifest["subject_model"] == p["subject_model"]) & (manifest["run"] == p["run"])
    for p in f["pairs"]:
        if not ((manifest["subject_model"] == p["subject_model"]) & (manifest["run"] == p["run"])).any():
            raise ValueError(f"frame pair not in the manifest: {p}")
    keep = pair_ok & manifest["to_hint"].fillna(False).astype(bool) & manifest["judge_label"].isin([0, 1])
    if f["require_clean"]:
        keep &= manifest["exclude_reason"].isna()
    frame = manifest[keep].copy()
    if frame.empty:
        raise ValueError("the frame is empty: no judged, clean to_hint rows for the configured pairs")
    return frame


def mention_flags(frame: pd.DataFrame, settings: dict) -> pd.Series:
    """The string judge's label (1 = hint mentioned) for every label-0 frame row."""
    flags = pd.Series(pd.NA, index=frame.index, dtype="object")
    label0 = frame[frame["judge_label"] == 0]
    if label0.empty:
        return flags
    n_error = 0
    for stem, csv_path, part, rows in rows_by_source(label0, settings):
        model_id = part["subject_model_id"].dropna().astype(str).iloc[0] if part["subject_model_id"].notna().any() else None
        delimiters = delimiters_for_source(csv_path, model_id)
        by_key = {(int(r["original_index"]), str(r["hint_name"])): r for _, r in rows.iterrows()}
        for idx, fr in part.iterrows():
            rec = string_judge_case_record(by_key[(int(fr["original_index"]), str(fr["hint_style"]))], delimiters=delimiters)
            if rec.get("error") is not None:
                n_error += 1
                flags[idx] = 0
            else:
                flags[idx] = int(rec["label"])
        print(f"[sample] {stem}: string judge on {len(part)} label-0 rows")
    if n_error:
        print(f"[sample] {n_error} label-0 rows had a blank trace for the string judge (counted as no mention)")
    return flags


def stage_sample(settings: dict) -> None:
    out = settings["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(settings["manifest"])
    frame = build_frame(manifest, settings)
    check_prompt_matches_frame(frame, settings)
    frame["mention"] = mention_flags(frame, settings)
    frame["stratum"] = assign_strata(frame, low_confidence_max=settings["strata"]["low_confidence_max"])
    frame = frame[FRAME_COLUMNS].sort_values("rollout_id", kind="stable").reset_index(drop=True)
    frame.to_csv(out / "frame.csv", index=False)

    counts = cell_counts(frame)
    s = settings["strata"]
    plan = plan_quotas(counts, s["targets"], oversample=s["oversample"], oversample_factor=s["oversample_factor"],
                       spill=s["spill"], min_per_style=s["min_per_style"])
    plan.to_csv(out / "sample_plan.csv", index=False)
    sample = draw_sample(frame, plan, seed=settings["seed"])
    sample.to_csv(out / "sample.csv", index=False)

    print(f"frame: {len(frame)} rows → {out / 'frame.csv'}")
    summary = plan.groupby(PAIR_KEYS + ["stratum"], sort=False)[["n_available", "n_take"]].sum()
    print(summary.to_string())
    print(f"sample: {len(sample)} rows → {out / 'sample.csv'}")
    print_cost_estimate(sample, settings)


def print_cost_estimate(rows: pd.DataFrame, settings: dict) -> None:
    a = settings["arbiter"]
    trace = pd.to_numeric(rows["trace_token_len"], errors="coerce").fillna(5000)
    prompt_tokens = float((trace + a["prompt_overhead_tokens"]).sum())
    output_tokens = float(len(rows) * a["expected_output_tokens"])
    line = f"estimate for {len(rows)} rows: ~{prompt_tokens / 1e6:.2f} M prompt tokens, ~{output_tokens / 1e6:.2f} M output tokens"
    if a["usd_per_m_input"] is not None and a["usd_per_m_output"] is not None:
        usd = prompt_tokens / 1e6 * float(a["usd_per_m_input"]) + output_tokens / 1e6 * float(a["usd_per_m_output"])
        line += f" ≈ ${usd:.0f} at {a['judge_model']} prices"
    print(line)


# --------------------------------------------------------------- judge ---

def judge_rows(frame_rows: pd.DataFrame, settings: dict, *, dry_run: bool) -> dict:
    """Judge ``frame_rows`` with the arbiter; returns {stem: records}."""
    a = settings["arbiter"]
    template = load_template(settings)
    print(f"[judge] arbiter {a['judge_model']}, prompt {a['judge_prompt_file'] or 'built-in'} "
          f"(hash {check_prompt_matches_frame(frame_rows, settings)}), {len(frame_rows)} rows")
    print_cost_estimate(frame_rows, settings)
    if not dry_run:
        if not check_judge_model(a["judge_model"]):
            print("[judge] preflight request failed transiently; continuing (retries apply per row)")
    results = {}
    for stem, csv_path, part, rows in rows_by_source(frame_rows, settings):
        model_id = part["subject_model_id"].dropna().astype(str).iloc[0] if part["subject_model_id"].notna().any() else None
        delimiters = delimiters_for_source(csv_path, model_id)
        cache_dir = settings["cache_dir"] / stem
        cached = load_judged_records(judge_cache_path(cache_dir, a["judge_model"], template))
        todo = [k for k in key_set(part) if cached.get(k, {}).get("error", "missing") is not None]
        print(f"[judge] {stem}: {len(part)} rows, {len(part) - len(todo)} cached, {len(todo)} to judge")
        if dry_run:
            if len(rows):
                row = rows.iloc[0]
                print(f"\n--- prompt for original_index={row['original_index']} hint={row['hint_name']} ---")
                print(build_judge_prompt(row, delimiters=delimiters, prompt_template=template))
                print("--- end prompt ---\n")
            results[stem] = cached
            continue
        results[stem] = run_judge_batch(rows, a["judge_model"], cache_dir, delimiters=delimiters,
                                        prompt_template=template, max_workers=a["workers"])
        recs = [results[stem].get(k) for k in key_set(part)]
        n_err = sum(1 for r in recs if r is None or r.get("error") is not None)
        cost = sum(float(r.get("cost_usd") or 0.0) for r in recs if r)
        print(f"[judge] {stem}: done, {n_err} errors, cumulative cost of these verdicts ${cost:.2f}")
    return results


def stage_judge(settings: dict, *, dry_run: bool) -> None:
    sample = pd.read_csv(settings["output_dir"] / "sample.csv")
    judge_rows(sample, settings, dry_run=dry_run)


# -------------------------------------------------------------- report ---

def arbiter_records(frame_rows: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """``frame_rows`` plus ``arbiter_*`` columns from the arbiter's caches."""
    a = settings["arbiter"]
    template = load_template(settings)
    out = frame_rows.copy()
    for field in ARBITER_FIELDS:
        out[f"arbiter_{field}"] = None
    for source_csv, part in frame_rows.groupby("source_csv", sort=True):
        cache = load_judged_records(judge_cache_path(settings["cache_dir"] / source_stem(str(source_csv)), a["judge_model"], template))
        for idx, row in part.iterrows():
            rec = cache.get((int(row["original_index"]), str(row["hint_style"])))
            if rec is None:
                continue
            for field in ARBITER_FIELDS:
                out.at[idx, f"arbiter_{field}"] = rec.get(field)
    errored = out["arbiter_error"].notna()
    out.loc[errored, "arbiter_label"] = None
    out["arbiter_label"] = pd.to_numeric(out["arbiter_label"], errors="coerce")
    out["arbiter_model"] = a["judge_model"]
    out["arbiter_prompt_hash"] = prompt_template_hash(template)
    return out


def setup_block(sample: pd.DataFrame, frame: pd.DataFrame, settings: dict) -> dict:
    a = settings["arbiter"]
    s = settings["strata"]
    flash = sorted(frame["judge_model"].dropna().astype(str).unique())
    prompts = sorted(frame["judge_prompt"].dropna().astype(str).unique())
    pairs = ", ".join(f"{p['subject_model']} / {p['run']}" for p in settings["frame"]["pairs"])
    return {
        "flash_model": ", ".join(flash), "prompt_file": ", ".join(prompts) or (a["judge_prompt_file"] or "built-in"),
        "prompt_hash": check_prompt_matches_frame(frame, settings), "arbiter_model": a["judge_model"],
        "cache_dir": str(settings["cache_dir"]), "seed": settings["seed"],
        "frame_description": f"judged (0/1) to_hint rows{' with exclude_reason null' if settings['frame']['require_clean'] else ''} of {pairs}",
        "n_frame": int(len(frame)), "n_sample": int(len(sample)),
        "strata_description": (
            f"label 0 split by the string judge's mention flag (unfaithful_mention / unfaithful_no_mention); "
            f"label 1 split at confidence <= {s['low_confidence_max']} (faithful_low_conf / faithful_high_conf). "
            f"Targets per (model, run): {s['targets']}; oversampled styles {dict((k, list(v)) for k, v in s['oversample'].items())} "
            f"x{s['oversample_factor']}; spill {[list(p) for p in s['spill']]}; floor {s['min_per_style']} per style."
        ),
        "targets": s["targets"], "pairs": settings["frame"]["pairs"], "primary": settings["frame"]["primary"],
    }


def attach_review_texts(sheet: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """The rendered hinted prompt, the trace and the flash reasoning for the review rows."""
    sheet = sheet.copy()
    for col in REVIEW_TEXT_COLUMNS:
        sheet[col] = ""
    if sheet.empty:
        return sheet
    for stem, csv_path, part, rows in rows_by_source(sheet, settings):
        model_id = part["subject_model_id"].dropna().astype(str).iloc[0] if part["subject_model_id"].notna().any() else None
        delimiters = delimiters_for_source(csv_path, model_id)
        by_key = {(int(r["original_index"]), str(r["hint_name"])): r for _, r in rows.iterrows()}
        for idx, fr in part.iterrows():
            row = by_key[(int(fr["original_index"]), str(fr["hint_style"]))]
            sheet.at[idx, "hinted_prompt_rendered"] = render_hinted_prompt(row.get("hinted_prompt"))
            sheet.at[idx, "reasoning_trace"] = row_reasoning(row, delimiters)
            reasoning = row.get("judge_reasoning")
            sheet.at[idx, "flash_reasoning"] = "" if pd.isna(reasoning) else str(reasoning)
    return sheet


def sheet_has_verdicts(path: Path) -> bool:
    """True when a review sheet on disk carries at least one researcher verdict."""
    if not path.exists():
        return False
    existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    return "researcher_verdict" in existing.columns and bool((existing["researcher_verdict"].str.strip() != "").any())


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def write_report(settings: dict) -> Path:
    """Render ``judge_validation_report.md`` from whatever artifacts exist."""
    out = settings["output_dir"]
    metrics = read_json(out / "metrics.json")
    if metrics is None:
        raise FileNotFoundError("run the report stage first (metrics.json is missing)")
    plan = pd.read_csv(out / "sample_plan.csv")
    finalize = read_json(out / "finalize.json")
    review = read_json(out / "review_scores.json")
    text = render_report(
        setup=metrics["setup"], plan=plan, breakdowns=metrics["breakdowns"],
        decision=read_json(out / "decision.json"), gap=read_json(out / "gap_audit.json"),
        review=review, final_fingerprint=(finalize or {}).get("final_fingerprint"), finalize=finalize,
        notes=settings.get("report_notes"),
    )
    path = out / "judge_validation_report.md"
    path.write_text(text)
    return path


def stage_report(settings: dict) -> None:
    out = settings["output_dir"]
    frame = pd.read_csv(out / "frame.csv")
    sample = pd.read_csv(out / "sample.csv")
    judged = arbiter_records(sample, settings)
    judged.to_csv(out / "sample_judged.csv", index=False)
    missing = int(judged["arbiter_label"].isna().sum())
    if missing:
        print(f"[report] {missing} of {len(judged)} sample rows have no usable arbiter verdict (unjudged or errored)")

    breakdowns = agreement_breakdowns(judged)
    decision = apply_decision_rules(breakdowns, primary=settings["frame"]["primary"], thresholds=settings["decision"])
    gap = gap_audit(judged, frame, seed=settings["seed"], n_boot=settings["gap_audit"]["n_boot"],
                    primary=settings["frame"]["primary"])
    metrics = {"setup": setup_block(sample, frame, settings), "breakdowns": breakdowns,
               "arbiter_cost_usd": float(pd.to_numeric(judged["arbiter_cost_usd"], errors="coerce").fillna(0).sum()),
               "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(out / "metrics.json", metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "gap_audit.json", gap)

    sheet = build_review_sheet(judged, seed=settings["seed"], n_agreements=settings["review"]["n_agreements"])
    sheet = attach_review_texts(sheet, settings)
    sheet_path = out / "review_sheet.csv"
    if sheet_has_verdicts(sheet_path):
        sheet.to_csv(out / "review_sheet_regenerated.csv", index=False)
        print("[report] review_sheet.csv holds researcher verdicts and was kept; the fresh sheet is review_sheet_regenerated.csv")
    else:
        sheet.to_csv(sheet_path, index=False)
    path = write_report(settings)

    pooled = breakdowns["pooled"]
    print(f"[report] pooled: n={pooled['n_binary']} agreement={_fmt(pooled['agreement'])} kappa={_fmt(pooled['cohen_kappa'])}")
    for key, block in breakdowns["by_model"].items():
        print(f"[report] {key}: n={block['n_binary']} agreement={_fmt(block['agreement'])} kappa={_fmt(block['cohen_kappa'])}")
    for key, entry in gap["models"].items():
        if not entry["overall"]["weights_check_ok"]:
            print(f"[report] WARNING {key}: design weights do not reproduce the population flash rate "
                  f"({entry['overall']['n_frame_rows_uncovered']} frame rows in cells the sample does not cover)")
    print(f"[report] decision: {decision['outcome']}")
    if "cross_model" in gap:
        c = gap["cross_model"]
        print(f"[report] gap {c['models'][0]} − {c['models'][1]}: flash {_fmt(c['flash_population_gap'])}, "
              f"arbiter {_fmt(c['arbiter_gap'])} CI {c['arbiter_gap_ci95']} → survives: {c['gap_survives']}")
    print(f"[report] review sheet: {len(sheet)} rows ({int((sheet['review_kind'] == 'disagreement').sum())} disagreements) → {out / 'review_sheet.csv'}")
    print(f"[report] → {path}")


# ------------------------------------------------------------ escalate ---

def escalation_slice(frame: pd.DataFrame, decision: dict, settings: dict) -> pd.DataFrame:
    """The frame rows the decision says to re-judge (may be empty)."""
    outcome = decision["outcome"]
    if outcome == "rejudge_primary_all":
        model, run = settings["frame"]["primary"].split("|", 1)
        return frame[(frame["subject_model"] == model) & (frame["run"] == run)]
    if outcome == "rejudge_slices":
        keep = pd.Series(False, index=frame.index)
        for cell in decision["rule2_slices"]:
            model, run = cell["subject_model|run"].split("|", 1)
            col = "hint_style" if cell["kind"] == "hint_style" else "stratum"
            keep |= (frame["subject_model"] == model) & (frame["run"] == run) & (frame[col] == cell["cell"])
        return frame[keep]
    return frame.iloc[0:0]


def stage_escalate(settings: dict, *, yes: bool, dry_run: bool) -> None:
    out = settings["output_dir"]
    frame = pd.read_csv(out / "frame.csv")
    decision = read_json(out / "decision.json")
    if decision is None:
        raise FileNotFoundError("run the report stage first (decision.json is missing)")
    rows = escalation_slice(frame, decision, settings)
    print(f"[escalate] decision {decision['outcome']}: {len(rows)} frame rows in scope")
    if rows.empty:
        return
    judged = arbiter_records(rows, settings)
    todo = judged[judged["arbiter_label"].isna()]
    print(f"[escalate] {len(rows) - len(todo)} already judged by the arbiter, {len(todo)} to buy")
    if todo.empty:
        return
    if dry_run or not yes:
        print_cost_estimate(todo, settings)
        if not dry_run:
            print("[escalate] pass --yes to judge these rows")
        return
    # Recorded before judging so an interrupted run still tags what it bought.
    rows.assign(override_reason=f"escalation:{decision['outcome']}")[["rollout_id", "override_reason"]].to_csv(
        out / "escalation_rows.csv", index=False)
    judge_rows(todo, settings, dry_run=False)


# ------------------------------------------------------------ finalize ---

def build_overrides(manifest: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """One override row per arbiter verdict cached for the frame's sources."""
    a = settings["arbiter"]
    template = load_template(settings)
    out = settings["output_dir"]
    reasons = {}
    sample_path = out / "sample.csv"
    if sample_path.exists():
        reasons.update({rid: "validation_sample" for rid in pd.read_csv(sample_path)["rollout_id"].astype(str)})
    esc_path = out / "escalation_rows.csv"
    if esc_path.exists():
        esc = pd.read_csv(esc_path)
        reasons.update({rid: reason for rid, reason in zip(esc["rollout_id"].astype(str), esc["override_reason"].astype(str))
                        if rid not in reasons})
    pair_ok = pd.Series(False, index=manifest.index)
    for p in settings["frame"]["pairs"]:
        pair_ok |= (manifest["subject_model"] == p["subject_model"]) & (manifest["run"] == p["run"])
    rows = []
    for source_csv, part in manifest[pair_ok].groupby("source_csv", sort=True):
        cache = load_judged_records(judge_cache_path(settings["cache_dir"] / source_stem(str(source_csv)), a["judge_model"], template))
        if not cache:
            continue
        by_key = {(int(i), str(h)): rid for i, h, rid in zip(part["original_index"], part["hint_style"], part["rollout_id"])}
        for key, rec in cache.items():
            rid = by_key.get(key)
            if rid is None or rec.get("error") is not None or rec.get("label") not in (-1, 0, 1):
                continue
            rows.append({
                "rollout_id": rid, "judge_label_final": int(rec["label"]), "judge_label_final_model": a["judge_model"],
                "judge_confidence_final": rec.get("confidence"), "judge_role_final": rec.get("hint_role") or "",
                "hint_quote_final": rec.get("hint_quote") or "", "override_reason": reasons.get(rid, "arbiter_cache"),
                "judged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
    return pd.DataFrame(rows, columns=OVERRIDE_COLUMNS)


def merge_overrides(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """``new`` replaces ``existing`` rows with the same rollout_id; others are kept."""
    if existing.empty:
        return new.reset_index(drop=True)
    kept = existing[~existing["rollout_id"].astype(str).isin(new["rollout_id"].astype(str))]
    kept = kept.assign(judged_utc=kept["judged_utc"].astype(str))
    return pd.concat([kept, new], ignore_index=True)


def final_fingerprint(manifest: pd.DataFrame, settings: dict) -> dict:
    frame = build_frame(manifest, settings)
    frame = frame[frame["judge_label_final"].isin([0, 1])]
    return population_rates(frame, label_col="judge_label_final")


RESEARCHER_MODEL = "researcher"


def overrides_from_review(sheet: pd.DataFrame, manifest_labels: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Overrides carrying the researcher's verdicts (provenance ``researcher``), plus counts.

    ``flash``/``arbiter`` are valid on disagreement rows only, ``both`` on
    agreement controls only; ``neither``/``unsure``/blank rows get no override
    and are listed in the counts. ``manifest_labels`` (``judge_label`` by
    rollout_id) detects a flash verdict re-judged after the manifest was built.
    """
    verdicts = sheet["researcher_verdict"].fillna("").astype(str).str.strip().str.lower()
    rows, counts = [], {"flash": 0, "arbiter": 0, "both": 0, "neither": 0, "unsure": 0, "undecided": 0,
                        "flash_label_differs": 0}
    neither_ids, unsure_ids, differs_ids = [], [], []
    for (_, r), verdict in zip(sheet.iterrows(), verdicts):
        if verdict in ("flash", "arbiter", "both"):
            kind = str(r.get("review_kind", ""))
            expected = "agreement_control" if verdict == "both" else "disagreement"
            if kind != expected:
                raise ValueError(f"{r['rollout_id']}: verdict {verdict!r} is only valid on a {expected} row, this is {kind!r}")
            source = "arbiter" if verdict == "arbiter" else "judge"
            label = pd.to_numeric(r[f"{source}_label"], errors="coerce")
            if pd.isna(label) or int(label) not in (-1, 0, 1):
                raise ValueError(f"{r['rollout_id']}: researcher sided with {verdict} but that label is {r[f'{source}_label']!r}")
            if verdict != "arbiter":
                manifest_label = manifest_labels.get(str(r["rollout_id"]))
                if manifest_label is not None and not pd.isna(manifest_label) and int(manifest_label) != int(label):
                    counts["flash_label_differs"] += 1
                    differs_ids.append(str(r["rollout_id"]))
            rows.append({
                "rollout_id": str(r["rollout_id"]), "judge_label_final": int(label),
                "judge_label_final_model": RESEARCHER_MODEL,
                "judge_confidence_final": r.get(f"{source}_confidence"),
                "judge_role_final": (r.get("arbiter_hint_role") or "") if verdict == "arbiter" else "",
                "hint_quote_final": (r.get("arbiter_hint_quote") or "") if verdict == "arbiter" else "",
                "override_reason": {"flash": "researcher_sided_with_flash", "arbiter": "researcher_sided_with_arbiter",
                                    "both": "researcher_confirmed_agreement"}[verdict],
                "judged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            counts[verdict] += 1
        elif verdict == "neither":
            counts["neither"] += 1
            neither_ids.append(str(r["rollout_id"]))
        elif verdict == "unsure":
            counts["unsure"] += 1
            unsure_ids.append(str(r["rollout_id"]))
        else:
            counts["undecided"] += 1
    counts["neither_rollout_ids"] = neither_ids
    counts["unsure_rollout_ids"] = unsure_ids
    counts["flash_label_differs_rollout_ids"] = differs_ids
    return pd.DataFrame(rows, columns=OVERRIDE_COLUMNS), counts


def stage_finalize(settings: dict, review_path: Path | None = None, *, overrides_only: bool = False) -> None:
    """Write the overrides file and (unless ``overrides_only``) the manifest's final labels.

    ``overrides_only`` reads the parquet without schema validation and leaves
    the manifest to the builder's next rebuild.
    """
    out = settings["output_dir"]
    if overrides_only:
        manifest = pd.read_parquet(settings["manifest"])
    else:
        manifest = read_manifest(settings["manifest"])
    noise_flip_min_votes = int(read_manifest_meta(settings["manifest"]).get("noise_flip_min_votes") or DEFAULT_NOISE_FLIP_MIN_VOTES)
    policy = settings["finalize"]["policy"]
    if policy not in FINALIZE_POLICIES:
        raise ValueError(f"finalize.policy must be one of {FINALIZE_POLICIES}")
    if policy == "researcher_adjudicated":
        sheet_path = review_path or (out / "review_sheet.csv")
        if not sheet_has_verdicts(sheet_path):
            raise ValueError(f"{sheet_path} carries no researcher verdicts; fill it in (or use policy arbiter_where_judged)")
        sheet = pd.read_csv(sheet_path, keep_default_na=False)
        manifest_labels = pd.Series(pd.to_numeric(manifest["judge_label"], errors="coerce").astype(float).to_numpy(),
                                    index=manifest["rollout_id"].astype(str))
        new, review_counts = overrides_from_review(sheet, manifest_labels)
        print(f"[finalize] policy {policy}: {len(new)} reviewed rows take the researcher's label "
              f"(sided with flash {review_counts['flash']}, with the arbiter {review_counts['arbiter']}, "
              f"confirmed agreements {review_counts['both']}); {review_counts['neither']} 'neither' and "
              f"{review_counts['unsure'] + review_counts['undecided']} unsure/undecided rows keep the scale judge's label")
        if review_counts["flash_label_differs"]:
            print(f"[finalize] {review_counts['flash_label_differs']} flash-sided rows carry a re-judged flash verdict "
                  f"that differs from the manifest's label — they change the manifest label too (ids in finalize.json)")
        if new.empty:
            raise ValueError(f"{sheet_path} holds no decided flash/arbiter/both verdict; nothing to finalize")
    else:
        new = build_overrides(manifest, settings)
        review_counts = None
        if new.empty:
            raise ValueError("no arbiter verdicts found for the frame's sources; run the judge stage first")
    existing = read_label_overrides(settings["label_overrides"])
    # The frame's pairs are replaced wholesale; rows re-derived with the same label and provenance are not stale.
    pair_ok = pd.Series(False, index=manifest.index)
    for pr in settings["frame"]["pairs"]:
        pair_ok |= (manifest["subject_model"] == pr["subject_model"]) & (manifest["run"] == pr["run"])
    frame_ids = set(manifest.loc[pair_ok, "rollout_id"].astype(str))
    in_pairs = existing["rollout_id"].astype(str).isin(frame_ids) if len(existing) else pd.Series([], dtype=bool)
    same = dict(zip(new["rollout_id"].astype(str), zip(new["judge_label_final"].astype(int), new["judge_label_final_model"].astype(str))))
    n_stale = sum(1 for rid, lab, mod in zip(existing.loc[in_pairs, "rollout_id"].astype(str),
                                             existing.loc[in_pairs, "judge_label_final"].astype(int),
                                             existing.loc[in_pairs, "judge_label_final_model"].astype(str))
                  if same.get(rid) != (lab, mod))
    existing = existing[~in_pairs] if len(existing) else existing
    merged = merge_overrides(existing, new)
    path = write_label_overrides(merged, settings["label_overrides"])
    # Same derivation as a rebuild: final = scale judge everywhere, then the overrides file.
    manifest["judge_label_final"] = manifest["judge_label"]
    manifest["judge_label_final_model"] = manifest["judge_model"]
    updated, report = apply_label_overrides(manifest, merged, noise_flip_min_votes=noise_flip_min_votes)
    finalized_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fingerprint = final_fingerprint(updated, settings)
    if overrides_only:
        print("[finalize] manifest left untouched (--overrides-only); the builder applies the overrides file on its next rebuild")
    else:
        meta = {k: v for k, v in read_manifest_meta(settings["manifest"]).items() if k not in _MANIFEST_SIDECAR_KEYS}
        meta["label_overrides"] = {"path": str(path), "present": True, **report,
                                   "finalized_utc": finalized_utc, "judge_validation_dir": str(out)}
        write_manifest(updated, settings["manifest"], meta)
    changed = updated[(updated["judge_label_final"].notna()) & (updated["judge_label"].notna())
                      & (updated["judge_label_final"] != updated["judge_label"])]
    payload = {
        "overrides_path": str(path), "n_overrides_new": int(len(new)), "n_overrides_total": int(len(merged)),
        "n_overrides_applied": report["n_applied"], "n_unknown": report["n_unknown"], "models": report["models"],
        "n_label_changed": int(len(changed)),
        "n_stale_overrides_replaced": n_stale,
        "policy": policy,
        "review_counts": review_counts,
        "manifest_written": not overrides_only,
        "by_reason": {k: int(v) for k, v in new["override_reason"].value_counts().items()},
        "final_fingerprint": fingerprint,
        "finalized_utc": finalized_utc,
    }
    write_json(out / "finalize.json", payload)
    print(f"[finalize] {len(new)} override rows → {path} ({len(merged)} override rows in total)")
    print(f"[finalize] manifest: {report['n_applied']} rows carry an override ({report['models']}), {len(changed)} of them differ from the scale judge")
    for key, entry in fingerprint.items():
        print(f"[finalize] {key}: final unfaithful rate {entry['overall']['rate']:.4f} (n0 {entry['overall']['n0']}, n1 {entry['overall']['n1']})")
    if (out / "metrics.json").exists():
        print(f"[finalize] → {write_report(settings)}")


# -------------------------------------------------------- score review ---

def stage_score_review(settings: dict, review_path: Path | None, fraction_override: float | None = None) -> None:
    out = settings["output_dir"]
    path = review_path or (out / "review_sheet.csv")
    sheet = pd.read_csv(path, dtype={"researcher_verdict": str, "researcher_notes": str}, keep_default_na=False)
    metrics = read_json(out / "metrics.json")
    pooled = (metrics or {}).get("breakdowns", {}).get("pooled", {})
    fraction = None if not pooled or pooled.get("agreement") is None else 1.0 - pooled["agreement"]
    if fraction_override is not None:
        fraction = float(fraction_override)
    scores = score_review(sheet, disagreement_fraction=fraction)
    scores["review_path"] = str(path)
    write_json(out / "review_scores.json", scores)
    print(f"[score-review] {scores['n_disagreements_decided']} of {scores['n_disagreements']} disagreements decided; "
          f"sided with flash on {scores['share_siding_with_flash']}; escalate: {scores['escalate_arbiter_choice']}")
    if metrics is not None:
        print(f"[score-review] → {write_report(settings)}")


# ---------------------------------------------------------------- main ---

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--dry-run", action="store_true", help="judge/escalate: print the plan and one prompt, call nothing")
    parser.add_argument("--yes", action="store_true", help="escalate: actually buy the escalation verdicts")
    parser.add_argument("--overrides-only", action="store_true",
                        help="finalize: write judge_label_overrides.csv but leave the manifest to the builder")
    parser.add_argument("--review", default=None, help="score-review / finalize: the filled review sheet (default: output_dir/review_sheet.csv)")
    parser.add_argument("--disagreement-fraction", type=float, default=None,
                        help="score-review: share of sample rows on which the judges disagree, for the precision "
                             "extrapolation (default: 1 - pooled agreement from metrics.json)")
    args = parser.parse_args(argv)
    settings = resolve_settings(load_config(args.config))
    stages = ["sample", "judge", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"== stage {stage} ==")
        if stage == "sample":
            stage_sample(settings)
        elif stage == "judge":
            stage_judge(settings, dry_run=args.dry_run)
            if args.dry_run:
                break
        elif stage == "report":
            stage_report(settings)
        elif stage == "escalate":
            stage_escalate(settings, yes=args.yes, dry_run=args.dry_run)
        elif stage == "finalize":
            stage_finalize(settings, resolve_data_path(args.review) if args.review else None, overrides_only=args.overrides_only)
        elif stage == "score-review":
            stage_score_review(settings, resolve_data_path(args.review) if args.review else None, args.disagreement_fraction)
    return 0


if __name__ == "__main__":
    sys.exit(main())
