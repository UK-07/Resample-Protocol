#!/usr/bin/env python3
"""Step 2 of the re-sampling design: pick the (question, hint) pairs to re-roll.

From the filtered manifest every ``to_hint`` rollout becomes a ``used_candidate``
and, per (subject_model, run, hint_style, case, stability bin), an equal-size
random sample of rollouts that kept the baseline answer becomes the matched
``control`` (:func:`src.lib.resample.select_resample_set`).

Writes under ``--output-dir``: ``resample_selection.csv`` (one row per selected
rollout, :data:`SELECTION_COLUMNS`), ``resample_selection_report.csv`` (per
match cell), ``resample_selection.meta.json`` (seed, k, sample seeds, manifest,
per-run plan with token estimates) and, unless ``--no-items``,
``items/<source stem>_resample_items.csv`` (+ ``.meta.json``): the selected
rows' prompt columns per source judged CSV, all the GPU stage needs.

``--extend`` keeps every run already stored (its re-rolls may exist) and draws
only the runs without a stored selection. ``--only`` is a regular expression.
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
    DEFAULT_K,
    DEFAULT_SAMPLE_SEEDS,
    extract_items,
    filter_manifest,
    items_meta,
    items_path,
    select_resample_set,
)
from src.lib.rollout_manifest import DEFAULT_MANIFEST_PATH, read_manifest, read_manifest_meta

DEFAULT_OUTPUT_DIR = "${DATA_ROOT}/resample"
SELECTION_FILENAME = "resample_selection.csv"


def source_recipe(judged_csv: Path) -> dict:
    """The collect recipe sidecar of a judged CSV ({} when absent)."""
    path = judged_csv.with_name(f"{judged_csv.stem.removesuffix('_judged')}.meta.json")
    return json.loads(path.read_text()) if path.exists() else {}


def plan_by_run(selection: pd.DataFrame, manifest: pd.DataFrame, meta: dict, k: int) -> pd.DataFrame:
    """Per (subject_model, run): items, rollouts to generate and a token estimate."""
    sources = {(s["subject_model"], s["run"]): s for s in meta.get("sources", [])}
    tok = manifest.groupby(["subject_model", "run"])["trace_token_len"].mean()
    orig = manifest.groupby(["subject_model", "run"]).agg(
        n_original=("rollout_id", "size"), original_tokens=("trace_token_len", "sum"))
    rows = []
    for (model, run), group in selection.groupby(["subject_model", "run"], sort=True):
        n_used = int((group["role"] == "used_candidate").sum())
        n_control = int((group["role"] == "control").sum())
        mean_tok = float(tok.get((model, run), float("nan")))
        src = sources.get((model, run), {})
        rows.append({
            "subject_model": model, "run": run,
            "model_name": src.get("subject_model_id"),
            "source_csv": src.get("source_csv"),
            "n_used": n_used, "n_control": n_control, "n_items": n_used + n_control,
            "n_rollouts": (n_used + n_control) * k,
            "mean_trace_tokens": round(mean_tok, 1),
            "est_tokens_M": round((n_used + n_control) * k * mean_tok / 1e6, 1),
            "n_original": int(orig.loc[(model, run), "n_original"]),
            "original_tokens_M": round(float(orig.loc[(model, run), "original_tokens"]) / 1e6, 1),
        })
    return pd.DataFrame(rows)


def read_stored(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """The selection already under ``out`` — rows, per-cell report, sidecar — or None.

    Cells are read as strings, so writing the rows back changes none of them.
    """
    path = out / SELECTION_FILENAME
    if not path.exists():
        return None
    selection = pd.read_csv(path, dtype=str, keep_default_na=False)
    report_path = out / "resample_selection_report.csv"
    report = pd.read_csv(report_path, dtype=str, keep_default_na=False) if report_path.exists() else pd.DataFrame()
    meta_path = out / "resample_selection.meta.json"
    return selection, report, json.loads(meta_path.read_text()) if meta_path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", default=None, help="only runs whose subject_model or run tag contains this")
    parser.add_argument("--seed", type=int, default=42, help="control draw seed (per cell)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="re-rolls per selected pair")
    parser.add_argument("--sample-seeds", default=",".join(str(s) for s in DEFAULT_SAMPLE_SEEDS),
                        help="comma-separated vLLM sampling seeds, one per re-roll")
    parser.add_argument("--controls-per-used", type=float, default=1.0,
                        help="matched controls per used candidate in each cell (default 1)")
    parser.add_argument("--rollouts-dir", default="${DATA_ROOT}/hinted_rollouts",
                        help="directory holding the source judged CSVs")
    parser.add_argument("--no-items", action="store_true", help="skip writing the per-source items files")
    parser.add_argument("--extend", action="store_true",
                        help="keep the stored selection's runs untouched and draw only the runs it lacks")
    parser.add_argument("--chunk-rows", type=int, default=20000, help="rows per judged-CSV read chunk")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.sample_seeds.split(",") if s.strip()]
    if len(seeds) != args.k:
        parser.error(f"--sample-seeds must name exactly k={args.k} seeds, got {seeds}")

    manifest = read_manifest(args.manifest)
    meta = read_manifest_meta(args.manifest)
    df = filter_manifest(manifest)
    if args.only:
        df = df[df["subject_model"].str.contains(args.only) | df["run"].str.contains(args.only)]
    if df.empty:
        print("no rows left after filtering", file=sys.stderr)
        return 1

    out = resolve_data_path(args.output_dir)
    stored = read_stored(out) if args.extend else None
    kept_runs: list[tuple[str, str]] = []
    if stored is not None:
        stored_k, stored_seeds = stored[2].get("k"), stored[2].get("sample_seeds")
        if (stored_k, stored_seeds) != (args.k, seeds):
            parser.error(f"--extend: the stored selection used k={stored_k} seeds={stored_seeds}, "
                         f"this run k={args.k} seeds={seeds}")
        kept_runs = sorted(set(zip(stored[0]["subject_model"], stored[0]["run"])))
        df = df[[key not in set(kept_runs) for key in zip(df["subject_model"], df["run"])]]
        if df.empty:
            print(f"--extend: every run in scope already has a stored selection ({len(kept_runs)} runs) — nothing drawn")
            return 0

    selection, report = select_resample_set(df, args.seed, controls_per_used=args.controls_per_used)
    plan = plan_by_run(selection, df, meta, args.k)
    drawn, drawn_report = selection, report  # the items pass and the closing summary cover the new runs only
    plan_records = plan.to_dict(orient="records")
    if stored is not None:
        selection = pd.concat([stored[0], selection.astype(object)], ignore_index=True)
        report = pd.concat([stored[1], report.astype(object)], ignore_index=True)
        plan_records = list(stored[2].get("plan") or []) + plan_records

    out.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out / SELECTION_FILENAME, index=False)
    report.to_csv(out / "resample_selection_report.csv", index=False)
    sidecar = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(resolve_data_path(args.manifest)),
        "manifest_written_utc": meta.get("written_utc"),
        "only": args.only,
        "seed": args.seed,
        "k": args.k,
        "sample_seeds": seeds,
        "controls_per_used": args.controls_per_used,
        "n_selected": int(len(selection)),
        "n_used": int((selection["role"] == "used_candidate").sum()),
        "n_control": int((selection["role"] == "control").sum()),
        "control_shortfall": int(pd.to_numeric(report["control_shortfall"]).sum()),
        "plan": plan_records,
    }
    if stored is not None:
        sidecar["extends"] = {"kept_runs": [f"{m}/{r}" for m, r in kept_runs],
                              "kept_written_utc": stored[2].get("written_utc")}
    (out / "resample_selection.meta.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    if not args.no_items:
        rollouts_dir = resolve_data_path(args.rollouts_dir)
        (out / "items").mkdir(exist_ok=True)
        sources = {(s["subject_model"], s["run"]): s for s in meta.get("sources", [])}
        for (model, run), group in drawn.groupby(["subject_model", "run"], sort=True):
            source = sources.get((model, run))
            if source is None or not source.get("source_csv"):
                print(f"  {model}/{run}: no source record in the manifest meta — items skipped", file=sys.stderr)
                continue
            judged_csv = rollouts_dir / source["source_csv"]
            baseline_csv = source.get("baseline_csv")
            if baseline_csv and not Path(baseline_csv).exists():
                print(f"  {model}/{run}: baseline {baseline_csv} missing — items carry no choices", file=sys.stderr)
                baseline_csv = None
            items = extract_items(judged_csv, group, chunk_rows=args.chunk_rows, baseline_csv=baseline_csv)
            target = items_path(out / "items", source["source_csv"])
            items.to_csv(target, index=False)
            target.with_suffix(".meta.json").write_text(
                json.dumps(items_meta(source, source_recipe(judged_csv), group, seeds), indent=2) + "\n")
            print(f"  {target.name}: {len(items)} items ({(group['role'] == 'used_candidate').sum()} used, "
                  f"{(group['role'] == 'control').sum()} control)")

    print(plan.to_string(index=False))
    n_used, n_control = ((drawn["role"] == role).sum() for role in ("used_candidate", "control"))
    print(f"\n{n_used} used candidates + {n_control} controls "
          f"(shortfall {int(drawn_report['control_shortfall'].sum())}) × k={args.k} = "
          f"{int(plan['n_rollouts'].sum())} rollouts, ~{plan['est_tokens_M'].sum():.0f}M tokens "
          f"(original: {int(plan['n_original'].sum())} rollouts, {plan['original_tokens_M'].sum():.0f}M tokens) → {out}")
    if stored is not None:
        print(f"--extend: kept {len(stored[0])} stored rows of {len(kept_runs)} run(s); "
              f"the selection now holds {len(selection)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
