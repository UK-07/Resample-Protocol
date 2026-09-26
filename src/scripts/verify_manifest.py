#!/usr/bin/env python3
"""Assert the rollout manifest reproduces ``unfaithfulness_metrics_summary.md``.

Parses the summary markdown (overview table, per-pair tables with totals row,
judge / prompt line) and recomputes every cell from ``pair_counts(manifest)``.
Prints each disagreement; exit status 1 when there is at least one, else 0.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.lib.paths import resolve_data_path
from src.lib.rollout_manifest import (
    DEFAULT_MANIFEST_PATH,
    PAIR_KEYS,
    pair_counts,
    read_manifest,
    read_manifest_meta,
)
from src.scripts.unfaithfulness_metrics_summary import excluded_cell, rate

DEFAULT_SUMMARY_PATH = "${DATA_ROOT}/hinted_rollouts/unfaithfulness_metrics_summary.md"

# The overview's trailing budget columns come from the recipe sidecar, not the manifest: parsed, not compared.
OVERVIEW_COLUMNS = [
    "model", "dataset", "rollouts", "changed", "switched", "unfaithful", "faithful",
    "unjudged", "unfaithful_rate", "truncated", "excluded", "judge", "judge_prompt",
    "max_tokens", "max_model_len",
]
OVERVIEW_COMPARED = OVERVIEW_COLUMNS[2:13]
PAIR_COLUMNS = [
    "hint", "case", "rollouts", "changed", "switched", "switch_rate",
    "unfaithful", "faithful", "incoherent", "unjudged", "unfaithful_rate",
    "truncated", "excluded",
]
TOTALS_KEY = ("**all**", "**both**")

_SECTION_RE = re.compile(r"^## (?P<model>.+?) — (?P<dataset>.+?)\s*$")
_JUDGE_RE = re.compile(r"^- judge: `(?P<judge>[^`]*)`\s+judge prompt: `(?P<prompt>[^`]*)`")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_row(line: str) -> bool:
    cells = _cells(line)
    return line.lstrip().startswith("|") and not all(re.fullmatch(r":?-+:?", c) for c in cells)


def parse_summary(text: str) -> dict:
    """{"overview": [dict], "pairs": {(model, dataset): {"judge", "prompt", "rows": [dict]}}}.

    Header rows are skipped by name, separator rows by shape; numbers stay
    strings, so the comparison is textual.
    """
    overview: list[dict] = []
    pairs: dict[tuple[str, str], dict] = {}
    section: str | None = None
    current: dict | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            m = _SECTION_RE.match(line)
            if m:
                section = "pair"
                current = pairs.setdefault((m["model"], m["dataset"]), {"judge": None, "prompt": None, "rows": []})
            else:
                section = "overview" if line.strip() == "## Overview" else None
                current = None
            continue
        if section == "pair" and current is not None:
            m = _JUDGE_RE.match(line)
            if m:
                current["judge"], current["prompt"] = m["judge"], m["prompt"]
                continue
        if not _is_row(line):
            continue
        cells = _cells(line)
        if section == "overview" and cells[0] != "model" and len(cells) == len(OVERVIEW_COLUMNS):
            overview.append(dict(zip(OVERVIEW_COLUMNS, cells)))
        elif section == "pair" and current is not None and cells[0] != "hint style" \
                and len(cells) == len(PAIR_COLUMNS):
            current["rows"].append(dict(zip(PAIR_COLUMNS, cells)))
    return {"overview": overview, "pairs": pairs}


def render_pair_row(hint: str, case: str, row) -> dict[str, str]:
    """A manifest count row in the summary table's textual form."""
    judged = int(row["n_unfaithful"] + row["n_faithful"])
    return {
        "hint": hint, "case": case,
        "rollouts": str(int(row["n_rollouts"])),
        "changed": str(int(row["n_changed"])),
        "switched": str(int(row["n_switched"])),
        "switch_rate": rate(int(row["n_switched"]), int(row["n_rollouts"])),
        "unfaithful": str(int(row["n_unfaithful"])),
        "faithful": str(int(row["n_faithful"])),
        "incoherent": str(int(row["n_incoherent"])),
        "unjudged": str(int(row["n_unjudged"])),
        "unfaithful_rate": rate(int(row["n_unfaithful"]), judged),
        "truncated": str(int(row["n_truncated"])),
        "excluded": excluded_cell(row),
    }


def render_overview_row(model: str, dataset: str, totals, judge: str, prompt: str) -> dict[str, str]:
    judged = int(totals["n_unfaithful"] + totals["n_faithful"])
    return {
        "model": model, "dataset": dataset,
        "rollouts": str(int(totals["n_rollouts"])),
        "changed": str(int(totals["n_changed"])),
        "switched": str(int(totals["n_switched"])),
        "unfaithful": str(int(totals["n_unfaithful"])),
        "faithful": str(int(totals["n_faithful"])),
        "unjudged": str(int(totals["n_unjudged"])),
        "unfaithful_rate": rate(int(totals["n_unfaithful"]), judged),
        "truncated": str(int(totals["n_truncated"])),
        "excluded": excluded_cell(totals),
        "judge": f"`{judge}`", "judge_prompt": f"`{prompt}`",
    }


def _diff(where: str, expected: dict, actual: dict, keys) -> list[str]:
    return [
        f"{where}: {k} summary={expected.get(k)!r} manifest={actual.get(k)!r}"
        for k in keys if expected.get(k) != actual.get(k)
    ]


def compare(summary: dict, manifest: pd.DataFrame, meta: dict) -> list[str]:
    """Every disagreement between the summary and the manifest, as one line each."""
    problems: list[str] = []
    counts = pair_counts(manifest)
    manifest_pairs = {tuple(k) for k in counts.reset_index()[PAIR_KEYS].drop_duplicates().itertuples(index=False)}
    summary_pairs = set(summary["pairs"])
    for pair in sorted(summary_pairs - manifest_pairs):
        problems.append(f"pair {pair[0]} / {pair[1]}: in the summary but not in the manifest")
    for pair in sorted(manifest_pairs - summary_pairs):
        problems.append(f"pair {pair[0]} / {pair[1]}: in the manifest but not in the summary")

    provenance = {
        (s["subject_model"], s["run"]): s for s in meta.get("sources", [])
    }
    overview = {(r["model"], r["dataset"]): r for r in summary["overview"]}
    for pair in sorted(summary_pairs & manifest_pairs):
        model, dataset = pair
        where = f"{model} / {dataset}"
        expected = summary["pairs"][pair]
        pc = counts.loc[pair]
        got = {(h, c): render_pair_row(h, c, row) for (h, c), row in pc.iterrows()}
        got[TOTALS_KEY] = render_pair_row(*TOTALS_KEY, pc.sum())
        want = {(r["hint"], r["case"]): r for r in expected["rows"]}
        for key in sorted(set(want) - set(got)):
            problems.append(f"{where}: row {key[0]} / {key[1]} in the summary but not in the manifest")
        for key in sorted(set(got) - set(want)):
            problems.append(f"{where}: row {key[0]} / {key[1]} in the manifest but not in the summary")
        for key in sorted(set(want) & set(got)):
            problems += _diff(f"{where}: {key[0]} / {key[1]}", want[key], got[key], PAIR_COLUMNS[2:])

        source = provenance.get(pair)
        judge = source["judge_model"] if source else None
        prompt = source["judge_prompt"] if source else None
        if source is None:
            problems.append(f"{where}: no provenance record in the manifest meta")
        else:
            if expected["judge"] != judge:
                problems.append(f"{where}: judge summary={expected['judge']!r} manifest={judge!r}")
            if expected["prompt"] != prompt:
                problems.append(f"{where}: judge prompt summary={expected['prompt']!r} manifest={prompt!r}")
            rows = manifest[(manifest["subject_model"] == model) & (manifest["run"] == dataset)]
            seen = set(rows["judge_model"].dropna())
            if seen and seen != set(judge.split(", ")):
                problems.append(f"{where}: rows judged by {sorted(seen)} but provenance says {judge!r}")

        if pair in overview:
            got_row = render_overview_row(model, dataset, pc.sum(), judge, prompt)
            problems += _diff(f"{where}: overview", overview[pair], got_row, OVERVIEW_COMPARED)
        else:
            problems.append(f"{where}: missing from the overview table")
    return problems


def stale_sources(meta: dict) -> list[str]:
    """Sources whose judged CSV size / mtime differ from what the manifest meta recorded."""
    root = meta.get("rollouts_dir")
    if not root:
        return []
    notes = []
    for source in meta.get("sources", []):
        path = Path(root) / str(source.get("source_csv", ""))
        if not path.is_file():
            notes.append(f"{path.name}: missing (was in the manifest)")
            continue
        stat = path.stat()
        mtime = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(timespec="seconds")
        if stat.st_size != source.get("source_size") or mtime != source.get("source_mtime_utc"):
            notes.append(f"{path.name}: changed since the manifest was built "
                         f"({source.get('source_mtime_utc')} → {mtime})")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH,
                        help=f"manifest parquet (default {DEFAULT_MANIFEST_PATH})")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH,
                        help=f"summary markdown (default {DEFAULT_SUMMARY_PATH})")
    args = parser.parse_args(argv)

    manifest_path = resolve_data_path(args.manifest)
    summary_path = resolve_data_path(args.summary)
    manifest = read_manifest(manifest_path)
    meta = read_manifest_meta(manifest_path)
    summary = parse_summary(Path(summary_path).read_text())
    if not summary["pairs"]:
        print(f"no per-pair tables found in {summary_path}", file=sys.stderr)
        return 1

    problems = compare(summary, manifest, meta)
    n_rows = sum(len(p["rows"]) for p in summary["pairs"].values())
    if problems:
        stale = stale_sources(meta)
        if stale:
            print(f"{len(stale)} source(s) changed since {manifest_path.name} was built — "
                  "rebuild it before reading the disagreements:")
            for line in stale:
                print(f"  {line}")
        print(f"{len(problems)} disagreement(s) between {summary_path.name} and {manifest_path.name}:")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"OK: {manifest_path.name} ({len(manifest)} rollouts) reproduces {summary_path.name} — "
          f"{len(summary['pairs'])} pairs, {n_rows} table rows, {len(summary['overview'])} overview rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
