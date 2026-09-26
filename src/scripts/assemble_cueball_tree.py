#!/usr/bin/env python3
"""Copy runs generated before the paper grid into the paper's data tree.

Full-split runs that follow the grid's recipe are copied from the shared tree ``${DATA_ROOT}``
into ``${DATA_ROOT}/cueball`` — copied, not linked, so a tool writing into the paper tree can
never write through into the shared one. Sidecars naming a baseline CSV are re-pointed at the
copy, the runs' re-sample selection rows are appended to the tree's ``resample_selection.csv``
(+ report, + plan), and ``sources.json`` lists every copied file with its source path, size and
mtime. The MMLU-Pro subsample cell is written by ``make_cueball_subsample.py`` instead.

    python -m src.scripts.assemble_cueball_tree
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.lib.config import utc_now
from src.lib.paths import resolve_data_path

RUNS = [  # rollouts stem, has re-rolls
    ("nemotron-nano-9b-v2_commonsense_qa-validation_0910_baseline_hinted_rollouts", True),
    ("nemotron-nano-9b-v2_gpqa-extended_0909_baseline_hinted_rollouts", True),
    ("nemotron-nano-9b-v2_medqa-test_0910_baseline_hinted_rollouts", True),
    ("qwen3.5-9b_commonsense_qa-validation_0909_baseline_hinted_rollouts", False),
    ("qwen3.5-9b_gpqa_extended_0828_baseline_hinted_rollouts", False),
    ("qwen3.5-9b_medqa-test_baseline_hinted_rollouts", False),
    ("qwen3-8b_medqa-test_0911_baseline_hinted_rollouts", False),
]
SEEDS = (43, 44, 45, 46)


def repoint_sidecar(path: Path, baseline: Path, *keys: str, source: str, now: str) -> None:
    """Point the sidecar's ``baseline_csv`` (top level and under ``keys``) at the copied baseline."""
    meta = json.loads(path.read_text())
    targets = [meta] + [meta[k] for k in keys if isinstance(meta.get(k), dict)]
    for block in targets:
        if block.get("baseline_csv"):
            block["baseline_csv"] = str(baseline)
    meta["copied_into_tree"] = {"utc": now, "source": source}
    path.write_text(json.dumps(meta, indent=2) + "\n")


class Copier:
    """Copies files into the tree and records every copy for ``sources.json``."""

    def __init__(self, tree: Path, now: str):
        self.tree = tree
        self.now = now
        self.copied: list[dict] = []

    def copy(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        st = src.stat()
        self.copied.append({
            "file": str(dst.relative_to(self.tree)), "source": str(src), "size": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        })

    def repoint(self, path: Path, baseline: Path, *keys: str) -> None:
        repoint_sidecar(path, baseline, *keys, source=self.copied[-1]["source"], now=self.now)


def copy_runs(data: Path, tree: Path, cp: Copier) -> None:
    for stem, has_rerolls in RUNS:
        src_dir, dst_dir = data / "hinted_rollouts", tree / "hinted_rollouts"
        recipe = json.loads((src_dir / f"{stem}.meta.json").read_text())
        src_baseline = Path(recipe["baseline_csv"])
        baseline = tree / "baselines" / src_baseline.name
        cp.copy(src_baseline, baseline)
        cp.copy(src_baseline.with_suffix(".meta.json"), baseline.with_suffix(".meta.json"))
        meta = json.loads(baseline.with_suffix(".meta.json").read_text())
        meta["output_csv"] = str(baseline)
        meta["copied_into_tree"] = {"utc": cp.now, "source": str(src_baseline)}
        baseline.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        cp.copy(src_dir / f"{stem}.csv", dst_dir / f"{stem}.csv")
        cp.copy(src_dir / f"{stem}_judged.csv", dst_dir / f"{stem}_judged.csv")
        cp.copy(src_dir / f"{stem}.meta.json", dst_dir / f"{stem}.meta.json")
        cp.repoint(dst_dir / f"{stem}.meta.json", baseline)
        cp.copy(src_dir / f"{stem}_judged_summary.json", dst_dir / f"{stem}_judged_summary.json")
        cp.repoint(dst_dir / f"{stem}_judged_summary.json", baseline, "provenance")
        if (src_dir / f"{stem}_judged.meta.json").exists():
            cp.copy(src_dir / f"{stem}_judged.meta.json", dst_dir / f"{stem}_judged.meta.json")
        print(f"{stem}: baseline + rollouts + judged copied")
        if not has_rerolls:
            continue
        rs_src, rs_dst = data / "resample", tree / "resample"
        items = f"{stem}_resample_items"
        cp.copy(rs_src / "items" / f"{items}.csv", rs_dst / "items" / f"{items}.csv")
        cp.copy(rs_src / "items" / f"{items}.meta.json", rs_dst / "items" / f"{items}.meta.json")
        cp.repoint(rs_dst / "items" / f"{items}.meta.json", baseline)
        for seed in SEEDS:
            name = f"{stem}_rs{seed}"
            for suffix in (".csv", "_judged.csv"):
                cp.copy(rs_src / "rollouts" / f"{name}{suffix}", rs_dst / "rollouts" / f"{name}{suffix}")
            cp.copy(rs_src / "rollouts" / f"{name}.meta.json", rs_dst / "rollouts" / f"{name}.meta.json")
            cp.repoint(rs_dst / "rollouts" / f"{name}.meta.json", baseline)
            cp.copy(rs_src / "rollouts" / f"{name}_judged_summary.json",
                    rs_dst / "rollouts" / f"{name}_judged_summary.json")
            cp.repoint(rs_dst / "rollouts" / f"{name}_judged_summary.json", baseline, "provenance")
        print(f"{stem}: items + {len(SEEDS)} re-roll files (raw + judged) copied")


def merge_selection(data: Path, tree: Path, now: str) -> None:
    """Append the copied runs' stored selection rows (and report / plan) to the tree's selection."""
    rs_src, rs_dst = data / "resample", tree / "resample"
    mine = pd.read_csv(rs_dst / "resample_selection.csv", dtype=str, keep_default_na=False)
    shared = pd.read_csv(rs_src / "resample_selection.csv", dtype=str, keep_default_na=False)
    wanted = {f"{stem}_judged.csv" for stem, has in RUNS if has}
    assert not (set(mine["source_csv"]) & wanted), "assemble_cueball_tree already ran"
    add = shared[shared["source_csv"].isin(wanted)]
    assert set(add["source_csv"]) == wanted
    for source_csv, group in add.groupby("source_csv"):
        items = pd.read_csv(rs_dst / "items" / f"{Path(source_csv).stem.removesuffix('_judged')}_resample_items.csv",
                            usecols=["rollout_id"])
        assert set(group["rollout_id"]) == set(items["rollout_id"]), source_csv
    selection = pd.concat([mine, add], ignore_index=True)
    selection.to_csv(rs_dst / "resample_selection.csv", index=False)

    keys = ["subject_model", "run", "hint_style", "case", "stability_bin"]
    report = pd.read_csv(rs_src / "resample_selection_report.csv", dtype=str, keep_default_na=False)
    report = report[report["run"].isin(set(add["run"])) & report["subject_model"].isin(set(add["subject_model"]))]
    cells = mine.groupby(keys + ["role"]).size().unstack("role", fill_value=0).reset_index()
    sub = pd.DataFrame({**{k: cells[k] for k in keys}, "n_used": cells.get("used_candidate", 0),
                        "n_control_available": "", "n_control": cells.get("control", 0)})
    sub["control_shortfall"] = (sub["n_used"] - sub["n_control"]).clip(lower=0)
    pd.concat([sub.astype(str), report], ignore_index=True).to_csv(rs_dst / "resample_selection_report.csv", index=False)

    meta = json.loads((rs_dst / "resample_selection.meta.json").read_text())
    shared_meta = json.loads((rs_src / "resample_selection.meta.json").read_text())
    n_used, n_control = int((mine["role"] == "used_candidate").sum()), int((mine["role"] == "control").sum())
    sub_run = mine.iloc[0]
    plan = [{"subject_model": sub_run["subject_model"], "run": sub_run["run"],
             "source_csv": sub_run["source_csv"], "n_used": n_used, "n_control": n_control,
             "n_items": n_used + n_control, "n_rollouts": (n_used + n_control) * int(meta["k"]),
             "note": "question subsample of the run's selection"}]
    plan += [p for p in shared_meta["plan"] if p.get("source_csv") in wanted]
    shortfall = pd.to_numeric(pd.read_csv(rs_dst / "resample_selection_report.csv")["control_shortfall"]).sum()
    meta.update({
        "written_utc": now, "only": None,
        "n_selected": int(len(selection)),
        "n_used": int((selection["role"] == "used_candidate").sum()),
        "n_control": int((selection["role"] == "control").sum()),
        "control_shortfall": int(shortfall), "plan": plan,
        "assembled": {"utc": now, "from": str(rs_src / "resample_selection.csv"),
                      "from_written_utc": shared_meta.get("written_utc"), "runs": sorted(set(add["run"]))},
    })
    (rs_dst / "resample_selection.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"selection: {len(mine)} rows already in the tree + {len(add)} rows of {len(wanted)} runs = {len(selection)}")


def main() -> None:
    data = resolve_data_path("${DATA_ROOT}")
    tree = data / "cueball"
    now = utc_now()
    cp = Copier(tree, now)
    copy_runs(data, tree, cp)
    merge_selection(data, tree, now)
    (tree / "sources.json").write_text(json.dumps({"assembled_utc": now, "files": cp.copied}, indent=2) + "\n")
    print(f"{len(cp.copied)} files copied → {tree}")


if __name__ == "__main__":
    main()
