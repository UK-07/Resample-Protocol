#!/usr/bin/env python3
"""The shared MMLU-Pro question set and one run's artifacts restricted to it.

The shared set is ``load_data("mmlu_pro", split="test", require_n_options=10)`` followed by
``df.sample(n=1000, random_state=42)`` — exactly what ``compute_baseline`` draws for
``max_samples: 1000, seed: 42``. pandas draws a permutation prefix, so the set is the head of any
larger draw with the same seed. Every artifact of the source run (baseline, hinted rollouts, judged
rollouts, re-sample items and re-rolls, selection rows) is filtered on ``original_index`` into
``${DATA_ROOT}/cueball`` with the standard file names, so rollout ids, label overrides and the
judge cache (keyed on the stem) stay valid. Cells are copied as strings, nothing is re-formatted.

    python -m src.scripts.make_cueball_subsample
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.lib.config import utc_now
from src.lib.dataset import load_data
from src.lib.paths import resolve_data_path
from src.scripts.batch_judge_rollouts import aggregate_judged

N = 1000
SEED = 42
STEM = "nemotron-nano-9b-v2_mmlu_pro-test_0909_baseline"
ROLLOUTS = f"{STEM}_hinted_rollouts"
OLMO_BASELINE = "olmo3-7b-think_mmlu_pro-test_0909_baseline.csv"
SEEDS = [43, 44, 45, 46]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def source_stat(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}


def subsample_block(tree: Path, source: Path, n_source: int, n_kept: int, now: str) -> dict:
    return {
        "question_list": str(tree / "mmlu_pro_1000_questions.csv"),
        "rule": f"original_index in mmlu_pro test (require_n_options=10) .sample(n={N}, random_state={SEED})",
        "source": source_stat(source), "n_source_rows": n_source, "n_rows": n_kept, "created_utc": now,
    }


def filter_csv(src: Path, dst: Path, keep: set[str]) -> tuple[pd.DataFrame, int]:
    """Keep the rows whose ``original_index`` is in ``keep``; cells are copied verbatim."""
    df = read(src)
    out = df[df["original_index"].isin(keep)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    back = read(dst)
    assert list(back.columns) == list(df.columns), dst
    assert back.reset_index(drop=True).equals(out.reset_index(drop=True)), f"round-trip changed cells: {dst}"
    print(f"{dst.name}: {len(out)} of {len(df)} rows, {out['original_index'].nunique()} questions")
    return out, len(df)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    data = resolve_data_path("${DATA_ROOT}")
    tree = data / "cueball"
    now = utc_now()

    # 1. the shared question set, re-derived from the loader and checked against both baselines
    df, _ = load_data("mmlu_pro", split="test", require_n_options=10)
    picked = df.sample(n=N, random_state=SEED)["original_index"].astype(int).tolist()
    nem_src = data / "baselines" / f"{STEM}.csv"
    nem = pd.read_csv(nem_src, usecols=["original_index"])["original_index"].tolist()
    olmo = pd.read_csv(data / "baselines" / OLMO_BASELINE, usecols=["original_index"])["original_index"].tolist()
    assert picked == nem[:N], "sample(1000) is not the first 1000 rows of the source baseline"
    assert picked[:len(olmo)] == olmo, "sample(1000) does not start with the 800-question baseline"
    keep = {str(i) for i in picked}

    tree.mkdir(exist_ok=True)
    questions = pd.DataFrame({"rank": range(N), "original_index": picked,
                              "in_olmo_800": [i < len(olmo) for i in range(N)]})
    questions.to_csv(tree / "mmlu_pro_1000_questions.csv", index=False)
    write_json(tree / "mmlu_pro_1000_questions.meta.json", {
        "dataset": {"name": "mmlu_pro", "params": {"split": "test", "require_n_options": 10}},
        "n_loader_rows": int(len(df)), "n": N, "seed": SEED,
        "rule": "df.sample(n=1000, random_state=42) on the loader frame == compute_baseline with max_samples: 1000, seed: 42",
        "checks": {"equals_first_1000_rows_of": str(nem_src), "starts_with_the_800_questions_of": OLMO_BASELINE},
        "created_utc": now,
    })
    print(f"mmlu_pro_1000_questions.csv: {N} questions")

    # 2. baseline
    dst = tree / "baselines" / f"{STEM}.csv"
    base, n_src = filter_csv(nem_src, dst, keep)
    assert base["original_index"].astype(int).tolist() == picked
    meta = json.loads(nem_src.with_suffix(".meta.json").read_text())
    counts = base["baseline_status"].value_counts().to_dict()
    meta.update({
        "n_rows": int(len(base)), "accuracy": float((base["baseline_status"] == "correct").mean()),
        "output_csv": str(dst), "baseline_status_counts": {k: int(v) for k, v in counts.items()},
        "subsample": {**subsample_block(tree, nem_src, n_src, len(base), now),
                      "source_accuracy": meta.get("accuracy"), "source_max_samples": meta.get("max_samples")},
    })
    write_json(dst.with_suffix(".meta.json"), meta)

    # 3. hinted rollouts, judged rollouts, summary
    src_dir, dst_dir = data / "hinted_rollouts", tree / "hinted_rollouts"
    raw, n_raw = filter_csv(src_dir / f"{ROLLOUTS}.csv", dst_dir / f"{ROLLOUTS}.csv", keep)
    recipe = json.loads((src_dir / f"{ROLLOUTS}.meta.json").read_text())
    recipe["baseline_csv"] = str(dst)
    recipe["subsample"] = subsample_block(tree, src_dir / f"{ROLLOUTS}.csv", n_raw, len(raw), now)
    write_json(dst_dir / f"{ROLLOUTS}.meta.json", recipe)

    judged, n_judged = filter_csv(src_dir / f"{ROLLOUTS}_judged.csv", dst_dir / f"{ROLLOUTS}_judged.csv", keep)
    assert len(judged) == len(raw)
    summary = json.loads((src_dir / f"{ROLLOUTS}_judged_summary.json").read_text())
    summary.update({
        "rollouts_csv": str(dst_dir / f"{ROLLOUTS}.csv"), "judged_csv": str(dst_dir / f"{ROLLOUTS}_judged.csv"),
        "provenance": recipe, "results": aggregate_judged(pd.read_csv(dst_dir / f"{ROLLOUTS}_judged.csv")),
        "subsample": subsample_block(tree, src_dir / f"{ROLLOUTS}_judged.csv", n_judged, len(judged), now),
    })
    write_json(dst_dir / f"{ROLLOUTS}_judged_summary.json", summary)

    # 4. re-sampling: items, the re-roll files (raw + judged + summaries), the selection
    rs_src, rs_dst = data / "resample", tree / "resample"
    items_name = f"{ROLLOUTS}_resample_items"
    items, n_items = filter_csv(rs_src / "items" / f"{items_name}.csv", rs_dst / "items" / f"{items_name}.csv", keep)
    items_meta = json.loads((rs_src / "items" / f"{items_name}.meta.json").read_text())
    items_meta.update({
        "baseline_csv": str(dst), "n_items": int(len(items)),
        "n_used": int((items["role"] == "used_candidate").sum()), "n_control": int((items["role"] == "control").sum()),
        "subsample": subsample_block(tree, rs_src / "items" / f"{items_name}.csv", n_items, len(items), now),
    })
    write_json(rs_dst / "items" / f"{items_name}.meta.json", items_meta)

    for seed in SEEDS:
        name = f"{ROLLOUTS}_rs{seed}"
        rolls, n_rolls = filter_csv(rs_src / "rollouts" / f"{name}.csv", rs_dst / "rollouts" / f"{name}.csv", keep)
        assert len(rolls) == len(items), (seed, len(rolls), len(items))
        rs_meta = json.loads((rs_src / "rollouts" / f"{name}.meta.json").read_text())
        rs_meta["baseline_csv"] = str(dst)
        rs_meta["subsample"] = subsample_block(tree, rs_src / "rollouts" / f"{name}.csv", n_rolls, len(rolls), now)
        write_json(rs_dst / "rollouts" / f"{name}.meta.json", rs_meta)

        rj, n_rj = filter_csv(rs_src / "rollouts" / f"{name}_judged.csv", rs_dst / "rollouts" / f"{name}_judged.csv", keep)
        assert len(rj) == len(rolls)
        rs_summary = json.loads((rs_src / "rollouts" / f"{name}_judged_summary.json").read_text())
        rs_summary.update({
            "rollouts_csv": str(rs_dst / "rollouts" / f"{name}.csv"),
            "judged_csv": str(rs_dst / "rollouts" / f"{name}_judged.csv"),
            "provenance": rs_meta,
            "results": aggregate_judged(pd.read_csv(rs_dst / "rollouts" / f"{name}_judged.csv")),
            "subsample": subsample_block(tree, rs_src / "rollouts" / f"{name}_judged.csv", n_rj, len(rj), now),
        })
        write_json(rs_dst / "rollouts" / f"{name}_judged_summary.json", rs_summary)

    selection = read(rs_src / "resample_selection.csv")
    mine = selection[(selection["source_csv"] == f"{ROLLOUTS}_judged.csv") & selection["original_index"].isin(keep)]
    assert set(mine["rollout_id"]) == set(items["rollout_id"]), "selection and items disagree"
    mine.to_csv(rs_dst / "resample_selection.csv", index=False)
    sel_meta = json.loads((rs_src / "resample_selection.meta.json").read_text())
    sel_meta.pop("plan", None)
    sel_meta.update({
        "n_selected": int(len(mine)), "n_used": int((mine["role"] == "used_candidate").sum()),
        "n_control": int((mine["role"] == "control").sum()),
        "control_shortfall": int((mine["role"] == "used_candidate").sum() - (mine["role"] == "control").sum()),
        "subsample": {**subsample_block(tree, rs_src / "resample_selection.csv", len(selection), len(mine), now),
                      "note": "rows of the source run on the shared questions; the controls are the members "
                              "of the original control draw that fall inside the set"},
    })
    write_json(rs_dst / "resample_selection.meta.json", sel_meta)
    print(f"resample_selection.csv: {len(mine)} selected ({sel_meta['n_used']} used, {sel_meta['n_control']} control)")


if __name__ == "__main__":
    main()
