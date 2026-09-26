"""Tests for src/scripts/build_probe_dataset.py and the I/O half of src/lib/probe_datasets.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.lib.config import load_config
from src.lib.paths import REPO_ROOT, resolve_data_path
from src.lib.probe_datasets import (
    DATASET_COLUMNS,
    RESAMPLE_EXTRA_COLUMNS,
    TEXT_COLUMNS,
    build_dataset,
    dataset_paths,
    dataset_sidecars,
    join_texts,
    load_dataset_manifest,
    parse_spec,
    plan_dataset,
    read_dataset,
    validate_dataset,
    write_dataset,
)
from src.scripts.build_probe_dataset import main
from src.scripts.collect_hinted_rollouts import OUTPUT_COLUMNS
from tests.scripts.resample_fixtures import SEEDS, probe_manifest

JUDGE_COLUMNS = ["judge_model", "judge_label", "judge_confidence", "judge_reasoning", "judge_role", "judge_hint_quote"]
CHOICES = '["a", "b", "c", "d"]'
LETTERS = '["A", "B", "C", "D"]'


def reasoning_for(original_index, style, seed) -> str:
    return f"cot {int(original_index)} {style} rs{seed}"


def spec_cfg(root: Path, **overrides) -> dict:
    cfg = {"name": "test_ds", "manifest": "${DATA_ROOT}/resample/resample_manifest.parquet",
           "predicate": "verbalised_vs_unverbalised", "subject_models": ["m1"], "runs": ["medqa-test_0911"],
           "output_dir": "${DATA_ROOT}/probe_datasets"}
    cfg.update(overrides)
    return cfg


def make_root(tmp: Path, *, csv_choices: bool = True, baseline: bool = False, baseline_skip=None) -> tuple[Path, pd.DataFrame]:
    """``tmp/data`` with a resample manifest, one re-roll CSV per (model, run, seed) and, with ``baseline``, a
    recipe sidecar per CSV naming a baseline CSV that carries ``choices`` for every question but ``baseline_skip``."""
    root = tmp / "data"
    resample = root / "resample"
    rollouts = resample / "rollouts"
    rollouts.mkdir(parents=True)
    (root / "baselines").mkdir()
    df = probe_manifest()
    reroll = df["provenance"].astype(str) == "resample_k4"
    df.loc[reroll, "source_csv"] = (df.loc[reroll, "subject_model"].astype(str) + "_" + df.loc[reroll, "run"].astype(str)
                                    + "_rs" + df.loc[reroll, "sample_seed"].astype(int).astype(str) + "_judged.csv")
    df.to_parquet(resample / "resample_manifest.parquet", index=False)
    (resample / "resample_manifest.meta.json").write_text(json.dumps({
        "written_utc": "2026-09-23T00:00:00+00:00", "n_rows": int(len(df)),
        "split": {"method": "signature_stratified_hash_v1", "seed": 7}}))
    for (model, run, seed), g in df[reroll].groupby(["subject_model", "run", "sample_seed"]):
        rows = []
        for _, r in g.drop_duplicates(["original_index", "hint_style"]).iterrows():
            rows.append({
                "original_index": int(r["original_index"]), "sample_type": r["case"], "hint_name": r["hint_style"],
                "prompt": f"Q{int(r['original_index'])}?", "hinted_prompt": f"hinted {r['hint_style']} {int(r['original_index'])}",
                "hinted_answer": r["target_option"], "baseline_answer": r["baseline_modal_answer"],
                "groundtruth": r["groundtruth"],
                "rollout": f"<think>{reasoning_for(r['original_index'], r['hint_style'], int(seed))}</think><answer>B</answer>",
                "reasoning": reasoning_for(r["original_index"], r["hint_style"], int(seed)), "final_answer": "B",
                "additional_fields": "{}", **{c: "" for c in JUDGE_COLUMNS},
                **({"choices": CHOICES} if csv_choices else {}),
            })
        stem = f"{model}_{run}_rs{int(seed)}"
        frame = pd.DataFrame(rows)[list(OUTPUT_COLUMNS) + JUDGE_COLUMNS + (["choices"] if csv_choices else [])]
        frame.to_csv(rollouts / f"{stem}_judged.csv", index=False)
        if baseline:
            baseline_csv = root / "baselines" / f"{model}_{run}_baseline.csv"
            idx = sorted(set(df.loc[df["run"] == run, "original_index"].astype(int)) - set(baseline_skip or ()))
            pd.DataFrame({"original_index": idx, "question": "q", "choices": CHOICES}).to_csv(baseline_csv, index=False)
            (rollouts / f"{stem}.meta.json").write_text(json.dumps({"max_tokens": 24576, "seed": int(seed),
                                                                    "baseline_csv": str(baseline_csv)}))
    return root, df


class BuildProbeDatasetTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.tmp = Path(self.td.name)
        # outside ``env(root)`` DATA_ROOT points at a path that does not exist, so no test can reach a real tree
        guard = patch.dict(os.environ, {"DATA_ROOT": str(self.tmp / "no_data_root")})
        guard.start()
        self.addCleanup(guard.stop)

    def env(self, root: Path):
        return patch.dict(os.environ, {"DATA_ROOT": str(root)})

    def plan(self, root: Path, **overrides):
        spec = parse_spec(spec_cfg(root, **overrides))
        with self.env(root):
            manifest, meta = load_dataset_manifest(spec.manifest)
        rows, report = plan_dataset(spec, manifest, meta)
        return spec, rows, report

    def run_main(self, root: Path, argv):
        out = io.StringIO()
        with self.env(root), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(list(argv))
        return code, out.getvalue()

    def write_config(self, root: Path, **overrides) -> Path:
        path = self.tmp / "spec.yaml"
        path.write_text(yaml.safe_dump(spec_cfg(root, **overrides)))
        return path

    # -- contract --------------------------------------------------------------------------------

    def test_extra_columns_match_resample_relabel(self):
        from src.scripts.resample_relabel import EXTRA_COLUMNS
        self.assertEqual(list(RESAMPLE_EXTRA_COLUMNS), list(EXTRA_COLUMNS))
        self.assertEqual(DATASET_COLUMNS[-len(TEXT_COLUMNS):], TEXT_COLUMNS)
        self.assertEqual(DATASET_COLUMNS[33 + 13:33 + 15], ("label", "balance_kept"))

    def test_dataset_paths(self):
        root, _ = make_root(self.tmp)
        spec = parse_spec(spec_cfg(root))
        with self.env(root):
            parquet, meta, report = dataset_paths(spec)
            by_name = dataset_paths("test_ds", "${DATA_ROOT}/probe_datasets")
        self.assertEqual(parquet, root / "probe_datasets" / "test_ds.parquet")
        self.assertEqual((meta, report), (parquet.with_name("test_ds.meta.json"), parquet.with_name("test_ds_report.json")))
        self.assertEqual(by_name, (parquet, meta, report))
        self.assertEqual(dataset_sidecars(parquet), (meta, report))
        with self.assertRaises(ValueError):
            dataset_paths("test_ds")

    # -- join_texts ------------------------------------------------------------------------------

    def test_join_texts_matches_by_index_and_hint(self):
        root, _ = make_root(self.tmp)
        _, rows, _ = self.plan(root)
        sources = []
        joined = join_texts(rows, source_dir=root / "resample" / "rollouts", chunk_rows=7, sources=sources)
        self.assertEqual(list(joined["rollout_id"]), list(rows["rollout_id"]))
        for _, r in joined.iterrows():
            self.assertEqual(r["reasoning"], reasoning_for(r["original_index"], r["hint_style"], int(r["sample_seed"])))
            self.assertEqual(r["prompt"], f"Q{int(r['original_index'])}?")
            self.assertTrue(r["rollout"].startswith("<think>"))
            self.assertEqual(r["choices"], CHOICES)
            self.assertEqual(r["option_letters"], LETTERS)
        self.assertEqual(len(sources), len(SEEDS))
        self.assertEqual(sum(s["n_rows_joined"] for s in sources), len(rows))
        self.assertTrue(all(s["baseline_csv"] is None and s["size"] > 0 for s in sources))
        # two dataset rows on one key within one source: refused
        dup = pd.concat([rows.iloc[:1].assign(rollout_id="dup"), rows], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "several dataset rows"):
            join_texts(dup, source_dir=root / "resample" / "rollouts")

    def test_join_texts_missing_row_is_an_error(self):
        root, _ = make_root(self.tmp)
        _, rows, _ = self.plan(root)
        rows = rows.copy()
        first_source = rows.index[rows["source_csv"] == rows["source_csv"].iloc[0]][:3]  # 3 rows of one CSV
        rows.loc[first_source, "original_index"] = [999001, 999002, 999003]
        with self.assertRaisesRegex(ValueError, r"3 dataset row\(s\) have no CSV row.*#rs") as cm:
            join_texts(rows, source_dir=root / "resample" / "rollouts")
        for rid in rows.loc[first_source, "rollout_id"]:
            self.assertIn(rid, str(cm.exception))
        with self.assertRaises(FileNotFoundError):
            join_texts(rows.assign(source_csv="nope.csv"), source_dir=root / "resample" / "rollouts")

    def test_join_texts_refuses_source_without_reasoning(self):
        root, _ = make_root(self.tmp)
        _, rows, _ = self.plan(root)
        rollouts = root / "resample" / "rollouts"
        for col in ("reasoning", "prompt"):
            path = rollouts / f"m1_medqa-test_0911_rs43_judged.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame.drop(columns=[col]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, f"lacks column\\(s\\) \\['{col}'\\]"):
                join_texts(rows, source_dir=rollouts)
            frame.to_csv(path, index=False)

    def test_join_texts_refuses_source_of_another_seed(self):
        root, _ = make_root(self.tmp)
        _, rows, _ = self.plan(root)
        rollouts = root / "resample" / "rollouts"
        # the hinted-once CSV carries every (original_index, hint) key, so only the seed guard can catch it
        hinted_once = rollouts / "m1_medqa-test_0911_judged.csv"
        pd.read_csv(rollouts / "m1_medqa-test_0911_rs43_judged.csv", dtype=str, keep_default_na=False).to_csv(hinted_once, index=False)
        wrong = rows.copy()
        wrong.loc[wrong["sample_seed"].astype(int) == 44, "source_csv"] = hinted_once.name
        with self.assertRaisesRegex(ValueError, "not the per-seed re-roll CSV.*sample_seed \[44\]"):
            join_texts(wrong, source_dir=rollouts)
        wrong.loc[wrong["sample_seed"].astype(int) == 44, "source_csv"] = "m1_medqa-test_0911_rs43_judged.csv"
        with self.assertRaisesRegex(ValueError, "not the per-seed re-roll CSV"):
            join_texts(wrong, source_dir=rollouts)
        join_texts(rows, source_dir=rollouts)  # the manifest's own per-seed names pass
        join_texts(rows.drop(columns=["sample_seed"]), source_dir=rollouts)  # no seed column: no check

    def test_join_texts_falls_back_to_baseline_for_choices(self):
        skipped = 5
        root, _ = make_root(self.tmp, csv_choices=False, baseline=True, baseline_skip=[skipped])
        _, rows, _ = self.plan(root)
        sources = []
        joined = join_texts(rows, source_dir=root / "resample" / "rollouts", sources=sources)
        hit = joined["original_index"].astype(int) == skipped
        self.assertTrue(hit.any())
        self.assertTrue(joined.loc[hit, "choices"].isna().all())
        self.assertTrue(joined.loc[hit, "option_letters"].isna().all())
        self.assertTrue((joined.loc[~hit, "choices"] == CHOICES).all())
        self.assertTrue((joined.loc[~hit, "option_letters"] == LETTERS).all())
        self.assertTrue(all(s["baseline_csv"].endswith("m1_medqa-test_0911_baseline.csv") for s in sources))
        # a sidecar naming a relative baseline: treated as none named, no error
        for sidecar in (root / "resample" / "rollouts").glob("*.meta.json"):
            sidecar.write_text(json.dumps({"baseline_csv": "relative_baseline.csv"}))
        self.assertTrue(join_texts(rows, source_dir=root / "resample" / "rollouts")["choices"].isna().all())
        # no sidecar at all: null choices, no error
        for sidecar in (root / "resample" / "rollouts").glob("*.meta.json"):
            sidecar.unlink()
        joined = join_texts(rows, source_dir=root / "resample" / "rollouts")
        self.assertTrue(joined["choices"].isna().all() and joined["option_letters"].isna().all())

    # -- build / write / read ---------------------------------------------------------------------

    def test_build_writes_schema_in_order(self):
        root, _ = make_root(self.tmp)
        spec = parse_spec(spec_cfg(root))
        with self.env(root):
            rows, report = build_dataset(spec, chunk_rows=5)
            parquet = write_dataset(rows, report, spec)
            back, meta = read_dataset(parquet)
        self.assertEqual(list(rows.columns), list(DATASET_COLUMNS))
        self.assertEqual(list(back.columns), list(DATASET_COLUMNS))
        self.assertEqual(str(rows["label"].dtype), "Int8")
        self.assertTrue(rows["balance_kept"].all() and back["balance_kept"].all())
        self.assertEqual(list(rows["rollout_id"]), sorted(rows["rollout_id"]))
        self.assertTrue(rows["rollout_id"].is_unique)
        self.assertEqual(set(rows["provenance"].astype(str)), {"resample_k4"})
        self.assertEqual(set(rows["subject_model"].astype(str)), {"m1"})
        self.assertEqual(list(back["rollout_id"]), list(rows["rollout_id"]))
        self.assertEqual(list(back["reasoning"]), list(rows["reasoning"]))
        self.assertEqual(back["label"].astype(int).tolist(), rows["label"].astype(int).tolist())
        self.assertEqual(meta["fingerprint"], report["fingerprint"])
        self.assertEqual(report["n_after_balance"], len(rows))
        self.assertEqual(sorted(report["fold_report"]), report["hint_styles"])

    def test_meta_and_report_written(self):
        root, _ = make_root(self.tmp)
        spec = parse_spec(spec_cfg(root, balance={"method": "ratio", "r": 1.0}))
        with self.env(root):
            rows, report = build_dataset(spec)
            parquet = write_dataset(rows, report, spec)
        meta_file, report_file = dataset_sidecars(parquet)
        meta = json.loads(meta_file.read_text())
        for key in ("spec", "predicate", "label_col", "manifest_path", "manifest_written_utc", "manifest_n_rows",
                    "split", "balance", "sources", "hint_styles", "built_utc", "git_sha", "fingerprint", "n_rows"):
            self.assertIn(key, meta)
        self.assertEqual(meta["spec"], spec.to_dict())
        self.assertEqual((meta["predicate"], meta["label_col"]), ("verbalised_vs_unverbalised", "judge_label_final"))
        self.assertEqual(meta["manifest_path"], str(root / "resample" / "resample_manifest.parquet"))
        self.assertEqual(meta["manifest_written_utc"], "2026-09-23T00:00:00+00:00")
        self.assertEqual(meta["split"]["seed"], 7)
        self.assertEqual(meta["balance"]["method"], "ratio")
        self.assertEqual(meta["balance"]["n_after"], len(rows))
        self.assertEqual(meta["hint_styles"], sorted(set(rows["hint_style"].astype(str))))
        self.assertEqual(meta["n_rows"], len(rows))
        self.assertEqual(len(meta["sources"]), len(SEEDS))
        for s in meta["sources"]:
            self.assertEqual(set(s), {"source_csv", "path", "size", "mtime", "n_rows_joined", "columns", "baseline_csv"})
        self.assertTrue(meta["git_sha"] is None or len(meta["git_sha"]) == 40)
        written = json.loads(report_file.read_text())
        for key in ("spec", "n_selected", "n_after_balance", "balance", "fold_report", "sources", "fingerprint",
                    "hint_styles", "built_utc"):
            self.assertIn(key, written)
        self.assertEqual(written["fingerprint"], meta["fingerprint"])
        self.assertLess(written["n_after_balance"], written["n_selected"])

    def test_refuses_overwrite_without_force(self):
        root, _ = make_root(self.tmp)
        spec = parse_spec(spec_cfg(root))
        with self.env(root):
            rows, report = build_dataset(spec)
            parquet = write_dataset(rows, report, spec)
            with self.assertRaises(FileExistsError):
                write_dataset(rows, report, spec)
            self.assertEqual(write_dataset(rows, report, spec, force=True), parquet)
            read_dataset(parquet)
        self.assertEqual([p.name for p in parquet.parent.iterdir() if p.name.startswith(".")], [])  # no temp files left
        old = os.umask(0o022)
        try:
            with self.env(root):
                write_dataset(rows, report, spec, force=True)
        finally:
            os.umask(old)
        for path in (parquet, *dataset_sidecars(parquet)):  # readable by teammates on the shared tree, not mkstemp's 0600
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o644, path.name)

    def test_read_dataset_validates(self):
        root, _ = make_root(self.tmp)
        spec = parse_spec(spec_cfg(root))
        with self.env(root):
            rows, report = build_dataset(spec)
            parquet = write_dataset(rows, report, spec)
        validate_dataset(rows)
        dup = rows.copy()
        dup.loc[dup.index[1], "rollout_id"] = dup["rollout_id"].iloc[0]
        with self.assertRaisesRegex(ValueError, "duplicate rollout_id"):
            validate_dataset(dup)
        bad = rows.copy()
        bad.loc[bad.index[0], "label"] = 2
        with self.assertRaisesRegex(ValueError, "label must be 0 or 1"):
            validate_dataset(bad)
        empty = rows.copy()
        empty.loc[empty.index[0], "reasoning"] = "  "
        with self.assertRaisesRegex(ValueError, "empty reasoning"):
            validate_dataset(empty)
        order = rows[list(DATASET_COLUMNS[1:]) + [DATASET_COLUMNS[0]]]
        with self.assertRaisesRegex(ValueError, "columns differ"):
            validate_dataset(order)
        unkept = rows.copy()
        unkept.loc[unkept.index[0], "balance_kept"] = False
        with self.assertRaisesRegex(ValueError, "balance_kept"):
            validate_dataset(unkept)
        original = rows.copy()
        original.loc[original.index[0], "provenance"] = "hinted_once"
        with self.assertRaisesRegex(ValueError, "re-rolls"):
            validate_dataset(original)
        # on disk: a rewritten parquet fails validation, a valid rewrite fails the fingerprint check
        dup.to_parquet(parquet, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate rollout_id"):
            read_dataset(parquet)
        rows.iloc[1:].to_parquet(parquet, index=False)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            read_dataset(parquet)
        dataset_sidecars(parquet)[0].unlink()
        rows.to_parquet(parquet, index=False)
        with self.assertRaises(FileNotFoundError):
            read_dataset(parquet)

    # -- CLI ------------------------------------------------------------------------------------

    def test_dry_run_writes_nothing(self):
        root, _ = make_root(self.tmp)
        config = self.write_config(root)
        code, out = self.run_main(root, ["--config", str(config), "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertIn("dry run — nothing written", out)
        self.assertIn("fold report", out)
        self.assertIn("test_ood", out)
        self.assertIn("ood_has_both_labels", out)
        self.assertIn("balance: none", out)
        self.assertIn("fingerprint ", out)
        self.assertFalse((root / "probe_datasets").exists())
        # a bad spec exits 1 with a message, never a traceback
        config.write_text(yaml.safe_dump(spec_cfg(root, predicate="dataset_C_incoherent")))
        code, out = self.run_main(root, ["--config", str(config), "--dry-run"])
        self.assertEqual(code, 1)
        self.assertIn("error:", out)
        config.write_text(yaml.safe_dump(spec_cfg(root, predicate=None)))  # an explicit null on a required key
        code, out = self.run_main(root, ["--config", str(config), "--dry-run"])
        self.assertEqual(code, 1)
        self.assertIn("missing required key(s) ['predicate']", out)

    def test_cli_manifest_override(self):
        root, _ = make_root(self.tmp)
        config = self.write_config(root, manifest="${DATA_ROOT}/resample/nope.parquet")
        code, out = self.run_main(root, ["--config", str(config)])
        self.assertEqual(code, 1)
        self.assertIn("does not exist", out)
        real = root / "resample" / "resample_manifest.parquet"
        code, out = self.run_main(root, ["--config", str(config), "--manifest", str(real), "--chunk-rows", "6"])
        self.assertEqual(code, 0, out)
        parquet = root / "probe_datasets" / "test_ds.parquet"
        self.assertIn(f"wrote {parquet}", out)
        with self.env(root):
            back, meta = read_dataset(parquet)
        self.assertEqual(meta["manifest_path"], str(real))
        self.assertEqual(meta["spec"]["manifest"], str(real))
        self.assertGreater(len(back), 0)
        # a second run refuses to overwrite without --force, and does so before any join
        code, out = self.run_main(root, ["--config", str(config), "--manifest", str(real)])
        self.assertEqual(code, 1)
        self.assertIn("--force", out)
        code, out = self.run_main(root, ["--config", str(config), "--manifest", str(real), "--force"])
        self.assertEqual(code, 0, out)

    def test_config_templates_parse(self):
        templates = sorted((REPO_ROOT / "configs" / "probe_datasets").glob("*.yaml"))
        names = [t.name for t in templates]
        for model in ("gemma4-12b-it", "olmo3-7b-think", "qwen3-8b"):
            for label in ("used_vs_ignored", "verbalised_vs_rest", "verbalised_vs_unverbalised"):
                self.assertIn(f"cueball_{model}_{label}.yaml", names)
        root = self.tmp / "data_root"
        for path in templates:
            spec = parse_spec(load_config(str(path)))
            # Every template is one subject model x one label configuration x that model's four runs.
            self.assertEqual(len(spec.subject_models), 1)
            self.assertEqual(spec.name, f"{spec.subject_models[0]}_{spec.predicate}")
            self.assertEqual(len(spec.runs), 4)
            self.assertIsNone(spec.hint_styles)
            self.assertEqual((spec.cases, spec.balance.method, spec.balance.r, spec.seed), ("both", "none", None, 42))
            self.assertEqual(spec.manifest, "${DATA_ROOT}/resample/resample_manifest.parquet")
            with self.env(root):
                self.assertEqual(resolve_data_path(spec.manifest), root / "resample" / "resample_manifest.parquet")
                self.assertEqual(dataset_paths(spec)[0], root / "probe_datasets" / f"{spec.name}.parquet")


if __name__ == "__main__":
    unittest.main()
