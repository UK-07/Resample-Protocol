"""Portable, read-only-input reproduction of the 2026-09-26 paper revision."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module, metadata
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

EDITION = "2026-09-26"
ARCHIVE_SHA256 = "d9ffbb1ff06e56be2302eaa532102a667cb0a7d2e8ca29d486a93737977f1ea1"
MANIFEST_SHA256 = "cf4321ae6389b8eef26adf6bbf12de7d65f9d383d4807e4e064fb34709593466"
FAMILIES = ("survival", "rates", "alpha_roles", "crosscheck", "dose", "judge_kappa")
EXPECTED_PDFS = {
    "fig_survival_vs_stability", "fig_survival_by_style", "fig_survival_by_dataset",
    "fig_survival_slice_case", "fig_survival_slice_cue_family", "fig_survival_slice_post_hoc",
    "fig_reliance_composition_ssp_unfaithful", "fig_reliance_composition_ssp_flips",
    "fig_reliance_composition_ssp_unfaithful_negative", "fig_reliance_composition_ssp_flips_negative",
    "fig_reroll_distribution", "fig_yield_per_style", "fig_rates_dumbbell",
    "fig_rates_dumbbell_by_dataset", "fig_rates_dumbbell_by_style_positive",
    "fig_rates_dumbbell_by_style_negative", "fig_susceptibility_vs_unfaithfulness",
    "fig_alpha_vs_measured_noise", "fig_alpha_vs_measured_noise_strict",
    "fig_alpha_vs_measured_noise_by_dataset", "fig_alpha_bias_vs_stability",
    "fig_alpha_negative_case_check", "fig_false_rejections", "fig_false_rejections_by_style",
    "fig_false_rejections_by_case", "fig_false_rejections_by_stability",
    "fig_noise_crosscheck_literal", "fig_noise_crosscheck_strict", "fig_dose_response",
    "fig_dose_response_by_case", "fig_judge_J1a", "fig_judge_J1b", "fig_judge_J1c",
}
EXPECTED_TABLES = {"rates_main.tex", "benchmark_main.tex", "benchmark_appendix.tex"}
EXPECTED_RECOMPUTED_TABLES = frozenset({
    'plot_tables/fig_rates_dumbbell_ssp_vs_rsp_clusterCI.csv',
    'plot_tables/fig_reliance_composition_ssp_flips.csv',
    'plot_tables/fig_reliance_composition_ssp_unfaithful.csv',
    'plot_tables/fig_survival_by_dataset_clusterCI.csv',
    'plot_tables/fig_survival_by_style_clusterCI.csv',
    'plot_tables/fig_survival_slice_case_clusterCI.csv',
    'plot_tables/fig_survival_vs_stability_positive_clusterCI.csv',
    'plot_tables/fig_survival_vs_stability_positive_model_overall_clusterCI.csv',
    'plot_tables_followup/benchmark_cells_model_cue_case_clusterCI.csv',
    'plot_tables_followup/benchmark_cells_model_dataset_case_clusterCI.csv',
    'plot_tables_followup/benchmark_cells_model_dataset_cue_case_clusterCI.csv',
    'plot_tables_followup/fig_alpha_bias_vs_stability_clusterCI.csv',
    'plot_tables_followup/fig_false_rejections_by_style_role_category_clusterCI.csv',
    'plot_tables_followup/fig_survival_slice_cue_family_clusterCI.csv',
    'plot_tables_followup/fig_survival_slice_post_hoc_clusterCI.csv',
    'plot_tables_followup/fig_yield_per_style_case_clusterCI.csv',
    'plot_tables_followup/fig_yield_per_style_clusterCI.csv',
    's1_persistence/alpha_vs_measured_noise.csv',
    's1_persistence/cells_ssp_flips_model_dataset_cue_case.csv',
    's1_persistence/cells_ssp_unfaithful_model_dataset_cue_case.csv',
    's1_persistence/persistence_counts_and_survival.csv',
    's2_roles/crosstab_emitted_role_x_coherence_by_model.csv',
    's2_roles/role_category_persistence.csv',
    's2_roles/role_coverage_and_inclusive_counts_by_model.csv',
})


@dataclass(frozen=True)
class RevisionPaths:
    bundle: Path
    output: Path

    @property
    def results(self) -> Path:
        return self.output / "analysis" / "results"

    @property
    def figures(self) -> Path:
        return self.output / "figures"

    @property
    def tables(self) -> Path:
        return self.output / "tables"

    @property
    def derived(self) -> Path:
        return self.output / "data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_file(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if not raw or not path.parts or path.is_absolute() or ".." in path.parts or "\\" in raw or ":" in raw:
        raise ValueError(f"Unsafe bundle path: {raw!r}")
    return path


def read_manifest(data: bytes) -> dict:
    manifest = json.loads(data)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Bundle manifest has no file entries")
    names = [str(relative_file(e["path"])) for e in files]
    if len(names) != len(set(names)):
        raise ValueError("Bundle manifest contains duplicate paths")
    if len(files) != manifest.get("n_files"):
        raise ValueError("Bundle manifest file count is inconsistent")
    return manifest


def verify_directory(root: Path) -> dict:
    """Verify every manifest-listed input, without importing archive scripts."""
    if root.is_symlink():
        raise ValueError("Bundle directory must not be a symlink")
    manifest_path = root / "MANIFEST.json"
    if manifest_path.is_symlink():
        raise ValueError("Bundle manifest must not be a symlink")
    manifest = read_manifest(manifest_path.read_bytes())
    for entry in manifest["files"]:
        path = root.joinpath(*relative_file(entry["path"]).parts)
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise ValueError(f"Symlink in bundle input: {entry['path']}")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Bundle input does not match manifest: {entry['path']}")
    return {"manifest_sha256": sha256(manifest_path), "verified_files": len(manifest["files"])}


def verify_archive(path: Path) -> dict:
    """Verify the released archive and refuse links, duplicates and traversal."""
    digest = sha256(path)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"This command targets the {EDITION} archive; SHA-256 mismatch: {digest}")
    with tarfile.open(path, "r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            name = relative_file(member.name)
            if not member.isdir() and not member.isfile():
                raise ValueError(f"Non-regular archive member: {member.name}")
            if name.parts[0] != "revision_analysis":
                raise ValueError(f"Unexpected archive root: {member.name}")
            if member.isfile():
                if member.name in files:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                files[member.name] = member
        data = archive.extractfile(files["revision_analysis/MANIFEST.json"]).read()
        manifest = read_manifest(data)
        expected = {"revision_analysis/MANIFEST.json"} | {"revision_analysis/" + e["path"] for e in manifest["files"]}
        if set(files) != expected:
            raise ValueError("Archive files do not match manifest")
        for entry in manifest["files"]:
            member = files["revision_analysis/" + entry["path"]]
            content = archive.extractfile(member).read()
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Archive member hash mismatch: {entry['path']}")
    return {"archive_sha256": digest, "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "verified_files": len(manifest["files"])}


def prepare_bundle(source: Path, output: Path) -> tuple[Path, dict]:
    if source.is_dir():
        return source, verify_directory(source)
    info = verify_archive(source)
    destination = output / "inputs"
    destination.mkdir(parents=True)
    # Every member has just been validated. Extract regular files manually,
    # avoiding archive extraction's symlink and absolute-path semantics.
    with tarfile.open(source, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                target = destination.joinpath(*relative_file(member.name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open("wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
    root = destination / "revision_analysis"
    verify_directory(root)
    return root, info


def verify_recomputed(reference: Path, generated: Path) -> dict:
    """Verify this edition's complete table set and bounded numerical correction.

    Integer-valued reference columns (including nullable counts parsed as floats)
    compare exactly. Other numeric columns use absolute tolerance 2e-7 and zero
    relative tolerance. Only the documented six alpha p-value and four BH q-value
    corrections are exempted, with explicit bounds and unchanged BH decisions.
    """
    import numpy as np
    import pandas as pd

    produced = {path.relative_to(generated).as_posix() for path in generated.rglob("*.csv")}
    if produced != EXPECTED_RECOMPUTED_TABLES:
        missing = sorted(EXPECTED_RECOMPUTED_TABLES - produced)
        unexpected = sorted(produced - EXPECTED_RECOMPUTED_TABLES)
        raise ValueError(f"Incomplete recomputed table set; missing={missing}, unexpected={unexpected}")
    report = {}
    for name in sorted(EXPECTED_RECOMPUTED_TABLES):
        path, original = generated / name, reference / name
        if not original.is_file():
            raise ValueError(f"No archived reference for recomputed table: {name}")
        got, expected = pd.read_csv(path), pd.read_csv(original)
        if got.shape != expected.shape or list(got.columns) != list(expected.columns):
            raise ValueError(f"Changed table rows or columns: {name}")
        corrections = {}
        corrected = {"p_strict": (0.0067, 6), "q_bh_strict": (0.0069, 4)} if name == "s1_persistence/alpha_vs_measured_noise.csv" else {}
        # The exact integer-numerator calculation corrects platform-dependent
        # float32 signs at mathematical ties. This pinned edition has precisely
        # six changed p-values and four propagated BH q-values. These bounds
        # allow that documented correction without exempting the columns from
        # regression validation or allowing any reported BH decision to change.
        for column, (maximum_delta, expected_changes) in corrected.items():
            a = got[column].to_numpy(dtype=float)
            b = expected[column].to_numpy(dtype=float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                raise ValueError(f"Changed undefined p/q values: {column}")
            if np.isinf(a).any() or np.isinf(b).any():
                raise ValueError(f"Non-finite alpha p/q value: {column}")
            finite = np.isfinite(a)
            if np.any((a[finite] < 0) | (a[finite] > 1)) or np.any((b[finite] < 0) | (b[finite] > 1)):
                raise ValueError(f"Alpha p/q value outside [0, 1]: {column}")
            differences = np.abs(a - b)
            if np.any(differences[finite] > maximum_delta):
                raise ValueError(f"Alpha numerical correction exceeds its documented bound: {column}")
            changed = np.flatnonzero(~np.isclose(a, b, equal_nan=True, rtol=0, atol=1e-12))
            if len(changed) != expected_changes:
                raise ValueError(f"Expected {expected_changes} corrected {column} values, found {len(changed)}")
            identifiers = [key for key in ("grouping", "subject_model", "dataset", "hint_style", "case") if key in expected]
            corrections[column] = [
                {"row": int(i), "keys": expected.loc[i, identifiers].fillna("").to_dict(),
                 "archived": float(b[i]), "recomputed": float(a[i])}
                for i in changed
            ]
            if column.startswith("q_") and not np.array_equal(a < .05, b < .05):
                raise ValueError("Alpha numerical correction changed a reported BH decision")
        max_error = 0.0
        exact_columns = []
        for column in expected:
            if column in corrected:
                continue
            if pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(expected[column]):
                a = got[column].to_numpy(dtype=float)
                b = expected[column].to_numpy(dtype=float)
                if not np.array_equal(np.isnan(a), np.isnan(b)):
                    raise ValueError(f"Changed missing-value mask: {name}:{column}")
                if not np.array_equal(np.isposinf(a), np.isposinf(b)) or not np.array_equal(np.isneginf(a), np.isneginf(b)):
                    raise ValueError(f"Changed infinite-value mask: {name}:{column}")
                finite = np.isfinite(b)
                if np.all(b[finite] == np.rint(b[finite])):
                    np.testing.assert_array_equal(a, b, err_msg=f"Integer-valued column differs: {name}:{column}")
                    exact_columns.append(column)
                else:
                    np.testing.assert_allclose(a, b, rtol=0, atol=2e-7, equal_nan=True,
                                               err_msg=f"Floating column differs: {name}:{column}")
                if finite.any():
                    max_error = max(max_error, float(np.max(np.abs(a[finite] - b[finite]))))
            else:
                pd.testing.assert_series_equal(got[column], expected[column], check_dtype=False,
                                               check_exact=True, obj=f"{name}:{column}")
        report[name] = {"rows": len(got), "max_absolute_numeric_difference": max_error,
                        "integer_columns_compared_exactly": exact_columns,
                        "alpha_sign_corrections": corrections}
    return report


def worker(phase: str, bundle: Path, output: Path) -> None:
    paths = RevisionPaths(bundle, output)
    if sha256(bundle / "MANIFEST.json") != MANIFEST_SHA256:
        raise ValueError("Extracted bundle manifest does not match the paper edition")
    if phase == "recompute":
        from . import recompute
        summary = recompute.build(bundle / "results", paths.results)
        if (summary["n_questions"], summary["weights_fingerprint"]) != (3982, "1a8b5e9a029dc3ce"):
            raise ValueError("Question frame/bootstrap weights differ from the paper edition")
        checks = verify_recomputed(bundle / "results", paths.results)
        paths.derived.mkdir(parents=True, exist_ok=True)
        (paths.derived / "recomputed_table_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
        from .validation import validate_counts
        validate_counts(bundle / "results", paths.derived / "count_validation.md")
    elif phase in FAMILIES:
        import_module(f"{__package__}.{phase}").build(paths)
    else:
        raise ValueError(f"Unknown phase {phase}")


def reproduce(source: Path, output: Path, *, dry_run: bool = False) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or (source.is_dir() and source in output.parents):
        raise ValueError("Choose an output directory outside the read-only bundle")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a fresh scratch directory")
    info = verify_directory(source) if source.is_dir() else verify_archive(source)
    if info["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Extracted bundle manifest does not match the paper edition")
    plan = {"edition": EDITION, "input": str(source), "output": str(output),
            "phases": ["recompute", *FAMILIES], "pdf_count": len(EXPECTED_PDFS),
            "latex_table_count": len(EXPECTED_TABLES), **info}
    if dry_run:
        return plan
    output.mkdir(parents=True, exist_ok=True)
    bundle, info = prepare_bundle(source, output)
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "cache" / "matplotlib"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # Each renderer gets an independent Matplotlib context, matching the original
    # builders and preventing figure-family style state from leaking.
    for phase in plan["phases"]:
        print(f"Reproducing {phase}…", flush=True)
        subprocess.run([sys.executable, "-m", "src.scripts.visualizations.paper_revision",
                        "--bundle", str(bundle), "--out-root", str(output), "--_phase", phase],
                       env=env, check=True)
    pdfs = {p.stem for p in (output / "figures").glob("*.pdf")}
    tables = {p.name for p in (output / "tables").glob("*.tex")}
    if pdfs != EXPECTED_PDFS or tables != EXPECTED_TABLES:
        raise ValueError(f"Incomplete reproduction: PDFs {pdfs ^ EXPECTED_PDFS}; tables {tables ^ EXPECTED_TABLES}")
    verify_directory(bundle)  # No input has changed during the calculations.
    package = Path(__file__).parent
    outputs = sorted([*(output / "figures").glob("*.pdf"), *(output / "tables").glob("*.tex"),
                      *(output / "data").glob("*"), *(output / "analysis/results").rglob("*.csv")])
    versions = {p: metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "matplotlib", "reportlab", "python-dotenv")}
    run = {**plan, **info, "status": "complete", "utc": datetime.now(timezone.utc).isoformat(),
           "python": sys.version, "dependencies": versions,
           "code_sha256": {p.name: sha256(p) for p in sorted(package.glob("*.py"))},
           "asset_sha256": {str(p.relative_to(package)): sha256(p) for p in sorted((package / "assets").glob("*")) if p.is_file()},
           "cli_sha256": sha256(package.parents[1] / "scripts/visualizations/paper_revision.py"),
           "outputs": {str(p.relative_to(output)): sha256(p) for p in outputs if p.is_file()},
           "scope": "33 revised PDFs and three LaTeX tables; existing legacy figures use the existing visualization commands. No generation or judging."}
    (output / "reproduction_manifest.json").write_text(json.dumps(run, indent=2) + "\n")
    return run
