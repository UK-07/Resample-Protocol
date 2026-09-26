"""Build paper figures and tables from released numeric records."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys

FAMILIES = ("survival", "rates", "alpha_roles", "crosscheck", "dose")
INPUT_FILES = ("pairs_master.parquet", "rerolls_long.parquet")
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
    "fig_dose_response_by_case",
}
EXPECTED_TABLES = {"rates_main.tex", "benchmark_main.tex", "benchmark_appendix.tex"}


@dataclass(frozen=True)
class FigurePaths:
    data: Path
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


def check_source(source: Path, *, from_tree: bool = False) -> None:
    if not source.is_dir():
        raise ValueError(f"Input must be a data directory: {source}")
    required = ("hinted_rollouts/rollout_manifest.parquet",
                "resample/resample_manifest.parquet", "resample/question_reliance.csv") if from_tree else INPUT_FILES
    missing = [name for name in required if not (source / name).is_file()]
    if from_tree:
        missing += [name + "/" for name in ("baselines", "binary_judge") if not (source / name).is_dir()]
    if missing:
        raise FileNotFoundError("Missing figure inputs: " + ", ".join(missing))


def worker(phase: str, source: Path, output: Path) -> None:
    paths = FigurePaths(source, output)
    if phase == "prepare":
        from .inputs import prepare
        prepare(source, output / "inputs")
    elif phase == "compute":
        from .recompute import build
        build(source, paths.results)
    elif phase in FAMILIES:
        import_module(f"{__package__}.{phase}").build(paths)
    else:
        raise ValueError(f"Unknown figure phase: {phase}")


def build(source: Path, output: Path, *, from_tree: bool = False, dry_run: bool = False) -> dict:
    """Read source data without changing it and write to a fresh output directory."""
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Choose an output directory outside the input data tree")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be empty; choose a fresh directory")
    check_source(source, from_tree=from_tree)
    phases = (["prepare"] if from_tree else []) + ["compute", *FAMILIES]
    plan = {"phases": phases, "pdf_count": len(EXPECTED_PDFS), "latex_table_count": len(EXPECTED_TABLES)}
    if dry_run:
        return plan
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "cache" / "matplotlib"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    prepared = output / "inputs" if from_tree else source
    for phase in phases:
        print(f"Building {phase}…", flush=True)
        flag = "--cueball-dir" if phase == "prepare" else "--data"
        data = source if phase == "prepare" else prepared
        subprocess.run([sys.executable, "-m", "src.scripts.visualizations.paper_figures",
                        flag, str(data), "--out-root", str(output), "--_phase", phase],
                       env=env, check=True)
    pdfs = {path.stem for path in (output / "figures").glob("*.pdf")}
    tables = {path.name for path in (output / "tables").glob("*.tex")}
    if pdfs != EXPECTED_PDFS or tables != EXPECTED_TABLES:
        raise ValueError(f"Incomplete figure build: PDFs {pdfs ^ EXPECTED_PDFS}; tables {tables ^ EXPECTED_TABLES}")
    return {**plan, "status": "complete"}
