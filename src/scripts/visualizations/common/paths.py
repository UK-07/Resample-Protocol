"""Locations of the released paper tree (``--cueball-dir``) and of the plot outputs.

Nothing here is a module-level path constant: every location is resolved when asked for, from the
``cueball`` argument or from ``${DATA_ROOT}`` (``src.lib.paths.resolve_data_path``), so a gather run against
another tree only needs ``--cueball-dir``.
"""

from __future__ import annotations

from pathlib import Path

from src.lib.paths import REPO_ROOT, resolve_data_path

DEFAULT_CUEBALL_DIR = "${DATA_ROOT}/cueball"
DEFAULT_JUDGE_CACHE_DIR = "${DATA_ROOT}/judge_cache"


def cueball_dir(raw: str | Path | None = None) -> Path:
    """The paper tree: ``raw`` (a ``${DATA_ROOT}`` string or an absolute path) or :data:`DEFAULT_CUEBALL_DIR`."""
    return resolve_data_path(raw if raw is not None else DEFAULT_CUEBALL_DIR)


def plots_dir(cueball: str | Path | None = None) -> Path:
    """``<cueball>/plots``: the root of every plot's outputs."""
    return cueball_dir(cueball) / "plots"


def plot_dir(name: str, cueball: str | Path | None = None) -> Path:
    """``<cueball>/plots/<name>``: one plot's tables (``data/``) and figures."""
    return plots_dir(cueball) / name


def cache_dir(cueball: str | Path | None = None) -> Path:
    """``<cueball>/plots/_cache``: the stitched-rows cache shared by every gather."""
    return plots_dir(cueball) / "_cache"


# The inputs every gather reads (the released tree's layout).

def manifest_path(cueball: str | Path | None = None) -> Path:
    return cueball_dir(cueball) / "hinted_rollouts" / "rollout_manifest.parquet"


def manifest_meta_path(cueball: str | Path | None = None) -> Path:
    return cueball_dir(cueball) / "hinted_rollouts" / "rollout_manifest.meta.json"


def resample_manifest_path(cueball: str | Path | None = None) -> Path:
    return cueball_dir(cueball) / "resample" / "resample_manifest.parquet"


def reliance_path(cueball: str | Path | None = None) -> Path:
    return cueball_dir(cueball) / "resample" / "question_reliance.csv"


def binary_dir(cueball: str | Path | None = None) -> Path:
    """``<cueball>/binary_judge``: the binary judge's ``<stem>_binary_judged.csv`` files."""
    return cueball_dir(cueball) / "binary_judge"


def hinted_dir(cueball: str | Path | None = None) -> Path:
    """``<cueball>/hinted_rollouts``: the hinted-once CSVs, their recipe sidecars and judged outputs."""
    return cueball_dir(cueball) / "hinted_rollouts"


def rollouts_dir(cueball: str | Path | None = None) -> Path:
    """``<cueball>/resample/rollouts``: the per-seed re-roll CSVs."""
    return cueball_dir(cueball) / "resample" / "rollouts"


def judge_cache_root() -> Path:
    """``${DATA_ROOT}/judge_cache``: the judge caches, keyed on the rollouts-CSV stem."""
    return resolve_data_path(DEFAULT_JUDGE_CACHE_DIR)


def repo_path(relative: str) -> Path:
    """A file of this repository by its repo-relative path (for code provenance)."""
    return REPO_ROOT / relative
