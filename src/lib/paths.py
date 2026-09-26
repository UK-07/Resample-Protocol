"""Resolution of data-artifact paths against a shared ``DATA_ROOT``.

Config path fields use the ``${DATA_ROOT}/...`` prefix; a path that does not
reference it must be absolute. ``DATA_ROOT`` defaults to ``<repo>/data``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT_ENV = "DATA_ROOT"
_DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def data_root() -> Path:
    """Return ``${DATA_ROOT}`` if set (must be absolute), else ``<repo>/data``."""
    root = os.environ.get(DATA_ROOT_ENV)
    if not root:
        # stderr: shell wrappers capture a resolved path from stdout.
        print(
            f"DATA_ROOT not found. Defaulting to {_DEFAULT_DATA_ROOT}",
            file=sys.stderr,
        )
        return _DEFAULT_DATA_ROOT
    path = Path(root)
    if not path.is_absolute():
        raise ValueError(
            f"{DATA_ROOT_ENV}={root!r} must be an absolute path."
        )
    return path


def resolve_data_path(raw: str | os.PathLike) -> Path:
    """Expand ``${DATA_ROOT}`` and any other ``$VAR``; the result must be absolute.

    Raises ``ValueError`` on an unresolved variable or a relative result.
    """
    text = os.fspath(raw)
    root = os.fspath(data_root())
    # Only the braced form: a bare "$DATA_ROOT" would match a longer "$DATA_ROOTX".
    text = text.replace("${DATA_ROOT}", root)
    expanded = os.path.expandvars(text)
    if "$" in expanded:
        raise ValueError(
            f"Unresolved environment variable in data path {os.fspath(raw)!r} "
            f"(expanded to {expanded!r}). Set every referenced variable."
        )
    path = Path(expanded)
    if not path.is_absolute():
        raise ValueError(
            f"Data path {os.fspath(raw)!r} is relative and does not reference "
            f"${{DATA_ROOT}}. Use a '${{DATA_ROOT}}/...' path or an absolute path."
        )
    return path
