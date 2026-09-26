"""Load the repo-root ``.env`` (secrets and ``DATA_ROOT``) into ``os.environ``.

Imported by ``src.lib.__init__``, so every script and notebook picks the file up.
A variable already set in the environment is never overridden by the file, and a
missing file is a no-op. See ``.env.example`` for the keys.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"

_loaded = False


def load_env(*, force: bool = False) -> Path | None:
    """Load ``ENV_FILE`` once (again with ``force``); returns the path loaded or ``None``."""
    global _loaded
    if _loaded and not force:
        return None
    _loaded = True
    if not ENV_FILE.is_file():
        return None
    load_dotenv(ENV_FILE, override=False)
    return ENV_FILE
