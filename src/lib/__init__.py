"""Shared library of the pipeline. Importing it loads the repo-root ``.env``."""

from src.lib.env import load_env

load_env()
