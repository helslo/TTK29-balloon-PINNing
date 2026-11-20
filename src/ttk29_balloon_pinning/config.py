"""Shared project configuration."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "TTK29_DATA_ROOT"
DEFAULT_DATA_ROOT = Path(
    "/Volumes/brainkyblab/Datasets/Dataset Mental Health Kavli/"
    "Helene -LA5c/ds000030_fmriprep"
)


def get_data_root() -> Path:
    """Return the shared dataset root, allowing an environment override."""
    configured_root = os.environ.get(DATA_ROOT_ENV)
    if configured_root is not None and not configured_root.strip():
        raise ValueError(f"{DATA_ROOT_ENV} must be a non-empty path")
    return Path(configured_root).expanduser() if configured_root else DEFAULT_DATA_ROOT
