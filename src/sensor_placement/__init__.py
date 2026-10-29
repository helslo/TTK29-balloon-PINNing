"""Sparse sensor placement and compressed sensing utilities for voxel time series."""

from .data import (
    RunFiles,
    SensorRun,
    discover_runs,
    load_clean_runs,
    temporal_holdout_split,
    participant_holdout_split,
)
from .methods import (
    l1_dct_placement,
    qr_pivot_placement,
    random_placement,
)

__all__ = [
    "RunFiles",
    "SensorRun",
    "discover_runs",
    "load_clean_runs",
    "temporal_holdout_split",
    "participant_holdout_split",
    "random_placement",
    "qr_pivot_placement",
    "l1_dct_placement",
]
