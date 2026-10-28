"""Sparse sensor placement and compressed sensing utilities for ROI time series."""

from .data import (
    build_roi_matrices,
    load_roi_data,
    temporal_holdout_split,
    participant_holdout_split,
    synthetic_roi_data,
)
from .methods import (
    lasso_placement,
    qr_pivot_placement,
    random_placement,
)

__all__ = [
    "build_roi_matrices",
    "load_roi_data",
    "temporal_holdout_split",
    "participant_holdout_split",
    "synthetic_roi_data",
    "random_placement",
    "qr_pivot_placement",
    "lasso_placement",
]
