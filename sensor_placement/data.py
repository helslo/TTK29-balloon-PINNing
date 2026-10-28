"""Utilities for loading and preprocessing ROI-averaged BOLD time series."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import h5py
import numpy as np


def zscore_signal(signal: Sequence[float] | np.ndarray) -> np.ndarray:
    """Standardize one ROI time series to zero mean and unit variance."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Signal must be 1D, got shape {values.shape}")
    if values.size == 0:
        return values.copy()
    std = values.std()
    if not np.isfinite(std) or std == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / std


def load_roi_data(path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load ROI time-series data from an HDF5 file.

    The expected layout is a dictionary-like structure where each participant is a
    key mapped to ROI names and the ROI names map to 1D arrays of BOLD values.
    This supports the README's assumed layout of ``{participant_ID: {ROI: signal}}``
    as well as common HDF5 group structures created from that dictionary.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"ROI-averaged data not found at {data_path}. "
            "Generate the HDF5 file or pass --data to a synthetic or custom dataset."
        )

    with h5py.File(data_path, "r") as handle:
        return _read_h5_group(handle)


def _read_h5_group(group: h5py.Group | h5py.File) -> Dict[str, Dict[str, np.ndarray]]:
    """Recursively read a participant/ROI mapping from an HDF5 group."""
    participants: Dict[str, Dict[str, np.ndarray]] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            continue
        roi_map: Dict[str, np.ndarray] = {}
        for roi_name, roi_data in value.items():
            if isinstance(roi_data, h5py.Dataset):
                roi_map[str(roi_name)] = np.asarray(roi_data, dtype=float).reshape(-1)
        participants[str(key)] = roi_map
    return participants


def build_roi_matrices(
    participant_data: Mapping[str, Mapping[str, Sequence[float]]],
    zscore: bool = True,
) -> Tuple[Dict[str, np.ndarray], list[str]]:
    """Build a matrix ``X`` of shape ``(n_rois, n_timepoints)`` for each participant."""
    matrices: Dict[str, np.ndarray] = {}
    roi_names: list[str] | None = None
    for participant_id, roi_map in participant_data.items():
        if not roi_map:
            raise ValueError(f"Participant {participant_id!r} contains no ROI signals")
        ordered_roi_names = list(roi_map)
        if roi_names is None:
            roi_names = ordered_roi_names
        elif ordered_roi_names != roi_names:
            ordered_roi_names = roi_names
        matrix = np.vstack([np.asarray(roi_map[roi_name], dtype=float).reshape(-1) for roi_name in ordered_roi_names])
        if zscore:
            matrix = np.vstack([zscore_signal(row) for row in matrix])
        matrices[str(participant_id)] = matrix
    if roi_names is None:
        raise ValueError("No ROI data was found")
    return matrices, roi_names


def temporal_holdout_split(X: np.ndarray, holdout_fraction: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/test time indices for a single participant's ROI matrix."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")
    n_time = X.shape[1]
    split_index = int(round((1.0 - holdout_fraction) * n_time))
    split_index = min(max(1, split_index), n_time - 1)
    train_idx = np.arange(split_index)
    test_idx = np.arange(split_index, n_time)
    return train_idx, test_idx


def participant_holdout_split(
    participant_matrices: Mapping[str, np.ndarray],
    n_folds: int = 5,
) -> list[Tuple[list[str], list[str]]]:
    """Generate cross-validation splits over participants.

    Each tuple is (train_participants, test_participants). The returned ordering is
    deterministic for reproducible comparisons across folds.
    """
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    participant_ids = sorted(participant_matrices)
    if len(participant_ids) < 2:
        raise ValueError("At least two participants are required for participant hold-out")
    n_participants = len(participant_ids)
    folds: list[Tuple[list[str], list[str]]] = []
    fold_size = max(1, n_participants // n_folds)
    for fold_index in range(n_folds):
        start = fold_index * fold_size
        end = start + fold_size
        if fold_index == n_folds - 1:
            end = n_participants
        test_ids = participant_ids[start:end]
        train_ids = [pid for pid in participant_ids if pid not in set(test_ids)]
        if not train_ids or not test_ids:
            continue
        folds.append((train_ids, test_ids))
    return folds


def synthetic_roi_data(
    n_participants: int = 8,
    n_rois: int = 32,
    n_timepoints: int = 160,
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create a low-rank synthetic ROI dataset when the HDF5 file is unavailable."""
    rng = np.random.default_rng(seed)
    low_rank_latents = rng.normal(size=(n_participants, 4, n_timepoints))
    basis = rng.normal(size=(n_rois, 4))
    participant_data: Dict[str, Dict[str, np.ndarray]] = {}
    for participant_index in range(n_participants):
        roi_map: Dict[str, np.ndarray] = {}
        for roi_index in range(n_rois):
            signal = 0.0
            for mode in range(4):
                signal += basis[roi_index, mode] * low_rank_latents[participant_index, mode]
            signal += 0.05 * rng.normal(size=n_timepoints)
            roi_map[f"ROI_{roi_index:03d}"] = signal
        participant_data[f"sub-{participant_index:02d}"] = roi_map
    return participant_data
