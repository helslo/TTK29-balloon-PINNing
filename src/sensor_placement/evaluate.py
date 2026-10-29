"""Temporal and participant hold-out evaluation for voxel sensor placement."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import SensorRun, participant_holdout_split, temporal_holdout_split
from .methods import (
    MAX_SVD_RANK,
    _low_rank_basis,
    l1_dct_placement,
    qr_pivot_placement,
    random_placement,
)


METHODS = ("random", "qr", "l1_random", "l1_qr")


def normalized_rmse(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """Compute relative Frobenius error over held-out voxel-time measurements."""
    if not np.isfinite(X_true).all() or not np.isfinite(X_pred).all():
        raise ValueError("Held-out truth and reconstructions must be finite")
    denominator = float(np.linalg.norm(X_true))
    if denominator == 0.0:
        raise ValueError("Normalized RMSE is undefined for an all-zero held-out matrix")
    return float(np.linalg.norm(X_true - X_pred) / denominator)


def mean_voxel_correlation(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """Average per-voxel Pearson correlation over the held-out time dimension."""
    true_centered = X_true - np.mean(X_true, axis=1, keepdims=True)
    pred_centered = X_pred - np.mean(X_pred, axis=1, keepdims=True)
    numerator = np.sum(true_centered * pred_centered, axis=1)
    denominator = np.sqrt(
        np.sum(true_centered**2, axis=1) * np.sum(pred_centered**2, axis=1)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    finite = correlations[np.isfinite(correlations)]
    if not finite.size:
        raise ValueError("Pearson correlation is undefined for constant voxel signals")
    return float(np.mean(finite))


def log_k_grid(n_voxels: int, fraction: float = 0.05, points: int = 8) -> list[int]:
    """Create unique integer sensor counts spaced logarithmically up to a voxel fraction."""
    if n_voxels < 1:
        raise ValueError("n_voxels must be positive")
    if not 0.0 < fraction <= 1.0 or points < 1:
        raise ValueError("fraction must be in (0, 1] and points must be positive")
    maximum = max(1, min(n_voxels, int(np.ceil(n_voxels * fraction))))
    return sorted(set(np.geomspace(1, maximum, num=points).round().astype(int).tolist()))


def _training_matrix(runs: Sequence[SensorRun], max_samples: int = 1000) -> np.ndarray:
    """Pool balanced time samples into a bounded-memory float32 SVD training matrix."""
    if not runs:
        raise ValueError("At least one training run is required")
    n_voxels = runs[0].X.shape[0]
    if any(run.X.shape[0] != n_voxels for run in runs):
        raise ValueError("All runs must use the same common-space voxel mask")
    quota = max(1, max_samples // len(runs))
    blocks = []
    for run in runs:
        sample_count = min(run.X.shape[1], quota)
        indices = np.linspace(0, run.X.shape[1] - 1, num=sample_count, dtype=int)
        blocks.append(np.asarray(run.X[:, indices], dtype=np.float32))
    pooled = np.concatenate(blocks, axis=1)
    if pooled.shape[1] > max_samples:
        indices = np.linspace(0, pooled.shape[1] - 1, num=max_samples, dtype=int)
        pooled = pooled[:, indices]
    return pooled


def _make_method(
    method: str,
    X_train: np.ndarray,
    k: int,
    rng: np.random.Generator,
    spatial_shape: tuple[int, int, int],
    brain_voxel_indices: np.ndarray,
    rank: int | None,
    l1_alpha: float | None,
    l1_iterations: int,
    basis: np.ndarray | None = None,
):
    """Construct one placement/reconstruction pair using training data only."""
    if method == "random":
        return random_placement(X_train, k, rng=rng, r=rank, basis=basis)
    if method == "qr":
        return qr_pivot_placement(X_train, k, r=rank)
    if method == "l1_random":
        return l1_dct_placement(
            X_train, k, spatial_shape, brain_voxel_indices,
            placement="random", rng=rng, alpha=l1_alpha, max_iter=l1_iterations,
        )
    if method == "l1_qr":
        return l1_dct_placement(
            X_train, k, spatial_shape, brain_voxel_indices,
            placement="qr", rng=rng, alpha=l1_alpha, max_iter=l1_iterations,
        )
    raise ValueError(f"Unknown sensor-placement method: {method}")


def _evaluate_split(
    X_train: np.ndarray,
    test_runs: Sequence[SensorRun],
    k_values: Iterable[int],
    holdout: str,
    fold: int,
    n_random_draws: int,
    random_seed: int,
    spatial_shape: tuple[int, int, int],
    brain_voxel_indices: np.ndarray,
    include_l1: bool,
    rank: int | None,
    l1_alpha: float | None,
    l1_iterations: int,
    l1_k_values: set[int],
    l1_test_timepoints: int,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    """Fit on one training split, evaluate held-out runs, and record chosen sensors."""
    records: list[dict[str, object]] = []
    selected_qr: list[np.ndarray] = []
    methods = METHODS if include_l1 else METHODS[:2]
    for k in k_values:
        for method in methods:
            if method.startswith("l1_") and int(k) not in l1_k_values:
                continue
            random_basis = None
            if method == "random":
                basis_rank = min(int(k), rank or int(k), MAX_SVD_RANK)
                random_basis, _, _ = _low_rank_basis(X_train, r=basis_rank)
            draws = min(n_random_draws, 3) if method == "l1_random" else (
                n_random_draws if method == "random" else 1
            )
            for draw in range(draws):
                rng = np.random.default_rng(random_seed + fold * 100_003 + int(k) * 997 + draw)
                sensors, reconstruct = _make_method(
                    method, X_train, int(k), rng, spatial_shape,
                    brain_voxel_indices, rank, l1_alpha, l1_iterations, random_basis,
                )
                if method == "qr":
                    selected_qr.append(sensors)
                for test_run in test_runs:
                    X_test = np.asarray(test_run.X, dtype=np.float32)
                    if method.startswith("l1_") and X_test.shape[1] > l1_test_timepoints:
                        time_indices = np.linspace(
                            0, X_test.shape[1] - 1, num=l1_test_timepoints, dtype=int
                        )
                        X_test = X_test[:, time_indices]
                    X_pred = reconstruct(X_test)
                    records.append(
                        {
                            "method": method,
                            "k": int(k),
                            "holdout": holdout,
                            "fold": fold,
                            "participant_id": test_run.participant_id,
                            "run_id": test_run.run_id,
                            "draw": draw,
                            "normalized_rmse": normalized_rmse(X_test, X_pred),
                            "mean_voxel_correlation": mean_voxel_correlation(X_test, X_pred),
                            "n_test_timepoints": int(X_test.shape[1]),
                            "l1_alpha": getattr(reconstruct, "alpha", np.nan),
                        }
                    )
    return records, selected_qr


def evaluate_all(
    runs: Sequence[SensorRun],
    k_values: Sequence[int],
    n_random_draws: int = 10,
    random_seed: int = 0,
    n_folds: int = 5,
    include_l1: bool = False,
    rank: int | None = None,
    l1_alpha: float | None = None,
    l1_iterations: int = 40,
    max_basis_samples: int = 1000,
    l1_k_values: Sequence[int] | None = None,
    l1_test_timepoints: int = 40,
) -> tuple[pd.DataFrame, list[np.ndarray], np.ndarray]:
    """Evaluate temporal and participant hold-outs without fitting on test data."""
    if n_random_draws < 1:
        raise ValueError("n_random_draws must be positive")
    if l1_test_timepoints < 1:
        raise ValueError("l1_test_timepoints must be positive")
    if not runs:
        raise ValueError("At least one cleaned run is required")
    first_masker = runs[0].masker
    mask_img = first_masker.mask_img_
    mask = np.asarray(mask_img.dataobj) > 0
    brain_voxel_indices = np.flatnonzero(mask.ravel())
    spatial_shape = tuple(int(size) for size in mask.shape)
    if len(brain_voxel_indices) != runs[0].X.shape[0]:
        raise ValueError("The mask and cleaned data have different voxel counts")
    selected_l1_k = set(int(value) for value in (l1_k_values or k_values))

    all_records: list[dict[str, object]] = []
    qr_selections: list[np.ndarray] = []
    fold = 0
    for run in runs:
        train_idx, test_idx = temporal_holdout_split(run.X)
        X_train = np.asarray(run.X[:, train_idx], dtype=np.float32)
        test_run = SensorRun(
            run.participant_id,
            run.run_id,
            run.X[:, test_idx],
            run.tr,
            run.masker,
            run.cache_path,
        )
        records, selections = _evaluate_split(
            X_train,
            [test_run],
            k_values,
            "temporal",
            fold,
            n_random_draws,
            random_seed,
            spatial_shape,
            brain_voxel_indices,
            include_l1,
            rank,
            l1_alpha,
            l1_iterations,
            selected_l1_k,
            l1_test_timepoints,
        )
        all_records.extend(records)
        fold += 1

    participants: dict[str, list[SensorRun]] = defaultdict(list)
    for run in runs:
        participants[run.participant_id].append(run)
    folds = participant_holdout_split(participants, n_folds=n_folds)
    for fold_index, (train_ids, test_ids) in enumerate(folds):
        train_runs = [run for pid in train_ids for run in participants[pid]]
        test_runs = [run for pid in test_ids for run in participants[pid]]
        X_train = _training_matrix(train_runs, max_samples=max_basis_samples)
        records, selections = _evaluate_split(
            X_train,
            test_runs,
            k_values,
            "participant",
            fold_index,
            n_random_draws,
            random_seed,
            spatial_shape,
            brain_voxel_indices,
            include_l1,
            rank,
            l1_alpha,
            l1_iterations,
            selected_l1_k,
            l1_test_timepoints,
        )
        all_records.extend(records)
        qr_selections.extend(selections)
    return pd.DataFrame.from_records(all_records), qr_selections, brain_voxel_indices
