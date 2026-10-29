"""Random, QR-pivoted, and spatial-DCT sparse sensor reconstruction methods."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import linalg
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import svds


Reconstruction = Callable[[np.ndarray], np.ndarray]
MAX_SVD_RANK = 64


def _low_rank_basis(
    X_train: np.ndarray,
    r: int | None = None,
    max_rank: int = MAX_SVD_RANK,
    explained_variance: float = 0.9,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate leading spatial modes with truncated Lanczos SVD without forming full U."""
    if X_train.ndim != 2 or min(X_train.shape) < 1:
        raise ValueError(f"Training data must be a non-empty 2D matrix, got {X_train.shape}")
    if not np.isfinite(X_train).all():
        raise ValueError("Training data must contain only finite values")
    n_voxels, n_timepoints = X_train.shape
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    rank_limit = min(n_voxels - 1, n_timepoints - 1, max_rank)
    if rank_limit < 1:
        raise ValueError("SVD training data needs at least two time samples")
    if r is not None:
        if r < 1:
            raise ValueError("r must be positive")
        if r > max_rank:
            raise ValueError(f"r cannot exceed the configured SVD rank limit ({max_rank})")
        rank_limit = min(rank_limit, int(r))
    u, singular_values, _ = svds(
        X_train,
        k=rank_limit,
        which="LM",
        random_state=random_state,
    )
    order = np.argsort(singular_values)[::-1]
    u = u[:, order]
    singular_values = singular_values[order]
    if r is None:
        total_energy = sum(
            float(np.sum(np.square(X_train[start:start + 4096], dtype=np.float64)))
            for start in range(0, n_voxels, 4096)
        )
        captured = np.cumsum(singular_values.astype(np.float64) ** 2)
        if total_energy > 0 and captured.size:
            rank = int(np.searchsorted(captured / total_energy, explained_variance) + 1)
            rank = min(max(rank, 1), rank_limit)
        else:
            rank = 1
    else:
        rank = rank_limit
    return u[:, :rank].astype(np.float32, copy=False), singular_values[:rank], rank


def _reconstruct_from_basis(
    basis: np.ndarray, sensors: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """Solve a small ridge-regularized least-squares problem in the fitted SVD modes."""
    system = basis[sensors, :].astype(np.float64, copy=False)
    y = np.asarray(observed[sensors, :], dtype=np.float64)
    if not np.isfinite(system).all() or not np.isfinite(y).all():
        raise ValueError("Sensor measurements and basis must contain only finite values")
    ridge = max(float(np.sum(system**2)) / max(system.shape[1], 1), 1.0) * 1e-8
    augmented_system = np.vstack(
        [system, np.sqrt(ridge) * np.eye(system.shape[1])]
    )
    augmented_y = np.vstack([y, np.zeros((system.shape[1], y.shape[1]))])
    coefficients = np.linalg.lstsq(augmented_system, augmented_y, rcond=None)[0]
    reconstructed = np.einsum(
        "ij,jt->it",
        np.asarray(basis, dtype=np.float64),
        coefficients,
        optimize=False,
    )
    if not np.isfinite(reconstructed).all():
        raise FloatingPointError("Low-rank reconstruction produced non-finite values")
    return reconstructed.astype(np.float32, copy=False)


def random_placement(
    X_train: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
    r: int | None = None,
    basis: np.ndarray | None = None,
) -> tuple[np.ndarray, Reconstruction]:
    """Sample voxel rows uniformly, then reconstruct their signals in a learned low-rank basis."""
    n_voxels = X_train.shape[0]
    if not 1 <= int(k) <= n_voxels:
        raise ValueError(f"k must be between 1 and {n_voxels}")
    if r is not None and r < 1:
        raise ValueError("r must be positive")
    generator = np.random.default_rng() if rng is None else rng
    sensors = np.sort(generator.choice(n_voxels, size=int(k), replace=False))
    requested_rank = int(k) if r is None else int(r)
    selected_rank = min(requested_rank, int(k), MAX_SVD_RANK)
    if basis is None:
        basis, _, _ = _low_rank_basis(X_train, r=selected_rank)
    elif basis.ndim != 2 or basis.shape[0] != n_voxels or basis.shape[1] < selected_rank:
        raise ValueError(f"basis must contain at least {selected_rank} modes for each voxel")
    else:
        basis = basis[:, :selected_rank]

    def reconstruction(observed: np.ndarray) -> np.ndarray:
        if observed.ndim != 2 or observed.shape[0] != n_voxels:
            raise ValueError(f"Expected a ({n_voxels}, n_time) matrix, got {observed.shape}")
        return _reconstruct_from_basis(basis, sensors, observed)

    return sensors, reconstruction


def qr_pivot_placement(
    X_train: np.ndarray,
    k: int,
    r: int | None = None,
    max_rank: int = MAX_SVD_RANK,
) -> tuple[np.ndarray, Reconstruction]:
    """Select informative voxel rows by pivoting QR over the leading spatial modes."""
    n_voxels = X_train.shape[0]
    if not 1 <= int(k) <= n_voxels:
        raise ValueError(f"k must be between 1 and {n_voxels}")
    basis, _, selected_rank = _low_rank_basis(X_train, r=r, max_rank=max_rank)
    basis = basis[:, :min(selected_rank, int(k))]
    _, _, pivots = linalg.qr(basis.T, mode="economic", pivoting=True)
    sensors = np.asarray(pivots[:int(k)], dtype=int)

    def reconstruction(observed: np.ndarray) -> np.ndarray:
        if observed.ndim != 2 or observed.shape[0] != n_voxels:
            raise ValueError(f"Expected a ({n_voxels}, n_time) matrix, got {observed.shape}")
        return _reconstruct_from_basis(basis, sensors, observed)

    return sensors, reconstruction


def _fista_dct_reconstruction(
    observed: np.ndarray,
    sensors: np.ndarray,
    spatial_shape: tuple[int, int, int],
    brain_voxel_indices: np.ndarray,
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    """Solve the masked voxel-sampling L1 objective independently at each time point."""
    if not np.isfinite(observed).all():
        raise ValueError("Sensor measurements must contain only finite values")
    sampled_voxels = brain_voxel_indices[sensors]
    estimates = np.empty_like(observed, dtype=np.float32)
    for time_index in range(observed.shape[1]):
        measurements = np.asarray(observed[sensors, time_index], dtype=np.float64)
        coefficients = np.zeros(spatial_shape, dtype=np.float64)
        accelerated = coefficients.copy()
        momentum = 1.0
        for _ in range(max_iter):
            volume = idctn(accelerated, type=2, norm="ortho")
            residual = np.zeros(spatial_shape, dtype=np.float64)
            residual.ravel()[sampled_voxels] = volume.ravel()[sampled_voxels] - measurements
            gradient = dctn(residual, type=2, norm="ortho")
            next_coefficients = np.sign(accelerated - gradient) * np.maximum(
                np.abs(accelerated - gradient) - alpha, 0.0
            )
            next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
            accelerated = next_coefficients + (
                (momentum - 1.0) / next_momentum
            ) * (next_coefficients - coefficients)
            coefficients = next_coefficients
            momentum = next_momentum
        volume = idctn(coefficients, type=2, norm="ortho")
        estimates[:, time_index] = volume.ravel()[brain_voxel_indices]
    return estimates


def l1_dct_placement(
    X_train: np.ndarray,
    k: int,
    spatial_shape: tuple[int, int, int],
    brain_voxel_indices: np.ndarray,
    placement: str = "qr",
    rng: np.random.Generator | None = None,
    alpha: float | None = None,
    max_iter: int = 40,
) -> tuple[np.ndarray, Reconstruction]:
    """Recover each volume with FISTA and an orthonormal 3D DCT sparsity penalty."""
    if X_train.ndim != 2 or not np.isfinite(X_train).all():
        raise ValueError("Training data must be a finite two-dimensional matrix")
    if len(spatial_shape) != 3 or any(size < 1 for size in spatial_shape):
        raise ValueError("spatial_shape must contain three positive dimensions")
    n_voxels = X_train.shape[0]
    if int(np.prod(spatial_shape)) <= int(np.max(brain_voxel_indices, initial=-1)):
        raise ValueError("Brain voxel indices do not fit inside spatial_shape")
    if brain_voxel_indices.shape != (n_voxels,):
        raise ValueError("brain_voxel_indices must map every matrix row to a volume voxel")
    if not 1 <= int(k) <= n_voxels:
        raise ValueError(f"k must be between 1 and {n_voxels}")
    if (alpha is not None and alpha <= 0) or max_iter <= 0:
        raise ValueError("alpha and max_iter must be positive")
    generator = np.random.default_rng() if rng is None else rng

    def select_sensors(training_data: np.ndarray) -> np.ndarray:
        if placement == "random":
            return np.sort(generator.choice(n_voxels, size=int(k), replace=False))
        if placement == "qr":
            return qr_pivot_placement(training_data, int(k))[0]
        raise ValueError("placement must be 'random' or 'qr'")

    if alpha is None:
        split_index = max(2, int(round(0.8 * X_train.shape[1])))
        if split_index < X_train.shape[1]:
            validation_sensors = select_sensors(X_train[:, :split_index])
            validation = X_train[:, split_index:]
            validation_indices = np.linspace(
                0, validation.shape[1] - 1,
                num=min(4, validation.shape[1]),
                dtype=int,
            )
            validation = validation[:, validation_indices]
            candidates = (1e-3, 1e-2, 1e-1)
            errors = []
            for candidate in candidates:
                estimate = _fista_dct_reconstruction(
                    validation,
                    validation_sensors,
                    spatial_shape,
                    brain_voxel_indices,
                    candidate,
                    min(max_iter, 15),
                )
                errors.append(float(np.linalg.norm(validation - estimate)))
            alpha = candidates[int(np.argmin(errors))]
        else:
            alpha = 1e-2
    selected_alpha = float(alpha)
    sensors = select_sensors(X_train)

    def reconstruction(observed: np.ndarray) -> np.ndarray:
        if observed.ndim != 2 or observed.shape[0] != n_voxels:
            raise ValueError(f"Expected a ({n_voxels}, n_time) matrix, got {observed.shape}")
        return _fista_dct_reconstruction(
            observed, sensors, spatial_shape, brain_voxel_indices, selected_alpha, max_iter
        )

    setattr(reconstruction, "alpha", selected_alpha)
    return np.asarray(sensors, dtype=int), reconstruction
