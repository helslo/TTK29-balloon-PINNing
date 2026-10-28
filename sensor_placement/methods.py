"""Sensor-placement strategies and reconstruction methods.

Each public function returns a tuple ``(sensor_indices, reconstruction_fn)``. The
``reconstruction_fn`` accepts the full ROI matrix ``X`` and returns the estimated
full matrix from the subset of observed rows.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy import linalg
from scipy.fft import dct
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


def _low_rank_basis(X: np.ndarray, r: int | None = None) -> tuple[np.ndarray, int]:
    """Project ROI data onto the leading spatial modes from its SVD.

    For a ROI-by-time matrix ``X``, the left singular vectors span the dominant
    spatial patterns. Choosing the leading ``r`` modes is equivalent to keeping the
    subspace that explains the largest fraction of spatial variance in the data.
    """
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    if r is None:
        if s.size == 0:
            r = 1
        else:
            explained = np.cumsum(s**2) / np.sum(s**2)
            r = int(np.searchsorted(explained, 0.9, side="left")) + 1
    r = min(max(1, r), min(X.shape[0], X.shape[1]))
    return u[:, :r], r


def _basis_from_dct(n_rois: int, n_features: int | None = None) -> np.ndarray:
    """Construct a DCT dictionary for sparse spatial recovery.

    The discrete cosine transform is a convenient basis in which smooth spatial
    variations are sparse, which matches the assumption behind compressed sensing.
    """
    n_features = n_rois if n_features is None else min(max(1, n_features), n_rois)
    identity = np.eye(n_rois)
    basis = dct(identity, type=2, norm="ortho", axis=0)
    return basis[:, :n_features]


def _stable_reconstruction(basis: np.ndarray, sensors: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Estimate the latent coefficients from observed rows with a small ridge.

    The least-squares problem is solved in the reduced basis, and the ridge term
    stabilizes the inversion when the selected rows are near-collinear or the basis
    is poorly conditioned.
    """
    system = basis[sensors, :]
    regularization = 1e-8 * np.eye(system.shape[1])
    coeffs = np.linalg.pinv(system.T @ system + regularization) @ (system.T @ observed)
    coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return basis @ coeffs


def random_placement(
    X_train: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Select ``k`` sensors uniformly at random and reconstruct with a low-rank basis.

    Random selection is used as a baseline: after choosing a subset of ROIs, we fit
    an SVD basis to the training data and solve a regularized least-squares problem
    to estimate the latent coefficients from the observed rows.
    """
    n_rois = X_train.shape[0]
    k = min(max(1, int(k)), n_rois)
    rng = np.random.default_rng() if rng is None else rng
    sensors = rng.choice(n_rois, size=k, replace=False)
    basis, _ = _low_rank_basis(X_train, r=min(k, n_rois))

    def reconstruction(X: np.ndarray) -> np.ndarray:
        observed = np.asarray(X, dtype=float)
        if observed.shape[0] != n_rois:
            raise ValueError(f"Expected {n_rois} rows; got {observed.shape[0]}")
        return _stable_reconstruction(basis, sensors, observed[sensors, :])

    return sensors, reconstruction


def qr_pivot_placement(
    X_train: np.ndarray,
    k: int,
    r: int | None = None,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Use SVD+QR pivoting to choose informative sensor rows.

    The QR factorization with column pivoting is applied to the transpose of the
    leading spatial modes. The pivots identify the rows whose columns are most
    linearly independent, which tends to maximize information content for a given
    sensor budget.
    """
    n_rois = X_train.shape[0]
    k = min(max(1, int(k)), n_rois)
    basis, selected_rank = _low_rank_basis(X_train, r=r)
    if r is None:
        basis = basis[:, : min(selected_rank, k)]
    else:
        basis = basis[:, : min(selected_rank, r)]
    _, _, pivots = linalg.qr(basis.T, mode="economic", pivoting=True)
    sensors = np.asarray(pivots[:k], dtype=int)

    def reconstruction(X: np.ndarray) -> np.ndarray:
        observed = np.asarray(X, dtype=float)
        if observed.shape[0] != n_rois:
            raise ValueError(f"Expected {n_rois} rows; got {observed.shape[0]}")
        return _stable_reconstruction(basis, sensors, observed[sensors, :])

    return sensors, reconstruction


def lasso_placement(
    X_train: np.ndarray,
    k: int,
    basis: str = "dct",
    alpha: float = 1e-3,
    placement: str = "qr",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Recover the signal from a sparse basis using a subset of measurement rows.

    This separates the placement problem from the reconstruction problem: the
    measurement rows are selected by either random or QR-pivoting, while LASSO is
    used to recover sparse coefficients in a DCT basis before mapping back to the
    original ROI space.
    """
    n_rois = X_train.shape[0]
    k = min(max(1, int(k)), n_rois)
    rng = np.random.default_rng(0) if rng is None else rng
    if placement == "random":
        sensors = rng.choice(n_rois, size=k, replace=False)
    elif placement == "qr":
        sensors, _ = qr_pivot_placement(X_train, k)
    else:
        raise ValueError("placement must be 'random' or 'qr'")

    n_features = min(n_rois, max(8, 2 * k))
    if basis == "dct":
        spatial_basis = _basis_from_dct(n_rois, n_features)
    else:
        spatial_basis = _low_rank_basis(X_train, r=min(n_features, n_rois))[0]

    measurement = np.zeros((k, n_rois), dtype=float)
    measurement[np.arange(k), sensors] = 1.0
    design = measurement @ spatial_basis

    def reconstruction(X: np.ndarray) -> np.ndarray:
        observed = np.asarray(X, dtype=float)
        if observed.shape[0] != n_rois:
            raise ValueError(f"Expected {n_rois} rows; got {observed.shape[0]}")
        y = observed[sensors, :]
        coefficients = np.empty((design.shape[1], y.shape[1]), dtype=float)
        for time_index in range(y.shape[1]):
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(design, y[:, time_index])
            coefficients[:, time_index] = np.nan_to_num(model.coef_, nan=0.0, posinf=0.0, neginf=0.0)
        coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return spatial_basis @ coefficients

    return np.asarray(sensors, dtype=int), reconstruction
