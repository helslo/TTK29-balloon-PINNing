"""Canonical-HRF general linear model baseline for task-fMRI time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nilearn.glm.first_level.hemodynamic_models import spm_hrf


@dataclass
class GLMFit:
    """OLS fit and predictions from a convolved task design."""

    coefficients: np.ndarray
    predictions: np.ndarray
    design: np.ndarray


def convolve_with_canonical_hrf(
    task_input: np.ndarray,
    repetition_time: float,
    oversampling: int = 16,
    hrf_duration: float = 32.0,
) -> np.ndarray:
    """Convolve a TR-averaged task input with Nilearn's canonical SPM HRF."""
    task_input = np.asarray(task_input, dtype=float)
    if task_input.ndim != 1 or task_input.size < 2 or not np.isfinite(task_input).all():
        raise ValueError("task_input must be a finite one-dimensional signal")
    if repetition_time <= 0 or not np.isfinite(repetition_time):
        raise ValueError("repetition_time must be finite and positive")
    if oversampling < 1:
        raise ValueError("oversampling must be positive")
    if hrf_duration <= 0 or not np.isfinite(hrf_duration):
        raise ValueError("hrf_duration must be finite and positive")

    hrf = spm_hrf(
        repetition_time,
        oversampling=oversampling,
        time_length=hrf_duration,
    )
    high_resolution_input = np.repeat(task_input, oversampling)
    response = np.convolve(high_resolution_input, hrf, mode="full")
    response = response[: high_resolution_input.size]
    return response[::oversampling]


def fit_canonical_hrf_glm(
    times: np.ndarray,
    task_regressors: np.ndarray,
    observed_bold: np.ndarray,
    fit_end: int | None = None,
) -> GLMFit:
    """Fit task scales plus an intercept and linear drift by ordinary least squares.

    ``fit_end`` allows fitting only an initial chronological block while
    returning predictions for the full input timeline.
    """
    times = np.asarray(times, dtype=float)
    task_regressors = np.asarray(task_regressors, dtype=float)
    observed_bold = np.asarray(observed_bold, dtype=float)
    if times.ndim != 1 or observed_bold.shape != times.shape:
        raise ValueError("times and observed_bold must be matching one-dimensional arrays")
    if task_regressors.ndim == 1:
        task_regressors = task_regressors[:, None]
    if task_regressors.ndim != 2 or task_regressors.shape[0] != times.size:
        raise ValueError("task_regressors must have one row per time point")
    if times.size < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("times must contain at least three strictly increasing samples")
    if not all(np.isfinite(values).all() for values in (times, task_regressors, observed_bold)):
        raise ValueError("GLM inputs must contain only finite values")
    if task_regressors.shape[1] < 1:
        raise ValueError("At least one task regressor is required")
    if fit_end is None:
        fit_end = times.size
    if not 3 <= fit_end <= times.size:
        raise ValueError("fit_end must leave at least three fitting samples")

    time_scale = max(float(times[-1] - times[0]), 1.0)
    linear_drift = (times - np.mean(times[:fit_end])) / time_scale
    design = np.column_stack(
        (task_regressors, np.ones(times.size, dtype=float), linear_drift)
    )
    coefficients = np.linalg.lstsq(design[:fit_end], observed_bold[:fit_end], rcond=None)[0]
    predictions = design @ coefficients
    return GLMFit(coefficients=coefficients, predictions=predictions, design=design)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return MSE, R², and variance-normalized RMSE for one evaluation segment."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.ndim != 1 or predicted.shape != observed.shape or observed.size == 0:
        raise ValueError("observed and predicted must be matching non-empty vectors")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("observed and predicted must contain only finite values")
    mse = float(np.mean((observed - predicted) ** 2))
    variance = float(np.mean((observed - np.mean(observed)) ** 2))
    if variance == 0.0:
        raise ValueError("R² and normalized RMSE are undefined for a constant target")
    return {
        "mse": mse,
        "r2": float(1.0 - mse / variance),
        "nrmse": float(np.sqrt(mse / variance)),
    }
