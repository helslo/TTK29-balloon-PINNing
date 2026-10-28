"""Evaluation utilities for sparse-sensor placement experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    from data import participant_holdout_split, temporal_holdout_split
    from methods import lasso_placement, qr_pivot_placement, random_placement
else:
    from .data import participant_holdout_split, temporal_holdout_split
    from .methods import lasso_placement, qr_pivot_placement, random_placement


def normalized_rmse(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """Return the relative Frobenius norm error used in the README."""
    denom = np.linalg.norm(X_true, ord="fro")
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(X_true - X_pred, ord="fro") / denom)


def _method_for_name(method_name: str):
    methods = {
        "random": random_placement,
        "qr": qr_pivot_placement,
        "lasso_random": lambda X, k: lasso_placement(X, k, placement="random"),
        "lasso_qr": lambda X, k: lasso_placement(X, k, placement="qr"),
    }
    return methods[method_name]


def evaluate_temporal_holdout(
    X: np.ndarray,
    k_values: Iterable[int],
    n_random_draws: int = 25,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Evaluate sensor-placement performance on a temporal train/test split."""
    train_idx, test_idx = temporal_holdout_split(X)
    X_train = X[:, train_idx]
    X_test = X[:, test_idx]
    records: list[dict[str, float | int | str]] = []
    for k in k_values:
        for method_name in ("random", "qr", "lasso_random", "lasso_qr"):
            if method_name == "random":
                errors: list[float] = []
                for draw in range(n_random_draws):
                    rng = np.random.default_rng(random_seed + draw + int(k))
                    sensors, reconstruct = random_placement(X_train, k, rng=rng)
                    X_hat = reconstruct(X_test)
                    errors.append(normalized_rmse(X_test, X_hat))
                records.append(
                    {
                        "method": method_name,
                        "k": int(k),
                        "holdout": "temporal",
                        "error_mean": float(np.mean(errors)),
                        "error_std": float(np.std(errors, ddof=0)),
                    }
                )
            else:
                sensors, reconstruct = _method_for_name(method_name)(X_train, k)
                X_hat = reconstruct(X_test)
                records.append(
                    {
                        "method": method_name,
                        "k": int(k),
                        "holdout": "temporal",
                        "error_mean": float(normalized_rmse(X_test, X_hat)),
                        "error_std": 0.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def evaluate_participant_holdout(
    participant_matrices: Mapping[str, np.ndarray],
    k_values: Iterable[int],
    n_random_draws: int = 25,
    random_seed: int = 0,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Evaluate participant-level generalization using cross-validation over subjects."""
    records: list[dict[str, float | int | str]] = []
    for train_ids, test_ids in participant_holdout_split(participant_matrices, n_folds=n_folds):
        X_train = np.concatenate([participant_matrices[pid] for pid in train_ids], axis=1)
        for k in k_values:
            for method_name in ("random", "qr", "lasso_random", "lasso_qr"):
                if method_name == "random":
                    errors: list[float] = []
                    for draw in range(n_random_draws):
                        rng = np.random.default_rng(random_seed + draw + int(k) + len(train_ids))
                        sensors, reconstruct = random_placement(X_train, k, rng=rng)
                        for pid in test_ids:
                            X_test = participant_matrices[pid]
                            X_hat = reconstruct(X_test)
                            errors.append(normalized_rmse(X_test, X_hat))
                    records.append(
                        {
                            "method": method_name,
                            "k": int(k),
                            "holdout": "participant",
                            "fold": len(records),
                            "error_mean": float(np.mean(errors)),
                            "error_std": float(np.std(errors, ddof=0)),
                        }
                    )
                else:
                    sensors, reconstruct = _method_for_name(method_name)(X_train, k)
                    fold_errors: list[float] = []
                    for pid in test_ids:
                        X_test = participant_matrices[pid]
                        X_hat = reconstruct(X_test)
                        fold_errors.append(normalized_rmse(X_test, X_hat))
                    records.append(
                        {
                            "method": method_name,
                            "k": int(k),
                            "holdout": "participant",
                            "fold": len(records),
                            "error_mean": float(np.mean(fold_errors)),
                            "error_std": float(np.std(fold_errors, ddof=0)),
                        }
                    )
    return pd.DataFrame.from_records(records)


def evaluate_all(
    participant_matrices: Mapping[str, np.ndarray],
    k_values: Sequence[int],
    n_random_draws: int = 25,
    random_seed: int = 0,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Evaluate both temporal and participant hold-out protocols for all requested k."""
    all_frames = []
    for pid, matrix in participant_matrices.items():
        all_frames.append(evaluate_temporal_holdout(matrix, k_values, n_random_draws=n_random_draws, random_seed=random_seed))
        break
    temporal_df = all_frames[0] if all_frames else pd.DataFrame()
    participant_df = evaluate_participant_holdout(participant_matrices, k_values, n_random_draws=n_random_draws, random_seed=random_seed, n_folds=n_folds)
    return pd.concat([temporal_df, participant_df], ignore_index=True)
