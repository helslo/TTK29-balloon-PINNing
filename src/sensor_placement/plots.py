"""Brain-space and performance plots for voxel sensor-placement experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.plotting import plot_markers, plot_stat_map


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    """Write a plot in both report-ready PNG and PDF formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=180)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def plot_error_curves(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot held-out reconstruction error against sensor count on logarithmic axes."""
    out_path = Path(output_dir)
    labels = {
        "random": "Random + least squares",
        "qr": "QR pivoting + least squares",
        "l1_random": "Random + 3D DCT L1",
        "l1_qr": "QR pivoting + 3D DCT L1",
    }
    for holdout in ("temporal", "participant"):
        subset = results[results["holdout"] == holdout]
        if subset.empty:
            continue
        summary = (
            subset.groupby(["method", "k"])["normalized_rmse"]
            .agg(["mean", "std"])
            .reset_index()
        )
        figure, axis = plt.subplots(figsize=(8, 5.5))
        for method, group in summary.groupby("method"):
            group = group.sort_values("k")
            label = labels.get(method, method)
            axis.plot(group["k"], group["mean"], marker="o", label=label)
            if method in {"random", "l1_random"}:
                spread = group["std"].fillna(0.0).to_numpy()
                axis.fill_between(
                    group["k"].to_numpy(),
                    group["mean"].to_numpy() - spread,
                    group["mean"].to_numpy() + spread,
                    alpha=0.18,
                )
        axis.set_xscale("log")
        axis.set_xlabel("Number of measured voxels (k)")
        axis.set_ylabel("Normalized RMSE")
        axis.set_title(f"Voxel reconstruction error: {holdout} hold-out")
        axis.grid(alpha=0.25, which="both")
        axis.legend()
        figure.tight_layout()
        _save_figure(figure, out_path, f"error_vs_k_{holdout}")
    return out_path


def plot_singular_value_spectrum(
    singular_values: np.ndarray,
    total_energy: float,
    output_dir: str | Path,
    r_chosen: int,
) -> Path:
    """Show cumulative energy in the randomized SVD modes used for sensor design."""
    out_path = Path(output_dir)
    energies = np.asarray(singular_values, dtype=np.float64) ** 2
    if total_energy <= 0:
        raise ValueError("Cannot plot a singular spectrum with zero total energy")
    cumulative = np.cumsum(energies) / total_energy
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(np.arange(1, cumulative.size + 1), cumulative, marker=".", linewidth=1.5)
    axis.axvline(r_chosen, color="tab:red", linestyle="--", label=f"chosen r={r_chosen}")
    axis.set_xlabel("Spatial mode")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_title("Training-data singular-value spectrum")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, out_path, "singular_value_spectrum")
    return out_path


def _voxel_coordinates(
    voxel_indices: np.ndarray, spatial_shape: tuple[int, int, int], affine: np.ndarray
) -> np.ndarray:
    """Convert flattened mask-grid indices into their MNI millimetre coordinates."""
    ijk = np.column_stack(np.unravel_index(voxel_indices, spatial_shape))
    return nib.affines.apply_affine(affine, ijk)


def plot_selected_sensors_on_brain(
    selected_sensors: np.ndarray,
    brain_voxel_indices: np.ndarray,
    masker,
    output_dir: str | Path,
    k: int,
) -> Path:
    """Display actual QR-selected voxel locations at their common-grid MNI coordinates."""
    out_path = Path(output_dir)
    mask_img = masker.mask_img_
    coords = _voxel_coordinates(
        brain_voxel_indices[selected_sensors],
        tuple(int(value) for value in mask_img.shape),
        mask_img.affine,
    )
    figure = plot_markers(
        np.ones(len(coords)),
        coords,
        node_size=20,
        node_cmap="autumn",
        display_mode="lyrz",
        title=f"QR-selected sensors (k={k})",
    )
    figure.savefig(out_path / f"selected_sensors_k{k}.png", dpi=180)
    figure.savefig(out_path / f"selected_sensors_k{k}.pdf")
    figure.close()
    return out_path


def plot_reconstruction_examples(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    selected_sensors: np.ndarray,
    tr: float,
    output_dir: str | Path,
    k: int,
) -> Path:
    """Compare held-out BOLD traces from one measured and several unmeasured voxels."""
    out_path = Path(output_dir)
    measured = [int(selected_sensors[0])] if selected_sensors.size else []
    unmeasured = [index for index in range(X_true.shape[0]) if index not in set(measured)]
    rows = (measured + unmeasured[:3])[:4]
    time_seconds = np.arange(X_true.shape[1]) * tr
    figure, axes = plt.subplots(len(rows), 1, figsize=(10, 2.2 * len(rows)), squeeze=False)
    for axis, voxel in zip(axes.ravel(), rows):
        axis.plot(time_seconds, X_true[voxel], label="measured truth", linewidth=1.5)
        axis.plot(time_seconds, X_pred[voxel], label="reconstruction", linestyle="--")
        axis.set_title(f"Voxel row {voxel}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("z-score")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.tight_layout()
    _save_figure(figure, out_path, f"reconstruction_example_k{k}")
    return out_path


def plot_per_voxel_error_map(
    per_voxel_error: np.ndarray,
    masker,
    output_dir: str | Path,
    k: int,
) -> Path:
    """Map held-out mean squared voxel error back to the common MNI brain grid."""
    out_path = Path(output_dir)
    error_img = masker.inverse_transform(np.asarray(per_voxel_error, dtype=np.float32))
    display = plot_stat_map(
        error_img,
        display_mode="ortho",
        title=f"Mean squared reconstruction error (k={k})",
        colorbar=True,
    )
    display.savefig(out_path / f"per_voxel_error_map_k{k}.png", dpi=180)
    display.savefig(out_path / f"per_voxel_error_map_k{k}.pdf")
    display.close()
    return out_path


def plot_sensor_stability(
    selection_history: Sequence[np.ndarray],
    brain_voxel_indices: np.ndarray,
    masker,
    output_dir: str | Path,
) -> Path:
    """Map how often QR pivoting selects each brain voxel across training folds."""
    out_path = Path(output_dir)
    frequencies = np.zeros(len(brain_voxel_indices), dtype=np.float32)
    for sensors in selection_history:
        frequencies[np.asarray(sensors, dtype=int)] += 1.0
    if selection_history:
        frequencies /= len(selection_history)
    stability_img = masker.inverse_transform(frequencies)
    display = plot_stat_map(
        stability_img,
        display_mode="ortho",
        title="QR sensor selection frequency across folds",
        colorbar=True,
    )
    display.savefig(out_path / "sensor_stability.png", dpi=180)
    display.savefig(out_path / "sensor_stability.pdf")
    display.close()
    return out_path
