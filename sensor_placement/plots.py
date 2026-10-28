"""Plotting helpers for sparse sensor-placement experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.plotting import plot_markers


def _make_roi_coords(n_rois: int, seed: int = 0) -> np.ndarray:
    """Generate synthetic 3D ROI coordinates used when no atlas is available."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, np.pi, size=n_rois)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_rois)
    radius = 40.0 + 20.0 * rng.random(n_rois)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack([x, y, z])


def plot_error_curves(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Create the main error-vs-k figures for temporal and participant hold-out.

    The result tables are aggregated across folds, so each method produces a mean
    curve with a shaded band showing the standard deviation across random draws or
    held-out participants.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for holdout_name in ("temporal", "participant"):
        subset = results[results["holdout"] == holdout_name].copy()
        fig, ax = plt.subplots(figsize=(7, 5))
        for method_name, group in subset.groupby("method"):
            group = group.sort_values("k")
            if method_name == "random":
                ax.plot(group["k"], group["error_mean"], label=method_name, linewidth=2)
                ax.fill_between(
                    group["k"],
                    group["error_mean"] - group["error_std"],
                    group["error_mean"] + group["error_std"],
                    alpha=0.2,
                )
            else:
                ax.plot(group["k"], group["error_mean"], label=method_name, linewidth=2)
        ax.set_title(f"Sensor reconstruction error ({holdout_name} hold-out)")
        ax.set_xlabel("Number of sensors k")
        ax.set_ylabel("Normalized RMSE")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / f"error_vs_k_{holdout_name}.png", dpi=200)
        fig.savefig(out_path / f"error_vs_k_{holdout_name}.pdf")
        plt.close(fig)
    return out_path


def plot_singular_value_spectrum(
    X: np.ndarray,
    output_dir: str | Path,
    r_chosen: int | None = None,
    title: str = "Singular value spectrum",
) -> Path:
    """Plot the singular-value spectrum and the cumulative explained variance.

    Low-rank signals show a rapid drop in singular values, so the chosen rank is a
    simple indicator of how many spatial modes carry most of the signal variance.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    cumulative = np.cumsum(s**2) / np.sum(s**2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(s) + 1), cumulative, marker="o", linewidth=2)
    if r_chosen is None:
        r_chosen = int(np.searchsorted(cumulative, 0.9, side="left")) + 1
    ax.axvline(r_chosen, color="tab:red", linestyle="--", label=f"chosen r={r_chosen}")
    ax.set_title(title)
    ax.set_xlabel("Number of spatial modes")
    ax.set_ylabel("Cumulative explained variance")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "singular_value_spectrum.png", dpi=200)
    fig.savefig(out_path / "singular_value_spectrum.pdf")
    plt.close(fig)
    return out_path


def plot_selected_sensors_on_brain(
    selected_sensors: np.ndarray,
    output_dir: str | Path,
    n_rois: int,
    k: int,
    seed: int = 0,
) -> Path:
    """Plot the selected sensor locations as points on a synthetic brain surface.

    The function degrades gracefully when an atlas is unavailable: it creates a
    synthetic 3D coordinate map and draws the chosen ROIs as highlighted markers.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    coords = _make_roi_coords(n_rois, seed=seed)
    all_selected = np.zeros(n_rois, dtype=float)
    all_selected[selected_sensors] = 1.0
    plot_markers(
        all_selected,
        coords,
        node_size=12,
        node_cmap="gray",
        display_mode="lzry",
        title=f"Selected sensors for k={k}",
        output_file=str(out_path / f"selected_sensors_k{k}.png"),
    )
    plot_markers(
        all_selected,
        coords,
        node_size=12,
        node_cmap="gray",
        display_mode="lzry",
        title=f"Selected sensors for k={k}",
        output_file=str(out_path / f"selected_sensors_k{k}.pdf"),
    )
    return out_path


def plot_reconstruction_examples(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    selected_sensors: np.ndarray,
    output_dir: str | Path,
    roi_names: list[str] | None = None,
    k: int | None = None,
) -> Path:
    """Plot a few ROIs of the true and reconstructed signal for a representative example.

    One selected sensor and several unobserved ROIs are shown together so the
    reader can see how the reconstruction behaves at low and moderate sensor counts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    indices = [int(selected_sensors[0])] if len(selected_sensors) else [0]
    extra = [max(0, idx) for idx in [0, 1, 2, 3] if idx < X_true.shape[0]]
    rows = sorted(set(indices + extra))[:4]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.2 * len(rows)), squeeze=False)
    for axis, row_idx in zip(axes.ravel(), rows):
        axis.plot(X_true[row_idx], label="true", linewidth=2)
        axis.plot(X_pred[row_idx], label="reconstructed", linestyle="--", linewidth=2)
        axis.set_title(f"ROI {row_idx if roi_names is None else roi_names[row_idx]}")
        axis.set_xlabel("time")
        axis.set_ylabel("z-score")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path / f"reconstruction_example_k{k if k is not None else 'selected'}.png", dpi=200)
    fig.savefig(out_path / f"reconstruction_example_k{k if k is not None else 'selected'}.pdf")
    plt.close(fig)
    return out_path


def plot_per_roi_error_map(
    per_roi_error: np.ndarray,
    output_dir: str | Path,
    n_rois: int,
    k: int,
    seed: int = 0,
) -> Path:
    """Plot a synthetic brain map colored by the per-ROI reconstruction error.

    This is a lightweight stand-in for a real atlas-based ROI error map when only a
    coordinate set is available.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    coords = _make_roi_coords(n_rois, seed=seed)
    plot_markers(
        np.asarray(per_roi_error, dtype=float),
        coords,
        node_size=18,
        node_cmap="viridis",
        display_mode="lzry",
        title=f"Per-ROI error at k={k}",
        output_file=str(out_path / f"per_roi_error_map_k{k}.png"),
    )
    plot_markers(
        np.asarray(per_roi_error, dtype=float),
        coords,
        node_size=18,
        node_cmap="viridis",
        display_mode="lzry",
        title=f"Per-ROI error at k={k}",
        output_file=str(out_path / f"per_roi_error_map_k{k}.pdf"),
    )
    return out_path


def plot_sensor_stability(
    selection_history: list[np.ndarray],
    n_rois: int,
    output_dir: str | Path,
    title: str = "Sensor selection frequency",
) -> Path:
    """Show how often each ROI is selected across folds or random draws.

    Higher-frequency ROIs are more stable and therefore less sensitive to the exact
    training subset or random seed.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    frequencies = np.zeros(n_rois, dtype=float)
    for sensors in selection_history:
        frequencies[np.asarray(sensors, dtype=int)] += 1.0
    frequencies /= max(1, len(selection_history))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(n_rois), frequencies, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("ROI index")
    ax.set_ylabel("Selection frequency")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path / "sensor_stability.png", dpi=200)
    fig.savefig(out_path / "sensor_stability.pdf")
    plt.close(fig)
    return out_path
