"""Command-line entry point for sparse sensor placement experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .data import build_roi_matrices, load_roi_data, synthetic_roi_data, temporal_holdout_split
    from .evaluate import evaluate_all
    from .methods import qr_pivot_placement
    from .plots import (
        plot_error_curves,
        plot_per_roi_error_map,
        plot_reconstruction_examples,
        plot_selected_sensors_on_brain,
        plot_sensor_stability,
        plot_singular_value_spectrum,
    )
except ImportError:  # pragma: no cover - allow running as a script from the repo root
    from data import build_roi_matrices, load_roi_data, synthetic_roi_data, temporal_holdout_split
    from evaluate import evaluate_all
    from methods import qr_pivot_placement
    from plots import (
        plot_error_curves,
        plot_per_roi_error_map,
        plot_reconstruction_examples,
        plot_selected_sensors_on_brain,
        plot_sensor_stability,
        plot_singular_value_spectrum,
    )


def _resolve_input_data(path: str | Path | None) -> tuple[dict[str, dict[str, float]], list[str], bool]:
    """Load real ROI data if it exists, otherwise fall back to synthetic data."""
    data_path = Path(path) if path else Path("results/ROI_averaged_data.h5")
    if data_path.exists():
        participant_data = load_roi_data(data_path)
        matrices, roi_names = build_roi_matrices(participant_data, zscore=True)
        return matrices, roi_names, True
    synthetic = synthetic_roi_data(n_participants=8, n_rois=24, n_timepoints=160, seed=123)
    matrices, roi_names = build_roi_matrices(synthetic, zscore=True)
    return matrices, roi_names, False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to ROI-averaged HDF5 data. Defaults to results/ROI_averaged_data.h5 if present.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum number of sensors to evaluate in the k sweep.")
    parser.add_argument("--n-random-draws", type=int, default=10, help="Number of random-placement draws for each k.")
    parser.add_argument("--holdout", choices=("both", "temporal", "participant"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sensor_placement"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    participant_matrices, roi_names, has_real_data = _resolve_input_data(args.data)
    k_values = list(range(1, min(args.k_max, len(roi_names)) + 1))
    results = evaluate_all(participant_matrices, k_values, n_random_draws=args.n_random_draws, random_seed=args.seed, n_folds=5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sensor_placement_results.csv"
    results.to_csv(output_file, index=False)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if args.holdout in {"both", "temporal", "participant"}:
        plot_error_curves(results, figures_dir)

    first_pid = next(iter(participant_matrices))
    X = participant_matrices[first_pid]
    train_idx, test_idx = temporal_holdout_split(X)
    X_train = X[:, train_idx]
    X_test = X[:, test_idx]
    sensors_5, recon_5 = qr_pivot_placement(X_train, 5)
    X_hat_5 = recon_5(X_test)
    plot_singular_value_spectrum(X_train, figures_dir, r_chosen=5)
    plot_selected_sensors_on_brain(sensors_5, figures_dir, X.shape[0], k=5, seed=args.seed)
    plot_reconstruction_examples(X_test, X_hat_5, sensors_5, figures_dir, roi_names=roi_names, k=5)
    per_roi_error = np.mean((X_test - X_hat_5) ** 2, axis=1)
    plot_per_roi_error_map(per_roi_error, figures_dir, X.shape[0], k=5, seed=args.seed)
    selection_history = [
        qr_pivot_placement(X_train, k)[0]
        for k in (3, 5, 10)
    ]
    plot_sensor_stability(selection_history, X.shape[0], figures_dir)

    print(f"Saved {len(results)} evaluation rows to {output_file}")
    print(f"Saved figures to {figures_dir}")
    print(f"Used {'real' if has_real_data else 'synthetic'} ROI data with {len(roi_names)} ROIs")


if __name__ == "__main__":
    main()
