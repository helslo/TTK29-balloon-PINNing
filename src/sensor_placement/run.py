"""Run sensor placement on cleaned, downsampled fMRIPrep BOLD data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_clean_runs, temporal_holdout_split
from .evaluate import evaluate_all, log_k_grid
from .methods import _low_rank_basis, qr_pivot_placement
from .plots import (
    plot_error_curves,
    plot_per_voxel_error_map,
    plot_reconstruction_examples,
    plot_selected_sensors_on_brain,
    plot_sensor_stability,
    plot_singular_value_spectrum,
)


def _parse_k_values(raw: str | None, n_voxels: int, fraction: float, points: int) -> list[int]:
    """Parse an explicit sensor grid or derive the planned logarithmic grid."""
    if raw is None:
        return log_k_grid(n_voxels, fraction=fraction, points=points)
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value < 1 or value > n_voxels for value in values):
        raise ValueError(f"Every k must be between 1 and {n_voxels}")
    return values


def _default_l1_k_values(k_values: list[int]) -> list[int]:
    """Choose a small low/medium/high subset for the costly iterative L1 solver."""
    if len(k_values) <= 3:
        return k_values
    indices = sorted({0, len(k_values) // 2, len(k_values) - 1})
    return [k_values[index] for index in indices]


def main(argv: list[str] | None = None) -> None:
    """Load real fMRIPrep inputs, evaluate sensor strategies, and save reproducible outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ds000030"))
    parser.add_argument("--task", default="rest", help="BIDS task label, default: rest")
    parser.add_argument("--max-participants", type=int, default=20)
    parser.add_argument("--include-all-diagnoses", action="store_true")
    parser.add_argument("--max-mean-fd", type=float, default=0.5, help="Exclude runs above this mean FD (mm)")
    parser.add_argument("--voxel-size-mm", type=float, default=4.0)
    parser.add_argument("--smoothing-fwhm-mm", type=float, default=6.0)
    parser.add_argument("--global-signal", action="store_true", help="Regress global signal in addition to default confounds")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--k-values", help="Comma-separated explicit voxel counts; defaults to a log-spaced grid")
    parser.add_argument("--k-fraction", type=float, default=0.05)
    parser.add_argument("--k-points", type=int, default=8)
    parser.add_argument(
        "--rank",
        type=int,
        help="Fixed number of spatial SVD modes (maximum 64); default selects up to 90%% variance",
    )
    parser.add_argument("--n-random-draws", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--include-l1", action="store_true", help="Run the expensive spatial 3D-DCT FISTA methods")
    parser.add_argument("--l1-k-values", help="Comma-separated reduced k grid for L1; defaults to low/medium/high k")
    parser.add_argument(
        "--l1-alpha",
        type=float,
        help="Override L1 regularization; by default it is selected on an inner training-only validation split",
    )
    parser.add_argument("--l1-iterations", type=int, default=40)
    parser.add_argument("--l1-test-timepoints", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sensor_placement"))
    args = parser.parse_args(argv)
    if args.rank is not None and not 1 <= args.rank <= 64:
        parser.error("--rank must be between 1 and 64")

    runs, exclusions = load_clean_runs(
        data_root=args.data_root,
        task=args.task,
        max_participants=args.max_participants,
        controls_only=not args.include_all_diagnoses,
        max_mean_fd=args.max_mean_fd,
        voxel_size_mm=args.voxel_size_mm,
        smoothing_fwhm_mm=args.smoothing_fwhm_mm,
        include_global_signal=args.global_signal,
        cache_dir=args.cache_dir,
    )
    n_voxels = runs[0].X.shape[0]
    k_values = _parse_k_values(args.k_values, n_voxels, args.k_fraction, args.k_points)
    l1_k_values = (
        _parse_k_values(args.l1_k_values, n_voxels, args.k_fraction, args.k_points)
        if args.l1_k_values
        else _default_l1_k_values(k_values)
    )
    if args.include_l1 and any(k not in k_values for k in l1_k_values):
        raise ValueError("--l1-k-values must be a subset of the evaluated --k-values")

    results, selection_history, brain_voxel_indices = evaluate_all(
        runs,
        k_values,
        n_random_draws=args.n_random_draws,
        random_seed=args.seed,
        n_folds=args.n_folds,
        include_l1=args.include_l1,
        rank=args.rank,
        l1_alpha=args.l1_alpha,
        l1_iterations=args.l1_iterations,
        l1_k_values=l1_k_values,
        l1_test_timepoints=args.l1_test_timepoints,
    )
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "sensor_placement_results.csv", index=False)
    plot_error_curves(results, figures_dir)

    representative = runs[0]
    train_idx, test_idx = temporal_holdout_split(representative.X)
    X_train = np.asarray(representative.X[:, train_idx], dtype=np.float32)
    X_test = np.asarray(representative.X[:, test_idx], dtype=np.float32)
    _, singular_values, chosen_rank = _low_rank_basis(X_train, r=args.rank)
    total_energy = sum(
        float(np.sum(np.square(X_train[start:start + 4096], dtype=np.float64)))
        for start in range(0, X_train.shape[0], 4096)
    )
    plot_singular_value_spectrum(
        singular_values, total_energy, figures_dir, r_chosen=chosen_rank
    )

    plot_sensor_counts = sorted(set(k_values[min(i, len(k_values) - 1)] for i in (2, 4, 6)))
    qr_selections: dict[int, np.ndarray] = {}
    for k in plot_sensor_counts:
        sensors, _ = qr_pivot_placement(X_train, k, r=args.rank)
        qr_selections[k] = sensors
        plot_selected_sensors_on_brain(
            sensors, brain_voxel_indices, representative.masker, figures_dir, k
        )

    example_k = plot_sensor_counts[-1]
    X_hat = None
    for k in sorted(set((plot_sensor_counts[0], example_k))):
        example_sensors, reconstruct = qr_pivot_placement(X_train, k, r=args.rank)
        prediction = reconstruct(X_test)
        plot_reconstruction_examples(
            X_test, prediction, example_sensors, representative.tr, figures_dir, k
        )
        if k == example_k:
            X_hat = prediction
    if X_hat is None:
        raise RuntimeError("Failed to generate the largest-k reconstruction example")
    plot_per_voxel_error_map(
        np.mean((X_test - X_hat) ** 2, axis=1),
        representative.masker,
        figures_dir,
        example_k,
    )
    plot_sensor_stability(
        [sensors for sensors in selection_history if len(sensors) == example_k],
        brain_voxel_indices,
        representative.masker,
        figures_dir,
    )

    metadata = {
        "task": args.task,
        "participants": sorted({run.participant_id for run in runs}),
        "runs": [{"participant_id": run.participant_id, "run_id": run.run_id} for run in runs],
        "excluded_runs": exclusions,
        "voxel_count": n_voxels,
        "voxel_size_mm": args.voxel_size_mm,
        "smoothing_fwhm_mm": args.smoothing_fwhm_mm,
        "global_signal_regressed": args.global_signal,
        "k_values": k_values,
        "l1_included": args.include_l1,
        "l1_k_values": l1_k_values if args.include_l1 else [],
        "l1_test_timepoints": args.l1_test_timepoints,
    }
    (output_dir / "sensor_placement_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Loaded {len(runs)} runs from {len(metadata['participants'])} participants")
    print(f"Excluded {len(exclusions)} runs")
    for exclusion in exclusions:
        print(f"  Excluded {exclusion['run']}: {exclusion['reason']}")
    print(f"Saved {len(results)} metric rows to {output_dir / 'sensor_placement_results.csv'}")
    print(f"Saved figures to {figures_dir}")
    print(f"Using {n_voxels} common-mask voxels at {args.voxel_size_mm:g} mm")


if __name__ == "__main__":
    main()
