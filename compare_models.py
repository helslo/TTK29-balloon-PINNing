"""Run the physical model, PINN, or both on one shared experiment."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from balloon_model import BalloonParams, build_input_function, fit_balloon, simulate_balloon
from data_pipeline import load_subject_timeseries, write_timeseries_csv
from pinn_model import PINNConfig, predict_pinn, train_pinn


def synthetic_data(duration: float = 30.0, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic dataset for development and model comparison."""
    input_function = build_input_function([2.0, 10.0, 18.0], duration, dt)
    times, _, observed_bold = simulate_balloon(input_function, BalloonParams(), (0.0, duration), dt)
    input_values = np.array([input_function(time) for time in times])
    return times, input_values, observed_bold


def load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CSV with columns ``time``, ``input``, and ``bold``."""
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"time", "input", "bold"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    values = np.array([[float(row[name]) for name in ("time", "input", "bold")] for row in rows])
    return values[:, 0], values[:, 1], values[:, 2]


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """Return coefficient of determination for one BOLD trace."""
    denominator = np.sum((target - np.mean(target)) ** 2)
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator) if denominator else float("nan")


def default_output_dir(mode: str, epochs: int, base_dir: Path = Path("results")) -> Path:
    """Return a timestamped directory for a command-line run."""
    timestamp = datetime.now().strftime("%d.%m.%y_%H.%M.%S")
    epoch_suffix = f"_epochs{epochs}" if mode in ("pinn", "compare") else ""
    return base_dir / f"{timestamp}_{mode}{epoch_suffix}"


def save_run(
    path: Path,
    times: np.ndarray,
    input_values: np.ndarray,
    observed_bold: np.ndarray,
    states: np.ndarray,
    bold: np.ndarray,
    metrics: Dict[str, float],
    parameters: Optional[Dict[str, float]] = None,
) -> None:
    """Save one model run in a portable NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        time=times,
        input=input_values,
        observed_bold=observed_bold,
        states=states,
        bold=bold,
        metrics=json.dumps(metrics),
        parameters=json.dumps(parameters or {}),
    )


def run(
    mode: str,
    data_path: Optional[Path],
    output_dir: Path,
    config: PINNConfig,
    fit_physical: bool = True,
) -> Dict[str, float]:
    """Run one requested model or both models on exactly the same arrays."""
    times, input_values, observed_bold = load_csv(data_path) if data_path else synthetic_data()
    params = BalloonParams()
    metrics: Dict[str, float] = {}

    if mode in ("physical", "compare"):
        input_function = lambda time: float(np.interp(time, times, input_values))
        if fit_physical:
            fit = fit_balloon(times, observed_bold, input_function, params)
            params = fit.params
            physical_time, physical_states, physical_bold = simulate_balloon(
                input_function, params, (times[0], times[-1]), float(np.median(np.diff(times)))
            )
        else:
            physical_time, physical_states, physical_bold = simulate_balloon(
                input_function, params, (times[0], times[-1]), float(np.median(np.diff(times)))
            )
        physical_bold = np.interp(times, physical_time, physical_bold)
        physical_states = np.vstack([np.interp(times, physical_time, state) for state in physical_states])
        physical_metrics = {"mse": float(np.mean((observed_bold - physical_bold) ** 2)), "r2": r2_score(observed_bold, physical_bold)}
        physical_parameters = {
            name: float(getattr(params, name))
            for name in ("kappa", "gamma", "tau", "alpha", "E0", "V0", "eps", "nu0", "r0", "epsilon_r", "TE")
        }
        save_run(output_dir / "physical_model.npz", times, input_values, observed_bold, physical_states, physical_bold, physical_metrics, physical_parameters)
        metrics.update({f"physical_{key}": value for key, value in physical_metrics.items()})

    if mode in ("pinn", "compare"):
        model, history = train_pinn(times, input_values, observed_bold, params, config)
        pinn_states, pinn_bold = predict_pinn(model, times, input_values)
        pinn_metrics = {"mse": float(np.mean((observed_bold - pinn_bold) ** 2)), "r2": r2_score(observed_bold, pinn_bold)}
        pinn_parameters = {
            name: float(getattr(params, name))
            for name in ("kappa", "gamma", "tau", "alpha", "E0", "V0", "eps", "nu0", "r0", "epsilon_r", "TE")
        }
        save_run(output_dir / "pinn_model.npz", times, input_values, observed_bold, pinn_states, pinn_bold, pinn_metrics, pinn_parameters)
        np.savez(output_dir / "pinn_loss.npz", **{key: np.asarray(value) for key, value in history.items()})
        metrics.update({f"pinn_{key}": value for key, value in pinn_metrics.items()})

    if mode == "compare":
        (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("physical", "pinn", "compare"), default="compare")
    parser.add_argument("--data", type=Path, help="CSV with time,input,bold; defaults to synthetic data")
    parser.add_argument("--subject", help="BIDS subject ID, for example sub-10159")
    parser.add_argument("--data-root", type=Path, default=Path("data/ds000030"))
    parser.add_argument("--bold-path", type=Path, help="Preprocessed BOLD NIfTI; overrides the raw subject path")
    parser.add_argument("--events-path", type=Path, help="Events TSV; defaults to the subject's raw BIDS events file")
    parser.add_argument("--roi-mask", type=Path, help="Aligned ROI mask NIfTI for --subject")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for results; defaults to a timestamped folder under results/",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--no-fit-physical", action="store_true", help="Use default Balloon parameters")
    args = parser.parse_args(argv)
    if args.data is not None and args.subject is not None:
        parser.error("use either --data or --subject, not both")
    if args.bold_path is not None and args.subject is None:
        parser.error("--bold-path requires --subject")
    output_dir = args.output_dir or default_output_dir(args.mode, args.epochs)
    data_path = args.data
    if args.subject is not None:
        if args.roi_mask is None:
            parser.error("--roi-mask is required with --subject")
        subject_dir = args.data_root / args.subject / "func"
        bold_path = args.bold_path or subject_dir / f"{args.subject}_task-stopsignal_bold.nii.gz"
        events_path = args.events_path or subject_dir / f"{args.subject}_task-stopsignal_events.tsv"
        times, input_values, observed_bold = load_subject_timeseries(bold_path, events_path, args.roi_mask)
        data_path = output_dir / f"{args.subject}_timeseries.csv"
        write_timeseries_csv(data_path, times, input_values, observed_bold)
    metrics = run(
        args.mode,
        data_path,
        output_dir,
        PINNConfig(epochs=args.epochs),
        fit_physical=not args.no_fit_physical,
    )
    print(json.dumps(metrics, indent=2))
    print("Ran successfully")


if __name__ == "__main__":
    main()