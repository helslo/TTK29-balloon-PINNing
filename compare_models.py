"""Run the physical model, PINN, or both on one shared experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from balloon_model import BalloonParams, build_input_function, simulate_balloon
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


def save_run(path: Path, times: np.ndarray, input_values: np.ndarray, states: np.ndarray, bold: np.ndarray, metrics: Dict[str, float]) -> None:
    """Save one model run in a portable NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, time=times, input=input_values, states=states, bold=bold, metrics=json.dumps(metrics))


def run(mode: str, data_path: Optional[Path], output_dir: Path, config: PINNConfig) -> Dict[str, float]:
    """Run one requested model or both models on exactly the same arrays."""
    times, input_values, observed_bold = load_csv(data_path) if data_path else synthetic_data()
    params = BalloonParams()
    metrics: Dict[str, float] = {}

    if mode in ("physical", "compare"):
        input_function = lambda time: float(np.interp(time, times, input_values))
        physical_time, physical_states, physical_bold = simulate_balloon(input_function, params, (times[0], times[-1]), float(np.median(np.diff(times))))
        physical_bold = np.interp(times, physical_time, physical_bold)
        physical_states = np.vstack([np.interp(times, physical_time, state) for state in physical_states])
        physical_metrics = {"mse": float(np.mean((observed_bold - physical_bold) ** 2)), "r2": r2_score(observed_bold, physical_bold)}
        save_run(output_dir / "physical_model.npz", times, input_values, physical_states, physical_bold, physical_metrics)
        metrics.update({f"physical_{key}": value for key, value in physical_metrics.items()})

    if mode in ("pinn", "compare"):
        model, history = train_pinn(times, input_values, observed_bold, params, config)
        pinn_states, pinn_bold = predict_pinn(model, times, input_values)
        pinn_metrics = {"mse": float(np.mean((observed_bold - pinn_bold) ** 2)), "r2": r2_score(observed_bold, pinn_bold)}
        save_run(output_dir / "pinn_model.npz", times, input_values, pinn_states, pinn_bold, pinn_metrics)
        np.savez(output_dir / "pinn_loss.npz", **{key: np.asarray(value) for key, value in history.items()})
        metrics.update({f"pinn_{key}": value for key, value in pinn_metrics.items()})

    if mode == "compare":
        (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("physical", "pinn", "compare"), default="compare")
    parser.add_argument("--data", type=Path, help="CSV with time,input,bold; defaults to synthetic data")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args(argv)
    metrics = run(args.mode, args.data, args.output_dir, PINNConfig(epochs=args.epochs))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()