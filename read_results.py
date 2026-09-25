"""Read, summarize, and optionally plot saved model results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


def load_json_value(value: np.ndarray) -> Dict[str, float]:
    """Decode a JSON object stored as a scalar NumPy string."""
    return json.loads(value.item())


def find_run_files(path: Path) -> Iterable[Path]:
    """Return result archives found in a run directory or at one file path."""
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got: {path}")
        return (path,)
    if path.is_dir():
        return sorted(path.glob("*_model.npz"))
    raise FileNotFoundError(f"Results path does not exist: {path}")


def print_model_summary(path: Path) -> Dict[str, float]:
    """Print the contents and metrics for one model archive."""
    with np.load(path, allow_pickle=False) as result:
        metrics = load_json_value(result["metrics"])
        parameters = load_json_value(result["parameters"])
        times = result["time"]
        states = result["states"]
        has_observed = "observed_bold" in result
        print(f"\n{path.name}")
        print(f"  samples: {len(times)}")
        print(f"  duration: {times[0]:.3f} to {times[-1]:.3f} s")
        print(f"  state shape: {states.shape} ([s, f, v, q])")
        print(f"  metrics: MSE={metrics.get('mse', float('nan')):.6g}, R2={metrics.get('r2', float('nan')):.6f}")
        print(f"  observed BOLD saved: {'yes' if has_observed else 'no (older archive)'}")
        if parameters:
            print("  fitted parameters: " + ", ".join(
                f"{name}={value:.4g}" for name, value in parameters.items()
            ))
    return metrics


def print_loss_summary(path: Path) -> None:
    """Print the first and final PINN losses when available."""
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as losses:
        print(f"\n{path.name}")
        print(f"  epochs: {len(losses['total'])}")
        for name in ("total", "data", "physics"):
            values = losses[name]
            print(f"  {name} loss: {values[0]:.6g} -> {values[-1]:.6g}")


def plot_run(run_dir: Path, model_files: Sequence[Path]) -> Path:
    """Save an observed-versus-predicted plot for a run directory."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    input_plotted = False
    for model_path in model_files:
        with np.load(model_path, allow_pickle=False) as result:
            time = result["time"]
            if "observed_bold" in result and not input_plotted:
                axes[0].plot(time, result["observed_bold"], color="black", label="observed")
            axes[0].plot(time, result["bold"], label=model_path.stem.replace("_model", ""))
            if not input_plotted:
                axes[1].plot(time, result["input"], color="tab:green", label="input")
                input_plotted = True
    axes[0].set_ylabel("BOLD")
    axes[1].set_ylabel("Input")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    output_path = run_dir / "results_summary.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Run directory or .npz result file")
    parser.add_argument("--plot", action="store_true", help="Save an observed-versus-predicted plot")
    args = parser.parse_args(argv)

    model_files = list(find_run_files(args.path))
    if not model_files:
        raise SystemExit(f"No model archives found in {args.path}")
    all_metrics = {path.stem.replace("_model", ""): print_model_summary(path) for path in model_files}
    print_loss_summary(args.path / "pinn_loss.npz" if args.path.is_dir() else args.path.parent / "pinn_loss.npz")

    if len(all_metrics) > 1:
        best_model = min(all_metrics, key=lambda name: all_metrics[name].get("mse", float("inf")))
        print(f"\nInterpretation: {best_model} has the lowest MSE.")
        best_r2 = max(all_metrics, key=lambda name: all_metrics[name].get("r2", float("-inf")))
        print(f"                 {best_r2} has the highest R2.")
    if args.plot:
        plot_dir = args.path if args.path.is_dir() else args.path.parent
        print(f"\nSaved plot: {plot_run(plot_dir, model_files)}")


if __name__ == "__main__":
    main()