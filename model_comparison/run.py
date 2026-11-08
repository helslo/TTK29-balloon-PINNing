"""Run the physical model, PINN, or both on one shared experiment."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .balloon_model import BalloonParams, build_input_function, fit_balloon, simulate_balloon
from .costa_model import CoSTAConfig, predict_costa, train_costa
from .data_pipeline import (
    SCHAEFER_NETWORKS,
    create_schaefer_network_mask,
    load_subject_timeseries,
    select_fmriprep_confounds,
    write_timeseries_csv,
)
from .pinn_model import PINNConfig, predict_pinn, train_pinn

DEFAULT_SUBJECT = "sub-10159"


def discover_fmriprep_bold(data_root: Path, subject: str, task: str = "stopsignal") -> Path:
    """Find one unambiguous fMRIPrep BOLD run for a subject and task."""
    derivative_root = data_root / "derivatives" / "fmriprep"
    func_dir = derivative_root / subject / "func"
    if not func_dir.is_dir():
        raise FileNotFoundError(
            f"fMRIPrep derivatives were not found for {subject} at {func_dir}. "
            "Run fMRIPrep or pass --bold-path and --confounds-path explicitly."
        )
    candidates = sorted(
        func_dir.glob(f"{subject}_task-{task}*desc-preproc_bold.nii.gz")
    )
    for template in ("space-MNI152NLin6Asym", "space-MNI152NLin2009cAsym"):
        standard_space = [path for path in candidates if template in path.name]
        if standard_space:
            candidates = standard_space
            break
    if not candidates:
        raise FileNotFoundError(
            f"No fMRIPrep preprocessed BOLD image for task-{task} was found in {func_dir}"
        )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple fMRIPrep BOLD runs match subject {subject} and task-{task}: "
            f"{names}. Select one explicitly with --bold-path."
        )
    return candidates[0]


def _fmriprep_confounds_for_bold(bold_path: Path) -> Path:
    """Resolve the matching fMRIPrep confounds file from BIDS entities."""
    if "_space-" in bold_path.name:
        prefix = bold_path.name.split("_space-", maxsplit=1)[0]
    else:
        prefix = bold_path.name.split("_desc-preproc_bold", maxsplit=1)[0]
    exact_path = bold_path.with_name(f"{prefix}_desc-confounds_timeseries.tsv")
    if exact_path.is_file():
        return exact_path
    candidates = sorted(
        bold_path.parent.glob(f"{prefix}*desc-confounds_timeseries.tsv")
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple fMRIPrep confounds files match {bold_path}")
    raise FileNotFoundError(f"No matching fMRIPrep confounds TSV found for {bold_path}")


def _fmriprep_brain_mask_for_bold(bold_path: Path) -> Optional[Path]:
    """Return the matching fMRIPrep brain mask when one exists."""
    suffix = "_desc-preproc_bold.nii.gz"
    if not bold_path.name.endswith(suffix):
        return None
    mask_path = bold_path.with_name(
        bold_path.name[: -len(suffix)] + "_desc-brain_mask.nii.gz"
    )
    return mask_path if mask_path.is_file() else None


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
    epoch_suffix = f"_epochs{epochs}" if mode in ("pinn", "costa", "compare") else ""
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
    use_synthetic: bool = False,
) -> Dict[str, float]:
    """Run one requested model or both models on exactly the same arrays."""
    if data_path is not None:
        times, input_values, observed_bold = load_csv(data_path)
    elif use_synthetic:
        times, input_values, observed_bold = synthetic_data()
    else:
        raise ValueError("Provide a CSV data path or set use_synthetic=True")
    params = BalloonParams()
    metrics: Dict[str, float] = {}

    if mode in ("physical", "costa", "compare"):
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
        if fit_physical:
            physical_bold = fit.fitted_bold
        physical_metrics = {"mse": float(np.mean((observed_bold - physical_bold) ** 2)), "r2": r2_score(observed_bold, physical_bold)}
        physical_parameters = {
            name: float(getattr(params, name))
            for name in ("kappa", "gamma", "tau", "alpha", "E0", "V0", "eps", "nu0", "r0", "epsilon_r", "TE")
        }
        if fit_physical:
            physical_parameters["observation_baseline"] = fit.observation_baseline
            physical_parameters["observation_linear_drift"] = fit.observation_linear_drift
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
        pinn_parameters["V0"] = float(model.v0.detach())
        save_run(output_dir / "pinn_model.npz", times, input_values, observed_bold, pinn_states, pinn_bold, pinn_metrics, pinn_parameters)
        np.savez(output_dir / "pinn_loss.npz", **{key: np.asarray(value) for key, value in history.items()})
        metrics.update({f"pinn_{key}": value for key, value in pinn_metrics.items()})

    if mode in ("costa", "compare"):
        costa_model, costa_history = train_costa(
            times,
            input_values,
            observed_bold,
            physical_states,
            physical_bold,
            CoSTAConfig(epochs=config.epochs),
        )
        costa_states, costa_bold = predict_costa(
            costa_model, times, input_values, physical_states, physical_bold
        )
        costa_metrics = {
            "mse": float(np.mean((observed_bold - costa_bold) ** 2)),
            "r2": r2_score(observed_bold, costa_bold),
        }
        save_run(
            output_dir / "costa_model.npz",
            times,
            input_values,
            observed_bold,
            costa_states,
            costa_bold,
            costa_metrics,
            physical_parameters,
        )
        np.savez(
            output_dir / "costa_loss.npz",
            **{key: np.asarray(value) for key, value in costa_history.items()},
        )
        metrics.update({f"costa_{key}": value for key, value in costa_metrics.items()})

    if mode == "compare":
        (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("physical", "pinn", "costa", "compare"), default="compare")
    parser.add_argument("--data", type=Path, help="CSV with time,input,bold")
    parser.add_argument(
        "--subject",
        help=f"BIDS subject ID; defaults to {DEFAULT_SUBJECT} when no other input is provided",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ds000030"))
    parser.add_argument("--bold-path", type=Path, help="Preprocessed BOLD NIfTI; overrides the raw subject path")
    parser.add_argument("--events-path", type=Path, help="Events TSV; defaults to the subject's raw BIDS events file")
    parser.add_argument(
        "--fmriprep",
        action="store_true",
        help="Use the subject's fMRIPrep task-stopsignal BOLD and matching confounds",
    )
    parser.add_argument(
        "--roi-mask",
        type=Path,
        help="ROI mask NIfTI; resampled to the BOLD grid. Without it, use a brain-wide mean",
    )
    parser.add_argument(
        "--schaefer-network",
        choices=SCHAEFER_NETWORKS,
        help=(
            "Build an ROI mask from the Schaefer 2018 Yeo-7 atlas. "
            "Requires MNI-space BOLD; mutually exclusive with --roi-mask"
        ),
    )
    parser.add_argument(
        "--schaefer-hemisphere",
        choices=("left", "right", "both"),
        default="both",
        help="Hemisphere(s) for --schaefer-network (default: both)",
    )
    parser.add_argument(
        "--confounds-path",
        type=Path,
        help="fMRIPrep confounds TSV; discovered automatically for --fmriprep BOLD",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic input instead of real subject or CSV data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for results; defaults to a timestamped folder under results/",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--no-fit-physical", action="store_true", help="Use default Balloon parameters")
    args = parser.parse_args(argv)
    has_subject_input = any(
        value is not None
        for value in (
            args.subject,
            args.bold_path,
            args.events_path,
            args.roi_mask,
            args.schaefer_network,
            args.confounds_path,
        )
    ) or args.fmriprep
    if args.data is not None and has_subject_input:
        parser.error("use either --data or --subject, not both")
    if args.synthetic and (args.data is not None or has_subject_input):
        parser.error("--synthetic cannot be combined with --data or subject input options")
    if args.fmriprep and args.bold_path is not None:
        parser.error("--fmriprep and --bold-path are alternatives; choose one")
    if args.roi_mask is not None and args.schaefer_network is not None:
        parser.error("--roi-mask and --schaefer-network are alternatives; choose one")
    if args.schaefer_network is not None and args.data is not None:
        parser.error("--schaefer-network requires subject BOLD input, not a CSV")
    if args.schaefer_network is not None and args.synthetic:
        parser.error("--schaefer-network cannot be combined with --synthetic")
    subject = args.subject
    if args.data is None and not args.synthetic and subject is None:
        subject = DEFAULT_SUBJECT
    if args.bold_path is not None and subject is None:
        parser.error("--bold-path requires --subject")
    output_dir = args.output_dir or default_output_dir(args.mode, args.epochs)
    data_path = args.data
    if subject is not None:
        subject_dir = args.data_root / subject / "func"
        bold_path = (
            discover_fmriprep_bold(args.data_root, subject)
            if args.fmriprep
            else args.bold_path or subject_dir / f"{subject}_task-stopsignal_bold.nii.gz"
        )
        events_path = args.events_path or subject_dir / f"{subject}_task-stopsignal_events.tsv"
        roi_mask_path = args.roi_mask
        atlas_metadata = None
        if args.schaefer_network is not None:
            if "space-MNI" not in bold_path.name:
                parser.error(
                    "--schaefer-network uses an MNI-space atlas. Select MNI-space "
                    "fMRIPrep BOLD with --fmriprep or pass a --bold-path whose "
                    "filename identifies its MNI space."
                )
            roi_mask_path, selected_labels = create_schaefer_network_mask(
                output_dir
                / (
                    "schaefer2018_400_7networks_"
                    f"{args.schaefer_network}_{args.schaefer_hemisphere}_2mm_mask.nii.gz"
                ),
                args.schaefer_network,
                args.schaefer_hemisphere,
            )
            atlas_metadata = {
                "atlas": "Schaefer 2018",
                "network": args.schaefer_network,
                "hemisphere": args.schaefer_hemisphere,
                "labels": list(selected_labels),
            }
        confounds_path = args.confounds_path
        if args.fmriprep and confounds_path is None:
            confounds_path = _fmriprep_confounds_for_bold(bold_path)
        elif (
            confounds_path is None
            and "desc-preproc_bold" in bold_path.name
        ):
            confounds_path = _fmriprep_confounds_for_bold(bold_path)
        brain_mask_path = (
            None if args.roi_mask is not None else _fmriprep_brain_mask_for_bold(bold_path)
        )
        times, input_values, observed_bold = load_subject_timeseries(
            bold_path,
            events_path,
            roi_mask_path,
            confounds_path,
            brain_mask_path,
        )
        data_path = output_dir / f"{subject}_timeseries.csv"
        write_timeseries_csv(data_path, times, input_values, observed_bold)
        metadata = {
            "subject": subject,
            "bold_path": str(bold_path),
            "events_path": str(events_path),
            "roi_mask_path": str(roi_mask_path) if roi_mask_path is not None else None,
            "roi_atlas": atlas_metadata,
            "brain_mask_path": (
                str(brain_mask_path) if brain_mask_path is not None else None
            ),
            "confounds_path": (
                str(confounds_path) if confounds_path is not None else None
            ),
            "masking": (
                "ROI mask (nearest-neighbor resampled)"
                if roi_mask_path is not None
                else "fMRIPrep brain mask"
                if brain_mask_path is not None
                else "EPI-derived brain mask"
            ),
            "signal": "mean voxel intensity within selected mask; fractional change; isolated-spike interpolation; confound regression when provided; linear detrending",
        }
        if confounds_path is not None:
            metadata["confound_columns"] = list(
                select_fmriprep_confounds(
                    pd.read_csv(confounds_path, sep="\t")
                ).columns
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "timeseries_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    metrics = run(
        args.mode,
        data_path,
        output_dir,
        PINNConfig(epochs=args.epochs),
        fit_physical=not args.no_fit_physical,
        use_synthetic=args.synthetic,
    )
    print(json.dumps(metrics, indent=2))
    print("Ran successfully")


if __name__ == "__main__":
    main()