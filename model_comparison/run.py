"""Run the physical model, PINN, or both on one shared experiment."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .balloon_model import BalloonParams, build_input_function, fit_balloon, simulate_balloon
from .costa_model import (
    CoSTAConfig,
    _temporal_validation_mask,
    predict_costa,
    train_costa,
)
from .data_pipeline import (
    SCHAEFER_NETWORKS,
    create_schaefer_network_mask,
    load_subject_timeseries,
    select_fmriprep_confounds,
    write_timeseries_csv,
)
from .pinn_model import PINNConfig, predict_pinn, train_pinn
from ttk29_balloon_pinning.config import get_data_root

DEFAULT_SUBJECT = "sub-10159"
PROJECT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def discover_events_tsv(data_root: Path, subject: str, task: str = "stopsignal") -> Path:
    """Find a subject's events TSV in the dataset or the project's local data."""
    filename = f"{subject}_task-{task}_events.tsv"
    expected_path = data_root / subject / "func" / filename
    if expected_path.is_file():
        return expected_path

    local_candidates = sorted(PROJECT_DATA_ROOT.rglob(filename))
    if len(local_candidates) == 1:
        return local_candidates[0]
    if len(local_candidates) > 1:
        paths = ", ".join(str(path) for path in local_candidates)
        raise ValueError(
            f"Multiple local events TSV files match subject {subject} and task-{task}: "
            f"{paths}. Select one explicitly with --events-path."
        )
    raise FileNotFoundError(
        f"No events TSV for subject {subject} and task-{task} was found at "
        f"{expected_path} or under {PROJECT_DATA_ROOT}. Pass --events-path explicitly."
    )


def discover_echo_time(
    data_root: Path,
    subject: str,
    task: str,
    bold_path: Path,
) -> Tuple[Optional[float], Optional[Path]]:
    """Find the acquisition EchoTime sidecar associated with a BOLD run."""
    func_dirs = (
        bold_path.parent,
        data_root / subject / "func",
        data_root / "derivatives" / "fmriprep" / subject / "func",
    )
    for func_dir in dict.fromkeys(func_dirs):
        if not func_dir.is_dir():
            continue
        exact_sidecar = func_dir / f"{subject}_task-{task}_bold.json"
        candidates = (
            [exact_sidecar]
            if exact_sidecar.is_file()
            else sorted(func_dir.glob(f"{subject}_task-{task}*_bold.json"))
        )
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(
                f"Multiple acquisition metadata files match subject {subject} "
                f"and task-{task}: {names}."
            )
        if candidates:
            metadata_path = candidates[0]
            metadata = json.loads(metadata_path.read_text())
            if "EchoTime" not in metadata:
                raise ValueError(f"{metadata_path} does not contain EchoTime")
            echo_time = float(metadata["EchoTime"])
            if not np.isfinite(echo_time) or echo_time <= 0:
                raise ValueError(
                    f"Invalid EchoTime in {metadata_path}: {metadata['EchoTime']}"
                )
            return echo_time, metadata_path

    local_candidates = sorted(
        PROJECT_DATA_ROOT.rglob(f"{subject}_task-{task}*_bold.json")
    )
    if len(local_candidates) > 1:
        names = ", ".join(str(path) for path in local_candidates)
        raise ValueError(
            f"Multiple local acquisition metadata files match subject {subject} "
            f"and task-{task}: {names}."
        )
    if local_candidates:
        metadata_path = local_candidates[0]
        metadata = json.loads(metadata_path.read_text())
        if "EchoTime" not in metadata:
            raise ValueError(f"{metadata_path} does not contain EchoTime")
        echo_time = float(metadata["EchoTime"])
        if not np.isfinite(echo_time) or echo_time <= 0:
            raise ValueError(f"Invalid EchoTime in {metadata_path}: {metadata['EchoTime']}")
        return echo_time, metadata_path
    return None, None


def discover_fmriprep_bold(data_root: Path, subject: str, task: str = "stopsignal") -> Path:
    """Find one unambiguous fMRIPrep BOLD run for a subject and task."""
    func_dir = data_root / subject / "func"
    if not func_dir.is_dir():
        func_dir = data_root / "derivatives" / "fmriprep" / subject / "func"
    if not func_dir.is_dir():
        raise FileNotFoundError(
            f"No subject functional directory was found for {subject} under "
            f"{data_root} (checked {data_root / subject / 'func'} and "
            f"{data_root / 'derivatives' / 'fmriprep' / subject / 'func'}). "
            "Run fMRIPrep or pass --bold-path and --confounds-path explicitly."
        )
    candidates = sorted(
        {
            *func_dir.glob(f"{subject}_task-{task}*desc-preproc_bold.nii.gz"),
            *func_dir.glob(f"{subject}_task-{task}*_bold_space-*_preproc.nii.gz"),
        }
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
    exact_paths = (
        bold_path.with_name(f"{prefix}_confounds.tsv"),
        bold_path.with_name(f"{prefix}_desc-confounds_timeseries.tsv"),
    )
    for exact_path in exact_paths:
        if exact_path.is_file():
            return exact_path
    candidates = sorted(bold_path.parent.glob(f"{prefix}*confounds*.tsv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple fMRIPrep confounds files match {bold_path}")
    raise FileNotFoundError(f"No matching fMRIPrep confounds TSV found for {bold_path}")


def _fmriprep_brain_mask_for_bold(bold_path: Path) -> Optional[Path]:
    """Return the matching fMRIPrep brain mask when one exists."""
    suffixes = (
        ("_preproc.nii.gz", "_brainmask.nii.gz"),
        ("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz"),
    )
    for suffix, mask_suffix in suffixes:
        if bold_path.name.endswith(suffix):
            mask_path = bold_path.with_name(
                bold_path.name[: -len(suffix)] + mask_suffix
            )
            return mask_path if mask_path.is_file() else None
    return None


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
    balloon_params: Optional[BalloonParams] = None,
) -> Dict[str, float]:
    """Run one requested model or both models on exactly the same arrays."""
    if data_path is not None:
        times, input_values, observed_bold = load_csv(data_path)
    elif use_synthetic:
        times, input_values, observed_bold = synthetic_data()
    else:
        raise ValueError("Provide a CSV data path or set use_synthetic=True")
    params = balloon_params or BalloonParams()
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
        costa_physical_states = physical_states
        costa_physical_bold = physical_bold
        costa_params = params
        costa_fit = None
        validation_start = int(
            np.flatnonzero(_temporal_validation_mask(len(times)))[0]
        )
        if fit_physical:
            costa_fit = fit_balloon(
                times[:validation_start],
                observed_bold[:validation_start],
                input_function,
                BalloonParams(TE=params.TE),
            )
            costa_params = costa_fit.params
            costa_time, costa_states, simulated_bold = simulate_balloon(
                input_function,
                costa_params,
                (float(times[0]), float(times[-1])),
                float(np.median(np.diff(times))),
            )
            costa_physical_states = np.vstack(
                [np.interp(times, costa_time, state) for state in costa_states]
            )
            costa_physical_bold = np.interp(times, costa_time, simulated_bold)
            normalized_time = (times - np.mean(times[:validation_start])) / max(
                float(times[validation_start - 1] - times[0]), 1.0
            )
            costa_physical_bold = (
                costa_physical_bold
                + costa_fit.observation_baseline
                + costa_fit.observation_linear_drift * normalized_time
            )
        costa_model, costa_history = train_costa(
            times,
            input_values,
            observed_bold,
            costa_physical_states,
            costa_physical_bold,
            CoSTAConfig(epochs=config.epochs),
        )
        costa_states, costa_bold = predict_costa(
            costa_model,
            times,
            input_values,
            costa_physical_states,
            costa_physical_bold,
        )
        validation = slice(validation_start, None)
        costa_metrics = {
            "mse": float(np.mean((observed_bold - costa_bold) ** 2)),
            "r2": r2_score(observed_bold, costa_bold),
            "validation_mse": float(
                np.mean((observed_bold[validation] - costa_bold[validation]) ** 2)
            ),
            "validation_r2": r2_score(
                observed_bold[validation], costa_bold[validation]
            ),
        }
        costa_parameters = {
            name: float(getattr(costa_params, name))
            for name in (
                "kappa", "gamma", "tau", "alpha", "E0", "V0", "eps",
                "nu0", "r0", "epsilon_r", "TE",
            )
        }
        if costa_fit is not None:
            costa_parameters["observation_baseline"] = costa_fit.observation_baseline
            costa_parameters["observation_linear_drift"] = (
                costa_fit.observation_linear_drift
            )
        save_run(
            output_dir / "costa_model.npz",
            times,
            input_values,
            observed_bold,
            costa_states,
            costa_bold,
            costa_metrics,
            costa_parameters,
        )
        np.savez(
            output_dir / "costa_loss.npz",
            **{key: np.asarray(value) for key, value in costa_history.items()},
        )
        metrics.update({f"costa_{key}": value for key, value in costa_metrics.items()})

    if mode == "compare":
        (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run_costa_extrapolation(
    times: np.ndarray,
    input_values: np.ndarray,
    observed_bold: np.ndarray,
    output_dir: Path,
    config: PINNConfig,
    fit_physical: bool = True,
    balloon_params: Optional[BalloonParams] = None,
    data_source: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate same-run future BOLD predictions with the final fifth untouched."""
    times = np.asarray(times, dtype=float)
    input_values = np.asarray(input_values, dtype=float)
    observed_bold = np.asarray(observed_bold, dtype=float)
    sample_count = len(times)
    if (
        times.ndim != 1
        or input_values.shape != (sample_count,)
        or observed_bold.shape != (sample_count,)
        or sample_count < 20
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Extrapolation evaluation requires at least 20 matching, ordered samples"
        )
    if not all(
        np.isfinite(values).all()
        for values in (times, input_values, observed_bold)
    ):
        raise ValueError("Extrapolation inputs must contain only finite values")

    train_end = int(np.floor(sample_count * 0.6))
    test_start = int(np.floor(sample_count * 0.8))
    validation_fraction = (test_start - train_end) / test_start
    if train_end < 10 or test_start - train_end < 2 or sample_count - test_start < 2:
        raise ValueError("The run is too short for the 60/20/20 temporal split")

    input_function = lambda time: float(np.interp(time, times, input_values))
    params = balloon_params or BalloonParams()
    if fit_physical:
        physical_fit = fit_balloon(
            times[:train_end],
            observed_bold[:train_end],
            input_function,
            params,
        )
        params = physical_fit.params
    else:
        physical_fit = None

    physical_time, physical_states, simulated_bold = simulate_balloon(
        input_function,
        params,
        (float(times[0]), float(times[-1])),
        float(np.median(np.diff(times))),
    )
    physical_states = np.vstack(
        [np.interp(times, physical_time, state) for state in physical_states]
    )
    physical_bold = np.interp(times, physical_time, simulated_bold)
    if physical_fit is not None:
        normalized_time = (times - np.mean(times[:train_end])) / max(
            float(times[train_end - 1] - times[0]), 1.0
        )
        physical_bold = (
            physical_bold
            + physical_fit.observation_baseline
            + physical_fit.observation_linear_drift * normalized_time
        )

    costa_model, history = train_costa(
        times[:test_start],
        input_values[:test_start],
        observed_bold[:test_start],
        physical_states[:, :test_start],
        physical_bold[:test_start],
        CoSTAConfig(epochs=config.epochs),
        validation_fraction=validation_fraction,
    )
    costa_states, costa_bold = predict_costa(
        costa_model,
        times,
        input_values,
        physical_states,
        physical_bold,
    )
    validation = slice(train_end, test_start)
    test = slice(test_start, sample_count)
    train_mean = float(np.mean(observed_bold[:train_end]))

    def summarize(target: np.ndarray, prediction: np.ndarray, prefix: str) -> None:
        metrics[f"{prefix}_mse"] = float(np.mean((target - prediction) ** 2))
        metrics[f"{prefix}_r2"] = r2_score(target, prediction)

    metrics: Dict[str, object] = {
        "evaluation": "same-participant, same-run chronological conditional extrapolation",
        "input_known_at_inference": True,
        "train_samples": train_end,
        "validation_samples": test_start - train_end,
        "test_samples": sample_count - test_start,
        "train_fraction": train_end / sample_count,
        "validation_fraction": (test_start - train_end) / sample_count,
        "test_fraction": (sample_count - test_start) / sample_count,
        "fit_physical_on_training_block_only": fit_physical,
        "epochs": config.epochs,
        "data_source": data_source,
    }
    test_target = observed_bold[test]
    summarize(test_target, physical_bold[test], "physical_test")
    summarize(test_target, costa_bold[test], "costa_test")
    summarize(test_target, np.full_like(test_target, train_mean), "train_mean_baseline_test")
    summarize(observed_bold[validation], physical_bold[validation], "physical_validation")
    summarize(observed_bold[validation], costa_bold[validation], "costa_validation")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extrapolation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    np.savez(
        output_dir / "extrapolation_predictions.npz",
        time=times,
        input=input_values,
        observed_bold=observed_bold,
        physical_bold=physical_bold,
        costa_bold=costa_bold,
        physical_states=physical_states,
        costa_states=costa_states,
        train_end=train_end,
        test_start=test_start,
    )
    np.savez(
        output_dir / "extrapolation_training_history.npz",
        **{key: np.asarray(value) for key, value in history.items()},
    )
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("physical", "pinn", "costa", "compare"), default="compare")
    parser.add_argument("--data", type=Path, help="CSV with time,input,bold")
    parser.add_argument(
        "--subject",
        help=f"BIDS subject ID; defaults to {DEFAULT_SUBJECT} when no other input is provided",
    )
    parser.add_argument("--data-root", type=Path, default=get_data_root())
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
        "--echo-time",
        type=float,
        help="BOLD echo time in seconds; defaults to the matching BIDS JSON EchoTime",
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
    parser.add_argument(
        "--extrapolation-test",
        action="store_true",
        help=(
            "Run CoSTA on a 60/20/20 chronological split: training, validation, "
            "and untouched final test block"
        ),
    )
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
    if args.echo_time is not None and args.echo_time <= 0:
        parser.error("--echo-time must be positive")
    if args.extrapolation_test and args.mode != "compare":
        parser.error("--extrapolation-test requires --mode compare")
    output_dir = args.output_dir or default_output_dir(args.mode, args.epochs)
    data_path = args.data
    echo_time = args.echo_time if args.echo_time is not None else BalloonParams().TE
    echo_time_source: Optional[Path] = None
    if subject is not None:
        subject_dir = args.data_root / subject / "func"
        bold_path = (
            discover_fmriprep_bold(args.data_root, subject)
            if args.fmriprep
            else args.bold_path or subject_dir / f"{subject}_task-stopsignal_bold.nii.gz"
        )
        events_path = args.events_path or discover_events_tsv(
            args.data_root, subject
        )
        if args.echo_time is None:
            discovered_echo_time, echo_time_source = discover_echo_time(
                args.data_root, subject, "stopsignal", bold_path
            )
            if discovered_echo_time is None:
                warnings.warn(
                    "No matching BIDS JSON EchoTime was found; using the model "
                    f"default TE={echo_time:g}s. Pass --echo-time to override.",
                    stacklevel=2,
                )
            else:
                echo_time = discovered_echo_time
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
            and bold_path.name.endswith(
                ("_preproc.nii.gz", "_desc-preproc_bold.nii.gz")
            )
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
            "echo_time_seconds": echo_time,
            "echo_time_source": (
                str(echo_time_source)
                if echo_time_source is not None
                else "command-line override"
                if args.echo_time is not None
                else "model default"
            ),
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
    if args.extrapolation_test:
        if data_path is not None:
            times, input_values, observed_bold = load_csv(data_path)
        elif args.synthetic:
            times, input_values, observed_bold = synthetic_data()
        else:
            parser.error("--extrapolation-test requires subject, --data, or --synthetic input")
        metrics = run_costa_extrapolation(
            times,
            input_values,
            observed_bold,
            output_dir,
            PINNConfig(epochs=args.epochs),
            fit_physical=not args.no_fit_physical,
            balloon_params=BalloonParams(TE=echo_time),
            data_source=str(data_path) if data_path is not None else "synthetic",
        )
        print(json.dumps(metrics, indent=2))
        print("Ran extrapolation test successfully")
        return
    metrics = run(
        args.mode,
        data_path,
        output_dir,
        PINNConfig(epochs=args.epochs),
        fit_physical=not args.no_fit_physical,
        use_synthetic=args.synthetic,
        balloon_params=BalloonParams(TE=echo_time),
    )
    print(json.dumps(metrics, indent=2))
    print("Ran successfully")


if __name__ == "__main__":
    main()