"""Convert BIDS BOLD and event files into the shared model input format."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker


def build_event_input(
    times: np.ndarray,
    events: pd.DataFrame,
    trial_type: str = "GO",
    outcome: Optional[str] = "SuccessfulGo",
) -> np.ndarray:
    """Sample event boxcars on the fMRI time grid."""
    selected = events[events["trial_type"].eq(trial_type)]
    if outcome is not None and "TrialOutcome" in selected:
        selected = selected[selected["TrialOutcome"].eq(outcome)]
    values = np.zeros(len(times), dtype=float)
    for event in selected.itertuples(index=False):
        onset = float(event.onset)
        duration = float(event.duration)
        values[(times >= onset) & (times < onset + duration)] += 1.0
    return values


def normalize_bold(bold: np.ndarray, baseline_volumes: int = 10) -> np.ndarray:
    """Convert ROI intensity to fractional baseline BOLD change."""
    if baseline_volumes <= 0 or baseline_volumes > len(bold):
        raise ValueError("baseline_volumes must be between 1 and the number of volumes")
    baseline = float(np.median(bold[:baseline_volumes]))
    if not np.isfinite(baseline) or baseline == 0.0:
        raise ValueError("ROI baseline must be finite and non-zero")
    return (bold - baseline) / baseline


def load_subject_timeseries(
    bold_path: Path,
    events_path: Path,
    roi_mask_path: Path,
    standardize: bool = False,
    baseline_volumes: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a baseline-normalized ROI trace and GO input from one subject.

    ``roi_mask_path`` must already be aligned to the BOLD image or to the same
    standard-space image supplied as ``bold_path``. It is deliberately explicit
    because resampling a standard atlas directly onto raw scanner-space BOLD
    is not a valid registration step.
    """
    bold_image = nib.load(bold_path)
    if len(bold_image.shape) != 4:
        raise ValueError(f"BOLD image must be four-dimensional: {bold_path}")
    repetition_time = float(bold_image.header.get_zooms()[3])
    times = np.arange(bold_image.shape[3], dtype=float) * repetition_time
    events = pd.read_csv(events_path, sep="\t")
    masker = NiftiMasker(mask_img=str(roi_mask_path), standardize=standardize, detrend=not standardize)
    extracted = masker.fit_transform(str(bold_path))
    if extracted.shape[1] == 0:
        raise ValueError(f"ROI mask contains no voxels in the BOLD image: {roi_mask_path}")
    bold = np.asarray(extracted.mean(axis=1), dtype=float)
    return times, build_event_input(times, events), normalize_bold(bold, baseline_volumes)


def write_timeseries_csv(
    path: Path,
    times: np.ndarray,
    input_values: np.ndarray,
    bold: np.ndarray,
) -> None:
    """Write the shared ``time,input,bold`` model input format."""
    if not (len(times) == len(input_values) == len(bold)):
        raise ValueError("times, input_values, and bold must have equal length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("time", "input", "bold"))
        writer.writerows(zip(times, input_values, bold))