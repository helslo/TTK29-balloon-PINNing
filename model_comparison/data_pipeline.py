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
    """Average event-boxcar drive over each fMRI acquisition interval.

    Integrating each interval preserves events that occur between volume
    timestamps instead of dropping them during point sampling.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("times must contain at least two finite samples")
    intervals = np.diff(times)
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0]):
        raise ValueError("times must be strictly increasing and uniformly sampled")
    repetition_time = float(intervals[0])

    selected = events[events["trial_type"].eq(trial_type)]
    if outcome is not None and "TrialOutcome" in selected:
        selected = selected[selected["TrialOutcome"].eq(outcome)]
    values = np.zeros(len(times), dtype=float)
    for event in selected.itertuples(index=False):
        onset = float(event.onset)
        duration = float(event.duration)
        if duration <= 0:
            continue
        interval_start = times
        interval_end = times + repetition_time
        overlap = np.maximum(
            0.0,
            np.minimum(interval_end, onset + duration) - np.maximum(interval_start, onset),
        )
        values += overlap / repetition_time
    return values


def normalize_bold(bold: np.ndarray) -> np.ndarray:
    """Convert raw ROI intensity to fractional change from its whole-run median."""
    bold = np.asarray(bold, dtype=float)
    if bold.ndim != 1 or len(bold) == 0 or not np.isfinite(bold).all():
        raise ValueError("bold must be a non-empty finite one-dimensional signal")
    baseline = float(np.median(bold))
    if not np.isfinite(baseline) or baseline == 0.0:
        raise ValueError("ROI baseline must be finite and non-zero")
    return (bold - baseline) / baseline


def load_subject_timeseries(
    bold_path: Path,
    events_path: Path,
    roi_mask_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a baseline-normalized ROI trace and GO input from one subject.

    ``roi_mask_path`` must already be aligned to the BOLD image. If no ROI mask
    is provided, an EPI-derived brain mask is computed and the whole-brain mean
    trace is extracted. The raw mean intensity is retained until fractional
    change is computed; detrending or standardizing first would invalidate the
    intensity baseline. Event durations are averaged over each TR interval.
    """
    bold_image = nib.load(bold_path)
    if len(bold_image.shape) != 4:
        raise ValueError(f"BOLD image must be four-dimensional: {bold_path}")
    repetition_time = float(bold_image.header.get_zooms()[3])
    times = np.arange(bold_image.shape[3], dtype=float) * repetition_time
    events = pd.read_csv(events_path, sep="\t")
    mask_img = str(roi_mask_path) if roi_mask_path is not None else None
    masker = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False)
    extracted = masker.fit_transform(str(bold_path))
    if extracted.shape[1] == 0:
        raise ValueError(f"ROI mask contains no voxels in the BOLD image: {roi_mask_path}")
    bold = np.asarray(extracted.mean(axis=1), dtype=float)
    return times, build_event_input(times, events), normalize_bold(bold)


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