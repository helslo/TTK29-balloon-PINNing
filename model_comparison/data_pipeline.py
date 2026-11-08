"""Convert BIDS BOLD and event files into the shared model input format."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.masking import compute_epi_mask
from nilearn.maskers import NiftiMasker
from scipy.signal import detrend as linear_detrend


def build_event_input(
    times: np.ndarray,
    events: pd.DataFrame,
    trial_type: Optional[str] = None,
    outcome: Optional[str] = None,
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

    selected = events[events["trial_type"].notna()]
    if trial_type is not None:
        selected = selected[selected["trial_type"].eq(trial_type)]
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


def interpolate_isolated_spikes(
    signal: np.ndarray,
    threshold: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate isolated one-sample excursions using robust difference scale.

    An excursion is flagged only when two adjacent first differences are both
    extreme and have opposite signs. This targets one-frame spikes without
    flattening sustained changes in the signal.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < 3 or not np.isfinite(signal).all():
        raise ValueError("signal must contain at least three finite one-dimensional samples")
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and positive")

    differences = np.diff(signal)
    center = float(np.median(differences))
    scale = max(
        1.4826 * float(np.median(np.abs(differences - center))),
        np.finfo(float).eps * max(float(np.max(np.abs(signal))), 1.0),
    )

    centered = differences - center
    spike_mask = (
        (np.abs(centered[:-1]) > threshold * scale)
        & (np.abs(centered[1:]) > threshold * scale)
        & (centered[:-1] * centered[1:] < 0)
    )
    spike_indices = np.flatnonzero(spike_mask) + 1
    cleaned = signal.copy()
    cleaned[spike_indices] = (signal[spike_indices - 1] + signal[spike_indices + 1]) / 2.0
    return cleaned, spike_indices


def load_subject_timeseries(
    bold_path: Path,
    events_path: Path,
    roi_mask_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a baseline-normalized ROI trace and GO input from one subject.

    ``roi_mask_path`` must already be aligned to the BOLD image. If no ROI mask
    is provided, an explicit EPI brain mask is computed and the brain mean
    trace is extracted. The raw mean intensity is retained until fractional
    change is computed; isolated single-volume spikes are interpolated before
    the normalized trace is linearly detrended. Labeled GO and STOP events are
    included regardless of outcome, and event durations are averaged over
    each TR interval.
    """
    bold_image = nib.load(bold_path)
    if len(bold_image.shape) != 4:
        raise ValueError(f"BOLD image must be four-dimensional: {bold_path}")
    repetition_time = float(bold_image.header.get_zooms()[3])
    times = np.arange(bold_image.shape[3], dtype=float) * repetition_time
    events = pd.read_csv(events_path, sep="\t")
    mask_img = (
        str(roi_mask_path)
        if roi_mask_path is not None
        else compute_epi_mask(bold_image)
    )
    masker = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False)
    extracted = masker.fit_transform(str(bold_path))
    if extracted.shape[1] == 0:
        raise ValueError(f"ROI mask contains no voxels in the BOLD image: {roi_mask_path}")
    bold = np.asarray(extracted.mean(axis=1), dtype=float)
    despiked_bold, _ = interpolate_isolated_spikes(bold)
    normalized_bold = normalize_bold(despiked_bold)
    return times, build_event_input(times, events), linear_detrend(normalized_bold, type="linear")


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