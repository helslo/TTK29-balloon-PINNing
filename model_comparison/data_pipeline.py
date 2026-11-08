"""Convert BIDS BOLD and event files into the shared model input format."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets as nilearn_datasets
from nilearn import image as nilearn_image
from nilearn.masking import compute_epi_mask
from nilearn.maskers import NiftiMasker
from nilearn.signal import clean as clean_signal
from scipy.signal import detrend as linear_detrend

SCHAEFER_NETWORKS = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)


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


def select_fmriprep_confounds(confounds: pd.DataFrame) -> pd.DataFrame:
    """Select and standardize standard motion, tissue, CompCor, and drift regressors.

    Global signal regression is deliberately excluded because it can remove
    neural signal and is not an uncontroversial default.
    """
    motion = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
    columns = [
        name
        for name in motion
        + tuple(f"{name}_derivative1" for name in motion)
        + tuple(f"{name}_power2" for name in motion)
        + tuple(f"{name}_derivative1_power2" for name in motion)
        + ("white_matter", "csf")
        if name in confounds.columns
    ]
    columns.extend(
        name
        for name in sorted(
            name for name in confounds.columns if name.startswith("a_comp_cor_")
        )[:6]
    )
    columns.extend(
        name for name in sorted(confounds.columns) if name.startswith("cosine")
    )
    columns.extend(
        name
        for name in sorted(confounds.columns)
        if name.startswith("non_steady_state_outlier")
    )
    if not columns:
        raise ValueError(
            "Confounds table has no supported fMRIPrep regressors "
            "(motion, white_matter/csf, a_comp_cor, cosine, or "
            "non_steady_state_outlier columns)"
        )

    selected = confounds.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    selected = selected.loc[:, selected.nunique(dropna=False) > 1]
    if selected.shape[1] == 0:
        raise ValueError("Selected fMRIPrep confounds are all constant")
    scales = selected.std(axis=0, ddof=1)
    selected = selected.loc[:, scales.notna() & scales.gt(0)]
    if selected.shape[1] == 0:
        raise ValueError("Selected fMRIPrep confounds cannot be standardized")
    return (selected - selected.mean(axis=0)) / selected.std(axis=0, ddof=1)


def create_schaefer_network_mask(
    output_path: Path,
    network: str = "Cont",
    hemisphere: str = "both",
) -> Tuple[Path, Tuple[str, ...]]:
    """Fetch Schaefer 2018 and save a binary mask for one Yeo-7 network.

    The Schaefer atlas is in MNI space. The resulting mask should therefore
    only be used with spatially normalized BOLD data.
    """
    if network not in SCHAEFER_NETWORKS:
        raise ValueError(
            f"Unknown Schaefer network {network!r}; choose from "
            f"{', '.join(SCHAEFER_NETWORKS)}"
        )
    if hemisphere not in ("left", "right", "both"):
        raise ValueError("hemisphere must be 'left', 'right', or 'both'")

    atlas = nilearn_datasets.fetch_atlas_schaefer_2018(
        n_rois=400,
        yeo_networks=7,
        resolution_mm=2,
    )
    atlas_image = nib.load(atlas.maps)
    atlas_values = np.asarray(atlas_image.dataobj)
    hemispheres = {"left": {"LH"}, "right": {"RH"}, "both": {"LH", "RH"}}
    selected_labels = tuple(
        label
        for label in atlas.labels
        if label != "Background"
        and len(label.split("_")) >= 4
        and label.split("_")[1] in hemispheres[hemisphere]
        and label.split("_")[2] == network
    )
    if not selected_labels:
        raise ValueError(
            f"No Schaefer parcels found for network {network!r}, "
            f"hemisphere {hemisphere!r}"
        )

    label_indices = {
        index
        for index, label in enumerate(atlas.labels)
        if label in selected_labels
    }
    mask = np.isin(atlas_values, list(label_indices)).astype(np.uint8)
    if not np.any(mask):
        raise ValueError(
            f"Schaefer atlas contains no voxels for network {network!r}, "
            f"hemisphere {hemisphere!r}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = nib.Nifti1Image(mask, atlas_image.affine, atlas_image.header)
    nib.save(mask_image, output_path)
    metadata = {
        "atlas": "Schaefer 2018",
        "n_rois": 400,
        "yeo_networks": 7,
        "resolution_mm": 2,
        "template": "MNI152",
        "network": network,
        "hemisphere": hemisphere,
        "labels": list(selected_labels),
        "mask_type": "binary union",
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return output_path, selected_labels


def _resample_mask(mask_path: Path, bold_image: nib.spatialimages.SpatialImage):
    """Resample a binary ROI mask to the functional image's voxel grid."""
    mask_image = nib.load(mask_path)
    if len(mask_image.shape) != 3:
        raise ValueError(f"Mask must be three-dimensional: {mask_path}")
    resampled = nilearn_image.resample_to_img(
        mask_image,
        bold_image,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    mask_data = np.asarray(resampled.dataobj)
    if not np.isfinite(mask_data).all() or not np.any(mask_data > 0):
        raise ValueError(
            f"Mask has no finite non-zero voxels overlapping the BOLD image: {mask_path}"
        )
    return resampled


def load_subject_timeseries(
    bold_path: Path,
    events_path: Path,
    roi_mask_path: Optional[Path] = None,
    confounds_path: Optional[Path] = None,
    brain_mask_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a baseline-normalized ROI trace and GO input from one subject.

    ``roi_mask_path`` is resampled to the BOLD grid with nearest-neighbor
    interpolation. If no ROI mask is provided, ``brain_mask_path`` is used
    when available, otherwise an explicit EPI brain mask is computed. Optional
    fMRIPrep confounds are regressed from the fractional-change trace with
    linear detrending. Labeled GO and STOP events are included regardless of
    outcome, and event durations are averaged over each TR interval.
    """
    bold_image = nib.load(bold_path)
    if len(bold_image.shape) != 4:
        raise ValueError(f"BOLD image must be four-dimensional: {bold_path}")
    repetition_time = float(bold_image.header.get_zooms()[3])
    times = np.arange(bold_image.shape[3], dtype=float) * repetition_time
    events = pd.read_csv(events_path, sep="\t")
    if roi_mask_path is not None:
        mask_img = _resample_mask(roi_mask_path, bold_image)
    elif brain_mask_path is not None:
        mask_img = _resample_mask(brain_mask_path, bold_image)
    else:
        mask_img = compute_epi_mask(bold_image)
    masker = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False)
    extracted = masker.fit_transform(str(bold_path))
    if extracted.shape[1] == 0:
        selected_mask = roi_mask_path or brain_mask_path
        raise ValueError(f"Mask contains no voxels in the BOLD image: {selected_mask}")
    bold = np.asarray(extracted.mean(axis=1), dtype=float)
    despiked_bold, _ = interpolate_isolated_spikes(bold)
    normalized_bold = normalize_bold(despiked_bold)
    if confounds_path is not None:
        confounds = pd.read_csv(confounds_path, sep="\t")
        if len(confounds) != len(normalized_bold):
            raise ValueError(
                f"Confounds row count ({len(confounds)}) does not match the "
                f"number of BOLD volumes ({len(normalized_bold)}): {confounds_path}"
            )
        selected_confounds = select_fmriprep_confounds(confounds)
        if selected_confounds.shape[1] >= len(normalized_bold) - 2:
            raise ValueError(
                "The selected confounds leave too few time points to estimate "
                f"a BOLD trace ({selected_confounds.shape[1]} regressors, "
                f"{len(normalized_bold)} volumes)"
            )
        normalized_bold = clean_signal(
            normalized_bold[:, np.newaxis],
            confounds=selected_confounds,
            detrend=True,
            standardize=False,
            standardize_confounds=False,
            filter=None,
        )[:, 0]
    else:
        normalized_bold = linear_detrend(normalized_bold, type="linear")
    return times, build_event_input(times, events), normalized_bold


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