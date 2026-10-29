"""Load, clean, cache, and split fMRIPrep BOLD runs for sensor placement."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_to_output
from nilearn.image import resample_img
from nilearn.maskers import NiftiMasker


@dataclass
class RunFiles:
    """Paths and identifying BIDS entities for one preprocessed functional run."""

    participant_id: str
    run_id: str
    bold_path: Path
    mask_path: Path
    confounds_path: Path
    metadata_path: Path


@dataclass
class SensorRun:
    """A cleaned voxel-by-time matrix and its common-space masker."""

    participant_id: str
    run_id: str
    X: np.ndarray
    tr: float
    masker: NiftiMasker
    cache_path: Path


def _read_participant_status(data_root: Path) -> dict[str, str]:
    """Read diagnoses so the default cohort can exclude non-control participants."""
    participants_path = data_root / "participants.tsv"
    if not participants_path.exists():
        return {}
    participants = pd.read_csv(participants_path, sep="\t")
    if not {"participant_id", "diagnosis"}.issubset(participants.columns):
        raise ValueError(f"{participants_path} must contain participant_id and diagnosis columns")
    return dict(zip(participants["participant_id"].astype(str), participants["diagnosis"].astype(str)))


def _sidecar_for_bold(bold_path: Path, data_root: Path, task: str) -> Path:
    """Resolve the BIDS JSON sidecar needed for the run's actual repetition time."""
    exact_sidecar = bold_path.with_name(bold_path.name.replace(".nii.gz", ".json"))
    if exact_sidecar.is_file():
        return exact_sidecar
    candidates = sorted(bold_path.parent.glob("*_bold.json"))
    candidates = [path for path in candidates if f"task-{task}" in path.name]
    if candidates:
        return candidates[0]
    participant_id = bold_path.name.split("_")[0]
    raw_func = data_root / participant_id / "func"
    candidates = sorted(raw_func.glob(f"*task-{task}*_bold.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No BIDS bold JSON sidecar found for {bold_path}")


def _confounds_for_bold(bold_path: Path) -> Path:
    """Match the fMRIPrep confounds table by its run-level BIDS entities."""
    prefix = bold_path.name.split("_space-")[0]
    direct = bold_path.with_name(f"{prefix}_desc-confounds_timeseries.tsv")
    if direct.exists():
        return direct
    candidates = sorted(bold_path.parent.glob(f"{prefix}*confounds*.tsv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No fMRIPrep confounds TSV found for {bold_path}")


def discover_runs(
    data_root: str | Path = "data/ds000030",
    task: str = "rest",
    max_participants: int = 20,
    controls_only: bool = True,
    max_mean_fd: float = 0.5,
) -> tuple[list[RunFiles], list[dict[str, str | float]]]:
    """Find matching MNI-space fMRIPrep runs and exclude high-motion runs."""
    root = Path(data_root)
    if max_participants < 1:
        raise ValueError("max_participants must be positive")
    derivative_root = root / "derivatives" / "fmriprep"
    if not derivative_root.is_dir():
        raise FileNotFoundError(
            f"fMRIPrep derivatives were not found at {derivative_root}. "
            "Run fMRIPrep for the requested task or pass --data-root pointing "
            "to a BIDS dataset containing derivatives/fmriprep."
        )

    diagnosis = _read_participant_status(root)
    if controls_only and not diagnosis:
        raise FileNotFoundError(
            f"{root / 'participants.tsv'} is required to select healthy controls; "
            "use --include-all-diagnoses to skip diagnosis filtering."
        )
    bold_paths = sorted(
        derivative_root.glob(
            f"sub-*/func/*task-{task}*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    )
    if not bold_paths:
        raise FileNotFoundError(
            f"No MNI-space preprocessed BOLD runs for task-{task} found under {derivative_root}"
        )

    participants = sorted({path.name.split("_")[0] for path in bold_paths})
    if controls_only and diagnosis:
        participants = [pid for pid in participants if diagnosis.get(pid, "").upper() == "CONTROL"]
    participants = participants[:max_participants]
    selected = set(participants)
    runs: list[RunFiles] = []
    exclusions: list[dict[str, str | float]] = []
    for bold_path in bold_paths:
        participant_id = bold_path.name.split("_")[0]
        if participant_id not in selected:
            continue
        mask_path = bold_path.with_name(
            bold_path.name.replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz")
        )
        if not mask_path.is_file():
            exclusions.append({"run": bold_path.name, "reason": "missing brain mask"})
            continue
        confounds_path = _confounds_for_bold(bold_path)
        confounds = pd.read_csv(confounds_path, sep="\t")
        if "framewise_displacement" not in confounds:
            exclusions.append({"run": bold_path.name, "reason": "missing framewise_displacement"})
            continue
        mean_fd = float(confounds["framewise_displacement"].mean())
        if not np.isfinite(mean_fd):
            exclusions.append({"run": bold_path.name, "reason": "invalid mean framewise displacement"})
            continue
        if mean_fd > max_mean_fd:
            exclusions.append(
                {"run": bold_path.name, "reason": "mean FD above threshold", "mean_fd": mean_fd}
            )
            continue
        metadata_path = _sidecar_for_bold(bold_path, root, task)
        run_id = re.sub(r".*_task-" + re.escape(task) + r"_?", "", bold_path.name)
        runs.append(
            RunFiles(
                participant_id,
                run_id,
                bold_path,
                mask_path,
                confounds_path,
                metadata_path,
            )
        )
    if not runs:
        raise ValueError(
            f"No eligible task-{task} runs remain after participant and motion filtering"
        )
    return runs, exclusions


def _load_tr(metadata_path: Path) -> float:
    """Read the run repetition time from its BIDS JSON sidecar."""
    metadata = json.loads(metadata_path.read_text())
    tr = float(metadata["RepetitionTime"])
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError(f"Invalid RepetitionTime in {metadata_path}: {tr}")
    return tr


def _confound_columns(confounds: pd.DataFrame, include_global_signal: bool) -> pd.DataFrame:
    """Select motion, derivative, white-matter, CSF, and optional global signal."""
    motion = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    columns = motion + [f"{name}_derivative1" for name in motion] + ["white_matter", "csf"]
    if include_global_signal:
        columns.append("global_signal")
    missing = [column for column in columns if column not in confounds.columns]
    if missing:
        raise ValueError(f"Confounds table is missing required columns: {', '.join(missing)}")
    selected = confounds[columns].apply(pd.to_numeric, errors="coerce")
    return selected.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _common_grid(mask_path: Path, voxel_size_mm: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Build one 4-mm-oriented target grid from the first MNI brain mask."""
    mask_img = nib.load(mask_path)
    reference = resample_to_output(
        mask_img,
        voxel_sizes=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
        order=0,
    )
    return np.asarray(reference.affine), tuple(int(size) for size in reference.shape[:3])


def _resampled_mask(
    mask_path: Path,
    target_affine: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Resample a run brain mask to the common coarse MNI grid."""
    image = resample_img(
        nib.load(mask_path),
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    return np.asarray(image.dataobj) > 0


def _cache_key(
    run: RunFiles,
    tr: float,
    voxel_size_mm: float,
    smoothing_fwhm_mm: float,
    include_global_signal: bool,
    common_mask_hash: str,
) -> str:
    """Hash source versions and cleaning options to invalidate stale caches."""
    sources = []
    for path in (run.bold_path, run.mask_path, run.confounds_path, run.metadata_path):
        stat = path.stat()
        sources.append((str(path), stat.st_size, stat.st_mtime_ns))
    payload = repr(
        (
            sources,
            tr,
            voxel_size_mm,
            smoothing_fwhm_mm,
            include_global_signal,
            common_mask_hash,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def load_clean_runs(
    data_root: str | Path = "data/ds000030",
    task: str = "rest",
    max_participants: int = 20,
    controls_only: bool = True,
    max_mean_fd: float = 0.5,
    voxel_size_mm: float = 4.0,
    smoothing_fwhm_mm: float = 6.0,
    include_global_signal: bool = False,
    cache_dir: str | Path = "data/cleaned",
) -> tuple[list[SensorRun], list[dict[str, str | float]]]:
    """Clean eligible fMRIPrep runs, use one group mask, and cache float32 matrices."""
    root = Path(data_root)
    run_files, exclusions = discover_runs(
        root, task, max_participants, controls_only, max_mean_fd
    )
    target_affine, target_shape = _common_grid(run_files[0].mask_path, voxel_size_mm)
    masks = [
        _resampled_mask(run.mask_path, target_affine, target_shape)
        for run in run_files
    ]
    common_mask = np.logical_and.reduce(masks)
    if not np.any(common_mask):
        raise ValueError("The intersection of run brain masks is empty on the common MNI grid")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    common_mask_hash = hashlib.sha256(np.packbits(common_mask).tobytes()).hexdigest()[:20]
    common_mask_path = cache_root / f"{task}_group-mask_{voxel_size_mm:g}mm.nii.gz"
    nib.save(
        nib.Nifti1Image(common_mask.astype(np.uint8), target_affine),
        common_mask_path,
    )

    runs: list[SensorRun] = []
    for run in run_files:
        tr = _load_tr(run.metadata_path)
        cache_key = _cache_key(
            run, tr, voxel_size_mm, smoothing_fwhm_mm, include_global_signal, common_mask_hash
        )
        safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run.run_id)
        cache_path = cache_root / f"{run.participant_id}_{safe_run_id}_{cache_key}.npy"
        masker = NiftiMasker(
            mask_img=str(common_mask_path),
            smoothing_fwhm=smoothing_fwhm_mm,
            standardize="zscore_sample",
            detrend=True,
            high_pass=0.01,
            t_r=tr,
            target_affine=target_affine,
            target_shape=target_shape,
            dtype=np.float32,
        )
        if cache_path.is_file():
            X = np.load(cache_path, mmap_mode="r")
            masker.fit()
        else:
            confounds = _confound_columns(
                pd.read_csv(run.confounds_path, sep="\t"), include_global_signal
            )
            cleaned_time_by_voxel = masker.fit_transform(
                str(run.bold_path), confounds=confounds
            )
            X = np.asarray(cleaned_time_by_voxel.T, dtype=np.float32)
            if not np.isfinite(X).all():
                raise ValueError(f"Cleaned BOLD contains non-finite values: {run.bold_path}")
            np.save(cache_path, X)
        if not np.isfinite(X).all():
            raise ValueError(f"Cached cleaned BOLD contains non-finite values: {cache_path}")
        if X.ndim != 2 or X.shape[0] != int(common_mask.sum()):
            raise ValueError(
                f"Cached data shape for {run.bold_path} does not match the common brain mask"
            )
        runs.append(SensorRun(run.participant_id, run.run_id, X, tr, masker, cache_path))
    return runs, exclusions


def temporal_holdout_split(
    X: np.ndarray, holdout_fraction: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Split time indices chronologically so future samples remain held out."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")
    split_index = int(round((1.0 - holdout_fraction) * X.shape[1]))
    if split_index < 1 or split_index >= X.shape[1]:
        raise ValueError("At least two time points are required for a temporal split")
    return np.arange(split_index), np.arange(split_index, X.shape[1])


def participant_holdout_split(
    participant_ids: Iterable[str], n_folds: int = 5
) -> list[tuple[list[str], list[str]]]:
    """Create deterministic folds whose test participants never enter training."""
    ids = sorted(set(participant_ids))
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if len(ids) < 2:
        raise ValueError("At least two participants are required for participant hold-out")
    folds = min(n_folds, len(ids))
    chunks = np.array_split(np.asarray(ids), folds)
    return [
        ([pid for pid in ids if pid not in set(test.tolist())], test.tolist())
        for test in chunks
    ]
