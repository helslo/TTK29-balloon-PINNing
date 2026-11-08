# Running the project code

The model workflows live in the `model_comparison` package, parallel to
`sensor_placement`. For the easiest comparison run, edit the settings at the top
of `run_comparison_analysis.py` and execute it from the repository root:

```bash
uv run python run_comparison_analysis.py
```

The script defaults to a full comparison with 2000 epochs using the real
`sub-10159` scan under `data/ds000030` and saves outputs in a timestamped folder
under `results/`. With no `ROI_MASK`, raw BOLD uses an explicit EPI-derived
brain mask and fMRIPrep BOLD uses its brain mask when available. Set
`MODE` to `"physical"`, `"pinn"`, or `"costa"` to run just one model.
`DATA_FILE`, `SUBJECT`, `ROI_MASK`, and `OUTPUT_DIR` are optional settings.
For synthetic test data, use
`uv run python -m ttk29_balloon_pinning compare --synthetic`.

The real-data loader preserves raw intensity until it converts the ROI signal
to fractional change using the whole-run median, interpolates isolated
one-volume spikes, then removes linear drift. Event durations for labeled GO
and STOP trials are averaged over each TR interval regardless of trial
outcome, preserving events between volume timestamps and avoiding a
successful-GO-only input.

## fMRIPrep and ROI comparison

By default, the comparison reads the raw stop-signal BOLD and events files
from `data/ds000030/<subject>/func/`. For example:

```bash
uv run python -m ttk29_balloon_pinning compare --subject sub-10159 --epochs 2000
```

This uses `sub-10159_task-stopsignal_bold.nii.gz` and
`sub-10159_task-stopsignal_events.tsv` directly. `--bold-path` and
`--events-path` can override either raw file. Without an ROI mask this runs on
the raw EPI-derived brain-mean signal.

The Schaefer atlas is in MNI space and cannot be correctly aligned to native
raw BOLD using resampling alone. For a Schaefer network ROI, normalized
MNI-space BOLD and its fMRIPrep confounds are required. If those derivatives
are available under `data/ds000030/derivatives/fmriprep`, run:

```bash
uv run python -m ttk29_balloon_pinning compare \
  --fmriprep --subject sub-10159 \
  --schaefer-network Cont \
  --epochs 2000
```

The `--fmriprep` option discovers a unique task-stopsignal preprocessed BOLD
image, the associated confounds TSV, and its brain mask if present. It stops
with an error if multiple runs match rather than choosing an arbitrary run.
For a manually selected run, use `--bold-path /path/to/desc-preproc_bold.nii.gz`
and `--confounds-path /path/to/desc-confounds_timeseries.tsv` instead. Use
`--events-path` if your events file is not at the raw BIDS default location.

`--schaefer-network` fetches/caches the 400-parcel, 7-network Schaefer 2018
atlas using Nilearn, then saves a binary network mask and its parcel labels in
the output directory. Supported networks include `Cont` and `SalVentAttn`;
`--schaefer-hemisphere left|right|both` selects the hemisphere (default:
`both`). This atlas is in MNI space, so the option rejects raw native-space
BOLD. A manually supplied `--roi-mask` remains available and is resampled to
the BOLD grid with nearest-neighbor interpolation; the mask must overlap the
BOLD image. If neither ROI option is supplied, the pipeline uses the
fMRIPrep brain mask when available, otherwise computes an EPI brain mask. The
selected signal is the mean across mask voxels. Global-signal regression is
excluded by default; available motion, tissue, aCompCor, cosine, and
non-steady-state regressors are used when a confounds TSV is supplied. The
output `timeseries_metadata.json` records the selected inputs and confound
columns. Schaefer networks are candidates, not validated stop-signal ROIs;
evaluate each separately against the whole-brain baseline on the same scan.

The project CLI remains available when you prefer command-line options:

## Quick start

```bash
uv run python -m ttk29_balloon_pinning physical --epochs 2000
uv run python -m ttk29_balloon_pinning pinn --epochs 2000
uv run python -m ttk29_balloon_pinning costa --epochs 2000
uv run python -m ttk29_balloon_pinning compare --epochs 2000
uv run python -m ttk29_balloon_pinning compare --synthetic
uv run python -m ttk29_balloon_pinning sensor --task rest
uv run python -m ttk29_balloon_pinning latest --plot
```

What each one does:
- `physical`: fits the classical Balloon-Windkessel model and saves a `.npz` result archive under `results/`
- `pinn`: trains the physics-informed neural network and saves the fitted output + loss curves
- `costa`: trains the residual-correction hybrid model
- `compare`: runs the physical model, PINN, and CoSTA on the same real data by default and writes a combined comparison summary; use `--synthetic` for generated test data
- `sensor`: runs voxel sensor placement on fMRIPrep derivatives and saves CSV + figure outputs under `results/sensor_placement/`; derivatives are required and synthetic data is never substituted
- `latest --plot`: opens the newest saved run in `results/` and creates a summary plot automatically

## Minimal result workflow

1. Run one of the model pipelines above.
2. Use the latest saved run automatically:

```bash
uv run python -m ttk29_balloon_pinning summary --latest
```

3. Or inspect a specific run directory or archive directly:

```bash
uv run python read_results.py results/your_run_directory
uv run python read_results.py results/your_run_directory/physical_model.npz --plot
```

## Package usage

The comparison implementation can also be imported from the package module:

```python
from model_comparison.run import run
```

The package also has a module CLI:

```bash
uv run python -m model_comparison --mode compare --epochs 2000
uv run python -m sensor_placement.run --task rest --output-dir results/sensor_placement
```

## Default behaviour

- `--epochs` defaults to `2000`.
- `--output-dir` defaults to a timestamped folder under `results/` for each pipeline.
- Comparison commands default to the real `sub-10159` BIDS scan; without `--roi-mask`, they extract an EPI-masked whole-brain mean.
- `latest --plot` uses the newest complete comparison run when available and plots all model archives in that run; if there is no comparison run, it falls back to the newest individual model archive.
- Pass a run directory explicitly to summarize or plot a particular run.
- The sensor workflow expects MNI-space fMRIPrep derivatives under `data/ds000030/derivatives/fmriprep`; the raw scans in `data/` alone are not sufficient.
- Sensor placement uses a 4 mm common brain mask, 6 mm smoothing, confound regression, 0.01 Hz high-pass filtering, and z-scoring by default. `--global-signal` optionally adds global-signal regression.
- `--include-l1` enables the slower 3D-DCT FISTA methods on a reduced k grid.

## Typical analysis goals

- Use `compare` to compare the baseline physical model against the hybrid methods.
- Use `pinn` with a high epoch budget to make the PINN training effect visible.
- Use `costa` with a lower or moderate number of epochs to show the hybrid correction can converge quickly.
- Use `sensor` to test how many selected voxels are needed before reconstruction quality drops sharply.
