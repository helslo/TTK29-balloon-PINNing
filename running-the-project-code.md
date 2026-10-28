# Running the project code

The model workflows live in the `model_comparison` package, parallel to
`sensor_placement`. For the easiest comparison run, edit the settings at the top
of `run_comparison_analysis.py` and execute it from the repository root:

```bash
uv run python run_comparison_analysis.py
```

The script defaults to a full comparison with 2000 epochs on deterministic
synthetic data, and saves the outputs in a timestamped folder under `results/`.
Set `MODE` to `"physical"`, `"pinn"`, or `"costa"` to run just one model.
`DATA_FILE`, `SUBJECT`, `ROI_MASK`, and `OUTPUT_DIR` are optional settings.

The project CLI remains available when you prefer command-line options:

## Quick start

```bash
uv run python -m ttk29_balloon_pinning physical --epochs 2000
uv run python -m ttk29_balloon_pinning pinn --epochs 2000
uv run python -m ttk29_balloon_pinning costa --epochs 2000
uv run python -m ttk29_balloon_pinning compare --epochs 2000
uv run python -m ttk29_balloon_pinning sensor --k-max 10
uv run python -m ttk29_balloon_pinning latest --plot
```

What each one does:
- `physical`: fits the classical Balloon-Windkessel model and saves a `.npz` result archive under `results/`
- `pinn`: trains the physics-informed neural network and saves the fitted output + loss curves
- `costa`: trains the residual-correction hybrid model
- `compare`: runs the physical model, PINN, and CoSTA on the same data and writes a combined comparison summary
- `sensor`: runs the sparse sensor placement / compressed sensing analysis and saves CSV + figure outputs under `results/sensor_placement/`
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
uv run python -m sensor_placement.run --k-max 10 --output-dir results/sensor_placement
```

## Default behaviour

- `--epochs` defaults to `2000`.
- `--output-dir` defaults to a timestamped folder under `results/` for each pipeline.
- When plotting or summarizing, the project prefers the newest result under `results/` unless an explicit path is supplied.
- The sensor workflow defaults to synthetic ROI data when no real ROI HDF5 file is available.

## Typical analysis goals

- Use `compare` to compare the baseline physical model against the hybrid methods.
- Use `pinn` with a high epoch budget to make the PINN training effect visible.
- Use `costa` with a lower or moderate number of epochs to show the hybrid correction can converge quickly.
- Use `sensor` to test how many selected ROIs are needed before reconstruction quality drops sharply.
