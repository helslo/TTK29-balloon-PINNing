# TTK29-balloon-PINNing

# TTK29 Project: Hybrid Modeling of Cerebral Hemodynamics

## 1. Goal

Compare a **pure physical model** of the BOLD hemodynamic response against **three hybrid physics+ML approaches** (PINN, CoSTA, SINDy), all fit to real task-fMRI data from a single brain region.

This is a course project (TTK29, 3.75 study points) designed to build modeling skills that feed directly into a related masters project on BOLD-signal analysis in mental health disorders.

Relevant articles:
- https://www.sciencedirect.com/science/article/pii/S105381190090630X

## 2. Quick start and project workflow

The project now exposes a single, minimal CLI entry point that routes to the main pipelines without remembering many raw script names:

```bash
# full comparison with editable settings at the top of the runner
uv run python run_comparison_analysis.py

# physical model baseline
uv run python -m ttk29_balloon_pinning physical --epochs 2000

# PINN training
uv run python -m ttk29_balloon_pinning pinn --epochs 2000

# CoSTA residual correction
uv run python -m ttk29_balloon_pinning costa --epochs 2000

# full comparison of all methods on the same dataset
uv run python -m ttk29_balloon_pinning compare --epochs 2000

# sparse sensor placement / compressed sensing analysis
uv run python -m ttk29_balloon_pinning sensor --k-max 10

# summarize and plot the newest saved run automatically
uv run python -m ttk29_balloon_pinning latest --plot
```

If no explicit run path is given to the summary tools, they automatically resolve the newest directory beneath `results/` and plot that run by default. This keeps the workflow lightweight and avoids manually finding the last comparison folder after a training run.

The model comparison implementation and its Balloon, PINN, CoSTA, and BIDS
timeseries modules are organized in the `model_comparison` package. For a
simple, readable run, edit the constants at the top of
`run_comparison_analysis.py` (mode, epochs, input data, and output directory)
and execute that file. The package can also be run with
`python -m model_comparison`.

## 3. Theory primer: the math behind the models

### 3.1 Balloon-Windkessel model

The baseline model is a reduced dynamical system for the neurovascular response. Let `u(t)` denote the neuronal drive, and let `s(t)`, `f(t)`, `v(t)`, and `q(t)` denote the vasodilatory signal, flow, volume, and deoxyhemoglobin content. The system is

```text
ds/dt = ε u(t) - κ s - γ (f - 1)
df/dt = s
dv/dt = (f - v^(1/α)) / τ
dq/dt = [f * E(u) / E0 - (v^(1/α) / v) q] / τ
```

where `E(u)` is the flow-dependent oxygen extraction term. The BOLD signal is then computed from the static observation equation

```text
BOLD(t) = V0 [k1 (1 - q) + k2 (1 - q / v) + k3 (1 - v)]
```

with `k1`, `k2`, and `k3` determined by the echo time and standard hemodynamic constants. This gives a compact nonlinear ODE that links neuronal input to the measured hemodynamic response.

### 3.2 PINN formulation

A physics-informed neural network does not replace the Balloon model; it learns a surrogate state trajectory while embedding the residual of the governing ODE into the loss. If the network predicts `s_hat(t), f_hat(t), v_hat(t), q_hat(t)`, the training objective includes both a data-fit term and the ODE residual term:

```text
L = L_data + λ L_physics
```

where `L_data` measures the mismatch to observed BOLD, and `L_physics` penalizes the residual of the differential equations. This is useful because it combines direct measurements with a mechanistic inductive bias: the solution is not just “close to the data,” but also consistent with the physically informed dynamics.

### 3.3 CoSTA residual correction

CoSTA can be viewed as a hybrid model in which the physical balloon solution remains the backbone and a learned correction term accounts for the part the physics model misses. In effect, the model estimates

```text
y_hat(t) = y_physical(t) + r_θ(t)
```

where `r_θ(t)` is a learned residual correction. This makes the method more interpretable than a fully data-driven model because the mechanistic structure remains present while the neural network adapts to systematic bias or unmodeled dynamics.

### 3.4 Sparse sensor placement and compressed sensing

For ROI-based fMRI analysis, suppose each ROI is a row of a matrix `X ∈ R^{n_rois × n_time}` after z-scoring over time. The goal is to observe only `k << n_rois` rows (sensors) and reconstruct the full matrix from those observations. We therefore solve a reduced problem of the form

```text
y = M X,
X_hat = A y,
```

where `M` selects the measured rows and `A` maps measurements back to the full ROI space. In the simplest formulation, the low-rank basis from the SVD provides the dominant spatial modes, and QR pivoting chooses the rows that best span that subspace. LASSO-based compressed sensing adds a sparse regularizer in a cosine-transform basis to recover coefficients from a limited number of measurements.

This is the core of the sensor-placement workflow: choose a small set of informative ROIs, reconstruct the rest, and evaluate how quickly the error rises as `k` becomes smaller.

## 4. Core physical model: the Balloon(-Windkessel) model

Reference: Buxton, Wong & Frank (1998), *Magnetic Resonance in Medicine* 39(6):855-864, extended by Friston, Mechelli, Turner & Price (2000), *NeuroImage* 12(4):466-477.

The model is a small system of nonlinear ODEs relating a neuronal "drive" input `u(t)` to the observed BOLD signal, via intermediate state variables:
- `s` — vasodilatory signal
- `f` — normalized cerebral blood flow (CBF)
- `v` — normalized venous blood volume
- `q` — normalized deoxyhemoglobin content

BOLD signal is then a static nonlinear function of `v` and `q`. This is a **reduced-order model (ROM)** of the true vascular physiology — a useful framing to connect to the ROM material in the course.

## 3. Methods to compare

| Method | Role in this project |
|---|---|
| Balloon model (classical fit) | Baseline: fit ODE parameters directly to data via optimization |
| PINN | Fit a neural network with the balloon ODE residual embedded in the loss |
| CoSTA | Balloon model as physical core + neural network correction term for what the physical model misses |
| SINDy | Attempt to discover governing equations directly from data, without assuming balloon structure; compare discovered dynamics against the known balloon ODEs |

Comparison axes: fit accuracy (e.g. R², residuals), generalization to held-out trials/subjects, robustness to noise, and (time permitting) interpretability of what each hybrid method adds beyond the pure physical model.

*(DMD, ROM framing, and other course methods can be added as secondary analyses — see Section 7.)*

## 4. Dataset: LA5c / ds000030 (OpenNeuro)

Chosen because it's the same dataset used in the related masters project (shared preprocessing/data-handling pipeline), and because it is a large, well-documented, public task-fMRI dataset with existing preprocessed derivatives.

- Raw data: https://openneuro.org/datasets/ds000030/
- GitHub mirror: https://github.com/OpenNeuroDatasets/ds000030
- Preprocessed derivatives (fMRIPrep + FreeSurfer, all tasks): AWS S3 `s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/` (no-sign-request), described in Gorgolewski, Durnez & Poldrack (2017), *F1000Research* 6:1262, https://doi.org/10.12688/f1000research.11964.2
- Unthresholded stat maps: https://neurovault.org/collections/2606/

### Task: Stop-Signal Task

Chosen over the other LA5c tasks (task-switching, spatial working memory, BART, etc.) because its **"Go" trials** give a comparatively simple, well-timed, repeated motor event that can be used as the neuronal drive input `u(t)` for the balloon model — a much cleaner starting point than paradigms with heavier trial-type mixing or parametric modulators. "Stop" trials can be added later as a secondary regressor if there's time (Section 7).

### Subjects: Healthy controls only (primary analysis)

Kept to healthy controls for the primary model comparison, so that inter-subject vascular/neural coupling variability doesn't get entangled with the question this project is actually asking (how well do different modeling approaches fit hemodynamic dynamics?). Comparing model fits across diagnostic groups is a different, follow-on question — see Section 7 for how this could extend toward the masters project.

## 5. Region of interest: Left primary motor cortex (M1, hand area)

Chosen because:
- The Go-trial response is a button press, so M1 (contralateral to the responding hand) has a strong, well-localized, high-SNR BOLD response tightly time-locked to the task events — ideal for a first pass at fitting and comparing models.
- It's the same region used in the original Buxton/Friston balloon-model papers, so there's a direct literature baseline for expected parameter ranges.
- **Note:** most LA5c subjects are right-handed (respond with the right hand), which is why left M1 is the default — but confirm per-subject response hand from the task event files, since a few subjects may differ.

## 6. Repo / deliverables

- Data loading + BIDS handling
- Design matrix / input function `u(t)` construction from Go-trial event timing
- Balloon model forward simulation + classical parameter fitting
- PINN implementation
- CoSTA implementation
- SINDy implementation
- Comparison metrics + figures
- Final written report

## 7. Open questions / possible extensions (not yet decided)

- Add "Stop" trials as a second regressor for a richer input function
- Extend the fitted framework to one clinical group (e.g. schizophrenia) from LA5c, to connect this project more directly to the masters work
- Bring in DMD as a purely data-driven baseline
- Explore DON (and possibly FMO — confirm what this refers to in the course) for operator-learning-based stimulus→BOLD mapping

## 8. Running the physical/PINN/CoSTA comparison

The first implementation keeps the physical model and PINN independent while
giving them the same experiment interface. Run from the repository root with
the project environment:

```bash
uv run python -m model_comparison --mode physical
uv run python -m model_comparison --mode pinn --epochs 2000
uv run python -m model_comparison --mode costa --epochs 2000
uv run python -m model_comparison --mode compare --epochs 2000
```

With no `--data` argument, these commands use a deterministic synthetic
Balloon-model trace. A real experiment can be supplied as a CSV containing
three columns: `time`, `input`, and `bold`. The same CSV is passed to both
models in `compare` mode, so the comparison does not change the input grid or
observations between methods.

Outputs are written to a timestamped folder under `results/` by default. For
example, a physical-model run at 25 September 2026 at 18:02:43 is saved in
`results/25.09.26_18.02.43_physical/`; a comparison with 1000 epochs uses
`results/25.09.26_18.02.43_compare_epochs1000/`. Each model produces a portable
`.npz` archive containing `time`, `input`, `states`, `bold`, and JSON-encoded
metrics, plus `observed_bold` for direct inspection. PINN and CoSTA runs also
save `pinn_loss.npz` and `costa_loss.npz`, respectively; combined runs
additionally save `comparison_metrics.json`.

Use `--output-dir` to choose a specific directory instead of the timestamped
default. Existing files in an explicitly supplied directory may be overwritten.

### Reading saved results

Use `read_results.py` to get a human-readable summary of metrics, fitted
parameters, array sizes, and PINN loss reduction:

```bash
uv run python3 read_results.py results/25.09.26_18.02.43_compare_epochs1000
```

Add `--plot` to save `results_summary.png` in that run directory. You can also
read one model archive directly:

```bash
uv run python3 read_results.py results/25.09.26_18.02.43_physical/physical_model.npz
```

Archives created before `observed_bold` was added can still be summarized, but
their plots cannot show the observed BOLD trace.

### PINN training details

The PINN uses a normalized time coordinate (`0` to `1`) internally. Its
derivatives are converted back to seconds before the Balloon ODE residual is
computed, which avoids saturating the Tanh network over a long 30-second time
interval. The BOLD data loss is normalized by the observed BOLD standard
deviation, and both data and physics losses are additionally normalized by
their initial values.

The flow, volume, and deoxyhemoglobin states use positive exponential
parameterizations with the required initial value of one. This prevents the
network from exploiting the previous artificial lower bound of `0.05` for
flow. ODE residuals are not evaluated exactly at input-pulse discontinuities,
where a smooth neural trajectory cannot satisfy both one-sided equations.

The default PINN uses a 64-unit, three-hidden-layer network, reduces the
learning rate when the total loss stops improving, and finishes with an LBFGS
refinement phase. The default weights are `data_weight=1` and
`physics_weight=1`; these can be adjusted in `PINNConfig` when experimenting
with the data/physics trade-off. More epochs are still useful for convergence,
for example:

```bash
uv run python -m model_comparison --mode compare --epochs 10000
```

The physical model remains the expected accuracy baseline: it directly
integrates the ODE and optimizes its parameters, while the PINN must learn a
neural approximation to the complete state trajectory.

### CoSTA model and training details

CoSTA keeps the fitted Balloon model as its physical core and learns an
additive correction in **BOLD observation space**. The correction network
receives normalized time, task input, the four physical states, and the
physical BOLD prediction. The saved CoSTA state trajectory is therefore the
physical model's trajectory; CoSTA changes the predicted BOLD signal, not the
Balloon ODE states. In `compare` mode all three models use the same observations
and input; CoSTA and PINN both use the fitted physical parameters as their
physical reference.

The correction network is initialized to output zero, so training starts at
the physical baseline. Its output is bounded to three training-residual
standard deviations and forced to zero at the start of the time series. Input
features and residual targets are scaled using training samples only. The
training objective combines normalized residual error with correction-size
and second-difference smoothness penalties; AdamW weight decay, gradient
clipping, a validation-driven learning-rate scheduler, and early stopping
limit an unconstrained correction from simply chasing each sample. The best
validation checkpoint is restored. Defaults are a 32-unit, two-layer network,
1,000 maximum epochs, and 100 epochs of patience; the shared `--epochs`
argument changes the maximum epoch count.

For an initial reproducible check, the current trainer was tested on synthetic
Balloon output with a smooth sinusoidal discrepancy added. On every fifth
sample held out from the residual fit, MSE decreased from `7.87e-6` for the
physical prediction to `1.69e-6` after correction, while the state trajectory
remained identical. This is a smoke check, not evidence of trial- or
subject-level generalization: adjacent fMRI samples are autocorrelated, so
scientific comparisons should later hold out complete trials or subjects and
report those results separately.

CoSTA saves `costa_model.npz` (corrected BOLD, physical states, metrics, and
physical parameters) and `costa_loss.npz` (training loss, validation loss, and
correction RMS). Run just the hybrid model with:

```bash
uv run python -m model_comparison --mode costa --epochs 1000
```

The earlier PINN implementation also needed several practical corrections
before it trained usefully: normalized time with derivatives converted back
to seconds, positive flow/volume/deoxyhemoglobin states with the correct
initial values, exclusion of input-pulse discontinuities from ODE residuals,
normalized losses, learning-rate reduction, and an LBFGS refinement phase.
Those constraints are retained in `model_comparison/pinn_model.py`; the CoSTA residual learner
uses separate safeguards suited to a supervised residual rather than
reusing the PINN's ODE-residual objective.

### Running a subject from BIDS files

Generate the left-M1 Julich-Brain v3.1 mask with siibra:

```bash
.venv/bin/python julich_parcellation.py \
	--output results/julich_left_m1_mask.nii.gz
```

The default mask is the binary union of `Area 4a (PreCG) left` and `Area 4p
(PreCG) left`. The command also writes a JSON sidecar describing the atlas,
space, and selected regions. This mask is in MNI 152 ICBM 2009c Nonlinear
Asymmetric space.

The repository also accepts a subject's BOLD NIfTI and event TSV directly:

```bash
uv run python -m model_comparison \
	--mode compare \
	--subject sub-10159 \
	--bold-path /path/to/fmriprep/sub-10159_task-stopsignal_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz \
	--roi-mask /path/to/sub-10159_left-precentral-mask.nii.gz \
	--epochs 1000 \
	--output-dir results/sub-10159
```

The mask must already be aligned to the BOLD image. With `--bold-path`, use an
fMRIPrep MNI-space BOLD image matching the Julich MNI mask. Without it, the
command uses the raw image under `--data-root`; an MNI mask must not be applied
to that scanner-space image directly.

The subject path creates `results/sub-10159/sub-10159_timeseries.csv` with
`time`, `input`, and baseline-normalized `bold` columns, then feeds that exact
series to both models. GO trials with `TrialOutcome == SuccessfulGo` form the
initial input; STOP trials can be added later as a second input channel.
The physical model is fitted by default. Use `--no-fit-physical` only for a
fixed-parameter forward simulation.

---







## Sparse Sensor Placement & Compressed Sensing Analysis

### Goal
Estimate how well whole-brain BOLD activity can be reconstructed from only a
subset of "sensors", and find which sensor locations are the most informative.
The fully sampled, fMRIPrep-preprocessed ROI-averaged data is treated as ground
truth. We pretend only k ROIs were measured, reconstruct the rest, and measure
the error. This is a proxy for sparse modalities (fNIRS, EEG, ECoG), not a
model of the MRI scanner itself.

### Input data
- File: `results/ROI_averaged_data.h5` (output of the ROI averaging step).
- Assumed layout: one BOLD time series per ROI per participant, originally
  stored as `{participant_ID: {ROI: signal}}`. Verify and adapt the loader.
- Build, per participant, a matrix `X` of shape `(n_rois, n_timepoints)`.
- Each ROI time series should be z-scored (zero mean, unit variance) before
  analysis. Document any further preprocessing in the code.

### Definitions
- **Sensor**: one ROI (a row of `X`) that we choose to observe.
- **k**: number of sensors selected.
- **Reconstruction**: estimate all `n_rois` signals from the `k` observed ones.
- **Error metric**: normalized RMSE, `||X - X_hat||_F / ||X||_F`, computed on
  held-out data only. Optionally also report mean per-ROI Pearson correlation.

### Methods to implement
Implement each method as a separate function in `sensor_placement/methods.py`
with a common interface: given training data and `k`, return the selected
sensor indices and a reconstruction function.

1. **Random placement (baseline).** Choose `k` ROIs uniformly at random.
   Repeat for many random draws (e.g. 50) and report mean and std of the
   error. Reconstruct with least squares in the low-rank basis from method 2.
2. **Optimal placement via SVD + QR pivoting (SSPOR style).**
   - Compute the SVD of the training data: `X_train = U S V^T`.
   - Keep the leading `r` spatial modes `U_r` (shape `n_rois x r`). Choose `r`
     from the singular value spectrum, and also try `r = k`.
   - Run QR factorization with column pivoting on `U_r^T`
     (`scipy.linalg.qr(..., pivoting=True)`). The first `k` pivots are the
     selected sensor indices.
   - Reconstruct by estimating mode coefficients from the observed rows
     (`a = pinv(U_r[sensors, :]) @ y`) and computing `x_hat = U_r @ a`.
     Consider a small ridge penalty if the system is ill-conditioned.
3. **Compressed sensing recovery with L1.**
   - Assume the signal is sparse in a DCT basis along the spatial axis
     (ROI index) or, alternatively, in a learned dictionary.
   - Measurement: select the `k` rows given by (a) random placement and
     (b) the QR-pivoting placement from method 2.
   - Recover coefficients with LASSO (`sklearn.linear_model.Lasso` or
     `LassoCV`), then map back to the full signal.
   - This lets us separate the effect of the placement strategy from the
     effect of the reconstruction method.

### Evaluation protocol
- Sweep `k` from 1 up to a sensible maximum (e.g. every value up to
  `n_rois // 2`, then coarser steps).
- Two held-out schemes, reported separately:
  - **Temporal hold-out**: fit the basis and placement on the first ~70% of
    time points per participant, test on the remaining ~30%.
  - **Participant hold-out**: fit the basis and placement on a training set of
    participants, test on unseen participants (use k-fold cross-validation over
    participants). This is the more honest test of whether one placement
    generalizes across brains.
- Never choose sensors or fit the basis using test data.
- Fix random seeds and save all results (per method, `k`, fold) to
  `results/sensor_placement/` as CSV or HDF5 so plots can be regenerated
  without recomputing.

### Visualizations
Save all figures to `results/sensor_placement/figures/` as PNG (and PDF for the
report). Use matplotlib; use nilearn plotting for brain figures. Label axes
with units, and add titles and legends.

1. **Error vs number of sensors.** Line plot, x = `k`, y = normalized RMSE.
   One line per method (random, QR-pivoting + least squares, random + LASSO,
   QR-pivoting + LASSO). Random baselines shown as mean with a shaded
   std band. Make one panel for temporal hold-out and one for participant
   hold-out.
2. **Singular value spectrum.** Cumulative explained variance vs number of
   modes, with a marker at the chosen `r`. Shows how low-rank the data is.
3. **Selected sensors on the brain.** Using the atlas, plot the first `k`
   ROIs chosen by QR pivoting (e.g. `k = 5, 10, 20`) on glass-brain or
   slice views, highlighting the selected ROIs. Use
   `nilearn.plotting.plot_roi` or `plot_glass_brain`.
4. **Reconstruction examples.** For one test participant, show true vs
   reconstructed time series for 3 to 4 ROIs (including one selected
   sensor and several unobserved ROIs) at a small and a large `k`.
5. **Per-ROI error map.** Brain map where each ROI is colored by its
   reconstruction error at a chosen `k`, to show which regions are hard
   to reconstruct.
6. **Sensor stability.** Heatmap or bar chart of how often each ROI is
   selected across cross-validation folds (selection frequency per ROI),
   showing whether the optimal placement is consistent across participants.
7. **(Optional) Group comparison.** If diagnosis labels are available for
   LA5c, compare reconstruction error and selected sensors between diagnosis
   groups (box plot of error at fixed `k`).

### Suggested file structure
```
sensor_placement/
    data.py        # load HDF5, build X, z-score, train/test splits
    methods.py     # random, qr_pivot, cs_lasso placement + reconstruction
    evaluate.py    # k-sweep, hold-out schemes, save results
    plots.py       # all figures listed above, reading saved results
    run.py         # CLI entry point, config (k range, r, seeds, paths)
```

### Implementation notes
- Add a docstring to every function explaining the math, not just the
  arguments.
- Keep each method readable and commented, since the goal is to understand
  every step.
- Runs on a single machine should be fine at ROI level; use Slurm only if
  extended to voxel level.
- Out of scope for now: voxel-level analysis and k-space undersampling.