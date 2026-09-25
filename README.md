# TTK29-balloon-PINNing

# TTK29 Project: Hybrid Modeling of Cerebral Hemodynamics

## 1. Goal

Compare a **pure physical model** of the BOLD hemodynamic response against **three hybrid physics+ML approaches** (PINN, CoSTA, SINDy), all fit to real task-fMRI data from a single brain region.

This is a course project (TTK29, 3.75 study points) designed to build modeling skills that feed directly into a related masters project on BOLD-signal analysis in mental health disorders.

## 2. Core physical model: the Balloon(-Windkessel) model

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

## 8. Running the current physical/PINN comparison

The first implementation keeps the physical model and PINN independent while
giving them the same experiment interface. Run from the repository root with
the project environment:

```bash
.venv/bin/python compare_models.py --mode physical
.venv/bin/python compare_models.py --mode pinn --epochs 1000
.venv/bin/python compare_models.py --mode compare --epochs 1000
```

With no `--data` argument, these commands use a deterministic synthetic
Balloon-model trace. A real experiment can be supplied as a CSV containing
three columns: `time`, `input`, and `bold`. The same CSV is passed to both
models in `compare` mode, so the comparison does not change the input grid or
observations between methods.

Outputs are written to `results/` by default. Each model produces a portable
`.npz` archive containing `time`, `input`, `states`, `bold`, and JSON-encoded
metrics. PINN runs also save `pinn_loss.npz`; combined runs additionally save
`comparison_metrics.json`.

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
.venv/bin/python compare_models.py \
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