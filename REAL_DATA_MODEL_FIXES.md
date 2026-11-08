# Real-data model failures: findings and first fixes

## What was wrong

The default comparison uses `sub-10159`, a raw 4D BOLD scan with 184 volumes
and a 2-second TR. Its events file has 96 GO trials and 32 STOP trials, plus
128 unlabeled timing rows. The loader previously selected only the 94
`SuccessfulGo` events, omitting unsuccessful GO and every STOP event from the
model input.

The raw mean BOLD signal also contained scan-wide drift, while the Balloon
fit treated the signal as if it had no observation baseline or drift. This
encouraged the fitted BOLD scale (`V0`) to compensate for differences that
were not hemodynamic. The PINN then inherited that fitted `V0` as a fixed
value, even though its own best observation scale can differ.

## Fixes made

- The default event drive now includes every labeled GO and STOP event,
  regardless of outcome, while continuing to ignore unlabeled timing rows.
  Each event is still averaged over the TR interval.
- Real-data traces are converted to fractional change first, then linearly
  detrended. The intensity baseline is therefore preserved during
  normalization.
- Balloon fitting profiles a constant observation baseline and linear scan
  drift alongside its physical parameters. Both nuisance coefficients are
  stored with the physical-model result.
- The PINN now learns its own bounded BOLD observation scale (`V0`) rather
  than freezing it to the physical model's estimate. The learned scale is
  included in the saved PINN parameters and loss history.

## Measured result on the default real scan

I ran the comparison on `sub-10159` with 1,000 epochs before and after the
changes. Metrics are in-sample, on the whole time series:

| Model | Before: R² / MSE | After: R² / MSE |
|---|---:|---:|
| Balloon physical | -0.0421 / 1.1645e-5 | 0.0313 / 1.0646e-5 |
| PINN | 0.1513 / 9.4835e-6 | 0.1438 / 9.4094e-6 |
| CoSTA | 0.0108 / 1.1054e-5 | 0.0565 / 1.0369e-5 |

The physical and CoSTA fits improved on this run. The PINN's MSE improved
slightly, but its R² did not; it remains a weak fit. These before/after scores
are indicative rather than a strict paired comparison because the event input
and target preprocessing changed. They do not establish generalization to
held-out volumes or subjects.

The PINN scale was highly influential in a direct scan-specific check: with a
fixed scale, `V0=0.001` gave R² 0.189, while the fitted physical scale
(`V0≈0.0047`) gave R² 0.131. This motivated making `V0` learnable, but the
current PINN still needs improvement beyond this first repair.

## Remaining limitations

The available scan is raw, and the default signal is an EPI-masked
whole-brain mean. This workflow does not currently regress motion or other
fMRI confounds, nor does it test ROI-specific signals. All reported metrics
are in-sample for one participant. The current positive R² values are modest,
so the models should not yet be treated as validated on real data. A useful
next step is to evaluate motion-corrected/confound-cleaned data, compare
task-relevant ROIs, and report held-out temporal or participant performance.

## Verification

The real-data CLI comparison completed successfully on `sub-10159`. All six
unit tests pass, covering event selection, TR-averaging, baseline/drift
fitting, and bounded trainable PINN scale.

## Second go: isolated BOLD spikes and CoSTA early stopping

### What the plotted run revealed

The saved 2,000-epoch plot showed two one-volume drops that immediately
rebounded, at volumes 28 and 72 (56 and 144 seconds). In the extracted
whole-brain mean these were approximately 202.19 and 202.55, compared with
neighbor-interpolated values of 206.50 and 205.86. They are isolated
single-volume excursions, not Balloon-like hemodynamic responses. The raw
subject folder contains the BOLD image, its JSON metadata, and events, but no
motion estimates or confound regressors; their exact cause cannot be
confirmed from the available files.

The comparison also exposed overly aggressive CoSTA early stopping. On the
same despiked time series, increasing patience from 100 to 200 let training
continue from 115 to 436 epochs. Its best interleaved validation loss
improved from 0.961 to 0.949 (normalized residual MSE), and in-sample R²
improved from 0.086 to 0.197.

### Changes made and measured result

- The real-data loader now finds isolated single-volume spikes using a robust
  scale estimate of adjacent differences. It only replaces a point if both
  adjacent differences exceed six robust scales and point in opposite
  directions, then linearly interpolates from the neighboring frames. On
  `sub-10159`, this flags only volumes 28 and 72. Synthetic inputs and CSV
  inputs are unchanged.
- CoSTA's default early-stopping patience is now 200 rather than 100.
- Added regression tests that cover an isolated spike and a sustained step;
  the filter must correct the former and leave the latter alone.

With the real-data compare CLI and 2,000 epochs, the new run gave:

| Model | Previous run R² / MSE | Second pass R² / MSE |
|---|---:|---:|
| Balloon physical | 0.0313 / 1.0646e-5 | 0.0548 / 7.4507e-6 |
| PINN | 0.1514 / 9.3260e-6 | 0.2042 / 6.2734e-6 |
| CoSTA | 0.0565 / 1.0369e-5 | 0.2042 / 6.2731e-6 |

These are in-sample scores on the same scan, but the second pass changes the
target by interpolating the two flagged samples, so this is not a strict
same-target comparison. The fit is improved, but R² around 0.20 remains
modest and should not be read as held-out performance.

### Remaining concern

The isolated-spike filter is a conservative fallback, not motion correction
or a substitute for fMRI confound regression. The dataset files available
here do not include fMRIPrep derivatives or motion confounds. The next
scientifically sound comparison should use motion-corrected, confound-cleaned
BOLD data and held-out temporal/participant evaluation; more flexible
single-run fitting alone risks modeling noise.

## Third go: correct the default brain mask

### Confirmed extraction bug

Inspecting the 20,000-epoch run showed that its target was still visually and
numerically noisy. I checked the actual mask generated by the default
`NiftiMasker` call: it contained **139,209 of 139,264 voxels** (99.96% of the
entire image). The image includes substantial low-intensity background, so the
whole-image mean was not a brain-only BOLD signal. This contradicted the
earlier assumption in this report that the default extraction already used an
EPI-derived brain mask.

Calling Nilearn's `compute_epi_mask` explicitly on this scan produces a
34,457-voxel mask (24.7% of the image), matching the visible brain-shaped
signal region in the mean EPI. The no-`--roi-mask` path now passes that mask
explicitly to `NiftiMasker`. User-provided aligned ROI masks remain unchanged.
A regression test verifies the explicit mask is passed to the masker.

### 20,000-epoch real-scan result

I reran the full comparison through the CLI with the corrected default mask,
the existing spike interpolation and event handling, and 20,000 PINN epochs:

| Model | Previous 20k run R² / MSE | Corrected-mask 20k R² / MSE |
|---|---:|---:|
| Balloon physical | 0.0548 / 7.4507e-6 | 0.0710 / 7.9779e-6 |
| PINN | 0.3244 / 5.3260e-6 | 0.4300 / 4.8955e-6 |
| CoSTA | 0.2042 / 6.2731e-6 | 0.2177 / 6.7186e-6 |

These are not same-target before/after scores: the corrected brain mask
changes the time series, and thus the R² denominator and MSE scale. On the
corrected-mask run, PINN explains about 43% of the in-sample variance, while
the physical model remains weak. These are still in-sample values for one
participant, not evidence of held-out prediction. The raw folder has no
motion/confound regressors, so further model fitting alone cannot address
unmodeled acquisition and motion noise.

### Verification

The corrected-mask CLI comparison completed successfully on `sub-10159`.
All eight tests pass, including the new mask-wiring test and existing
isolated-spike, event, and fitting tests.
