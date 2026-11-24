import unittest

import numpy as np

from model_comparison.glm_baseline import (
    convolve_with_canonical_hrf,
    fit_canonical_hrf_glm,
    regression_metrics,
)


class CanonicalHRFGLMTests(unittest.TestCase):
    def test_hrf_convolution_returns_one_sample_per_input_and_is_delayed(self) -> None:
        task_input = np.zeros(80)
        task_input[2:4] = 1.0

        response = convolve_with_canonical_hrf(task_input, repetition_time=1.0)

        self.assertEqual(response.shape, task_input.shape)
        self.assertTrue(np.isfinite(response).all())
        self.assertGreater(int(np.argmax(response)), 2)

    def test_fit_recovers_task_scale_baseline_and_drift(self) -> None:
        times = np.arange(0.0, 180.0, 2.0)
        task_input = np.zeros(times.size)
        task_input[[2, 15, 35, 57]] = 1.0
        regressor = convolve_with_canonical_hrf(task_input, repetition_time=2.0)
        design = np.column_stack(
            (regressor, np.ones(times.size), (times - times.mean()) / times[-1])
        )
        observed = design @ np.array([0.75, 0.2, -0.05])

        fit = fit_canonical_hrf_glm(times, regressor, observed)

        np.testing.assert_allclose(fit.coefficients, [0.75, 0.2, -0.05], atol=1e-10)
        self.assertAlmostEqual(regression_metrics(observed, fit.predictions)["r2"], 1.0)

    def test_heldout_targets_do_not_change_training_fit(self) -> None:
        times = np.arange(0.0, 200.0, 2.0)
        task_input = np.zeros(times.size)
        task_input[[1, 10, 30, 55, 75]] = 1.0
        regressor = convolve_with_canonical_hrf(task_input, repetition_time=2.0)
        design = np.column_stack(
            (regressor, np.ones(times.size), (times - times.mean()) / times[-1])
        )
        observed = design @ np.array([0.4, 0.1, 0.02])
        changed = observed.copy()
        changed[80:] += 1.0

        original_fit = fit_canonical_hrf_glm(times, regressor, observed, fit_end=80)
        changed_fit = fit_canonical_hrf_glm(times, regressor, changed, fit_end=80)

        np.testing.assert_allclose(
            original_fit.predictions, changed_fit.predictions, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
