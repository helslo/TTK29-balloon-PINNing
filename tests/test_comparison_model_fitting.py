import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_comparison.balloon_model import (
    BalloonParams,
    build_input_function,
    fit_balloon,
    simulate_balloon,
)
from model_comparison.costa_model import _temporal_validation_mask
from model_comparison.pinn_model import BalloonPINN, PINNConfig
from model_comparison.run import run_costa_extrapolation, synthetic_data


class BalloonFitTests(unittest.TestCase):
    def test_costa_validation_is_a_contiguous_future_block(self) -> None:
        validation = _temporal_validation_mask(20)

        np.testing.assert_array_equal(np.flatnonzero(validation), [16, 17, 18, 19])

    def test_costa_validation_supports_explicit_temporal_fraction(self) -> None:
        validation = _temporal_validation_mask(147, 37 / 147)

        np.testing.assert_array_equal(np.flatnonzero(validation), np.arange(110, 147))

    def test_extrapolation_test_targets_do_not_change_model_predictions(self) -> None:
        times, input_values, observed_bold = synthetic_data()
        test_start = int(np.floor(len(times) * 0.8))
        changed_target = observed_bold.copy()
        changed_target[test_start:] += 0.05

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_costa_extrapolation(
                times,
                input_values,
                observed_bold,
                root / "original",
                PINNConfig(epochs=5),
                fit_physical=False,
            )
            run_costa_extrapolation(
                times,
                input_values,
                changed_target,
                root / "changed",
                PINNConfig(epochs=5),
                fit_physical=False,
            )

            original = np.load(root / "original" / "extrapolation_predictions.npz")
            changed = np.load(root / "changed" / "extrapolation_predictions.npz")
            original_metrics = json.loads(
                (root / "original" / "extrapolation_metrics.json").read_text()
            )

        self.assertEqual(original_metrics["train_samples"], 180)
        self.assertEqual(original_metrics["validation_samples"], 60)
        self.assertEqual(original_metrics["test_samples"], 61)
        self.assertTrue(original_metrics["input_known_at_inference"])
        np.testing.assert_allclose(
            original["costa_bold"][test_start:],
            changed["costa_bold"][test_start:],
        )

    def test_fit_profiles_constant_baseline_and_linear_drift(self) -> None:
        times = np.arange(0.0, 30.0, 0.2)
        input_function = build_input_function([2.0, 10.0, 18.0], 30.0, 0.05)
        _, _, physical_bold = simulate_balloon(
            input_function, BalloonParams(), (times[0], times[-1]), 0.2
        )
        observed_bold = physical_bold + 0.02 + 0.01 * (
            (times - np.mean(times)) / (times[-1] - times[0])
        )

        fit = fit_balloon(times, observed_bold, input_function)

        self.assertLess(
            float(np.mean((fit.fitted_bold - observed_bold) ** 2)),
            float(np.mean((physical_bold - observed_bold) ** 2)) * 0.01,
        )
        self.assertAlmostEqual(fit.observation_baseline, 0.02, delta=0.001)
        self.assertAlmostEqual(fit.observation_linear_drift, 0.01, delta=0.001)

    def test_pinn_bold_scale_is_trainable_and_bounded(self) -> None:
        model = BalloonPINN(BalloonParams(V0=0.005), duration=30.0)

        model.v0.backward()

        self.assertGreater(float(model.observation_v0.grad), 0.0)
        self.assertGreaterEqual(float(model.v0.detach()), 0.001)
        self.assertLessEqual(float(model.v0.detach()), 0.2)


if __name__ == "__main__":
    unittest.main()
