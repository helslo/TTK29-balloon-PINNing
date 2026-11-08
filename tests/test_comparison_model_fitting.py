import unittest

import numpy as np

from model_comparison.balloon_model import (
    BalloonParams,
    build_input_function,
    fit_balloon,
    simulate_balloon,
)
from model_comparison.pinn_model import BalloonPINN


class BalloonFitTests(unittest.TestCase):
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
