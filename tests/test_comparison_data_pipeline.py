import unittest

import numpy as np
import pandas as pd

from model_comparison.data_pipeline import build_event_input, normalize_bold


class EventInputTests(unittest.TestCase):
    def test_event_between_frames_is_preserved_as_interval_average(self) -> None:
        times = np.arange(0.0, 8.0, 2.0)
        events = pd.DataFrame(
            {
                "onset": [1.5, 4.5],
                "duration": [1.5, 1.5],
                "trial_type": ["GO", "GO"],
                "TrialOutcome": ["SuccessfulGo", "SuccessfulGo"],
            }
        )

        values = build_event_input(times, events)

        np.testing.assert_allclose(values, [0.25, 0.5, 0.75, 0.0])
        self.assertAlmostEqual(float(values.sum() * 2.0), 3.0)

    def test_normalization_uses_full_run_median(self) -> None:
        raw_signal = np.array([202.0, 198.0, 200.0, 200.0])

        normalized = normalize_bold(raw_signal)

        np.testing.assert_allclose(normalized, [0.01, -0.01, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
