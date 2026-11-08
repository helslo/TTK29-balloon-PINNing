import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import model_comparison.data_pipeline as data_pipeline
from model_comparison.data_pipeline import (
    build_event_input,
    interpolate_isolated_spikes,
    load_subject_timeseries,
    normalize_bold,
)


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

    def test_default_input_includes_labeled_events_regardless_of_outcome(self) -> None:
        times = np.arange(0.0, 8.0, 2.0)
        events = pd.DataFrame(
            {
                "onset": [0.0, 2.0, 4.0, 6.0],
                "duration": [1.0, 1.0, 1.0, 2.0],
                "trial_type": ["GO", "GO", "STOP", None],
                "TrialOutcome": ["SuccessfulGo", "UnsuccessfulGo", "SuccessfulStop", None],
            }
        )

        values = build_event_input(times, events)

        np.testing.assert_allclose(values, [0.5, 0.5, 0.5, 0.0])

    def test_event_input_can_still_select_a_trial_and_outcome(self) -> None:
        times = np.arange(0.0, 8.0, 2.0)
        events = pd.DataFrame(
            {
                "onset": [0.0, 2.0, 4.0],
                "duration": [1.0, 1.0, 1.0],
                "trial_type": ["GO", "GO", "STOP"],
                "TrialOutcome": ["SuccessfulGo", "UnsuccessfulGo", "SuccessfulStop"],
            }
        )

        values = build_event_input(times, events, "GO", "SuccessfulGo")

        np.testing.assert_allclose(values, [0.5, 0.0, 0.0, 0.0])

    def test_normalization_uses_full_run_median(self) -> None:
        raw_signal = np.array([202.0, 198.0, 200.0, 200.0])

        normalized = normalize_bold(raw_signal)

        np.testing.assert_allclose(normalized, [0.01, -0.01, 0.0, 0.0])

    def test_isolated_spike_is_interpolated_without_changing_other_samples(self) -> None:
        expected = np.linspace(-0.002, 0.002, 21)
        signal = expected.copy()
        signal[10] -= 0.02

        cleaned, spike_indices = interpolate_isolated_spikes(signal)

        np.testing.assert_array_equal(spike_indices, [10])
        np.testing.assert_allclose(cleaned, expected, atol=1e-12)

    def test_sustained_signal_change_is_not_classified_as_an_isolated_spike(self) -> None:
        signal = np.concatenate((np.zeros(10), np.full(10, 0.01)))

        cleaned, spike_indices = interpolate_isolated_spikes(signal)

        np.testing.assert_array_equal(spike_indices, [])
        np.testing.assert_array_equal(cleaned, signal)

    def test_default_loader_passes_explicit_epi_mask_to_masker(self) -> None:
        class FakeHeader:
            def get_zooms(self) -> tuple[float, float, float, float]:
                return (3.0, 3.0, 4.0, 2.0)

        class FakeImage:
            shape = (2, 2, 2, 10)
            header = FakeHeader()

        expected_mask = object()
        observed_masks = []
        signal = np.linspace(100.0, 101.0, 10)

        class FakeMasker:
            def __init__(self, mask_img: object, standardize: bool, detrend: bool) -> None:
                observed_masks.append(mask_img)
                self.standardize = standardize
                self.detrend = detrend

            def fit_transform(self, image_path: str) -> np.ndarray:
                return np.column_stack((signal, signal, signal))

        events = pd.DataFrame(
            {"onset": [], "duration": [], "trial_type": []}
        )
        with patch.object(data_pipeline.nib, "load", return_value=FakeImage()):
            with patch.object(data_pipeline.pd, "read_csv", return_value=events):
                with patch.object(data_pipeline, "compute_epi_mask", return_value=expected_mask):
                    with patch.object(data_pipeline, "NiftiMasker", FakeMasker):
                        load_subject_timeseries(Path("bold.nii.gz"), Path("events.tsv"))

        self.assertEqual(observed_masks, [expected_mask])


if __name__ == "__main__":
    unittest.main()
