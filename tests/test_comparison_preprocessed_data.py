import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

import model_comparison.data_pipeline as data_pipeline
from model_comparison.data_pipeline import (
    create_schaefer_network_mask,
    load_subject_timeseries,
    select_fmriprep_confounds,
)
from model_comparison.run import (
    _fmriprep_confounds_for_bold,
    discover_fmriprep_bold,
)


class SchaeferAtlasMaskTests(unittest.TestCase):
    def test_creates_binary_network_mask_and_records_selected_parcels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            atlas_path = root / "atlas.nii.gz"
            atlas_image = nib.Nifti1Image(
                np.array([[[0]], [[1]], [[2]], [[3]]], dtype=np.int16),
                np.eye(4),
            )
            nib.save(atlas_image, atlas_path)
            atlas = SimpleNamespace(
                maps=str(atlas_path),
                labels=[
                    "Background",
                    "7Networks_LH_Cont_PFCl_1",
                    "7Networks_RH_Cont_PFCl_1",
                    "7Networks_LH_Vis_1",
                ],
            )
            output_path = root / "control_mask.nii.gz"

            with patch.object(
                data_pipeline.nilearn_datasets,
                "fetch_atlas_schaefer_2018",
                return_value=atlas,
            ) as fetch:
                mask_path, labels = create_schaefer_network_mask(
                    output_path, "Cont", "both"
                )

            fetch.assert_called_once_with(
                n_rois=400,
                yeo_networks=7,
                resolution_mm=2,
            )
            np.testing.assert_array_equal(
                nib.load(mask_path).get_fdata().reshape(-1), [0, 1, 1, 0]
            )
            self.assertEqual(
                labels,
                ("7Networks_LH_Cont_PFCl_1", "7Networks_RH_Cont_PFCl_1"),
            )
            metadata = json.loads(
                output_path.with_suffix(output_path.suffix + ".json").read_text()
            )
            self.assertEqual(metadata["network"], "Cont")
            self.assertEqual(metadata["hemisphere"], "both")

    def test_rejects_unknown_network_and_hemisphere(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Schaefer network"):
            create_schaefer_network_mask(Path("mask.nii.gz"), "Control")
        with self.assertRaisesRegex(ValueError, "hemisphere"):
            create_schaefer_network_mask(Path("mask.nii.gz"), "Cont", "bilateral")


class FmriprepConfoundTests(unittest.TestCase):
    def test_selects_standard_motion_tissue_compcor_and_drift_regressors(self) -> None:
        confounds = pd.DataFrame(
            {
                "trans_x": np.arange(10.0),
                "trans_x_derivative1": [np.nan, *np.arange(9.0)],
                "white_matter": np.linspace(0.0, 1.0, 10),
                "csf": np.linspace(1.0, 0.0, 10),
                "a_comp_cor_00": np.sin(np.arange(10.0)),
                "cosine00": np.cos(np.arange(10.0)),
                "non_steady_state_outlier00": [1.0, *np.zeros(9)],
                "framewise_displacement": np.arange(10.0),
                "global_signal": np.arange(10.0),
                "constant": np.ones(10),
            }
        )

        selected = select_fmriprep_confounds(confounds)

        self.assertEqual(
            list(selected.columns),
            [
                "trans_x",
                "trans_x_derivative1",
                "white_matter",
                "csf",
                "a_comp_cor_00",
                "cosine00",
                "non_steady_state_outlier00",
            ],
        )
        self.assertTrue(np.isfinite(selected.to_numpy()).all())
        np.testing.assert_allclose(selected.mean(axis=0).to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            selected.std(axis=0, ddof=1).to_numpy(), 1.0, atol=1e-12
        )

    def test_rejects_tables_without_supported_regressors(self) -> None:
        with self.assertRaisesRegex(ValueError, "no supported fMRIPrep regressors"):
            select_fmriprep_confounds(pd.DataFrame({"framewise_displacement": [0.0, 1.0]}))


class FmriprepDiscoveryTests(unittest.TestCase):
    def test_discovers_unique_preprocessed_bold_and_matching_confounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            func_dir = data_root / "derivatives" / "fmriprep" / "sub-10159" / "func"
            func_dir.mkdir(parents=True)
            bold_path = func_dir / (
                "sub-10159_task-stopsignal_space-MNI152NLin2009cAsym_"
                "desc-preproc_bold.nii.gz"
            )
            bold_path.touch()
            confounds_path = func_dir / (
                "sub-10159_task-stopsignal_desc-confounds_timeseries.tsv"
            )
            confounds_path.touch()

            self.assertEqual(
                discover_fmriprep_bold(data_root, "sub-10159"), bold_path
            )
            self.assertEqual(_fmriprep_confounds_for_bold(bold_path), confounds_path)

    def test_requires_explicit_selection_when_multiple_runs_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            func_dir = data_root / "derivatives" / "fmriprep" / "sub-10159" / "func"
            func_dir.mkdir(parents=True)
            for run in (1, 2):
                (func_dir / (
                    f"sub-10159_task-stopsignal_run-{run}_"
                    "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                )).touch()

            with self.assertRaisesRegex(ValueError, "Multiple fMRIPrep BOLD runs"):
                discover_fmriprep_bold(data_root, "sub-10159")

    def test_reports_missing_fmriprep_derivatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "fMRIPrep derivatives"):
                discover_fmriprep_bold(Path(temporary_directory), "sub-10159")


class SubjectTimeseriesConfoundTests(unittest.TestCase):
    def test_resamples_roi_and_regresses_confounds(self) -> None:
        class FakeHeader:
            def get_zooms(self) -> tuple[float, float, float, float]:
                return (3.0, 3.0, 3.0, 2.0)

        class FakeBoldImage:
            shape = (2, 2, 2, 10)
            header = FakeHeader()

        class FakeMaskImage:
            shape = (2, 2, 2)
            dataobj = np.ones(shape, dtype=np.uint8)

        class FakeMasker:
            def __init__(self, mask_img: object, standardize: bool, detrend: bool) -> None:
                self.mask_img = mask_img
                observed_masks.append(mask_img)

            def fit_transform(self, image_path: str) -> np.ndarray:
                signal = np.linspace(100.0, 101.0, 10)
                return np.column_stack((signal, signal, signal))

        events = pd.DataFrame({"onset": [], "duration": [], "trial_type": []})
        confounds = pd.DataFrame(
            {
                "trans_x": np.linspace(0.0, 1.0, 10),
                "white_matter": np.sin(np.arange(10.0)),
            }
        )
        observed_masks = []
        resampled_mask = FakeMaskImage()
        cleaned_values = []

        def fake_clean(
            signal: np.ndarray,
            confounds: pd.DataFrame,
            detrend: bool,
            standardize: bool,
            standardize_confounds: bool,
            filter: object,
        ) -> np.ndarray:
            cleaned_values.append(
                (signal.copy(), confounds.copy(), standardize_confounds, filter)
            )
            return signal

        with patch.object(data_pipeline.nib, "load", side_effect=[FakeBoldImage(), FakeMaskImage()]):
            with patch.object(
                data_pipeline.nilearn_image, "resample_to_img", return_value=resampled_mask
            ) as resample:
                with patch.object(
                    data_pipeline.pd,
                    "read_csv",
                    side_effect=[events, confounds],
                ):
                    with patch.object(data_pipeline, "NiftiMasker", FakeMasker):
                        with patch.object(data_pipeline, "clean_signal", side_effect=fake_clean):
                            times, inputs, bold = load_subject_timeseries(
                                Path("bold.nii.gz"),
                                Path("events.tsv"),
                                Path("roi.nii.gz"),
                                Path("confounds.tsv"),
                            )

        resample.assert_called_once()
        self.assertEqual(resample.call_args.kwargs["interpolation"], "nearest")
        self.assertEqual(observed_masks, [resampled_mask])
        self.assertEqual(len(cleaned_values), 1)
        self.assertEqual(list(cleaned_values[0][1].columns), ["trans_x", "white_matter"])
        self.assertFalse(cleaned_values[0][2])
        self.assertIsNone(cleaned_values[0][3])
        self.assertEqual(len(times), 10)
        self.assertEqual(len(inputs), 10)
        self.assertEqual(len(bold), 10)

    def test_rejects_confounds_with_a_different_number_of_volumes(self) -> None:
        class FakeHeader:
            def get_zooms(self) -> tuple[float, float, float, float]:
                return (3.0, 3.0, 3.0, 2.0)

        class FakeImage:
            shape = (2, 2, 2, 10)
            header = FakeHeader()

        class FakeMasker:
            def __init__(self, mask_img: object, standardize: bool, detrend: bool) -> None:
                pass

            def fit_transform(self, image_path: str) -> np.ndarray:
                signal = np.linspace(100.0, 101.0, 10)
                return np.column_stack((signal, signal, signal))

        events = pd.DataFrame({"onset": [], "duration": [], "trial_type": []})
        confounds = pd.DataFrame({"trans_x": np.arange(9.0)})
        with patch.object(data_pipeline.nib, "load", return_value=FakeImage()):
            with patch.object(data_pipeline.pd, "read_csv", side_effect=[events, confounds]):
                with patch.object(data_pipeline, "compute_epi_mask", return_value=object()):
                    with patch.object(data_pipeline, "NiftiMasker", FakeMasker):
                        with self.assertRaisesRegex(ValueError, "does not match"):
                            load_subject_timeseries(
                                Path("bold.nii.gz"),
                                Path("events.tsv"),
                                confounds_path=Path("confounds.tsv"),
                            )


if __name__ == "__main__":
    unittest.main()
