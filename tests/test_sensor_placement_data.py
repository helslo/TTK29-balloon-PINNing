import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

from sensor_placement.data import (
    _confound_columns,
    _load_tr,
    discover_runs,
)
from sensor_placement.methods import (
    _low_rank_basis,
    _reconstruct_from_basis,
    qr_pivot_placement,
)


class SharedDatasetSensorDataTests(unittest.TestCase):
    def test_precomputed_qr_basis_matches_internal_basis_fit(self) -> None:
        rng = np.random.default_rng(42)
        training = rng.normal(size=(24, 12)).astype(np.float32)
        basis, _, _ = _low_rank_basis(training, r=4)

        sensors_internal, reconstruct_internal = qr_pivot_placement(
            training, k=8, r=4
        )
        sensors_cached, reconstruct_cached = qr_pivot_placement(
            training, k=8, r=4, basis=basis
        )
        observed = rng.normal(size=(24, 5)).astype(np.float32)

        np.testing.assert_array_equal(sensors_cached, sensors_internal)
        np.testing.assert_allclose(
            reconstruct_cached(observed), reconstruct_internal(observed), rtol=1e-5
        )

    def test_regularizes_ill_conditioned_low_rank_sensor_system(self) -> None:
        basis = np.array([[1.0, 0.0], [0.0, 1e-3], [1.0, 1.0]])
        observed = np.array([[1.0], [1.0], [0.0]])

        reconstructed = _reconstruct_from_basis(
            basis, np.array([0, 1]), observed
        )

        self.assertTrue(np.isfinite(reconstructed).all())
        self.assertLess(float(np.max(np.abs(reconstructed))), 2.0)

    def test_selects_shared_motion_white_matter_and_compcor_columns(self) -> None:
        values = np.arange(10.0)
        columns = (
            "X",
            "Y",
            "Z",
            "RotX",
            "RotY",
            "RotZ",
            "WhiteMatter",
            "aCompCor00",
            "aCompCor01",
            "aCompCor02",
            "aCompCor03",
            "aCompCor04",
            "aCompCor05",
            "GlobalSignal",
        )
        confounds = pd.DataFrame(
            {name: values + index for index, name in enumerate(columns)}
        )

        selected = _confound_columns(confounds, include_global_signal=False)
        selected_with_global = _confound_columns(confounds, include_global_signal=True)

        self.assertEqual(list(selected.columns), list(columns[:13]))
        self.assertEqual(list(selected_with_global.columns), list(columns))

    def test_discovers_shared_preprocessed_bold_and_brainmask_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            func_dir = root / "sub-10159" / "func"
            func_dir.mkdir(parents=True)
            bold_path = func_dir / (
                "sub-10159_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"
            )
            bold_path.touch()
            (
                func_dir
                / "sub-10159_task-rest_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"
            ).touch()
            pd.DataFrame({"FramewiseDisplacement": [0.1, 0.2]}).to_csv(
                func_dir / "sub-10159_task-rest_bold_confounds.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                {"participant_id": ["sub-10159"], "diagnosis": ["CONTROL"]}
            ).to_csv(root / "participants.tsv", sep="\t", index=False)

            runs, exclusions = discover_runs(root, task="rest")

            self.assertEqual(exclusions, [])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].bold_path, bold_path)
            self.assertEqual(
                runs[0].mask_path.name,
                "sub-10159_task-rest_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz",
            )
            self.assertIsNone(runs[0].metadata_path)

    def test_loads_tr_from_nifti_header_when_json_sidecar_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            bold_path = Path(temporary_directory) / "bold.nii.gz"
            image = nib.Nifti1Image(np.zeros((2, 2, 2, 5), dtype=np.float32), np.eye(4))
            image.header.set_zooms((1.0, 1.0, 1.0, 2.0))
            nib.save(image, bold_path)

            with patch("sensor_placement.data.PROJECT_DATA_ROOT", Path(temporary_directory) / "missing"):
                self.assertEqual(_load_tr(None, bold_path), 2.0)


if __name__ == "__main__":
    unittest.main()
