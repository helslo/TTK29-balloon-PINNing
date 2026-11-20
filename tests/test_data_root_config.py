import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ttk29_balloon_pinning import _build_parser
from ttk29_balloon_pinning.config import DATA_ROOT_ENV, DEFAULT_DATA_ROOT, get_data_root


class DataRootConfigTests(unittest.TestCase):
    def test_defaults_to_configured_shared_dataset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_data_root(), DEFAULT_DATA_ROOT)
            self.assertEqual(
                _build_parser().parse_args(["compare"]).data_root,
                DEFAULT_DATA_ROOT,
            )

    def test_environment_path_is_used_by_default(self) -> None:
        configured_root = Path("/shared/fmri/ds000030")
        with patch.dict(os.environ, {DATA_ROOT_ENV: str(configured_root)}, clear=True):
            self.assertEqual(get_data_root(), configured_root)
            self.assertEqual(
                _build_parser().parse_args(["compare"]).data_root,
                configured_root,
            )

    def test_explicit_cli_path_overrides_environment(self) -> None:
        configured_root = Path("/shared/fmri/ds000030")
        cli_root = Path("/another/dataset")
        with patch.dict(os.environ, {DATA_ROOT_ENV: str(configured_root)}, clear=True):
            self.assertEqual(
                _build_parser()
                .parse_args(["compare", "--data-root", str(cli_root)])
                .data_root,
                cli_root,
            )

    def test_rejects_empty_environment_path(self) -> None:
        with patch.dict(os.environ, {DATA_ROOT_ENV: "  "}, clear=True):
            with self.assertRaisesRegex(ValueError, DATA_ROOT_ENV):
                get_data_root()


if __name__ == "__main__":
    unittest.main()
