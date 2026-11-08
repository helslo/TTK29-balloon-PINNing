"""Edit the settings below, then run this file to launch a model comparison."""

from __future__ import annotations

from pathlib import Path

from model_comparison.run import main


# Select "physical", "pinn", "costa", or "compare".
MODE = "compare"
EPOCHS = 2000

# Leave DATA_FILE and SUBJECT unset to use the default real BIDS subject.
DATA_FILE: Path | None = None
SUBJECT: str | None = None
ROI_MASK: Path | None = None
BOLD_PATH: Path | None = None
CONFOUNDS_PATH: Path | None = None
USE_FMRIPREP = False
SCHAEFER_NETWORK: str | None = None
SCHAEFER_HEMISPHERE = "both"

# Leave OUTPUT_DIR unset to create a timestamped folder under results/.
OUTPUT_DIR: Path | None = None
FIT_PHYSICAL_MODEL = True


def main_from_config() -> None:
    """Translate the editable settings above into the package CLI."""
    arguments = ["--mode", MODE, "--epochs", str(EPOCHS)]
    if DATA_FILE is not None:
        arguments.extend(["--data", str(DATA_FILE)])
    if SUBJECT is not None:
        arguments.extend(["--subject", SUBJECT])
    if ROI_MASK is not None:
        arguments.extend(["--roi-mask", str(ROI_MASK)])
    if SCHAEFER_NETWORK is not None:
        arguments.extend(["--schaefer-network", SCHAEFER_NETWORK])
    if SCHAEFER_HEMISPHERE != "both":
        arguments.extend(["--schaefer-hemisphere", SCHAEFER_HEMISPHERE])
    if BOLD_PATH is not None:
        arguments.extend(["--bold-path", str(BOLD_PATH)])
    if CONFOUNDS_PATH is not None:
        arguments.extend(["--confounds-path", str(CONFOUNDS_PATH)])
    if USE_FMRIPREP:
        arguments.append("--fmriprep")
    if OUTPUT_DIR is not None:
        arguments.extend(["--output-dir", str(OUTPUT_DIR)])
    if not FIT_PHYSICAL_MODEL:
        arguments.append("--no-fit-physical")
    main(arguments)


if __name__ == "__main__":
    main_from_config()
