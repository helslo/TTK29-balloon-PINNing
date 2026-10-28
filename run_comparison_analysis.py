"""Edit the settings below, then run this file to launch a model comparison."""

from __future__ import annotations

from pathlib import Path

from model_comparison.run import main


# Select "physical", "pinn", "costa", or "compare".
MODE = "compare"
EPOCHS = 2000

# Leave DATA_FILE unset to use deterministic synthetic data.
DATA_FILE: Path | None = None
SUBJECT: str | None = None
ROI_MASK: Path | None = None

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
    if OUTPUT_DIR is not None:
        arguments.extend(["--output-dir", str(OUTPUT_DIR)])
    if not FIT_PHYSICAL_MODEL:
        arguments.append("--no-fit-physical")
    main(arguments)


if __name__ == "__main__":
    main_from_config()
