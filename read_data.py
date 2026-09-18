import pandas as pd
import openneuro as on
from nilearn import datasets
from typing import Optional


DATASET = "ds000030"
PARTICIPANTS_FILE = "data/participants.tsv"
TARGET_DIR = "data/ds000030"
TASK = "stopsignal"
ROI = "Precentral Gyrus"
N: Optional[int] = 15


def select_participants(
    participants_file: str = PARTICIPANTS_FILE,
    n: Optional[int] = N,
) -> pd.DataFrame:
    """Return healthy controls with stop-signal data, in TSV order."""
    if n is not None and n < 0:
        raise ValueError("n must be non-negative or None")

    participants = pd.read_csv(participants_file, sep="\t")
    eligible = participants[
        participants["diagnosis"].eq("CONTROL")
        & participants["stopsignal"].eq(1)
    ]
    return eligible if n is None else eligible.head(n)


def download_selected_data(participants: pd.DataFrame) -> None:
    """Download stop-signal functional files for the selected participants."""
    on.download(
        dataset=DATASET,
        target_dir=TARGET_DIR,
        include=[
            f"{participant_id}/func/*task-{TASK}*"
            for participant_id in participants["participant_id"]
        ],
    )


def load_roi():
    """Fetch the Harvard-Oxford atlas used to identify left M1."""
    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    if ROI not in atlas["labels"]:
        raise ValueError(f"ROI {ROI!r} was not found in the atlas labels")
    return atlas


if __name__ == "__main__":
    selected = select_participants()
    print(f"Selected {len(selected)} participants for {TASK}: {ROI}")
    print(", ".join(selected["participant_id"]))
    load_roi()
    download_selected_data(selected)

