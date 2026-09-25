"""Create Julich-Brain v3.1 ROI masks with siibra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import siibra


JULICH_V31 = "Julich-Brain Cytoarchitectonic Atlas (v3.1)"
MNI_SPACE = "MNI 152 ICBM 2009c Nonlinear Asymmetric"
LEFT_M1_REGIONS = ("Area 4a (PreCG) left", "Area 4p (PreCG) left")


def create_julich_mask(
    output_path: Path,
    regions: Tuple[str, ...] = LEFT_M1_REGIONS,
    space: str = MNI_SPACE,
) -> Path:
    """Fetch and union Julich regions into one binary NIfTI mask."""
    if not regions:
        raise ValueError("at least one Julich region is required")
    parcellation = siibra.parcellations.get(JULICH_V31)
    images = []
    for region_name in regions:
        region = parcellation.get_region(region_name)
        if region is None:
            raise ValueError(f"Julich region not found: {region_name}")
        images.append(region.get_regional_mask(space).fetch("nii"))

    reference = images[0]
    combined = np.zeros(reference.shape, dtype=np.uint8)
    for image in images:
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise ValueError("Julich region masks do not share a common grid")
        combined |= np.asarray(image.dataobj) > 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(combined, reference.affine, reference.header), output_path)
    metadata = {
        "atlas": JULICH_V31,
        "space": space,
        "regions": list(regions),
        "mask_type": "binary union",
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/julich_left_m1_mask.nii.gz"))
    parser.add_argument("--region", action="append", dest="regions", help="Julich region name; repeat to union regions")
    parser.add_argument("--space", default=MNI_SPACE)
    args = parser.parse_args(argv)
    path = create_julich_mask(args.output, tuple(args.regions) if args.regions else LEFT_M1_REGIONS, args.space)
    print(path)


if __name__ == "__main__":
    main()