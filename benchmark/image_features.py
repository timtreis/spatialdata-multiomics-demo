"""Image features of every cell (cp_measure through squidpy) from the DAPI and the 27 protein stains.

    python image_features.py DATA_DIR STORE.zarr OUT.h5ad [--crop UM]

Uses the cell segmentation and the "stains" image of `notebooks/helpers.load`, written once to STORE.zarr: squidpy
sends the image to worker processes, which works for zarr-backed images but not for the lazily read OME-TIFFs. With
--crop, only a central window of UM x UM µm is featurized (to time it and look at the features). Worker processes
follow SLURM_CPUS_PER_TASK.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spatialdata as sd
import squidpy as sq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import helpers  # noqa: E402

FEATURES = ["cp_measure:intensity", "cp_measure:sizeshape", "cp_measure:texture", "cp_measure:granularity",
            "cp_measure:radial"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--crop", type=float, default=None)
    args = parser.parse_args()
    sdata = helpers.load(args.data_dir)
    table = sdata["table"]
    center = np.median(table.obsm["spatial"], axis=0) / helpers.PIXEL_SIZE
    # label value of each cell in the segmentation -> cell_id, to align the features with the table
    instance_key = table.uns["spatialdata_attrs"]["instance_key"]
    cell_id = pd.Series(table.obs["cell_id"].to_numpy().astype(str), index=table.obs[instance_key].to_numpy())
    sdata = sdata.subset(["stains", "cell_labels"])
    if args.crop is not None:
        cx, cy = center
        half = args.crop / 2 / helpers.PIXEL_SIZE
        sdata = sdata.query.bounding_box(axes=("x", "y"), min_coordinate=[cx - half, cy - half],
                                         max_coordinate=[cx + half, cy + half], target_coordinate_system="global")
    if not args.store.exists():
        start = time.time()
        sdata.write(args.store)
        print(f"wrote {args.store} in {time.time() - start:.0f}s", flush=True)
    sdata = sd.read_zarr(args.store)
    start = time.time()
    features = sq.experimental.im.calculate_image_features(
        # each stain scaled to 0-1 by its own range (intensity_range="auto"), which cp_measure's texture needs
        sdata, image_key="stains", labels_key="cell_labels", scale="scale0", features=FEATURES, intensity_range="auto",
        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())), inplace=False,
    )
    print(f"{features.n_obs} cells x {features.n_vars} features in {time.time() - start:.0f}s", flush=True)
    features.obs["cell_id"] = cell_id.reindex(features.obs["label_id"].to_numpy()).to_numpy()
    features.write_h5ad(args.out)


if __name__ == "__main__":
    main()
