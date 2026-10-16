"""QC, pathology region of each cell, and the folds of held-out features and test cells.

    python prep.py DATA_DIR WORK_DIR [--n-cells N]

DATA_DIR is the directory filled by `notebooks/helpers.download`. Writes WORK_DIR/cells.h5mu (the QC-filtered cells,
or a random subset of N of them) and WORK_DIR/folds.json.
"""

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from shapely.affinity import affine_transform
from spatialdata_io import xenium

warnings.filterwarnings("ignore")

MIN_TRANSCRIPTS, MIN_AREA, MAX_AREA = 20, 10, 600  # µm²; the same thresholds as the tutorial
PIXEL_SIZE = 0.2125
N_FOLDS = 3
# regions drawn on top of the tumor annotation take precedence over it
REGION_ORDER = ["Tumor", "Adipose tissue", "Necrosis", "Immune infiltration"]


def pathology_regions(data_dir: Path, xy_um: np.ndarray) -> np.ndarray:
    """The pathology region of each cell centroid (µm), from the annotation drawn on the H&E."""
    m = np.loadtxt(data_dir / "he_imagealignment.csv", delimiter=",") * PIXEL_SIZE  # H&E pixels -> µm
    ann = gpd.read_file(data_dir / "pathology_annotation.geojson")[["name", "geometry"]].set_crs(None, allow_override=True)
    ann["geometry"] = ann.geometry.apply(lambda p: affine_transform(p, [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]]))
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy_um[:, 0], xy_um[:, 1]))
    region = np.full(len(points), "unannotated", dtype=object)
    for name in REGION_ORDER:
        region[gpd.sjoin(points, ann[ann["name"] == name], predicate="within").index.unique()] = name
    return region


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--n-cells", type=int, default=None, help="random subset of the QC-filtered cells")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sdata = xenium(args.data_dir / "outs", multimodal=True, transcripts=False, morphology_mip=False,
                   morphology_focus=False, aligned_images=False, cells_labels=False, nucleus_labels=False,
                   cells_boundaries=False, nucleus_boundaries=False)
    mdata = sdata["table"]
    obs = mdata.obs
    keep = (obs["transcript_counts"] >= MIN_TRANSCRIPTS) & obs["cell_area"].between(MIN_AREA, MAX_AREA)
    print("QC keeps", int(keep.sum()), "of", len(keep), "cells")
    mdata.obs["pathology"] = pd.Categorical(pathology_regions(args.data_dir, mdata.obsm["spatial"]))

    idx = np.flatnonzero(keep.to_numpy())
    if args.n_cells is not None:
        idx = np.sort(rng.choice(idx, args.n_cells, replace=False))
    cells = mdata[idx].copy()
    rna, prot = cells.mod["rna"], cells.mod["prot"]
    rna.X = sp.csr_matrix(rna.X).astype(np.float32)
    prot.X = np.asarray(prot.X.toarray() if sp.issparse(prot.X) else prot.X, dtype=np.float32)
    print(cells.obs["pathology"].value_counts().to_dict())

    # proteins split into 3 groups of 9; genes detected in >= 10% of cells split into 3; 20% test cells per fold
    proteins = rng.permutation(prot.var_names.to_numpy())
    detected = np.asarray((rna.X > 0).mean(0)).ravel()
    genes = rng.permutation(rna.var_names[detected >= 0.10].to_numpy())
    folds = []
    for f in range(N_FOLDS):
        test = rng.random(cells.n_obs) < 0.2
        folds.append({"prot": sorted(proteins[f::N_FOLDS].tolist()), "genes": sorted(genes[f::N_FOLDS].tolist()),
                      "test_cells": np.flatnonzero(test).tolist()})
        print(f"fold {f}: {len(folds[-1]['prot'])} proteins, {len(folds[-1]['genes'])} genes, {test.sum()} test cells")
    json.dump(folds, open(args.work_dir / "folds.json", "w"))
    cells.write_h5mu(args.work_dir / "cells.h5mu")


if __name__ == "__main__":
    main()
