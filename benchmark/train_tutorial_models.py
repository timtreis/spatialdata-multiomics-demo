"""Train the tutorial's joint models on all QC-filtered cells and all features, and save their outputs for the notebook.

    python train_tutorial_models.py WORK_DIR OUT_DIR [totalvi] [mofa]

Reads WORK_DIR/cells.h5mu (from `prep.py` without --n-cells) and writes OUT_DIR/totalvi.npz and OUT_DIR/mofa.npz,
keyed by cell_id, so that the notebook can load them (OUT_DIR is `notebooks/cache` in this repository). The
preprocessing and the training are those of the notebook (`notebooks/helpers.py`). The models train on a GPU.
"""

import sys
import time
import warnings
from pathlib import Path

import mudata as md

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import helpers  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    work_dir, out_dir, which = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:] or ["totalvi", "mofa"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = md.read_h5mu(work_dir / "cells.h5mu")
    rna, prot = cells.mod["rna"].copy(), cells.mod["prot"].copy()
    rna.obs["cell_id"] = cells.obs["cell_id"].to_numpy()
    helpers.normalize(rna, prot)
    for name in which:
        start = time.time()
        getattr(helpers, f"train_{name}")(rna, prot, out_dir / f"{name}.npz")
        print(f"{name}: {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
