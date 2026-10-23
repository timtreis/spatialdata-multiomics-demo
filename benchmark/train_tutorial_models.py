"""Train the tutorial's joint model (MOFA+ on RNA and morphology) on all QC-filtered cells, for the notebook.

    python train_tutorial_models.py WORK_DIR MORPH.h5ad OUT_DIR

Reads WORK_DIR/cells.h5mu (from `prep.py` without --n-cells) and the morphology features (`helpers.morphology`, the
notebook's data/morph.h5ad), and writes OUT_DIR/mofa.npz keyed by cell_id (OUT_DIR is `notebooks/cache` in this
repository). The preprocessing and the training are those of the notebook (`notebooks/helpers.py`), on the cells that
have both modalities. MOFA+ trains on a GPU.
"""

import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import mudata as md
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import helpers  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    work_dir, morph_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = md.read_h5mu(work_dir / "cells.h5mu")
    rna = cells.mod["rna"]
    raw = ad.read_h5ad(morph_file)
    raw.obs_names = raw.obs["cell_id"].astype(str).to_numpy()
    raw = raw[raw.obs_names.isin(rna.obs_names)].copy()
    centroids = pd.DataFrame(cells.obsm["spatial"], index=cells.obs_names)
    profiles = helpers.morphology_profiles(raw, centroids)
    morph = helpers.preprocess_morphology(profiles, helpers.stain_image_shape(morph_file.parent))
    rna = rna[morph.obs_names].copy()
    helpers.normalize_rna(rna)
    print(f"{rna.n_obs} cells, {rna.n_vars} genes, {morph.n_vars} morphology features", morph.uns["feature_select"])
    start = time.time()
    helpers.train_mofa({"rna": rna, "morph": morph}, out_dir / "mofa.npz")
    print(f"mofa: {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
