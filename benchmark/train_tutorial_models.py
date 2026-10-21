"""Train the tutorial's joint models on all QC-filtered cells and all features, and save their outputs for the notebook.

    python train_tutorial_models.py WORK_DIR OUT_DIR [totalvi] [mofa]

Reads WORK_DIR/cells.h5mu (from `prep.py` without --n-cells) and writes OUT_DIR/totalvi.npz and OUT_DIR/mofa.npz,
keyed by cell_id, so that the notebook can load them (OUT_DIR is `notebooks/cache` in this repository). The
preprocessing is the one of the notebook. The models train on a GPU.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
N_LATENT = 20


def main():
    work_dir, out_dir, which = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:] or ["totalvi", "mofa"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = md.read_h5mu(work_dir / "cells.h5mu")
    rna, prot = cells.mod["rna"].copy(), cells.mod["prot"].copy()
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    prot.layers["intensity"] = prot.X.copy()
    prot.X = np.log1p(prot.X)
    sc.pp.scale(prot, max_value=10)
    cell_id = cells.obs["cell_id"].to_numpy().astype(str)

    for name in which:
        start = time.time()
        if name == "totalvi":
            import scvi
            import torch

            torch.set_num_threads(N_THREADS)
            scvi.settings.seed = 0
            adata = rna.copy()
            # totalVI models protein counts; Xenium protein values are mean intensities, rounded here
            adata.obsm["prot_counts"] = pd.DataFrame(np.round(prot.layers["intensity"]), index=adata.obs_names,
                                                     columns=prot.var_names)
            scvi.model.TOTALVI.setup_anndata(adata, layer="counts", protein_expression_obsm_key="prot_counts")
            model = scvi.model.TOTALVI(adata, n_latent=N_LATENT)
            model.train(accelerator="gpu", enable_progress_bar=False)
            np.savez_compressed(out_dir / "totalvi.npz", cell_id=cell_id,
                                latent=model.get_latent_representation().astype(np.float16))
        elif name == "mofa":
            import muon as mu

            mdata = md.MuData({"rna": rna, "prot": prot})
            outfile = out_dir / "mofa_model.hdf5"
            mu.tl.mofa(mdata, n_factors=N_LATENT, use_var=None, scale_views=True, convergence_mode="fast", seed=0,
                       use_float32=True, gpu_mode=True, outfile=str(outfile),
                       save_data=False)
            variance = mdata.uns["mofa"]["variance"]
            weights = mdata.varm["LFs"].astype(np.float32)  # rows: the RNA features, then the protein features
            np.savez_compressed(
                out_dir / "mofa.npz",
                cell_id=cell_id,
                factors=mdata.obsm["X_mofa"].astype(np.float16),
                genes=rna.var_names.to_numpy().astype(str),
                weights_rna=weights[: rna.n_vars],
                proteins=prot.var_names.to_numpy().astype(str),
                weights_prot=weights[rna.n_vars :],
                r2_rna=np.asarray(variance["rna"], dtype=np.float32),
                r2_prot=np.asarray(variance["prot"], dtype=np.float32),
            )
            outfile.unlink()
        print(f"{name}: {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
