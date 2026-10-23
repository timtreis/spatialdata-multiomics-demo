"""Fit one method on one fold, without that fold's held-out proteins and genes, and save the cell representation.

    python methods.py METHOD FOLD WORK_DIR

Reads WORK_DIR/cells.h5mu and WORK_DIR/folds.json (from prep.py); writes WORK_DIR/cache/METHOD_fFOLD.npy (an
embedding) or .npz (the WNN graph). Existing outputs are skipped. Threads follow SLURM_CPUS_PER_TASK when set; scVI and
totalVI and MOFA+ train on a GPU.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
N_LATENT = 20
METHODS = ["rna_pca", "scvi", "prot_markers", "concat", "wnn", "mofa", "totalvi", "image", "prot_image",
           "rna_prot_image"]  # MOFA-FLEX: mofaflex_retest.py


def inputs(cells, fold):
    """Model inputs without the held-out features: RNA counts and log-normalized, protein log1p z-scores."""
    rna = cells.mod["rna"][:, ~cells.mod["rna"].var_names.isin(fold["genes"])].copy()
    prot = cells.mod["prot"][:, ~cells.mod["prot"].var_names.isin(fold["prot"])].copy()
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    prot.layers["intensity"] = prot.X.copy()
    prot.X = np.log1p(prot.X)
    sc.pp.scale(prot, max_value=10)
    return rna, prot


def rna_pca(rna, prot, cells, work_dir):
    sc.tl.pca(rna, n_comps=N_LATENT, random_state=0)
    return rna.obsm["X_pca"]


def prot_markers(rna, prot, cells, work_dir):
    return prot.X


def _stack(*blocks):
    # each block scaled to unit total variance, so that neither modality dominates the distances
    return np.hstack([b / np.sqrt(b.var(0).sum()) for b in blocks])


def concat(rna, prot, cells, work_dir):
    return _stack(rna_pca(rna, prot, cells, work_dir), prot.X)


def image(rna, prot, cells, work_dir):
    """PCA of the morphology features (`helpers.morphology`) of the cell shape, the DAPI and the training stains only,
    preprocessed as in the notebook with mantispy's single-cell workflow (`helpers.preprocess_morphology`)."""
    import anndata as ad
    from sklearn.decomposition import PCA

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import helpers

    raw = ad.read_h5ad(work_dir / "image_features.h5ad")
    raw.obs_names = raw.obs["cell_id"].astype(str).to_numpy()
    raw = raw[raw.obs_names.isin(cells.obs_names)].copy()
    profiles = helpers.morphology_profiles(raw, pd.DataFrame(cells.obsm["spatial"], index=cells.obs_names))
    image_shape = helpers.stain_image_shape(Path(__file__).resolve().parents[1] / "data")
    morph = helpers.preprocess_morphology(profiles, image_shape, channels=["DAPI", *prot.var_names])
    found = cells.obs_names.isin(morph.obs_names)
    print(f"{(~found).sum()} of {len(found)} cells without image features or failing their QC (set to the mean); "
          f"{morph.n_vars} features selected, removed per operation: {morph.uns['feature_select']}")
    X = np.zeros((cells.n_obs, morph.n_vars), dtype=np.float32)
    X[found] = morph[cells.obs_names[found]].X
    return PCA(n_components=N_LATENT, svd_solver="randomized", random_state=0).fit_transform(X)


def prot_image(rna, prot, cells, work_dir):
    return _stack(prot.X, image(rna, prot, cells, work_dir))


def rna_prot_image(rna, prot, cells, work_dir):
    return _stack(rna_pca(rna, prot, cells, work_dir), prot.X, image(rna, prot, cells, work_dir))


def wnn(rna, prot, cells, work_dir):
    import muon as mu

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import helpers

    helpers.patch_muon_wnn()
    rna.obsm["X_pca"] = rna_pca(rna, prot, cells, work_dir)
    sc.pp.neighbors(rna, n_neighbors=20, use_rep="X_pca")
    prot.obsm["X_markers"] = prot.X.copy()
    sc.pp.neighbors(prot, n_neighbors=20, use_rep="X_markers")
    mdata = md.MuData({"rna": rna, "prot": prot})
    mu.pp.neighbors(mdata, n_neighbors=20, key_added="wnn")
    print("mean modality weights", mdata.obs[["rna:mod_weight", "prot:mod_weight"]].mean().round(3).to_dict())
    return mdata.obsp["wnn_distances"].tocsr()


def mofa(rna, prot, cells, work_dir):
    import muon as mu

    mdata = md.MuData({"rna": rna, "prot": prot})
    outfile = work_dir / "cache" / f"mofa_tmp_{os.getpid()}.hdf5"
    mu.tl.mofa(mdata, n_factors=N_LATENT, use_var=None, scale_views=True, convergence_mode="fast", seed=0,
               use_float32=True, gpu_mode=True, outfile=str(outfile), save_data=False)
    outfile.unlink()
    return mdata.obsm["X_mofa"]


def _train(model):
    # scvi-tools defaults: the epoch heuristic and no early stopping (with the KL warm-up, the validation ELBO can
    # rise in the first epochs and stop the training after a few of them)
    model.train(accelerator="gpu", enable_progress_bar=False)
    return model.get_latent_representation()


def scvi(rna, prot, cells, work_dir):
    import scvi as sv

    sv.settings.seed = 0
    sv.model.SCVI.setup_anndata(rna, layer="counts")
    return _train(sv.model.SCVI(rna, n_latent=N_LATENT))


def totalvi(rna, prot, cells, work_dir):
    # totalVI models protein counts; Xenium protein values are mean intensities, rounded here: an approximation
    import scvi as sv

    sv.settings.seed = 0
    rna.obsm["prot_counts"] = pd.DataFrame(np.round(prot.layers["intensity"]), index=rna.obs_names,
                                           columns=prot.var_names)
    sv.model.TOTALVI.setup_anndata(rna, layer="counts", protein_expression_obsm_key="prot_counts")
    return _train(sv.model.TOTALVI(rna, n_latent=N_LATENT))


def main():
    method, fold_index, work_dir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    out = work_dir / "cache" / (f"{method}_f{fold_index}" + (".npz" if method == "wnn" else ".npy"))
    if out.exists():
        print("exists:", out)
        return
    out.parent.mkdir(exist_ok=True)
    import torch

    torch.set_num_threads(N_THREADS)
    cells = md.read_h5mu(work_dir / "cells.h5mu")
    fold = json.load(open(work_dir / "folds.json"))[fold_index]
    rna, prot = inputs(cells, fold)
    start = time.time()
    result = globals()[method](rna, prot, cells, work_dir)
    if method == "wnn":
        sp.save_npz(out, result)
    else:
        np.save(out, result)
    print(f"{method} fold {fold_index}: {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
