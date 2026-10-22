"""Fit one method on one fold, without that fold's held-out proteins and genes, and save the cell representation.

    python methods.py METHOD FOLD WORK_DIR

Reads WORK_DIR/cells.h5mu and WORK_DIR/folds.json (from prep.py); writes WORK_DIR/cache/METHOD_fFOLD.npy (an
embedding) or .npz (the WNN graph). Existing outputs are skipped. Threads follow SLURM_CPUS_PER_TASK when set; scVI and
totalVI and MOFA+ train on a GPU.
"""

import inspect
import json
import os
import sys
import textwrap
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


# image features that record where a cell is rather than what it looks like (they would leak the tissue region)
POSITIONAL = ("Center_", "BoundingBoxMinimum", "BoundingBoxMaximum", "Location_", "SpatialMoment_")


def image(rna, prot, cells, work_dir):
    """cp_measure features (image_features.py) of the cell shape, the DAPI and the training stains only: scaled and
    selected with mantispy (pycytominer's operations), then reduced by PCA."""
    import anndata as ad
    import mantispy as mt
    from sklearn.decomposition import PCA

    features = ad.read_h5ad(work_dir / "image_features.h5ad")
    channel = np.array([name.rsplit("__", 1)[1] if "__" in name else "" for name in features.var_names])
    keep = np.isin(channel, ["", "DAPI", *prot.var_names]) & ~features.var_names.str.startswith(POSITIONAL)
    position = pd.Index(features.obs["cell_id"]).get_indexer(cells.obs["cell_id"].astype(str))
    found = position >= 0
    features = features[position[found]][:, keep].copy()
    print(f"{(~found).sum()} of {len(found)} cells without image features (set to the mean)")
    # z-scores rather than mantispy's default median/MAD: on single cells most are negative for a sparse stain, its MAD
    # is 0 and mad_robustize discards its features (for PD-L1, CD20 and granzyme B, 10 of 155 kept instead of 25)
    mt.pp.normalize(features, method="standardize", by=None, reference=None)
    # the selection is fit on 50,000 random cells: on all of them the correlation pass takes long and keeps the same
    sample = np.sort(np.random.default_rng(0).choice(features.n_obs, min(50_000, features.n_obs), replace=False))
    subset = features[sample].copy()
    mt.pp.feature_select(subset)
    selected = subset.var["selected"].to_numpy()
    print(f"{selected.sum()} of {keep.sum()} image features selected; removed per operation:",
          subset.uns["mantispy"]["feature_select"])
    X = np.zeros((cells.n_obs, selected.sum()), dtype=np.float32)
    # z-scores: a missing value (a cell without image features, or an undefined feature of a cell) is set to the mean
    X[found] = np.nan_to_num(np.clip(np.asarray(features.X[:, selected], dtype=np.float32), -10, 10), nan=0)
    return PCA(n_components=N_LATENT, svd_solver="randomized", random_state=0).fit_transform(X)


def prot_image(rna, prot, cells, work_dir):
    return _stack(prot.X, image(rna, prot, cells, work_dir))


def rna_prot_image(rna, prot, cells, work_dir):
    return _stack(rna_pca(rna, prot, cells, work_dir), prot.X, image(rna, prot, cells, work_dir))


def patch_muon_wnn():
    """muon 0.2.0's WNN returns NaN for cells whose neighbors in a modality are at distance 0 (identical profiles)."""
    import muon.pp

    source = textwrap.dedent(inspect.getsource(muon.pp.neighbors))
    fixes = {
        # include the neighbors at distance 0, which `.nonzero()` drops
        "X[neighbordistances[cell, :].nonzero()[1], :]": (
            "X[neighbordistances.indices[neighbordistances.indptr[cell] : neighbordistances.indptr[cell + 1]], :]"
        ),
        # keep the kernel bandwidth above the nearest-neighbor distance, which is 0 for identical profiles
        "        currtheta = 0\n": "        csigmas = np.maximum(csigmas, nndistances + eps)\n        currtheta = 0\n",
    }
    for old, new in fixes.items():
        assert source.count(old) == 1, "the patch does not apply to this muon version"
        source = source.replace(old, new)
    exec(compile(source, muon.pp.__file__, "exec"), muon.pp.__dict__)


def wnn(rna, prot, cells, work_dir):
    import muon as mu

    patch_muon_wnn()
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
