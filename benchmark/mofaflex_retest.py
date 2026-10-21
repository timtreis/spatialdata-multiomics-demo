"""MOFA-FLEX on fold 0, in a few settings; not a benchmark arm, since it ignores most of the RNA here (README).

    python mofaflex_retest.py WORK_DIR CONFIG

Writes WORK_DIR/cache/mfx-CONFIG_f0.npy, which evaluate.py scores with the other methods, and prints the variance
explained per factor and view, and how many factors copy a single protein (|r| > 0.9 with one protein input).
"""
import json
import sys
import time
import warnings
from pathlib import Path

import mofaflex as mf
import mudata as md
import numpy as np
from methods import N_LATENT, inputs

warnings.filterwarnings("ignore")

work_dir, cfg = Path(sys.argv[1]), sys.argv[2]
CONFIGS = {
    # the setting of the first test (on CPU)
    "handover": dict(model=dict(weight_prior="Horseshoe", likelihoods="Normal"),
                     train=dict(batch_size=5000, lr=0.05, max_epochs=1000, early_stopper_patience=50)),
    # package defaults (Normal priors, lr 1e-3), full batch, PCA init
    "default-pca": dict(model=dict(weight_prior="Normal", likelihoods="Normal", init_factors="pca"),
                        train=dict(batch_size=0, lr=1e-3)),
    "horseshoe-pca": dict(model=dict(weight_prior="Horseshoe", likelihoods="Normal", init_factors="pca"),
                          train=dict(batch_size=0, lr=1e-3)),
    # RNA counts with the negative binomial, protein Gaussian
    "nb-pca": dict(model=dict(weight_prior="Horseshoe", likelihoods={"rna": "NegativeBinomial", "prot": "Normal"},
                              init_factors="pca"),
                   data=dict(layer={"rna": "counts", "prot": None}), train=dict(batch_size=0, lr=1e-3)),
}
c = CONFIGS[cfg]
cells = md.read_h5mu(work_dir / "cells.h5mu")
fold = json.load(open(work_dir / "folds.json"))[0]
rna, prot = inputs(cells, fold)
mdata = md.MuData({"rna": rna.copy(), "prot": prot.copy()})
start = time.time()
model = mf.MOFAFLEX(mdata, mf.DataOptions(subset_var=None, plot_data_overview=False, **c.get("data", {})),
                    mf.ModelOptions(n_factors=N_LATENT, **c["model"]),
                    mf.TrainingOptions(device="cuda", seed=0, **c["train"]))
z = next(iter(model.get_factors().values())).loc[mdata.obs_names].to_numpy(dtype=np.float32)
print(cfg, f"{time.time() - start:.0f}s, epochs {len(model.training_loss)}")
print("R2", model.get_r2())
zc = (z - z.mean(0)) / (z.std(0) + 1e-8)
pc = (prot.X - prot.X.mean(0)) / (prot.X.std(0) + 1e-8)
r = np.abs(zc.T @ pc / len(z))
print("factors copying one protein:", int((r.max(1) > 0.9).sum()), "of", z.shape[1])
(work_dir / "cache").mkdir(exist_ok=True)
np.save(work_dir / "cache" / f"mfx-{cfg}_f0.npy", z)
