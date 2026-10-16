"""kNN prediction of held-out proteins and genes, and of the pathology region, from each cell representation.

    python evaluate.py WORK_DIR

For each fold, a test cell's held-out values are predicted as the mean of those of its K nearest training cells in
the representation, and scored by the Pearson correlation with the measured values (per feature, averaged). Writes
results_per_fold.csv, results_per_feature.csv and results_summary.csv to WORK_DIR.
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import mudata as md, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

K = 20
work_dir = sys.argv[1]
sub = md.read_h5mu(f"{work_dir}/cells.h5mu")
folds = json.load(open(f"{work_dir}/folds.json"))

# targets: protein log1p z-scores, gene log-normalized expression (library size from all genes)
prot = sub.mod["prot"].copy(); prot.X = np.log1p(prot.X); sc.pp.scale(prot, max_value=10)
rna = sub.mod["rna"].copy(); sc.pp.normalize_total(rna); sc.pp.log1p(rna)
P = pd.DataFrame(prot.X, columns=prot.var_names)
G = pd.DataFrame(rna.X.toarray(), columns=rna.var_names)
region = sub.obs["pathology"].astype(str).to_numpy()
evaluable = np.isin(region, ["Tumor", "Immune infiltration"])  # necrosis/adipose: too few cells


def neighbors(rep, train, test):
    """Indices (into train) of each test cell's K nearest training cells."""
    if sp.issparse(rep):  # a precomputed graph (WNN): keep each test cell's neighbors that are training cells
        pos = np.full(rep.shape[0], -1); pos[train] = np.arange(len(train))
        rows = rep[test]
        nbrs = [pos[rows.indices[rows.indptr[i]:rows.indptr[i + 1]]] for i in range(len(test))]
        return [n[n >= 0] for n in nbrs]
    nn = NearestNeighbors(n_neighbors=K).fit(rep[train])
    return list(nn.kneighbors(rep[test], return_distance=False))


def pearson(y, yhat):
    y, yhat = y - y.mean(0), yhat - yhat.mean(0)
    return (y * yhat).sum(0) / np.sqrt((y ** 2).sum(0) * (yhat ** 2).sum(0))


rows, per_feature = [], []
for path in sorted(glob.glob(f"{work_dir}/cache/*_f[0-9].np[yz]")):
    method, f = os.path.basename(path).rsplit("_f", 1); f = int(f[0])
    rep = sp.load_npz(path) if path.endswith(".npz") else np.load(path)
    test = np.array(folds[f]["test_cells"]); train = np.setdiff1d(np.arange(sub.n_obs), test)
    nbrs = neighbors(rep, train, test)
    empty = np.mean([len(n) == 0 for n in nbrs])
    res = {"method": method, "fold": f, "no_train_neighbors": empty}
    for kind, Y, feats in [("protein", P, folds[f]["prot"]), ("gene", G, folds[f]["genes"])]:
        Ytr = Y[feats].to_numpy()[train]
        yhat = np.stack([Ytr[n].mean(0) if len(n) else Ytr.mean(0) for n in nbrs])
        r = pearson(Y[feats].to_numpy()[test], yhat)
        res[f"{kind}_r"] = np.nanmean(r)
        per_feature += [{"method": method, "fold": f, "kind": kind, "feature": ft, "r": v} for ft, v in zip(feats, r)]
    lab_tr = region[train]
    pred = np.array([pd.Series(lab_tr[n]).mode()[0] if len(n) else "Tumor" for n in nbrs])
    m = evaluable[test]
    res["region_bal_acc"] = balanced_accuracy_score(region[test][m], pred[m])
    rows.append(res)

df = pd.DataFrame(rows)
df.to_csv(f"{work_dir}/results_per_fold.csv", index=False)
pd.DataFrame(per_feature).to_csv(f"{work_dir}/results_per_feature.csv", index=False)
summary = df.drop(columns="fold").groupby("method").agg(["mean", "std"]).round(3)
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.sort_values("protein_r_mean", ascending=False)
summary.to_csv(f"{work_dir}/results_summary.csv")
print(summary.to_string())
