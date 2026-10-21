"""Data access for the Xenium gene + protein renal cell carcinoma tutorial.

The 10x Genomics bundle of this dataset is a 36 GB zip, most of it full-resolution morphology images. `download`
fetches only the files the tutorial uses (about 22 GB), reading single members of the remote zips with HTTP range
requests. `load` assembles them into a SpatialData object with a MuData table; the images are read lazily from their
OME-TIFF pyramids, without converting them.
"""

import io
import shutil
import zipfile
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import requests
import tifffile
import xarray as xr
import zarr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.transformations import Affine, Identity, Scale, Sequence
from spatialdata_io import xenium

BASE_URL = "https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Human_Kidney_FFPE_Protein_updated"
PREFIX = "Xenium_V1_Human_Kidney_FFPE_Protein_updated"
# the full output bundle (cell table, segmentation, count matrix) and the Xenium Explorer bundle (morphology images)
OUTS_ZIP = f"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/{PREFIX}/{PREFIX}_outs.zip"
EXPLORER_ZIP = f"{BASE_URL}/{PREFIX}_xe_outs.zip"
OUTS_FILES = [
    "experiment.xenium",
    "gene_panel.json",
    "protein_panel.json",
    "metrics_summary.csv",
    "cells.parquet",
    "cell_boundaries.parquet",
    "nucleus_boundaries.parquet",
    "cell_feature_matrix.h5",
    "cells.zarr.zip",
]
# DAPI and the 27 protein stains (the background, 18S and multiplexed segmentation stains are not downloaded)
STAINS = {"DAPI": "ch0000_dapi"} | {
    name: f"ch{i:04d}_{name.lower()}"
    for i, name in [(8, "CD4"), (9, "CD20"), (10, "CD8A"), (11, "CD3E"), (12, "CD138"), (13, "HLA-DR"), (14, "CD11c"),
                    (15, "CD68"), (16, "CD16"), (17, "GranzymeB"), (18, "CD163"), (19, "PD-1"), (20, "VISTA"),
                    (21, "PD-L1"), (22, "LAG-3"), (23, "CD45RA"), (24, "PCNA"), (25, "CD45RO"), (26, "Ki-67"),
                    (27, "Beta-catenin"), (28, "CD31"), (29, "PTEN"), (30, "PanCK"), (31, "Vimentin"),
                    (32, "alphaSMA"), (33, "CD45"), (34, "E-Cadherin")]
}
EXTRA_FILES = {  # file name -> URL suffix
    "he_image.ome.tif": "he_image.ome.tif",
    "he_imagealignment.csv": "he_imagealignment.csv",
    "pathology_annotation.geojson": "annotation.geojson",
}
PIXEL_SIZE = 0.2125  # µm per pixel of the morphology images (the "global" coordinate system)


class _RemoteFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests, so that zipfile reads only the members it needs."""

    def __init__(self, url: str):
        self.url, self.pos, self.session = url, 0, requests.Session()
        self.size = int(self.session.head(url, allow_redirects=True).headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        response = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=600)
        response.raise_for_status()
        buffer[: len(response.content)] = response.content
        self.pos += len(response.content)
        return len(response.content)


def _extract(url: str, members: list[str], out: Path) -> None:
    missing = [m for m in members if not (out / m).exists()]
    if not missing:
        return
    with zipfile.ZipFile(io.BufferedReader(_RemoteFile(url), buffer_size=64 << 20)) as z:
        for member in missing:
            target = out / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(f"{target}.part", "wb") as dst:
                shutil.copyfileobj(src, dst, 64 << 20)
            Path(f"{target}.part").rename(target)
            print("downloaded", member)


def download(data_dir: str | Path) -> Path:
    """Download the files of the tutorial into `data_dir` (about 22 GB); files already present are skipped."""
    data_dir = Path(data_dir)
    _extract(OUTS_ZIP, OUTS_FILES, data_dir / "outs")
    _extract(EXPLORER_ZIP, [f"morphology_focus/{f}.ome.tif" for f in STAINS.values()], data_dir)
    for name, suffix in EXTRA_FILES.items():
        target = data_dir / name
        if not target.exists():
            with requests.get(f"{BASE_URL}/{PREFIX}_{suffix}", stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(f"{target}.part", "wb") as f:
                    shutil.copyfileobj(r.raw, f, 64 << 20)
            Path(f"{target}.part").rename(target)
            print("downloaded", name)
    return data_dir


def _pyramid(path: Path, n_levels: int) -> list[da.Array]:
    group = zarr.open(tifffile.imread(path, aszarr=True, is_ome=False), mode="r")
    return [da.from_zarr(group[str(i)]) for i in range(n_levels)]


def _lazy_image(levels: list[da.Array], c_coords: list[str], transformation) -> xr.DataTree:
    """A multiscale image from existing pyramid levels (c, y, x), read lazily."""
    scale0 = Image2DModel.parse(levels[0], dims=("c", "y", "x"), c_coords=c_coords,
                                transformations={"global": transformation})
    tree = {"scale0": xr.Dataset({"image": scale0})}
    for i, level in enumerate(levels[1:], start=1):
        factors = [levels[0].shape[1] / level.shape[1], levels[0].shape[2] / level.shape[2]]
        # pixel centers in full-resolution pixels, as in the pyramids that spatialdata builds (needed by queries)
        coords = {"c": c_coords, "y": (np.arange(level.shape[1]) + 0.5) * factors[0],
                  "x": (np.arange(level.shape[2]) + 0.5) * factors[1]}
        image = xr.DataArray(level, dims=("c", "y", "x"), coords=coords)
        image.attrs["transform"] = {"global": Sequence([Scale(factors, axes=("y", "x")), transformation])}
        tree[f"scale{i}"] = xr.Dataset({"image": image})
    tree = xr.DataTree.from_dict(tree)
    Image2DModel().validate(tree)
    return tree


def load(data_dir: str | Path) -> SpatialData:
    """The dataset as SpatialData: segmentation, cell boundaries, a MuData table (RNA + protein), stains, H&E, pathology."""
    data_dir = Path(data_dir)
    sdata = xenium(
        data_dir / "outs",
        multimodal=True,  # the table is a MuData object with an "rna" and a "prot" modality
        transcripts=False,
        morphology_mip=False,
        morphology_focus=False,  # the stains are added below, without the background channels
        aligned_images=False,
        nucleus_labels=False,
    )
    # the stains, in the coordinate system of the segmentation (morphology pixels)
    stains = [_pyramid(data_dir / "morphology_focus" / f"{f}.ome.tif", 5) for f in STAINS.values()]
    levels = [da.stack([s[i] for s in stains]) for i in range(5)]
    sdata["stains"] = _lazy_image(levels, list(STAINS), Identity())
    # the H&E, registered to the morphology pixels by the alignment matrix provided by 10x
    he_to_morphology = Affine(
        np.loadtxt(data_dir / "he_imagealignment.csv", delimiter=","), input_axes=("x", "y"), output_axes=("x", "y")
    )
    he = [level.transpose(2, 0, 1) for level in _pyramid(data_dir / "he_image.ome.tif", 6)]  # (y, x, rgb) -> (c, y, x)
    sdata["he"] = _lazy_image(he, ["r", "g", "b"], he_to_morphology)
    # pathology annotation, drawn on the H&E
    pathology = gpd.read_file(data_dir / "pathology_annotation.geojson")[["name", "geometry"]]
    pathology = pathology.rename(columns={"name": "region"}).set_crs(None, allow_override=True)
    sdata["pathology"] = ShapesModel.parse(gpd.GeoDataFrame(pathology), transformations={"global": he_to_morphology})
    return sdata


# regions drawn on top of the tumor annotation take precedence over it
REGION_ORDER = ["Tumor", "Adipose tissue", "Necrosis", "Immune infiltration"]


def pathology_region(sdata: SpatialData, table_name: str = "table") -> np.ndarray:
    """The pathology region of each cell of the table, from its centroid and the annotation drawn on the H&E."""
    from spatialdata import transform

    regions = transform(sdata["pathology"], to_coordinate_system="global")
    xy = sdata[table_name].obsm["spatial"] / PIXEL_SIZE  # µm -> morphology pixels
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]))
    region = np.full(len(points), "unannotated", dtype=object)
    for name in REGION_ORDER:
        inside = gpd.sjoin(points, regions[regions["region"] == name], predicate="within").index.unique()
        region[inside] = name
    return region


def normalize(rna, prot) -> None:
    """In place: RNA counts to log-normalized expression, protein intensities to log1p z-scores (raw ones in layers)."""
    import scanpy as sc
    import scipy.sparse as sp

    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    prot.layers["intensity"] = np.asarray(prot.X.toarray() if sp.issparse(prot.X) else prot.X, dtype=np.float32)
    prot.X = np.log1p(prot.layers["intensity"])
    sc.pp.scale(prot, max_value=10)


N_LATENT = 20


def train_totalvi(rna, prot, out: Path) -> None:
    """totalVI on a GPU; saves its latent space keyed by cell_id. The protein intensities are rounded to counts."""
    import pandas as pd
    import scvi

    scvi.settings.seed = 0
    adata = rna.copy()
    adata.obsm["prot_counts"] = pd.DataFrame(
        np.round(prot.layers["intensity"]), index=adata.obs_names, columns=prot.var_names
    )
    scvi.model.TOTALVI.setup_anndata(adata, layer="counts", protein_expression_obsm_key="prot_counts")
    model = scvi.model.TOTALVI(adata, n_latent=N_LATENT)
    # scvi-tools defaults: the epoch heuristic, no early stopping (with the KL warm-up it stops after a few epochs)
    model.train(accelerator="gpu", enable_progress_bar=False)
    np.savez_compressed(out, cell_id=rna.obs["cell_id"].to_numpy().astype(str),
                        latent=model.get_latent_representation().astype(np.float16))


def train_mofa(rna, prot, out: Path) -> None:
    """MOFA+ on a GPU; saves the factors (keyed by cell_id), the weights and the variance explained per view."""
    import h5py
    import mudata as md
    import muon as mu

    mdata = md.MuData({"rna": rna.copy(), "prot": prot.copy()})
    model_file = Path(out).with_suffix(".hdf5")
    mu.tl.mofa(mdata, n_factors=N_LATENT, use_var=None, scale_views=True, convergence_mode="fast", seed=0,
               use_float32=True, gpu_mode=True, outfile=str(model_file), save_data=False)
    variance = mdata.uns["mofa"]["variance"]
    # the weights per view, from the model file: in mdata.varm the order of the views is not that of mdata.mod
    with h5py.File(model_file) as f:
        weights = {view: f[f"expectations/W/{view}"][:].T.astype(np.float32) for view in ("rna", "prot")}
        names = {view: f[f"features/{view}"][:].astype(str) for view in ("rna", "prot")}
    assert (names["rna"] == rna.var_names).all() and (names["prot"] == prot.var_names).all()
    np.savez_compressed(
        out,
        cell_id=rna.obs["cell_id"].to_numpy().astype(str),
        factors=mdata.obsm["X_mofa"].astype(np.float16),
        genes=names["rna"],
        weights_rna=weights["rna"],
        proteins=names["prot"],
        weights_prot=weights["prot"],
        r2_rna=np.asarray(variance["rna"], dtype=np.float32),
        r2_prot=np.asarray(variance["prot"], dtype=np.float32),
    )
    model_file.unlink()
