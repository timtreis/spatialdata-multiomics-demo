"""Data access for the Xenium gene + protein renal cell carcinoma tutorial.

The 10x Genomics bundle of this dataset is a 36 GB zip, most of it full-resolution morphology images. `download`
fetches only the files the tutorial uses (about 9 GB), reading single members of the remote zips with HTTP range
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
# DAPI and a few protein stains, to look at the tissue behind the quantification
STAINS = {
    "DAPI": "ch0000_dapi",
    "CD20": "ch0009_cd20",
    "CD8A": "ch0010_cd8a",
    "CD68": "ch0015_cd68",
    "PD-1": "ch0019_pd-1",
    "CD31": "ch0028_cd31",
    "PanCK": "ch0030_panck",
    "alphaSMA": "ch0032_alphasma",
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
    """Download the files of the tutorial into `data_dir` (about 9 GB); files already present are skipped."""
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
        image = xr.DataArray(level, dims=("c", "y", "x"), coords={"c": c_coords})
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
        morphology_focus=False,  # only some stains were downloaded: added below
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
