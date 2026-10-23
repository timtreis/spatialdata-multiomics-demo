"""Data access for the Xenium gene + protein renal cell carcinoma tutorial.

The 10x Genomics bundle of this dataset is a 36 GB zip, most of it full-resolution morphology images. `download`
fetches only the files the tutorial uses (about 22 GB), reading single members of the remote zips with HTTP range
requests. `load` assembles them into a SpatialData object with a MuData table; the images are read lazily from their
OME-TIFF pyramids, without converting them.
"""

import io
import os
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
    """The dataset as SpatialData: segmentation, cell boundaries, a MuData table (RNA), stains, H&E, pathology.

    The morphology modality is added to the table by :func:`morphology`.
    """
    data_dir = Path(data_dir)
    sdata = xenium(
        data_dir / "outs",
        multimodal=True,  # the table is a MuData object, with one modality per measurement of the cells
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
    # the protein of each cell is measured from the stains by morphology(); 10x's own per-cell mean intensities go
    table = sdata["table"]
    del table.mod["prot"]
    table.update()
    del table.obsm["prot"], table.varm["prot"]  # left behind by update()
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


CP_FEATURES = ["cp_measure:intensity", "cp_measure:sizeshape", "cp_measure:texture", "cp_measure:granularity",
               "cp_measure:radial"]


def stain_ranges(sdata: SpatialData, scale: str = "scale2") -> dict[str, tuple[float, float]]:
    """Per stain, the ``(low, high)`` intensities that cp_measure's input is scaled to 0-1 by.

    The protein stains sit on a camera offset of about 99 counts. ``low`` is the background, the median of the tissue
    pixels outside the cells; ``high`` the 99.99th percentile of the pixels in cells, so that the rare positive cells
    of a sparse stain are not clipped. From a downsampled level of the image pyramid (reading the full resolution
    takes an hour).
    """
    image = sdata["stains"][scale]["image"]
    pixels = np.asarray(image.values)
    cells = np.asarray(sdata["cell_labels"][scale]["image"].values) > 0
    background = ~cells & (pixels[0] > 0)  # outside the scanned area every channel is 0
    return {str(c): (float(np.median(x[background])), float(np.percentile(x[cells], 99.99)))
            for c, x in zip(image.c.values, pixels, strict=True)}


def morphology(sdata: SpatialData, data_dir: str | Path, n_jobs: int | None = None):
    """The raw cp_measure features of every cell, from the DAPI and the 27 protein stains (squidpy's names).

    Computed once (about 2 hours on 64 CPU cores for this section) and cached as ``data_dir/morph.h5ad``. The stains
    are first written to a zarr store, which squidpy's worker processes read; it is deleted afterwards.
    """
    import anndata as ad
    import pandas as pd
    import squidpy as sq
    from spatialdata import read_zarr

    out = Path(data_dir) / "morph.h5ad"
    if not out.exists():
        store = Path(data_dir) / "morph_images.zarr"
        if not store.exists():
            SpatialData(images={"stains": sdata["stains"]}, labels={"cell_labels": sdata["cell_labels"]}).write(store)
        features = sq.experimental.im.calculate_image_features(
            # each stain scaled to 0-1 from its background to its brightest cells: cp_measure's texture quantizes it
            # into 256 gray levels, and its radial and intensity ratios need the camera offset removed
            read_zarr(store), image_key="stains", labels_key="cell_labels", scale="scale0", features=CP_FEATURES,
            intensity_range=stain_ranges(sdata), n_jobs=n_jobs or os.cpu_count(), inplace=False,
        )
        table = sdata["table"]
        instance_key = table.uns["spatialdata_attrs"]["instance_key"]
        cell_id = pd.Series(table.obs["cell_id"].to_numpy().astype(str), index=table.obs[instance_key].to_numpy())
        features.obs["cell_id"] = cell_id.reindex(features.obs["label_id"].to_numpy()).to_numpy()
        features.write_h5ad(out)
        shutil.rmtree(store)
    features = ad.read_h5ad(out)
    features.obs_names = features.obs["cell_id"].astype(str).to_numpy()
    return features


def _cellprofiler_name(name: str) -> str:
    """A squidpy cp_measure column name in CellProfiler's grammar (``Cells_<group>_<feature>_<stain>[_<params>]``),
    which mantispy parses into its feature annotation."""
    if "__" not in name:
        return f"Cells_AreaShape_{name}"
    feature, stain = name.rsplit("__", 1)
    group, _, rest = feature.partition("_")
    if group in ("Intensity", "Location", "Granularity"):
        return f"Cells_{group}_{rest}_{stain}"
    if group == "RadialDistribution":
        kind, _, params = rest.partition("_")
        return f"Cells_RadialDistribution_{kind}_{stain}_{params}"
    texture, params = feature.rsplit("_", 3)[0], "_".join(feature.rsplit("_", 3)[1:])  # Contrast_3_00_256
    return f"Cells_Texture_{texture}_{stain}_{params}"


def stain_image_shape(data_dir: str | Path) -> tuple[int, int]:
    """(height, width) in pixels of the full-resolution stain images, read from the header of the DAPI image."""
    with tifffile.TiffFile(Path(data_dir) / "morphology_focus" / f"{STAINS['DAPI']}.ome.tif") as tif:
        return tuple(tif.series[0].shape[-2:])


def morphology_profiles(raw, centroids):
    """The raw features as a mantispy object of single cells, indexed by cell_id.

    mantispy's cell-level objects identify a cell by its plate, well, image and object number, as in a screen; the
    section is one plate, one well and one image, and the object number is the cell's label in the segmentation.
    ``centroids`` (cell_id x [x, y], in µm, as in the table's ``obsm["spatial"]``) places each cell in the image, for
    mantispy's border check.
    """
    import pandas as pd
    from mantispy.io._profiles import from_dataframe  # the in-memory reader, not exported from mantispy.io yet

    # mantispy drops the location and centroid features itself; the raw spatial moments also depend on position
    raw = raw[:, ~raw.var_names.str.startswith("SpatialMoment_")]
    raw = raw[:, ~raw.var_names.str.endswith(tuple(f"__{s}" for s in EXCLUDED_STAINS))]
    values = pd.DataFrame(np.asarray(raw.X, dtype=np.float32), columns=[_cellprofiler_name(n) for n in raw.var_names])
    ids = pd.DataFrame({"Metadata_Plate": "rcc_section", "Metadata_Well": "A01", "Metadata_ImageID": "rcc_section",
                        "Metadata_ObjectType": "Cells",
                        "Metadata_ObjectNumber": raw.obs["label_id"].astype(int).to_numpy()})
    channels = [s for s in STAINS if s not in EXCLUDED_STAINS]
    profiles = from_dataframe(pd.concat([ids, values], axis=1), channels=channels, resolution="object")
    profiles.obs_names = raw.obs["cell_id"].astype(str).to_numpy()
    xy = pd.DataFrame(centroids).loc[profiles.obs_names].to_numpy() / PIXEL_SIZE  # µm -> image pixels
    profiles.obs["Metadata_Center_X"], profiles.obs["Metadata_Center_Y"] = xy[:, 0], xy[:, 1]
    return profiles


#: stains left out of the morphology modality: 7.5% of the cells (after QC) have no HLA-DR signal above its
#: background, which leaves its texture undefined in them; removing those cells would remove only HLA-DR-negative ones
EXCLUDED_STAINS = ("HLA-DR",)

#: feature_select's steps for single cells in mantispy's documentation (the default ones, plus drop_outliers)
SINGLE_CELL_OPERATIONS = ("drop_degenerate", "variance_threshold", "correlation_threshold", "drop_na_columns",
                          "blocklist", "drop_outliers")


#: cells closer than this to the edge of the scanned area have stains cut off (no signal in any stain)
BORDER_UM = 20


def preprocess_morphology(profiles, image_shape: tuple[int, int], channels: list[str] | None = None):
    """mantispy's single-cell workflow: cell QC, feature QC, normalization, feature selection.

    Three choices differ from mantispy's defaults, because on single cells of a tissue they lose signal (the
    notebook shows each comparison): z-scores instead of median/MAD normalization (most cells are negative for a
    sparse stain, its MAD is 0 and its features are flagged degenerate); the few cells with an undefined feature are
    removed, so that the selection's ``na_cutoff=0`` does not drop most texture features for them; and cells with an
    unusual area are flagged (``obs["qc_area_outlier"]``) but kept, since in a tissue they are large or small cell
    types, not segmentation errors, which the QC on cell area has already removed. Cells within ``BORDER_UM`` of the
    edge of the image (``image_shape``, in pixels) are removed: the stains are cut off there. ``channels`` limits the
    features to those stains (and the cell shape).
    """
    import mantispy as mt

    a = profiles.copy()
    if channels is not None:
        a = a[:, a.var["channel"].isna().to_numpy() | a.var["channel"].isin(channels).to_numpy()].copy()
    mt.pp.calculate_qc_metrics(a, image_shape=image_shape, border_margin=round(BORDER_UM / PIXEL_SIZE),
                               max_nan_fraction=0.0)
    # area outliers kept, see above
    a.obs["qc_pass"] = (a.obs["qc_nan_fraction"].to_numpy() == 0) & ~a.obs["qc_is_border"].to_numpy()
    mt.pp.filter_cells(a, min_cells_per_well=0)
    mt.pp.filter_features(a, min_variance=1e-8)
    mt.pp.normalize(a, method="standardize", by="Metadata_Plate", reference=None)
    mt.pp.feature_select(a, operations=SINGLE_CELL_OPERATIONS, na_cutoff=0.0)
    a.uns["feature_select"] = dict(a.uns["mantispy"]["feature_select"])
    return mt.pp.subset_features(a)


def normalize_rna(rna) -> None:
    """In place: RNA counts to log-normalized expression (the counts kept in ``layers["counts"]``)."""
    import scanpy as sc

    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)


N_LATENT = 20


def train_mofa(views: dict, out: Path) -> None:
    """MOFA+ on a GPU over the given views (name -> AnnData of the same cells); saves the factors (keyed by cell_id),
    the weights and the variance explained per view."""
    import h5py
    import mudata as md
    import muon as mu

    mdata = md.MuData({name: view.copy() for name, view in views.items()})
    model_file = Path(out).with_suffix(".hdf5")
    mu.tl.mofa(mdata, n_factors=N_LATENT, use_var=None, scale_views=True, convergence_mode="fast", seed=0,
               use_float32=True, gpu_mode=True, outfile=str(model_file), save_data=False)
    variance = mdata.uns["mofa"]["variance"]
    result = {"cell_id": mdata.obs_names.to_numpy().astype(str), "factors": mdata.obsm["X_mofa"].astype(np.float16),
              "views": np.array(list(views))}
    # the weights per view, from the model file: in mdata.varm the order of the views is not that of mdata.mod
    with h5py.File(model_file) as f:
        for name, view in views.items():
            features = f[f"features/{name}"][:].astype(str)
            assert (features == view.var_names).all()
            result[f"features_{name}"] = features
            result[f"weights_{name}"] = f[f"expectations/W/{name}"][:].T.astype(np.float32)
            result[f"r2_{name}"] = np.asarray(variance[name], dtype=np.float32)
    np.savez_compressed(out, **result)
    model_file.unlink()


def patch_muon_wnn():
    """muon 0.2.0's WNN returns NaN for cells whose neighbors in a modality are at distance 0 (identical profiles)."""
    import inspect
    import textwrap

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
