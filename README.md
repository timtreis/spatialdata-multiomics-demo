# spatialdata-multiomics-demo

Development environment and demo notebooks for **multimodal tables in SpatialData**: a table of a `SpatialData`
object can be a [`MuData`](https://mudata.scverse.org) object, with one modality per assay (e.g. RNA and protein
measured on the same cells), instead of an `AnnData` object.

The support is developed on the `claude/mudata-support-sdata-c2tfjf` branches of:

| package            | repository                                                                   | what it adds                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `spatialdata`      | [scverse/spatialdata](https://github.com/scverse/spatialdata)                | MuData tables: models, IO, joins, queries, `get_values` (`"<modality>:<key>"`), `concatenate`, `sanitize_table` |
| `spatialdata-io`   | [timtreis/spatialdata-io](https://github.com/timtreis/spatialdata-io) (fork) | `xenium(..., multimodal=True)`: Xenium gene + protein tables as MuData (`"rna"`, `"prot"`)                      |
| `spatialdata-plot` | [scverse/spatialdata-plot](https://github.com/scverse/spatialdata-plot)      | coloring shapes and labels by the values of any modality (`color="prot:VISTA"`)                                 |

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/timtreis/spatialdata-multiomics-demo
cd spatialdata-multiomics-demo
uv sync                  # creates .venv with the development branches
uv run jupyter lab       # or select the .venv kernel in your editor
```

`uv.lock` pins the commit of each branch (and every other dependency), so `uv sync` reproduces the environment the
notebooks were run in; `uv lock --upgrade` (then `uv sync`) picks up new commits of the branches. Besides the MuData
branches, squidpy comes from a branch that scales the stains for cp_measure (scverse/squidpy#1321) and mantispy from
its main branch, neither released yet. On Linux, `cupy` is installed for MOFA+ on a GPU.

Installed from the fork, `spatialdata-io` reports a version like `0.1.devN`: the fork lacks the upstream release tags that
the version is derived from. The code is that of the branch; pushing the `v0.7.1` tag to the fork fixes the version.

## Tutorial

[`notebooks/rcc_multimodal.ipynb`](notebooks/rcc_multimodal.ipynb): immune cells around a kidney tumor, read from the
transcripts and the stained images of the same cells. A 10x Genomics Xenium Protein section of FFPE clear cell renal
cell carcinoma (408,798 cells after QC, 405 genes, DAPI and 27 protein stains, H&E and a pathologist's annotation),
analysed as one SpatialData object with a MuData table of two modalities: `rna`, the Xenium transcripts, and `morph`,
morphology features of every cell in DAPI and 26 of the stains (cp_measure through squidpy, each stain scaled from its
background; HLA-DR is left out, its texture being undefined in the many cells without its signal), preprocessed with
the mantispy single-cell workflow. Each modality is preprocessed on its own, their latent spaces are compared, and they are then
modelled jointly (MOFA+, weighted nearest neighbors), giving cell types, the organisation of the immune cells relative
to the tumor, and where the protein stains rescue the sparse RNA. Written for biologists; every figure comes with its
interpretation.

The data (about 22 GB of the 36 GB bundle) are downloaded by the notebook. The morphology features (`data/morph.h5ad`),
the MOFA+ model and the clustering are cached, so the notebook only computes them once; `RETRAIN = True` trains the
models again (MOFA+ on a GPU).

## Benchmark

[`benchmark/`](benchmark/README.md): does modelling RNA and protein jointly describe cells better than either
modality alone, and do image features add to the mean stain intensities? Ten representations, scored by how well
they predict held-out proteins, genes and pathology regions. Joint representations beat either modality alone; the
simple ones (concatenation, WNN) match the models (MOFA+, totalVI); image features carry information but add little
to the markers. Run on a SLURM cluster with GPUs.

## Notes

- A MuData table keeps the SpatialData annotation (`region`, `region_key`, `instance_key`) in its global `obs`/`uns`,
  so all the modalities annotate the same cells; a modality may cover a subset of them.
- Following the MuData convention, global columns taken from a modality are named `"<modality>:<key>"`; `":"` is
  stored as `"%3A"` on disk, since it is not allowed in Windows file names.
- muon 0.2.0's weighted nearest neighbors (`mu.pp.neighbors`) returns NaN for cells with identical profiles in a
  modality (e.g. no signal for any protein of a small panel): it drops their neighbors at distance 0 and their kernel
  bandwidth is 0. `helpers.patch_muon_wnn` patches `muon.pp.neighbors` in place (two lines, checked against the installed version).
