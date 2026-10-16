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
uv sync                  # creates .venv with the three development branches
uv run jupyter lab       # or select the .venv kernel in your editor
```

`uv.lock` pins the commit of each branch (and every other dependency), so `uv sync` reproduces the environment the
notebooks were run in; `uv lock --upgrade` (then `uv sync`) picks up new commits of the branches.

Installed from the fork, `spatialdata-io` reports a version like `0.1.devN`: the fork lacks the upstream release tags that
the version is derived from. The code is that of the branch; pushing the `v0.7.1` tag to the fork fixes the version.

## Notebooks

- [`notebooks/mudata_xenium_gene_protein.ipynb`](notebooks/mudata_xenium_gene_protein.ipynb): reads the public 10x
  Genomics *Xenium V1 Protein Human Kidney tiny* dataset (gene + protein, 21 MB, downloaded by the notebook) as a
  SpatialData object with a MuData table, and analyses both modalities: QC, RNA clustering and annotation, protein
  normalization, joint RNA + protein clustering (WNN), proteins across cell types, gene vs protein concordance, spatial
  plots of both modalities, spatial queries, and saving/reloading.

## Notes

- A MuData table keeps the SpatialData annotation (`region`, `region_key`, `instance_key`) in its global `obs`/`uns`,
  so all the modalities annotate the same cells; a modality may cover a subset of them.
- Following the MuData convention, global columns taken from a modality are named `"<modality>:<key>"`; `":"` is
  stored as `"%3A"` on disk, since it is not allowed in Windows file names.
- muon 0.2.0's weighted nearest neighbors (`mu.pp.neighbors`) returns NaN for cells with identical profiles in a
  modality (e.g. no signal for any protein of a small panel): it drops their neighbors at distance 0 and their kernel
  bandwidth is 0. The notebook patches `muon.pp.neighbors` in place (two lines, checked against the installed version).
