# spAlignDE Interactive Region Pairing Tool

This Streamlit application compares two spatial datasets side by side. It can
select or group regions, refine correspondences, and export the curated pairs
as CSV for the public `spAlignDE.align_st_to_allen_atlas_from_ui_pairs` API.

## Install

Create the validated repository environment from the repository root, or add
the UI dependencies to another supported environment:

```bash
conda env create -f environment.yml
conda activate spAlignDE-notebooks
python -m pip install --no-deps --no-build-isolation -e .

# Smaller package-only installation
# python -m pip install -e ".[ui]"
```

## Configure Allen CCF data

The 3-D Allen annotation is intentionally not committed to GitHub. Download
the Allen CCF 2022 files described by the Atlas tutorial and use either form:

```bash
export SPALIGNDE_ALLEN_CCF_DIR=/path/to/allen_ccf_2022

# Or set the files separately:
# export SPALIGNDE_ALLEN_ANNOTATION=/path/to/annotation_10.nrrd
# export SPALIGNDE_ALLEN_STRUCTURE_TABLE=/path/to/voxel_count_and_differences.csv
```

`SPALIGNDE_ALLEN_CCF_DIR` must contain `annotation_10.nrrd`. The repository
already includes the small validated structure table at
`ui/data/voxel_count_and_differences.csv`; an explicitly configured table
takes precedence.

## Start

From any working directory:

```bash
streamlit run /path/to/spAlignDE/ui/app.py
```

Streamlit prints a local browser URL, normally `http://localhost:8501`. The UI
component and data locations are resolved relative to `ui/app.py`, not the
terminal's current directory.

By default uploaded working files are saved under `ui/uploaded_datasets/`,
which is excluded from Git. Set `SPALIGNDE_UI_UPLOAD_DIR` to store them
elsewhere.

## Supported user inputs

- Point/cluster CSV: one cell or spot per row with numeric x, y and label
  columns selected in the UI.
- 2-D label-image NPY: a numeric two-dimensional label image; `NaN` and `-1`
  are treated as background.
- Allen CCF atlas: the configured annotation volume, slice and orientation.

The app downloads pairing and custom-region tables through the browser. A
many-to-many correspondence produces several rows with the same `group_id`;
the alignment API reconstructs one deformation channel per group and skips
automatic pair discovery and scoring. Mask validation, filtering, signed
distance processing, S-LDDMM input construction and final alignment are still
performed.
