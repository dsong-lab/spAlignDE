Run differential expression on your own aligned data
====================================================

Alignment returns an AnnData object. ``prepare_inference`` takes a pandas
DataFrame with one row per spot or cell: sample identity, batch, original and
aligned coordinates, and one expression column per gene. The example below
constructs that table and runs local differential expression.

Build the input table
---------------------

Use an aligned H5AD with ``sample_id``, ``batch``, ``x_aligned`` and
``y_aligned`` in ``.obs``, and original coordinates in ``.obsm["spatial"]``.
``batch`` records experimental batches; use one common label for a single-batch
dataset. The raw-count H5AD must have the same unique observation identifiers
and unique gene names. Observation identifiers should include the sample when
barcodes are reused across sections.

Keep original, nonnegative, unlogged counts for inference. The kidney
alignment H5AD's ``.X`` contains gene-filtered, quantile-normalized clustering
features, so that example reloads its original counts. The code below likewise
reads expression separately and joins it to the alignment by observation ID.

.. code-block:: python

   import anndata as ad
   import numpy as np
   from spalignde import fit_local_de, prepare_inference

   aligned = ad.read_h5ad("aligned.h5ad")
   raw = ad.read_h5ad("raw_counts.h5ad")
   raw = raw[aligned.obs_names, :]

   # Replace these genes and the reference sample with your own choices.
   genes = ["Cbr1", "Cd44", "Myo5a"]
   reference = "NL3"

   # A broad candidate panel, rather than only the genes being tested.
   detected = np.asarray((raw.X > 0).sum(axis=0)).ravel()
   totals = np.asarray(raw.X.sum(axis=0)).ravel()
   risk_genes = raw.var_names[(detected >= 10) & (totals >= 10)].tolist()
   expression_genes = list(dict.fromkeys([*risk_genes, *genes]))
   expression = raw[:, expression_genes].to_df()

   coordinate_data = aligned.obs[
       ["sample_id", "batch", "x_aligned", "y_aligned"]
   ].copy()
   coordinate_data[["x", "y"]] = np.asarray(aligned.obsm["spatial"])
   inference_data = coordinate_data.join(expression)

If raw counts are already preserved in ``aligned.layers["counts"]``, replace
the two lines loading and subsetting ``raw`` with:

.. code-block:: python

   raw = aligned.copy()
   raw.X = raw.layers["counts"]

The detection/count cutoffs above are example choices. Supply at least ten
expression genes and a broad risk-gene candidate set; ``prepare_inference``
screens that pool for stable, informative genes. All samples' aligned
coordinates must use the same coordinate system and units.

Prepare and fit local tests
---------------------------

.. code-block:: python

   prepared = prepare_inference(
       inference_data,
       reference=reference,
       genes=genes,
       risk_genes=risk_genes,
       coordinate_key=("x", "y"),
       aligned_coordinate_key=("x_aligned", "y_aligned"),
       cell_type_key=None,
       density_energy_share=0.25,
       library_size=10_000,
       n_jobs=1,
       random_state=1,
   )
   result = fit_local_de(
       prepared,
       genes=genes,
       contrast="vs_reference",
       mismatch_aware=True,
       technical_adjustment=True,
       cell_type_adjustment=False,
       global_offset=False,
       region_cleanup=False,
       n_jobs=1,
       random_state=1,
   )

``vs_reference`` compares each other sample with ``reference``.
This example disables cell-type adjustment because no cell-type annotations
were supplied. See :doc:`post_alignment_inference` for the statistical model,
result summaries and the dataset-specific settings used by the saved examples.

Visium inputs
-------------

For Visium, the existing ``build_visium_inference_table`` helper can replace
the manual table construction. It reads sample identity and coordinates from
the aligned AnnData, joins separate raw 10x counts by sample and barcode, and
selects the risk-gene candidates:

.. code-block:: python

   from spalignde.datasets import build_visium_inference_table

   inputs = build_visium_inference_table(
       aligned,
       raw_counts={
           "NL3": "NL3_filtered_feature_bc_matrix.h5",
           "IL3": "IL3_filtered_feature_bc_matrix.h5",
       },
       genes=genes,
       batch="kidney_pair",
   )
   inference_data = inputs.data
   risk_genes = inputs.risk_genes

This helper uses original ``x`` and ``y`` columns in ``aligned.obs`` and accepts
an H5AD path in place of ``aligned``. Use actual batch labels for your study;
``batch`` also accepts a sample-to-batch mapping.

Study data and recorded examples
--------------------------------

The `spAlignDE study archive <https://zenodo.org/records/21983769>`_ describes
post-alignment materials containing expression data, annotations, original and
aligned coordinates, and analysis notebooks for kidney and aging brain.
For the website's saved results, follow the
:doc:`executed inference notebooks <../source_notebooks/post_alignment_inference>`.
The kidney example uses its packaged fixed coordinates and reloads the public
raw 10x counts; its H5AD coordinates are multiplied by 50 for the recorded
inference coordinate scale.
