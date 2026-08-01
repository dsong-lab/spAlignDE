For every sample, place one metadata file and one cell/spot-by-gene expression
file in the same directory. Use exactly the same ``<sample_id>`` in each
matched filename:

.. code-block:: text

   cell_metadata_<sample_id>.csv
   cell_by_gene_<sample_id>.csv

The metadata file must begin with ``cell_id``, ``x`` and ``y`` columns. It may
contain additional observation-level columns:

.. code-block:: text

   cell_id,x,y,...

The expression file must begin with ``cell_id`` followed by numeric gene
columns:

.. code-block:: text

   cell_id,GeneA,GeneB,...

Within each sample, rows are matched by ``cell_id`` rather than row position.
Across samples, genes are aligned by column name; genes absent from a sample
are filled with zero. Cell IDs must be unique within each file, coordinate and
expression values must be finite, and expression values must be non-negative.
