Reproducibility with fixed seeds
================================

The public tutorials use a fixed seed for every stochastic workflow. To repeat
a reported analysis, use:

1. the same cleaned input observations in the same order;
2. the same workflow parameters; and
3. the same seed, set before the first stochastic step.

No ``PYTHONHASHSEED`` or thread environment variables are required by these
tutorials. The joint-clustering API limits Harmony itself to one PyTorch thread
and restores the prior thread count afterwards. The supplied environment file
lists the package versions used for the examples, but reproducibility is tested
from the scientific outputs, not merely from a successful exit status. When a
change affects reported labels, maintainers repeat that affected workflow and
compare labels observation by observation.

Seeds used in the tutorials
---------------------------

.. list-table:: Fixed workflow seeds
   :header-rows: 1
   :widths: 55 15 30

   * - Workflow
     - Seed
     - Main stochastic steps
   * - Single-sample BANKSY and Atlas/ATAC workflows
     - ``1234``
     - Randomized PCA, clustering and alignment.
   * - Joint cross-sample workflows and uncertainty repeats
     - ``1000``
     - Joint PCA, Harmony, Leiden and repeat sampling.
   * - H&E feature processing and alignment
     - ``0``
     - Feature subsampling and image-region clustering.
   * - Post-alignment inference and RCTD reference subsampling
     - ``1``
     - Reference subsampling and trajectory clustering.

Every computational notebook calls ``spAlignDE.set_random_seed`` before its
first stochastic operation and passes ``random_state`` to the relevant public
API configuration. For example:

.. code-block:: python

   import spAlignDE

   WORKFLOW_SEED = 1234
   spAlignDE.set_random_seed(WORKFLOW_SEED)

The selected seed is also stored with the workflow settings. The general
joint-clustering API uses ``leidenalg`` with unlimited iterations, whereas the
large Xenium example uses Scanpy's ``igraph`` implementation with two
iterations. Use the backend documented for the example because changing the
clustering method can change its labels.

``JointClusteringConfig`` defaults to ``harmony_device="cpu"`` and
``harmony_threads=1``. In harmonypy 0.2.0, parallel CPU and accelerator
reductions can vary in the last bits even with a fixed seed; cells near a graph
partition boundary can then receive different labels. Set
``harmony_threads=None`` or select an accelerator only for a speed-first run.

Independent-run check
---------------------

The complete public notebook set was run from fresh kernels with the documented
inputs. Repeat runs were targeted to clustering paths affected by a
repeatability change or observed run-to-run drift, and their labels were
compared observation by observation. Downstream alignment and inference
notebooks were checked against their reported scientific summaries. GPU
calculations can differ slightly in the last digits of transformed coordinates
without changing those summaries.

The main fixed-seed examples produced:

.. list-table:: Reported fixed-seed outputs
   :header-rows: 1
   :widths: 55 45

   * - Workflow
     - Fresh validation result
   * - MERFISH S2R1 single clustering
     - 25 final structures.
   * - ST to Allen CCF slice 675
     - 18 final matched pairs.
   * - Xenium S2R1 to H&E
     - 21 cleaned image structures and 2 accepted pairs.
   * - Spatial ATAC to MERFISH S3R1
     - 17 ATAC structures and 8 accepted pairs.
   * - MERFISH S2R3 to S2R2
     - 28 refined shared structures.
   * - Kidney IL3 to NL3
     - 4 structures and label agreement ``0.6631`` to ``0.7413``.
   * - Breast cancer Rep2 to Rep1
     - 11 shared structures.
   * - MERFISH alignment uncertainty
     - 10 seeded 80% repeats and the same pointwise summary table.

The three automatic Atlas stages progressively refine only the ST partition.
All stages search the same Allen hierarchy depths 2--10. The scheduled stages
retain 3, 8 and 16 pairs, and continuation increases the set to 18 before the
stopping cycle. The public UI example instead uses the correspondences saved
from the validated interactive session.

The website aging-brain inference example continues from five sections of the
saved 19-query fixed-seed alignment; it does not rerun alignment. The kidney
inference example continues from the fixed-seed manual-prealignment result.

Checking an edited tutorial
---------------------------

For an ordinary notebook edit, run the affected notebook once from a fresh
kernel and inspect its results. Maintainers changing a reported scientific
result should also repeat it independently with the same inputs, parameters
and seed, then compare labels, accepted pairs and reported summary values. The
repository helpers execute and inspect the canonical public notebooks:

.. code-block:: bash

   python tools/execute_fixed_seed_tutorials.py --only path/to/notebook.ipynb
   python tools/audit_source_notebooks.py source_notebooks

The execution helper stops when a cell fails and updates the canonical notebook
after a successful run. The audit checks portable source, saved execution
state and consistency between each declared seed and its execution metadata.
