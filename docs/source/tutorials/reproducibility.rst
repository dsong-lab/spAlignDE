Reproducibility with fixed seeds
================================

The public tutorials use a fixed seed for every stochastic workflow. To repeat
a reported analysis, use:

1. the same cleaned input observations in the same order;
2. the same workflow parameters; and
3. the same seed, set before the first stochastic step.

The supplied environment file lists the package versions used for the examples.
Reproducibility is assessed from the scientific outputs.

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
     - Recorded result
   * - MERFISH S2R1 single clustering
     - 25 final structures.
   * - ST to Allen CCF slice 675
     - 18 final matched pairs.
   * - Xenium Rep1 to H&E
     - 21 cleaned image structures and 2 accepted pairs.
   * - Spatial ATAC to MERFISH S3R1
     - 17 ATAC structures and 8 accepted pairs.
   * - MERFISH S2R3 to S2R2
     - 27 refined shared structures.
   * - Kidney IL3 to NL3
     - 4 structures and label agreement ``0.6587`` to ``0.7417``, using the same saved coordinates as kidney inference.
   * - Breast cancer Rep2 to Rep1
     - 10 shared structures.
   * - MERFISH alignment uncertainty
     - 10 seeded 80% repeats; median ``dist_mean`` 40.25, in aligned-coordinate units.

The three automatic Atlas stages progressively refine only the ST partition.
All stages search the same Allen hierarchy depths 2--10. The scheduled stages
retain 3, 8 and 16 pairs, and continuation increases the set to 18 before the
stopping cycle. The public UI example instead uses the correspondences saved
from the validated interactive session.

The website aging-brain inference example continues from five sections of the
saved 19-query fixed-seed alignment; it does not rerun alignment. The kidney
inference example continues from the fixed-seed manual-prealignment result.

Numerical comparisons
---------------------

Independent runs reproduced the raw, refined and final labels for MERFISH
single clustering, MERFISH joint clustering, kidney and ATAC. MERFISH joint
clustering produced 28 raw and 27 refined/final structures; kidney produced
four. The uncertainty workflow's two complete ten-repeat executions produced
identical pointwise tables with the final optimizer iterate retained.

For the continuous alignment outputs:

- **Kidney:** compare float32 CUDA coordinates with a ``0.01``-unit tolerance
  in the original coordinate system, keeping labels, observation order,
  manual initialization and S-LDDMM settings fixed. The website alignment and
  inference examples use the same saved result.
- **ATAC to ST:** the accepted pair table and scores matched exactly across
  two independent runs; float64 aligned coordinates agreed within ``1e-12``.
- **H&E:** the S-LDDMM configuration used in the manuscript uses float64.
  Two independent runs from the same HIPT features reproduced image labels,
  pre-alignment and accepted pairs exactly. Aligned coordinates differed by
  at most ``0.00114`` feature-grid unit and pairwise Dice by at most ``9.23e-5``.

The kidney and five-section aging-brain inference examples use seed ``1`` and
request ``n_jobs=1``. The two auto-geometry passes always use one worker;
subsequent stages use the requested worker count. Two independent runs
reproduced the saved summaries, with 6,187 shared-grid locations for kidney
and 74,908 for aging brain. Parallel workers can change diagnostic message
order without changing the fitted scientific results.

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
