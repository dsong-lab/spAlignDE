Reproducibility and fixed random seeds
======================================

The public tutorials use explicit workflow seeds and a pinned environment.
The seed is set inside the package before BANKSY feature construction and PCA,
not only at the Leiden partition step. This distinction matters because the
validated pyBANKSY release delegates large-matrix PCA to randomized SVD.

Canonical seeds
---------------

.. list-table:: Fixed tutorial controls
   :header-rows: 1
   :widths: 40 15 45

   * - Workflow
     - Seed
     - Controlled stochastic steps
   * - Single-sample BANKSY for MERFISH or spatial ATAC
     - ``1234``
     - Python, NumPy, Torch, randomized PCA and graph partitioning.
   * - Joint BANKSY/Harmony/Leiden cross-sample clustering
     - ``1000``
     - Python, NumPy, Torch, joint PCA, Harmony and Leiden.
   * - H&E feature processing and image-region clustering
     - ``0``
     - Model/runtime seed, point subsampling and K-means.
   * - Post-alignment RCTD/reference subsampling
     - ``1``
     - Reference-cell subsampling.

Every executable analysis notebook calls ``spAlignDE.set_random_seed`` before
its first stochastic operation and also passes ``random_state`` explicitly to
the corresponding configuration. The returned seed metadata are stored with
clustering provenance under ``adata.uns["spAlignDE"]``.

The general joint-clustering API pins Leiden to ``leidenalg`` with unlimited
iterations (``-1``), matching the cross-sample validation runs. The dedicated
large Xenium runner pins Scanpy's ``igraph`` implementation with two
iterations, matching its 10-cluster validated result. Backend substitution is
treated as a configuration change, not as an automatic fallback.

Launch-time controls
--------------------

``PYTHONHASHSEED`` must be set before Python starts. For the closest match to
the validated Linux/CUDA environment, launch Jupyter from the repository root
with:

.. code-block:: bash

   # Match this value to the workflow seed in the table above.
   export PYTHONHASHSEED=1234
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export NUMBA_NUM_THREADS=1
   jupyter lab

Inside a script or notebook, reset the workflow-level generators before the
analysis call:

.. code-block:: python

   import spAlignDE

   seed_controls = spAlignDE.set_random_seed(
       1234,
       deterministic_torch=True,
       warn_only=True,
   )

Setting ``PYTHONHASHSEED`` inside an already running notebook does not change
Python's current hash order. Re-run from a freshly launched kernel when
testing reproducibility.

Before publishing notebook edits, run:

.. code-block:: bash

   python tools/audit_tutorial_reproducibility.py

The audit checks every computational notebook, explicit backend controls,
seed forwarding to external runners, reproducibility metadata and the
byte-identical ``source_notebooks``/Sphinx mirror contract. The interactive
region-pairing notebook only records manual selections and is therefore
classified separately from stochastic computational workflows.

What should reproduce exactly
-----------------------------

For a fixed input checksum, observation order, configuration and dependency
environment, the following are expected to be exact between fresh processes:

- raw, refined and selected cluster labels;
- ST hierarchy memberships;
- masks, candidate tables and accepted structure-pair identities; and
- deterministic pre-alignment parameters and table row order.

Continuous CUDA coordinates are assessed with a declared tolerance. PyTorch
reports that ``grid_sampler_2d_backward_cuda`` has no deterministic CUDA
implementation, so setting a seed does not justify a claim of bitwise-identical
deformation fields. Float64 reduced the observed H&E and ATAC coordinate
differences below ``1e-12``. Large cross-sample runs in float32 reproduced
within one coordinate unit, at most one thirtieth of the 30-unit raster grid.

Validated fixed-seed results
----------------------------

.. list-table:: August 2026 two-run validation
   :header-rows: 1
   :widths: 31 31 38

   * - Workflow
     - Fixed result
     - Repeat criterion
   * - MERFISH S2R1 single clustering
     - 25 raw/refined/final clusters
     - Exact labels.
   * - ST to Allen CCF slice 675
     - ST levels 7/16/25; stage pairs 3/7/15; continuation 15→17 and 17→17
     - Exact discrete tables; float64 coordinates within tolerance.
   * - Xenium S2R1 to H&E
     - 24 image structures; 2 accepted pairs
     - Exact masks/pairs; float64 coordinates within tolerance.
   * - Spatial ATAC to MERFISH S3R1
     - 17 ATAC structures; 5 accepted pairs
     - Exact labels/pairs; float64 coordinates within tolerance.
   * - MERFISH S2R3 to S2R2
     - 28 raw and 27 refined/final shared clusters
     - Exact labels; scale-aware coordinate tolerance.
   * - Kidney IL3 to NL3
     - 4 raw/refined/final shared clusters
     - Exact labels; scale-aware coordinate tolerance.
   * - Aging brain to age 4.3
     - 27 raw/refined/final shared clusters; 19 section alignments
     - Exact labels; all coordinates within 1/30 raster pixel.
   * - Breast cancer Rep2 to Rep1
     - 10 shared clusters
     - Exact labels; scale-aware coordinate tolerance.

Atlas terminology
-----------------

The three Atlas stages make only the ST partition progressively finer. At
every stage, automatic pairing searches the same Allen hierarchy candidate
depths 2–10; there is no stage-specific Atlas-depth restriction. The final
scheduled stage accepts 15 pairs. Continuation re-scores at the completed
coordinates, increases the set to 17, runs one stopping cycle at 17, and
retains that cycle's final iterate.

Reporting checklist
-------------------

Archive the input SHA-256 checksums, observation order, full configuration,
seed metadata, package versions, device, dtype, discrete-output hashes and the
predeclared continuous-coordinate tolerance with every reported run. Do not
mix figures or biological metrics produced from a previous uncontrolled
clustering with a new fixed-seed alignment.
