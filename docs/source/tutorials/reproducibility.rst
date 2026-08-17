Reproducibility and fixed random seeds
======================================

The public tutorials use fixed workflow seeds and a specified software
environment. The seed is set inside the package before BANKSY feature
construction and PCA, not only at the Leiden partition step. This distinction
matters because the pyBANKSY version used here delegates large-matrix PCA to
randomized SVD.

Seeds used in the tutorials
---------------------------

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
the corresponding configuration. The selected seed is stored with the
clustering settings under ``adata.uns["spAlignDE"]``.

The general joint-clustering API uses ``leidenalg`` with unlimited iterations
(``-1``), as in the cross-sample analyses. The dedicated large Xenium runner
uses Scanpy's ``igraph`` implementation with two iterations, as in the
reported 10-cluster result. Changing the backend may change the clustering.

Launch-time controls
--------------------

``PYTHONHASHSEED`` must be set before Python starts. To match the Linux/CUDA
environment used for the tutorials, launch Jupyter from the repository root
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

   # Execute each computational notebook in a fresh kernel, in dependency order.
   python tools/execute_fixed_seed_tutorials.py

   # Check notebook execution, saved outputs, seeds and documentation copies.
   python tools/audit_tutorial_reproducibility.py

The execution script uses the package source in the current checkout and runs
each notebook in a fresh kernel. It stops if a cell fails and saves the
notebook only after all cells finish. The checking script reports incomplete
runs, saved errors, inconsistent seeds or differences between the notebook and
documentation copies. The interactive region-pairing notebook records manual
selections and is therefore not included with the computational notebooks.

Each tutorial's Data section lists the required external inputs. The public
aging-brain notebook continues from five sections selected from the saved
fixed-seed alignments for 19 queries; it does not rerun alignment.

The public spAlignDE 0.1.0 environment uses AnnData ``0.10.9``. Some earlier
manuscript-scale cross-sample analyses were generated with AnnData ``0.10.8``
and retain that version in their saved run information. Manuscript Methods
therefore distinguish the current public environment from the environment used
for those analyses.

Expected repeat behavior
------------------------

With identical input data, observation order, configuration and dependency
environment, the following are expected to be exact between fresh processes:

- raw, refined and selected cluster labels;
- ST hierarchy memberships;
- masks, candidate tables and accepted structure-pair identities; and
- deterministic pre-alignment parameters and table row order.

Continuous CUDA coordinates are assessed with a declared tolerance. PyTorch
reports that ``grid_sampler_2d_backward_cuda`` has no deterministic CUDA
implementation, so a seed alone cannot make every CUDA coordinate identical.
With final-iterate checkpoint handling, repeated float64
ATAC runs differed by at most ``9.10e-13`` coordinate units; repeated H&E runs
differed by at most ``0.00114`` feature-grid unit and are assessed against a
conservative ``0.05``-unit tolerance. Large cross-sample runs in float32
reproduced within one coordinate unit, at most one thirtieth of the 30-unit
raster grid.

Results from repeated runs
--------------------------

.. list-table:: August 2026 repeated-run checks
   :header-rows: 1
   :widths: 31 31 38

   * - Workflow
     - Fixed result
     - Expected agreement
   * - MERFISH S2R1 single clustering
     - 25 raw/refined/final clusters
     - Exact labels.
   * - ST to Allen CCF slice 675
     - ST levels 7/16/25; stage pairs 3/8/16; continuation 17→18 and 18→18
     - Exact discrete tables; float64 coordinates within tolerance.
   * - Xenium S2R1 to H&E
     - 26-region merge target; 21 final cleaned image structures; 2 accepted pairs
     - Exact masks/pairs; float64 coordinates within tolerance.
   * - Spatial ATAC to MERFISH S3R1
     - 17 ATAC structures; 8 accepted pairs
     - Exact labels/pairs; float64 coordinates within tolerance.
   * - MERFISH S2R3 to S2R2
     - 28 raw and 27 refined/final shared clusters
     - Exact labels; scale-aware coordinate tolerance.
   * - Kidney IL3 to NL3
     - 4 raw/refined/final shared clusters; fixed manual transform; label agreement 0.6637→0.7366
     - Exact labels, observation order and manual pre-alignment; float32 CUDA
       coordinates use an original-unit tolerance of ``0.01``.
   * - Breast cancer Rep2 to Rep1
     - 10 shared clusters
     - Exact labels; scale-aware coordinate tolerance.
   * - MERFISH subsampling-based transformation variability
     - 10 independent 80% repeats; median ``dist_var`` 429.54; 95th percentile 2,352.74
     - Two full final-iterate executions produced identical pointwise tables.

Atlas terminology
-----------------

The three Atlas stages make only the ST partition progressively finer. At
every stage, automatic pairing searches the same Allen hierarchy candidate
depths 2–10; there is no stage-specific Atlas-depth restriction. The final
scheduled stage accepts 16 pairs. Continuation first identifies 17 pairs at
the completed coordinates, increases the set to 18, runs one stopping cycle
at 18, and
retains that cycle's final iterate.

The UI-paired Atlas example consumes the same seed-1234 clustered H5AD. Its
Allen selections and deformation groups are preserved from the validated UI
session, while ST cluster IDs are revalidated by cell overlap against the
fixed-seed labels. This avoids restoring an older uncontrolled label vector
inside an otherwise fixed-seed workflow.

Manuscript analyses not run in the website tutorials
----------------------------------------------------

The manuscript mouse-brain structure-resolution sweep used Leiden resolutions
``0.6``, ``0.8``, ``1.0``, ``1.2`` and ``1.4``. These produced
``15/17/21/24/28`` raw clusters and ``15/17/21/24/27`` refined/final
structures, respectively. The manuscript reports the refined/final range
``15–27`` because those labels define the alignment channels.

The selected cross-sample clustering configurations are dataset specific.
Mouse brain used BANKSY ``lambda=0.8``, 20 principal components, 50 SNN
neighbors, Harmony ``theta=2``/``max_iter=30`` and Leiden resolution ``1.4``.
Kidney used ``lambda=0.2``, 30 principal components, 100 SNN neighbors,
Harmony ``theta=2``/``max_iter=20`` and resolution ``0.2``. Aging brain used
``lambda=0.8``, 20 principal components, 50 SNN neighbors, Harmony
``theta=2``/``max_iter=30`` and resolution ``0.8``. Breast cancer used
``lambda=0.2``, 30 principal components, 50 SNN neighbors, Harmony
``theta=4``/``max_iter=30`` and resolution ``0.3``; its specified igraph Leiden
configuration used two iterations and no boundary refinement.

The public Atlas notebook executes the primary S2R1 example. The additional
S1R1 and S3R1 manuscript analyses use the same fixed-seed automatic pipeline
but are not separate public tutorial notebooks. Their final fixed results are
12 pairs for S1R1 (stage counts ``3/7/11``, followed by ``12→12``) and 11
pairs for S3R1 (stage counts ``4/5/10``, followed by ``10→11`` and
``11→11``).

The full 20-section aging-brain alignment is likewise a manuscript-scale
analysis rather than a public executable tutorial. The reported result uses
Leiden resolution ``0.8`` (19 joint structures), seed ``1000`` and 800
S-LDDMM iterations for each of the 19 source-to-4.3-month alignments. The
public aging-brain notebook demonstrates post-alignment inference from
packaged fixed-alignment coordinates. An independent full-cohort repeat through
the public API kept the fixed joint labels and reproduced pre-alignment to
within ``2.28e-12`` original coordinate units. All 19 final aligned coordinate
sets were within ``2.17`` units of the selected outputs. We therefore compare
coordinates using a ``3.0``-unit tolerance, one tenth of the 30-unit raster
spacing; GPU interpolation can produce small numerical differences even when
all random seeds and discrete inputs are fixed. With the current public API,
the timed scope from automatic pre-alignment through rasterization and
synchronized S-LDDMM took
``2.100`` minutes in total (mean ``6.633`` seconds for 19 alignments), with a
peak PyTorch allocation of ``96.994`` MiB (``0.0947`` GiB) on the recorded
NVIDIA RTX PRO 6000 Blackwell Max-Q system. Clustering, coordinate-quality
checks, plotting and serialization were outside this timing scope.

The manuscript simulation uses base seed ``2026`` and replicate identifier
``1`` (main seed ``2027``), with deterministic operation-specific offsets. The
executed fitted-model bank contains 300 genes including *Gamt*, and the
specified negative-effect multiplier is ``0.25``.

What to record
--------------

For each reported run, retain the input file identities, observation order,
full configuration, random seed, package versions, device, dtype and the
specified continuous-coordinate tolerance. Figures and biological metrics
should come from the same fixed-seed clustering and alignment run.
