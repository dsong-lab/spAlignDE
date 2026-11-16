Clustering
==========

spAlignDE applies BANKSY clustering to incorporate both gene-expression
similarity and local spatial neighborhood information. Single clustering is
used for cross-modality alignment, while joint clustering is used for
cross-sample alignment; the joint workflow also uses Harmony embeddings to
reduce potential slide-specific batch effects.

Both notebooks state their CSV/AnnData input format, display raw and refined
spatial maps, and write the clustered H5AD used by the next notebook. Tune
``banksy_lambda`` and ``resolution`` first, keep the random seed fixed, and
select spatially coherent structures rather than matching example cluster
numbers. See :doc:`../tutorials/parameter_tuning` for boundary-refinement and
cross-sample integration guidance.

For seeds and comparisons across runs, see :doc:`../tutorials/reproducibility`.

- :doc:`Single clustering for cross-modality alignment (CSV or AnnData)
  <clustering/clustering_single_nb>`
- :doc:`Joint Clustering - Two Adult Mouse Brain Sections from MERFISH
  <clustering/clustering_joint_nb>`

.. toctree::
   :hidden:
   :maxdepth: 1

   clustering/clustering_single_nb
   clustering/clustering_joint_nb
