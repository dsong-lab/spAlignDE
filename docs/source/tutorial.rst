Tutorial
========

This section documents reproducible workflows for the spAlignDE pipeline. The
tutorials follow the manuscript structure: first alignment workflows that place
samples or modalities into a common coordinate system, then post-alignment
local inference on the aligned coordinates. Read the parameter guide before
adapting a validated example to a new coordinate system or tissue. Before
running the notebooks, create and verify the :doc:`validated notebook
environment <installation>`.

.. toctree::
   :maxdepth: 2

   tutorials/cross_sample_alignment
   tutorials/cross_modality_alignment
   tutorials/reproducibility
   tutorials/parameter_tuning
   tutorials/post_alignment_inference
