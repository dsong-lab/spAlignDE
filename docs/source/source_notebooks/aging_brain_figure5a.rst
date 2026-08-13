Aging Mouse Brain - Figure 5A
=============================

This self-contained notebook reproduces the four mismatch-aware ``Gamt``
local-statistic maps from Figure 5A. It compares the 6.6-, 15.8-, 30.9-, and
34.5-month MERFISH sections with the 4.3-month reference in their precomputed
shared alignment space.

The package includes raw counts for the complete 300-gene mismatch-risk
candidate panel, original and aligned coordinates, and cell-type labels for all
five sections. No external download or alignment run is required. The notebook
uses the manuscript settings, including the automatic shared-grid resolution,
local technical covariates, gene-specific global offset, mismatch-aware
variance adjustment, connected-region cleanup, and grid-level FDR threshold
``q <= 0.05``.

Source notebook
---------------

.. toctree::
   :maxdepth: 1

   mouse_aging_brain_figure5a
