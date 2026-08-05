import unittest

import numpy as np
import pandas as pd

from spalignde import (
    LocalDEResult,
    PreparedInference,
    acat_pvalue,
    gene_level_acat_pvalue,
)


def _result(p_by_time):
    prepared = PreparedInference(
        shared={},
        data=pd.DataFrame(),
        genes=("gene_a",),
        reference="ref",
        library_size=None,
        density_energy_share=0.25,
        alignment_uncertainty_key=None,
    )
    return LocalDEResult(
        fits={"gene_a": {"terrain_data": {
            "time_ids": list(p_by_time),
            "p_by_time": p_by_time,
        }}},
        prepared=prepared,
        alpha=0.05,
        contrast="vs_reference",
        mismatch_aware=True,
        technical_adjustment=True,
    )


class GeneLevelAcatTests(unittest.TestCase):
    def test_acat_uses_stable_small_p_branch_and_ignores_boundaries(self):
        observed = acat_pvalue([1e-30, 0.20, 1.0, 0.0, np.nan])
        expected = acat_pvalue([1e-15, 0.20])
        self.assertTrue(np.isfinite(observed))
        self.assertAlmostEqual(observed, expected, places=16)

    def test_single_contrast_combines_local_pvalues(self):
        local = np.array([0.01, 0.20, np.nan, 0.80])
        observed = gene_level_acat_pvalue(_result({"query": local}), "gene_a")
        expected = acat_pvalue([0.01, 0.20, 0.80])
        self.assertAlmostEqual(observed, expected)

    def test_multiple_contrasts_use_valid_grid_counts_as_weights(self):
        result = _result({
            "q1": np.array([0.01, 0.20]),
            "q2": np.array([0.03, 0.40, 0.80, 0.90]),
        })
        expected = acat_pvalue(
            [acat_pvalue([0.01, 0.20]), acat_pvalue([0.03, 0.40, 0.80, 0.90])],
            weights=[2, 4],
        )
        self.assertAlmostEqual(gene_level_acat_pvalue(result, "gene_a"), expected)


if __name__ == "__main__":
    unittest.main()
