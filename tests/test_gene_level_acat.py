import unittest

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from spAlignDE import (
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

    def test_compact_fit_reconstructs_local_pvalues_from_t_and_df(self):
        result = _result({"query": np.array([0.25])})
        terrain = result.fits["gene_a"]["terrain_data"]
        terrain.pop("p_by_time")
        terrain["stat_by_time"] = {"query": np.array([2.0, -1.0, np.nan])}
        terrain["df_by_time"] = {"query": 12.0}

        local = 2.0 * student_t.sf(np.array([2.0, 1.0]), 12.0)
        expected = acat_pvalue(local)
        self.assertAlmostEqual(gene_level_acat_pvalue(result, "gene_a"), expected)


if __name__ == "__main__":
    unittest.main()
