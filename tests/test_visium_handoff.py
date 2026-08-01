import unittest

import anndata as ad
import numpy as np
import pandas as pd

from spAlignDE.datasets import (
    build_visium_coordinate_table,
    build_visium_inference_table,
    canonical_visium_barcodes,
)


class VisiumHandoffTests(unittest.TestCase):
    def test_build_visium_inference_table_uses_barcode_not_row_order(self):
        genes = [f"gene_{index}" for index in range(12)]
        aligned = ad.AnnData(
            X=np.zeros((4, 1), dtype=float),
            obs=pd.DataFrame(
                {
                    "sample_id": ["NL3", "NL3", "IL3", "IL3"],
                    "x": [1.0, 2.0, 1.0, 2.0],
                    "y": [3.0, 4.0, 3.0, 4.0],
                    "x_aligned": [1.0, 2.0, 1.1, 2.1],
                    "y_aligned": [3.0, 4.0, 3.1, 4.1],
                },
                index=[
                    "AAAA-1|NL3.csv",
                    "CCCC-1|NL3.csv",
                    "AAAA-1|IL3.csv",
                    "CCCC-1|IL3.csv",
                ],
            ),
            var=pd.DataFrame(index=["unused"]),
        )
        nl3 = ad.AnnData(
            X=np.vstack([np.full(12, 2.0), np.full(12, 1.0)]),
            obs=pd.DataFrame(index=["CCCC-1", "AAAA-1"]),
            var=pd.DataFrame(index=genes),
        )
        il3 = ad.AnnData(
            X=np.vstack([np.full(12, 4.0), np.full(12, 3.0)]),
            obs=pd.DataFrame(index=["CCCC-1", "AAAA-1"]),
            var=pd.DataFrame(index=genes),
        )

        observed = build_visium_inference_table(
            aligned,
            {"NL3": nl3, "IL3": il3},
            genes=["gene_0"],
            min_detected_spots=1,
            min_total_counts=1,
            batch="kidney_pair",
        )

        self.assertEqual(observed.sample_sizes, {"NL3": 2, "IL3": 2})
        self.assertEqual(observed.n_common_genes, 12)
        self.assertEqual(len(observed.risk_genes), 12)
        self.assertEqual(observed.data["barcode"].tolist(), [
            "AAAA-1", "CCCC-1", "AAAA-1", "CCCC-1"
        ])
        self.assertEqual(observed.data["gene_0"].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(set(observed.data["batch"]), {"kidney_pair"})

    def test_canonical_visium_barcodes_accepts_common_prefixes(self):
        observed = canonical_visium_barcodes(
            [
                "NL3__AAACAAGTATCTCCCA-1",
                "CTRL3_AAACAGAGCGACTCCT-1",
                "AAACAGCTTTCAGAAG-1",
                "AAACAGGGTCTATATT-1|IL3_spatial_filtered.csv",
            ]
        )
        self.assertEqual(
            observed.tolist(),
            [
                "AAACAAGTATCTCCCA-1",
                "AAACAGAGCGACTCCT-1",
                "AAACAGCTTTCAGAAG-1",
                "AAACAGGGTCTATATT-1",
            ],
        )

    def test_build_visium_coordinate_table_matches_by_barcode(self):
        positions = pd.DataFrame(
            {
                "barcode": [
                    "CTRL3_AAACAAGTATCTCCCA-1",
                    "CTRL3_AAACAGAGCGACTCCT-1",
                ],
                "array_row": [102, 94],
                "array_col": [50, 14],
            }
        )
        aligned = pd.DataFrame(
            {
                "cell_id": [
                    "NL3__AAACAGAGCGACTCCT-1",
                    "NL3__AAACAAGTATCTCCCA-1",
                ],
                "x": [14.0, 50.0],
                "y": [94.0, 102.0],
                "sample_id": ["NL3", "NL3"],
            }
        )

        observed = build_visium_coordinate_table(
            positions,
            aligned,
            sample_id="NL3",
        )

        self.assertEqual(
            observed["barcode"].tolist(),
            [
                "AAACAAGTATCTCCCA-1",
                "AAACAGAGCGACTCCT-1",
            ],
        )
        self.assertEqual(observed["x_aligned"].tolist(), [50.0, 14.0])
        self.assertEqual(observed["y_aligned"].tolist(), [102.0, 94.0])

    def test_build_visium_coordinate_table_rejects_unmatched_spots(self):
        positions = pd.DataFrame(
            {
                "barcode": ["CTRL3_AAACAAGTATCTCCCA-1"],
                "array_row": [102],
                "array_col": [50],
            }
        )
        aligned = pd.DataFrame(
            {
                "cell_id": ["NL3__AAACAGAGCGACTCCT-1"],
                "x": [14.0],
                "y": [94.0],
                "sample_id": ["NL3"],
            }
        )

        with self.assertRaisesRegex(ValueError, "do not match one-to-one"):
            build_visium_coordinate_table(
                positions,
                aligned,
                sample_id="NL3",
            )


if __name__ == "__main__":
    unittest.main()
