import unittest

import numpy as np

from spAlignDE.datasets import (
    KIDNEY_SAMPLES,
    canonical_visium_barcodes,
    kidney_alignment_metadata,
    load_kidney_aligned_coordinates,
)


class KidneyDatasetTests(unittest.TestCase):
    def test_packaged_alignment_coordinates_are_complete(self):
        expected_spots = {"NL3": 3215, "IL3": 2965}

        self.assertEqual(KIDNEY_SAMPLES, ("NL3", "IL3"))
        for sample_id, n_expected in expected_spots.items():
            frame = load_kidney_aligned_coordinates(sample_id)

            self.assertEqual(
                frame.columns.tolist(),
                ["cell_id", "sample_id", "x", "y"],
            )
            self.assertEqual(len(frame), n_expected)
            self.assertEqual(set(frame["sample_id"]), {sample_id})
            self.assertFalse(frame["cell_id"].duplicated().any())
            self.assertTrue(
                np.isfinite(frame[["x", "y"]].to_numpy(float)).all()
            )
            barcodes = canonical_visium_barcodes(frame["cell_id"])
            self.assertFalse(barcodes.duplicated().any())

    def test_packaged_alignment_metadata_matches_files(self):
        metadata = kidney_alignment_metadata()

        self.assertEqual(
            metadata["coordinate_version"],
            "manuscript aligned_317",
        )
        for sample_id in KIDNEY_SAMPLES:
            frame = load_kidney_aligned_coordinates(sample_id)
            self.assertEqual(
                metadata["samples"][sample_id]["n_spots"],
                len(frame),
            )

    def test_unknown_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample_id must be one of"):
            load_kidney_aligned_coordinates("unknown")


if __name__ == "__main__":
    unittest.main()
