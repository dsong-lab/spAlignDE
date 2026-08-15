import hashlib
import unittest
from importlib.resources import files

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
            "fixed-seed cross-sample tutorial alignment",
        )
        self.assertEqual(metadata["alignment_seed"], 1000)
        self.assertEqual(metadata["inference_seed"], 1)
        self.assertEqual(metadata["coordinate_scale_factor"], 50)
        self.assertEqual(
            metadata["public_data"]["alignment_source"]["record"],
            "20647680",
        )
        self.assertEqual(
            metadata["public_data"]["inference_expression_source"][
                "record"
            ],
            "17676992",
        )
        self.assertEqual(
            metadata["validation"]["nearest_label_agreement_aligned"],
            0.736593591905565,
        )
        self.assertFalse(metadata["slddmm"]["restore_best_checkpoint"])
        for sample_id in KIDNEY_SAMPLES:
            frame = load_kidney_aligned_coordinates(sample_id)
            self.assertEqual(
                metadata["samples"][sample_id]["n_spots"],
                len(frame),
            )
            resource = files("spAlignDE.datasets.kidney").joinpath(
                f"aligned_coords_{sample_id}.csv.gz"
            )
            with resource.open("rb") as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            self.assertEqual(
                metadata["samples"][sample_id][
                    "coordinate_file_sha256"
                ],
                digest,
            )

    def test_unknown_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample_id must be one of"):
            load_kidney_aligned_coordinates("unknown")


if __name__ == "__main__":
    unittest.main()
