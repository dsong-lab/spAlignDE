import json
import unittest
from pathlib import Path

import numpy as np

from spAlignDE.datasets import (
    AGING_BRAIN_QUERIES,
    AGING_BRAIN_REFERENCE,
    AGING_BRAIN_SAMPLES,
    aging_brain_genes,
    aging_brain_metadata,
    load_aging_brain,
)


class AgingBrainDatasetTests(unittest.TestCase):
    def test_metadata_and_gene_panel(self):
        metadata = aging_brain_metadata()
        genes = aging_brain_genes()

        self.assertEqual(AGING_BRAIN_REFERENCE, "age_4.3")
        self.assertEqual(
            AGING_BRAIN_QUERIES,
            ("age_6.6", "age_15.8", "age_30.9", "age_34.5"),
        )
        self.assertEqual(metadata["reference"], AGING_BRAIN_REFERENCE)
        self.assertEqual(tuple(metadata["queries"]), AGING_BRAIN_QUERIES)
        self.assertEqual(
            metadata["source"]["processed_data_url"],
            "https://doi.org/10.5281/zenodo.13883177",
        )
        self.assertEqual(
            metadata["source"]["subset"],
            "Five coronal sections selected from the public aging cohorts",
        )
        self.assertEqual(metadata["alignment"]["method"], "spAlignDE")
        self.assertEqual(metadata["alignment"]["random_seed"], 1000)
        self.assertEqual(
            metadata["alignment"]["verification"],
            "two independent fixed-seed runs",
        )
        self.assertEqual(len(genes), 300)
        self.assertEqual(len(set(genes)), len(genes))
        self.assertIn("Gamt", genes)

        settings = metadata["manuscript_settings"]
        self.assertEqual(settings["gene"], "Gamt")
        self.assertEqual(settings["alpha"], 0.05)
        self.assertEqual(settings["density_energy_share"], 0.25)
        self.assertEqual(settings["library_size"], 250)
        self.assertIsNone(settings["grid_n"])
        self.assertTrue(settings["mismatch_aware"])
        self.assertTrue(settings["technical_adjustment"])
        self.assertFalse(settings["cell_type_adjustment"])
        self.assertTrue(settings["global_offset"])
        self.assertTrue(settings["region_cleanup"])
        self.assertEqual(settings["contrast"], "vs_reference")
        self.assertEqual(settings["random_state"], 1)

    def test_subset_load_has_counts_coordinates_and_annotations(self):
        frame = load_aging_brain(
            samples=["age_4.3"],
            genes=["Gamt"],
        )

        self.assertEqual(len(frame), 79824)
        self.assertEqual(set(frame["sample_id"]), {"age_4.3"})
        self.assertFalse(frame["cell_id"].duplicated().any())
        self.assertTrue((frame["Gamt"] >= 0).all())
        self.assertTrue(np.equal(frame["Gamt"], np.floor(frame["Gamt"])).all())
        self.assertTrue(
            np.isfinite(
                frame[["x", "y", "x_aligned", "y_aligned"]].to_numpy(float)
            ).all()
        )
        self.assertFalse(frame["celltype"].isna().any())

    def test_selection_validation(self):
        with self.assertRaisesRegex(ValueError, "Unknown samples"):
            load_aging_brain(samples=["age_unknown"], genes=["Gamt"])
        with self.assertRaisesRegex(ValueError, "Unknown genes"):
            load_aging_brain(samples=["age_4.3"], genes=["NotAGene"])

    def test_tutorial_has_analysis_settings_and_valid_json(self):
        tutorial_path = (
            Path(__file__).resolve().parents[1]
            / "source_notebooks"
            / "post_alignment_inference_aging_brain_nb.ipynb"
        )
        notebook = json.loads(tutorial_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertEqual(notebook["nbformat"], 4)
        self.assertIn("density_energy_share=0.25", source)
        self.assertIn("library_size=250", source)
        self.assertIn("alpha=0.05", source)
        self.assertIn("mismatch_aware=True", source)
        self.assertIn("technical_adjustment=True", source)
        self.assertIn("cell_type_adjustment=False", source)
        self.assertIn("global_offset=True", source)
        self.assertIn("region_cleanup=True", source)
        self.assertIn("show_expression=False", source)
        self.assertEqual(
            AGING_BRAIN_SAMPLES,
            ("age_4.3", "age_6.6", "age_15.8", "age_30.9", "age_34.5"),
        )


if __name__ == "__main__":
    unittest.main()
