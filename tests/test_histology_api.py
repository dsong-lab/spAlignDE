from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import spAlignDE


class HistologyContractTests(unittest.TestCase):
    def make_adata(self) -> ad.AnnData:
        coordinates = np.array(
            [[1.0, 2.0], [3.0, 2.0], [1.0, 4.0], [3.0, 4.0]],
            dtype=float,
        )
        return ad.AnnData(
            X=np.ones((4, 3), dtype=np.float32),
            obs=pd.DataFrame(
                {"cluster": pd.Categorical(["0", "0", "1", "1"])},
                index=[f"cell_{index}" for index in range(4)],
            ),
            obsm={"spatial": coordinates},
        )

    def make_histology(self, directory: Path) -> spAlignDE.HistologyClusteringResult:
        image_path = directory / "he.jpg"
        Image.new("RGB", (32, 32), color=(245, 235, 240)).save(image_path)
        labels = np.full((2, 2), -1, dtype=np.int32)
        labels[0, 0] = 0
        labels[1, 1] = 1
        return spAlignDE.HistologyClusteringResult(
            image_path=image_path,
            feature_path=directory / "embeddings-hist-vit.pickle",
            labels_raw=labels.copy(),
            labels_filled=labels.copy(),
            labels_merged=labels.copy(),
            labels_clean=labels.copy(),
            tissue_mask=labels >= 0,
            summary={"regions": {"labels_merged_clean": 2}},
        )

    def test_histology_api_is_exported_from_canonical_package(self):
        from spAlignDE.alignment import prealign_st_to_histology

        self.assertIs(prealign_st_to_histology, spAlignDE.prealign_st_to_histology)
        self.assertTrue(callable(spAlignDE.extract_histology_features))
        self.assertTrue(callable(spAlignDE.cluster_histology_features))
        self.assertTrue(callable(spAlignDE.build_st_histology_structures))
        self.assertEqual(spAlignDE.STHistologyStructureConfig().n_levels, 5)

    def test_image_preparation_uses_only_the_image_and_records_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (225, 227), color=(255, 255, 255)).save(source)
            prepared, manifest = spAlignDE.prepare_histology_image(
                source,
                root / "prepared",
            )

            with Image.open(prepared) as image:
                self.assertEqual(image.size, (448, 448))
            self.assertEqual(manifest["feature_grid_shape_hw"], [28, 28])
            self.assertEqual(manifest["pad_right_px"], 223)
            self.assertEqual(manifest["pad_bottom_px"], 221)

    def test_official_hipt_clone_layout_is_resolved(self):
        from spAlignDE.alignment.histology import _resolve_hipt_assets

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "HIPT_4K"
            checkpoints = source / "Checkpoints"
            checkpoints.mkdir(parents=True)
            for filename in (
                "hipt_4k.py",
                "hipt_model_utils.py",
                "vision_transformer.py",
                "vision_transformer4k.py",
            ):
                (source / filename).touch()
            model256 = checkpoints / "vit256_small_dino.pth"
            model4k = checkpoints / "vit4k_xs_dino.pth"
            model256.touch()
            model4k.touch()

            resolved = _resolve_hipt_assets(
                spAlignDE.HistologyFeatureConfig(hipt_dir=root)
            )
            self.assertEqual(resolved, (source, model256, model4k))

    def test_ome_pixel_size_is_used_without_spatial_assay_inputs(self):
        import tifffile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.ome.tif"
            tifffile.imwrite(
                source,
                np.zeros((20, 10, 3), dtype=np.uint8),
                photometric="rgb",
                metadata={
                    "axes": "YXS",
                    "PhysicalSizeX": 0.25,
                    "PhysicalSizeY": 0.25,
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeYUnit": "µm",
                },
            )
            _, manifest = spAlignDE.prepare_histology_image(
                source,
                root / "prepared",
                target_microns_per_pixel=0.5,
            )

            self.assertEqual(manifest["resized_size_wh"], [5, 10])
            self.assertEqual(manifest["padded_size_wh"], [224, 224])
            self.assertAlmostEqual(manifest["source_microns_per_pixel"], 0.25)
            self.assertAlmostEqual(manifest["output_microns_per_pixel"], 0.5)

    def test_manual_prealignment_preserves_input_and_standardizes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            histology = self.make_histology(Path(temporary))
            adata = self.make_adata()
            original = np.asarray(adata.obsm["spatial"]).copy()
            result = spAlignDE.prealign_st_to_histology(
                adata,
                histology,
                config=spAlignDE.HistologyPrealignmentConfig(
                    method="manual",
                    manual=spAlignDE.ManualPrealignmentConfig(
                        scale=2.0,
                        theta_deg=0.0,
                        translation_x=5.0,
                        translation_y=-1.0,
                    ),
                ),
            )

            expected = original * 2.0 + np.array([5.0, -1.0])
            np.testing.assert_allclose(
                result.adata.obs[["x_prealigned", "y_prealigned"]], expected
            )
            np.testing.assert_allclose(
                result.adata.obs[["x_aligned", "y_aligned"]], expected
            )
            np.testing.assert_array_equal(adata.obsm["spatial"], original)
            self.assertNotIn("spAlignDE", adata.uns)
            self.assertEqual(
                result.adata.uns["spAlignDE"]["st_to_histology"]
                ["prealignment"]["method"],
                "manual_similarity",
            )
            result.adata.write_h5ad(Path(temporary) / "manual_prealigned.h5ad")

    def test_histology_plots_have_fixed_image_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            histology = self.make_histology(Path(temporary))
            fig, axes = spAlignDE.plot_histology_feature_clusters(histology)
            self.assertEqual(axes[0, 0].get_title(), "High-resolution H&E image")
            self.assertEqual(axes[1, 1].get_title(), "Cleaned structures (n=2)")
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
