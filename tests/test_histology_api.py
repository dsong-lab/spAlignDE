from __future__ import annotations

import inspect
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
        self.assertTrue(callable(spAlignDE.plot_st_histology_pair_overlap))
        self.assertEqual(spAlignDE.STHistologyStructureConfig().n_levels, 5)

    def test_histology_pairing_defaults_match_paper_and_exclude_asd(self):
        from spAlignDE.alignment.histology import _histology_pairing_weights
        from spAlignDE.alignment import _histology_alignment_core as core

        config = spAlignDE.STHistologyAlignmentConfig()
        self.assertFalse(config.restore_best_checkpoint)
        self.assertEqual(config.kernel_scale, 60.0)
        self.assertEqual(config.velocity_grid_spacing, 6.0)
        weights = _histology_pairing_weights(config)
        self.assertEqual(
            weights,
            {
                "sdf_corr": 0.20,
                "chamfer_sim": 0.40,
                "dice": 0.15,
                "area_sim": 0.25,
                "thick_sim": 0.00,
            },
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        metrics = core.compute_all_metrics(
            np.array([[1, 1], [1, 0]], dtype=np.uint8),
            np.array([[1, 0], [1, 1]], dtype=np.uint8),
        )
        self.assertIn("asd", metrics)
        self.assertNotIn("asd_sim", metrics)
        self.assertIn("thick_sim", metrics)

    def test_histology_pairing_weights_are_user_configurable_and_normalized(self):
        from spAlignDE.alignment.histology import _histology_pairing_weights

        weights = _histology_pairing_weights(
            spAlignDE.STHistologyAlignmentConfig(
                pairing_weight_sdf=0.25,
                pairing_weight_chamfer=0.35,
                pairing_weight_dice=0.20,
                pairing_weight_area=0.15,
                pairing_weight_thickness=0.05,
            )
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            _histology_pairing_weights(
                spAlignDE.STHistologyAlignmentConfig(pairing_weight_sdf=0.30)
            )

    def test_slddmm_core_has_no_landmark_arguments(self):
        from spAlignDE.alignment import _atlas_core, _slddmm_core

        for function in (
            _slddmm_core.LDDMM_shooting,
            _atlas_core.LDDMM_shooting,
            _atlas_core.LDDMM_shooting_mixture,
        ):
            parameters = inspect.signature(function).parameters
            self.assertNotIn("pointsI", parameters)
            self.assertNotIn("pointsJ", parameters)

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

    def test_pair_overlap_plot_reports_before_and_after_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            histology = self.make_histology(Path(temporary))
            target_0 = np.zeros((16, 16), dtype=np.uint8)
            target_1 = np.zeros((16, 16), dtype=np.uint8)
            target_0[2:7, 2:7] = 1
            target_1[9:14, 9:14] = 1
            before_0 = np.zeros_like(target_0)
            before_1 = np.zeros_like(target_1)
            before_0[2:7, 0:5] = 1
            before_1[9:14, 11:16] = 1
            whole = np.ones_like(target_0)
            result = spAlignDE.STHistologyAlignmentResult(
                adata=self.make_adata(),
                histology=histology,
                matched_pairs=pd.DataFrame(
                    {"st": ["0", "1"], "he": [0, 1]}
                ),
                prealignment_parameters={},
                context={
                    "source_binary": np.stack((before_0, before_1, whole)),
                    "target_binary": np.stack((target_0, target_1, whole)),
                    "final_source_binary": np.stack((target_0, target_1)),
                    "final_target_binary": np.stack((target_0, target_1)),
                },
            )

            fig, axes, metrics = spAlignDE.plot_st_histology_pair_overlap(result)
            self.assertEqual(axes.shape, (2, 3))
            self.assertEqual(len(metrics), 4)
            before = metrics[metrics["stage"] == "before"].sort_values(
                "pair_order"
            )
            after = metrics[metrics["stage"] == "after"].sort_values(
                "pair_order"
            )
            self.assertTrue(
                np.all(after["dice"].to_numpy() > before["dice"].to_numpy())
            )
            np.testing.assert_allclose(after["dice"], 1.0)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
