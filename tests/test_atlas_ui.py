from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import spAlignDE
from spalignde.alignment.atlas_ui import _initialize_st_coordinates


class UIAtlasPairingTests(unittest.TestCase):
    def make_export(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group_id": ["group_1", "group_1", "group_2"],
                "left_dataset_kind": ["st", "st", "st"],
                "right_dataset_kind": ["atlas", "atlas", "atlas"],
                "left_selected_id": [2, 2, 5],
                "right_selected_id": [101, 102, 201],
                "left_raw_selected_ids": ["[2]", "[2]", "[5, 6]"],
                "right_raw_selected_ids": [
                    "[101, 102]",
                    "[101, 102]",
                    "[201]",
                ],
                "left_selected_name": ["cluster_2", "cluster_2", "custom"],
                "right_selected_name": ["CA1", "CA3", "DG"],
                "atlas_z_slice": [675, 675, 675],
            }
        )

    def test_group_id_becomes_one_deformation_channel(self):
        pairing = spAlignDE.load_ui_atlas_pairing(
            self.make_export(),
            expected_atlas_slice=675,
        )
        self.assertEqual(pairing.st_side, "left")
        self.assertEqual(pairing.atlas_slice_index, 675)
        self.assertEqual(len(pairing.deformation_groups), 2)
        self.assertEqual(len(pairing.matched_pairs), 3)
        first = pairing.deformation_groups.iloc[0]
        self.assertEqual(first["st_cluster_ids"], [2])
        self.assertEqual(first["atlas_labels_union"], [101, 102])
        second_clusters = pairing.matched_pairs.loc[
            pairing.matched_pairs["group_id"].eq("group_2"), "cluster"
        ].tolist()
        self.assertEqual(second_clusters, ["5", "6"])

    def test_reversed_panels_are_supported(self):
        frame = self.make_export().rename(
            columns={
                column: column.replace("left_", "temporary_").replace(
                    "right_", "left_"
                ).replace("temporary_", "right_")
                for column in self.make_export().columns
                if column.startswith(("left_", "right_"))
            }
        )
        pairing = spAlignDE.load_ui_atlas_pairing(frame)
        self.assertEqual(pairing.st_side, "right")
        self.assertEqual(len(pairing.deformation_groups), 2)

    def test_slice_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            spAlignDE.load_ui_atlas_pairing(
                self.make_export(),
                expected_atlas_slice=700,
            )

    def test_validated_ui_alignment_defaults_are_public(self):
        config = spAlignDE.UIAtlasAlignmentConfig()
        self.assertEqual(config.prealignment_mode, "mask")
        self.assertEqual(config.kernel_scale, 200.0)
        self.assertEqual(config.velocity_grid_spacing, 50.0)
        self.assertEqual(config.iterations, 500)
        self.assertEqual(config.time_steps, 5)
        self.assertFalse(config.restore_best_checkpoint)

    def test_provided_manual_prealignment_is_used_without_mask_search(self):
        table = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
                "x_prealigned": [10.0, 20.0],
                "y_prealigned": [30.0, 40.0],
            },
            index=["a", "b"],
        )
        initialized, parameters = _initialize_st_coordinates(
            table,
            atlas_info={},
            core_config=None,
            config=spAlignDE.UIAtlasAlignmentConfig(prealignment_mode="provided"),
        )
        np.testing.assert_allclose(
            initialized[["x_prealigned", "y_prealigned"]],
            [[10.0, 30.0], [20.0, 40.0]],
        )
        self.assertEqual(parameters["method"], "provided_coordinates")


if __name__ == "__main__":
    unittest.main()
