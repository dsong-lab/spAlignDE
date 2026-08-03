import unittest

import pandas as pd
from shapely.geometry import box

from spalignde.inference._legacy_core import (
    _resolve_grid_n_for_tissue,
    _resolve_grid_n_for_tissue_exact,
    _typical_sample_size,
)


class SharedGridResolutionTests(unittest.TestCase):
    def test_r_driven_resolution_is_retained_inside_range(self):
        resolved = _resolve_grid_n_for_tissue(50, 2_000, 0.8)
        self.assertEqual(resolved["grid_n"], 50)
        self.assertEqual(resolved["selection"], "r_driven")

    def test_resolution_increases_below_n_typ(self):
        resolved = _resolve_grid_n_for_tissue(30, 2_000, 0.8)
        self.assertEqual(resolved["grid_n"], resolved["grid_n_bounds"][0])
        self.assertEqual(resolved["selection"], "n_typ_lower_bound")
        self.assertGreaterEqual(0.8 * resolved["grid_n"] ** 2, 2_000)

    def test_resolution_decreases_above_two_n_typ(self):
        resolved = _resolve_grid_n_for_tissue(100, 2_000, 0.8)
        self.assertEqual(resolved["grid_n"], resolved["grid_n_bounds"][1])
        self.assertEqual(resolved["selection"], "two_n_typ_upper_bound")
        self.assertLessEqual(0.8 * resolved["grid_n"] ** 2, 4_000)

    def test_two_sample_n_typ_is_median_observation_count(self):
        frame = pd.DataFrame({
            "sample_id": ["a"] * 100 + ["b"] * 200,
            "x": range(300),
            "y": range(300),
        })
        self.assertEqual(_typical_sample_size(frame), 150.0)

    def test_multiple_sample_n_typ_is_median_observation_count(self):
        frame = pd.DataFrame({
            "sample_id": ["a"] * 100 + ["b"] * 200 + ["c"] * 500,
            "x": range(800),
            "y": range(800),
        })
        self.assertEqual(_typical_sample_size(frame), 200.0)

    def test_explicit_grid_n_overrides_automatic_rule(self):
        resolved = _resolve_grid_n_for_tissue(
            100,
            2_000,
            0.8,
            explicit_grid_n=73,
        )
        self.assertEqual(resolved["grid_n"], 73)
        self.assertEqual(resolved["selection"], "explicit")

    def test_exact_rule_retains_r_driven_count_inside_range(self):
        resolved = _resolve_grid_n_for_tissue_exact(box(0, 0, 2, 1), 50, 1_300)
        self.assertEqual(resolved["grid_n"], 50)
        self.assertEqual(resolved["m_grid"], 2_500)
        self.assertEqual(resolved["selection"], "r_driven")

    def test_exact_rule_increases_actual_count_to_lower_boundary(self):
        resolved = _resolve_grid_n_for_tissue_exact(box(0, 0, 2, 1), 20, 1_200)
        self.assertEqual(resolved["selection"], "n_typ_lower_bound")
        self.assertGreaterEqual(resolved["m_grid"], 1_200 - 50)
        self.assertLessEqual(resolved["m_grid"], 2_400)

    def test_exact_rule_decreases_actual_count_to_upper_boundary(self):
        resolved = _resolve_grid_n_for_tissue_exact(box(0, 0, 2, 1), 100, 1_200)
        self.assertEqual(resolved["selection"], "two_n_typ_upper_bound")
        self.assertGreaterEqual(resolved["m_grid"], 1_200)
        self.assertLessEqual(abs(resolved["m_grid"] - 2_400), 2 * resolved["grid_n"])

    def test_exact_rule_uses_rectangular_bbox_occupancy(self):
        resolved = _resolve_grid_n_for_tissue_exact(box(0, 0, 4, 1), 100, 1_200)
        self.assertAlmostEqual(resolved["tissue_occupancy"], 1.0)
        self.assertLessEqual(abs(resolved["m_grid"] - 2_400), 2 * resolved["grid_n"])

    def test_exact_explicit_grid_n_is_not_clamped(self):
        resolved = _resolve_grid_n_for_tissue_exact(
            box(0, 0, 4, 1), 100, 1_200, explicit_grid_n=73
        )
        self.assertEqual(resolved["grid_n"], 73)
        self.assertEqual(resolved["m_grid"], 73 ** 2)
        self.assertEqual(resolved["selection"], "explicit")


if __name__ == "__main__":
    unittest.main()
