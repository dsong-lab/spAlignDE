import unittest

from spAlignDE.inference._legacy_core import _resolve_risk_map_radius


class RiskMapRadiusTests(unittest.TestCase):
    def test_default_radius_tracks_grid_spacing(self):
        self.assertAlmostEqual(_resolve_risk_map_radius(27.195), 40.7925)
        self.assertAlmostEqual(_resolve_risk_map_radius(0.515), 0.7725)

    def test_default_radius_is_scale_equivariant(self):
        original = _resolve_risk_map_radius(25.728)
        rescaled = _resolve_risk_map_radius(25.728 / 50.0)
        self.assertAlmostEqual(50.0 * rescaled, original)

    def test_explicit_radius_is_retained(self):
        self.assertEqual(
            _resolve_risk_map_radius(2.0, configured_radius=7.0),
            7.0,
        )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_risk_map_radius(0.0)
        with self.assertRaises(ValueError):
            _resolve_risk_map_radius(1.0, grid_multiplier=-1.0)


if __name__ == "__main__":
    unittest.main()
