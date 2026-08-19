# -*- coding: utf-8 -*-
import unittest
import numpy as np
from ai_chemistry.preprocessing import GreenBorderNormalizer, IdentityNormalizer, srgb_to_linear, linear_to_srgb


class TestGreenBorderRegression(unittest.TestCase):
    def test_identity_normalizer(self):
        norm = IdentityNormalizer()
        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = 255  # Blue channel
        rgb = norm(img_bgr)
        self.assertEqual(rgb.shape, (100, 100, 3))
        self.assertEqual(rgb.dtype, np.float32)
        self.assertAlmostEqual(float(rgb[0, 0, 2]), 1.0, places=5)
        self.assertAlmostEqual(float(rgb[0, 0, 0]), 0.0, places=5)

    def test_greenborder_numerical_constancy(self):
        norm = GreenBorderNormalizer()
        # Create deterministic synthetic pattern with green reference border
        np.random.seed(42)
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        # Green border (BGR: 20, 210, 30)
        img[:15, :] = [20, 210, 30]
        img[-15:, :] = [20, 210, 30]
        img[:, :15] = [20, 210, 30]
        img[:, -15:] = [20, 210, 30]
        # Inner color patch (BGR: 50, 100, 200)
        img[30:90, 30:90] = [50, 100, 200]

        out = norm(img)
        self.assertEqual(out.shape, (120, 120, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))

        # Check exact numerical output at known coordinates (center patch)
        center_rgb = out[60, 60]
        self.assertAlmostEqual(float(center_rgb[0]), 0.9999999, delta=0.05)
        self.assertAlmostEqual(float(center_rgb[1]), 0.4819786, delta=0.05)
        self.assertAlmostEqual(float(center_rgb[2]), 0.8197855, delta=0.05)

    def test_srgb_linear_roundtrip(self):
        x = np.linspace(0.0, 1.0, 100).astype(np.float32)
        lin = srgb_to_linear(x)
        rec = linear_to_srgb(lin)
        np.testing.assert_allclose(x, rec, atol=1e-5)


if __name__ == "__main__":
    unittest.main()