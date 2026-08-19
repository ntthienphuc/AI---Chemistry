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

    def test_greenborder_numerical_stability(self):
        norm = GreenBorderNormalizer()
        # Create synthetic image with a green border
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # BGR: Green is (0, 200, 0)
        img[:20, :] = [0, 200, 0]
        img[-20:, :] = [0, 200, 0]
        img[:, :20] = [0, 200, 0]
        img[:, -20:] = [0, 200, 0]
        # Center patch
        img[50:150, 50:150] = [100, 100, 200]

        out = norm(img)
        self.assertEqual(out.shape, (200, 200, 3))
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))

    def test_srgb_linear_roundtrip(self):
        x = np.linspace(0.0, 1.0, 100).astype(np.float32)
        lin = srgb_to_linear(x)
        rec = linear_to_srgb(lin)
        np.testing.assert_allclose(x, rec, atol=1e-5)


if __name__ == "__main__":
    unittest.main()