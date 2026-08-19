# -*- coding: utf-8 -*-
import unittest
from api.config import MODEL_ZOO, MODEL_ZOO_EXPLICIT, VALID_CALIB_MODES
from api.main import resolve_model_key, infer_calib_mode_from_model, normalize_calib_mode


class TestApiCompatibility(unittest.TestCase):
    def test_legacy_unsuffixed_aliases(self):
        # Verify old model aliases resolve correctly to MODEL_ZOO
        legacy_aliases = ["convnext10k", "mnv313k", "effb03k", "nfnet10k", "swint13k", "tfb310k"]
        for alias in legacy_aliases:
            self.assertIn(alias, MODEL_ZOO, f"Legacy alias '{alias}' missing from MODEL_ZOO")
            resolved = resolve_model_key(alias, None, None, None)
            self.assertEqual(resolved, alias)

    def test_explicit_keys(self):
        explicit_keys = [
            "convnext10k_green", "convnext10k_none",
            "mnv313k_green", "mnv313k_none",
            "effb03k_green", "effb03k_none",
        ]
        for key in explicit_keys:
            self.assertIn(key, MODEL_ZOO, f"Explicit key '{key}' missing")
            resolved = resolve_model_key(key, None, None, None)
            self.assertEqual(resolved, key)

    def test_triplet_resolution(self):
        # Resolve via data_type, model_type, train_calib
        key = resolve_model_key(None, "10k", "convnext", "green")
        self.assertEqual(key, "convnext10k_green")

        key_none = resolve_model_key(None, "13k", "mnv3", "none")
        self.assertEqual(key_none, "mnv313k_none")

    def test_calibration_inference(self):
        self.assertEqual(infer_calib_mode_from_model("convnext10k_none"), "none")
        self.assertEqual(infer_calib_mode_from_model("convnext10k_green"), "greenborder")

    def test_normalize_calib_mode(self):
        self.assertEqual(normalize_calib_mode("greenborder"), "greenborder")
        self.assertEqual(normalize_calib_mode("none"), "none")
        with self.assertRaises(ValueError):
            normalize_calib_mode("invalid_mode")


if __name__ == "__main__":
    unittest.main()