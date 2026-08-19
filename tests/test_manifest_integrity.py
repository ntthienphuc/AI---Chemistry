# -*- coding: utf-8 -*-
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from ai_chemistry.data.loaders import load_publication_splits, SPLIT_NAMES

EXPECTED_COUNTS = {
    "3k": {"train": 2064, "val": 476, "test": 301},
    "10k": {"train": 7227, "val": 1795, "test": 900},
    "13k": {"train": 7495, "val": 2095, "test": 901},
}


class TestManifestIntegrity(unittest.TestCase):
    def setUp(self):
        self.manifests_dir = Path(__file__).resolve().parents[1] / "data" / "manifests"

    def test_all_manifest_counts_and_schemas(self):
        for dataset, expected in EXPECTED_COUNTS.items():
            _, train_df, val_df, test_df = load_publication_splits(self.manifests_dir, dataset)
            self.assertEqual(len(train_df), expected["train"], f"Train count mismatch in {dataset}")
            self.assertEqual(len(val_df), expected["val"], f"Val count mismatch in {dataset}")
            self.assertEqual(len(test_df), expected["test"], f"Test count mismatch in {dataset}")

            # Check zero leakage
            all_paths = {}
            for sname, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                # Required columns
                for col in ["path", "chemical", "ppm"]:
                    self.assertIn(col, df.columns, f"Missing column {col} in {dataset}/{sname}")

                # Chemical domain
                chems = set(df["chemical"].unique())
                self.assertTrue(chems.issubset({"NH4", "NO2"}), f"Invalid chemicals in {dataset}/{sname}")

                # PPM range
                self.assertTrue((df["ppm"] >= 0).all(), f"Negative PPM in {dataset}/{sname}")
                self.assertFalse(df["ppm"].isna().any(), f"NaN PPM in {dataset}/{sname}")

                # Path uniqueness across splits
                for p in df["path"].astype(str).str.lower():
                    self.assertNotIn(p, all_paths, f"Leakage: Path {p} in both {all_paths.get(p)} and {sname} ({dataset})")
                    all_paths[p] = sname


if __name__ == "__main__":
    unittest.main()