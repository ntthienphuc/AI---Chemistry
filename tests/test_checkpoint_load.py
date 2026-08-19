# -*- coding: utf-8 -*-
import unittest
import torch
from pathlib import Path
from ai_chemistry.modeling import MultiTaskHetero, MultiTaskHeteroFlexible, build_meta_from_ckpt, strip_state_dict_prefix, infer_head_variant

class TestCheckpointLoading(unittest.TestCase):
    def setUp(self):
        # Point to real weights folder
        self.weights_dir = Path(r"E:\Project\Chemistry Research\ver 2 - 2025-10-26 - AI Chemistry\weights")

    def test_load_mnv3_13k_none(self):
        ckpt_path = self.weights_dir / "runs_multitask_13k" / "MNV3_seed0_l2.0_none.pt"
        if not ckpt_path.is_file():
            self.skipTest(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = strip_state_dict_prefix(ckpt.get("state_dict", ckpt))
        meta = build_meta_from_ckpt(ckpt)

        model = MultiTaskHetero(
            timm_name=meta.timm_name,
            num_classes=meta.num_classes,
            pretrained=False,
            image_size=meta.image_size,
        )
        model.load_state_dict(state, strict=True)
        self.assertTrue(True)

    def test_load_convnext_10k_none(self):
        ckpt_path = self.weights_dir / "runs_multitask_10k" / "ConvNext_seed0_l2.0_none.pt"
        if not ckpt_path.is_file():
            self.skipTest(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = strip_state_dict_prefix(ckpt.get("state_dict", ckpt))
        meta = build_meta_from_ckpt(ckpt)

        model = MultiTaskHetero(
            timm_name=meta.timm_name,
            num_classes=meta.num_classes,
            pretrained=False,
            image_size=meta.image_size,
        )
        model.load_state_dict(state, strict=True)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()