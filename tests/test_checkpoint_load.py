# -*- coding: utf-8 -*-
import os
import unittest
from pathlib import Path
import torch

from ai_chemistry.modeling import (
    MultiTaskHeteroFlexible,
    build_meta_from_ckpt,
    infer_head_in_features,
    infer_head_variant,
    infer_reg_out_dim,
    strip_state_dict_prefix,
)


class TestCheckpointLoading(unittest.TestCase):
    def setUp(self):
        # Resolve weights directory dynamically from environment or default repo location
        env_w = os.getenv("AI_CHEMISTRY_WEIGHTS")
        if env_w and Path(env_w).is_dir():
            self.weights_dir = Path(env_w).resolve()
        else:
            self.weights_dir = Path(__file__).resolve().parents[1] / "weights"

    def test_all_discovered_checkpoints(self):
        checkpoints = sorted(self.weights_dir.rglob("*.pt"))
        if not checkpoints:
            self.skipTest(f"No .pt checkpoints found in {self.weights_dir}")

        for cp in checkpoints:
            if cp.name == "best.pt":
                continue  # YOLO model

            with self.subTest(checkpoint=cp.name):
                ckpt = torch.load(str(cp), map_location="cpu", weights_only=False)
                state = strip_state_dict_prefix(ckpt.get("state_dict", ckpt))
                meta = build_meta_from_ckpt(ckpt)

                head_variant = infer_head_variant(state)
                head_in = infer_head_in_features(state)
                reg_out_dim = infer_reg_out_dim(state)

                model = MultiTaskHeteroFlexible(
                    timm_name=meta.timm_name,
                    num_classes=meta.num_classes,
                    pretrained=False,
                    drop=meta.drop,
                    drop_path=meta.drop_path,
                    image_size=meta.image_size,
                    head_variant=head_variant,
                    reg_out_dim=reg_out_dim,
                    expected_feat_dim=head_in,
                )
                # Must load state_dict strictly
                model.load_state_dict(state, strict=True)
                self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()