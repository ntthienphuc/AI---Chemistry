# -*- coding: utf-8 -*-
import unittest
import torch
import torch.nn as nn
from ai_chemistry.modeling import (
    MultiTaskHetero,
    MultiTaskHeteroFlexible,
    infer_head_variant,
    strip_state_dict_prefix,
    PAPER_BACKBONES,
)


class TestModelArchitecture(unittest.TestCase):
    def test_canonical_mlp2_structure(self):
        # Test architecture initialization and MLP2 heads
        # Using a fast lightweight backbone for unit testing
        model = MultiTaskHetero(
            timm_name="mobilenetv3_large_100.ra_in1k",
            num_classes=2,
            pretrained=False,
            image_size=224,
        )

        # Check Head structure
        self.assertIsInstance(model.head_cls[0], nn.Linear)
        self.assertEqual(model.head_cls[0].out_features, 512)
        self.assertIsInstance(model.head_cls[1], nn.ReLU)
        self.assertIsInstance(model.head_cls[2], nn.Dropout)
        self.assertIsInstance(model.head_cls[3], nn.Linear)
        self.assertEqual(model.head_cls[3].out_features, 2)

        self.assertIsInstance(model.head_reg_NH4[0], nn.Linear)
        self.assertEqual(model.head_reg_NH4[0].out_features, 512)
        self.assertEqual(model.head_reg_NH4[3].out_features, 2)

        self.assertIsInstance(model.head_reg_NO2[0], nn.Linear)
        self.assertEqual(model.head_reg_NO2[0].out_features, 512)
        self.assertEqual(model.head_reg_NO2[3].out_features, 2)

        # Forward pass
        x = torch.zeros(2, 3, 224, 224)
        logits, rNH4, rNO2 = model(x)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(rNH4.shape, (2, 2))
        self.assertEqual(rNO2.shape, (2, 2))

    def test_infer_head_variant(self):
        # MLP2 state dict keys
        mlp2_state = {
            "head_cls.0.weight": torch.zeros(512, 1024),
            "head_cls.3.weight": torch.zeros(2, 512),
        }
        self.assertEqual(infer_head_variant(mlp2_state), "mlp2")

        # Linear state dict keys
        linear_state = {
            "head_cls.1.weight": torch.zeros(2, 1024),
        }
        self.assertEqual(infer_head_variant(linear_state), "linear")

    def test_strip_state_dict_prefix(self):
        state = {
            "module.backbone.conv.weight": torch.zeros(1),
            "model.head_cls.0.weight": torch.zeros(1),
            "norm.weight": torch.zeros(1),
        }
        stripped = strip_state_dict_prefix(state)
        self.assertIn("backbone.conv.weight", stripped)
        self.assertIn("head_cls.0.weight", stripped)
        self.assertIn("norm.weight", stripped)


if __name__ == "__main__":
    unittest.main()