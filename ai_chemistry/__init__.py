# -*- coding: utf-8 -*-
"""
AI-Chemistry: Smartphone-AI-WebGIS for Inorganic Nitrogen Monitoring.
"""

from .modeling import (
    MultiTaskHetero,
    MultiTaskHeteroFlexible,
    ModelMeta,
    PAPER_BACKBONES,
    build_meta_from_ckpt,
    infer_feat_dim,
    infer_head_variant,
    strip_state_dict_prefix,
)
from .preprocessing import (
    IdentityNormalizer,
    GreenBorderNormalizer,
    get_normalizer,
    srgb_to_linear,
    linear_to_srgb,
    make_train_transform,
    make_eval_transform,
)

__version__ = "1.0.0"
__all__ = [
    "MultiTaskHetero",
    "MultiTaskHeteroFlexible",
    "ModelMeta",
    "PAPER_BACKBONES",
    "build_meta_from_ckpt",
    "infer_feat_dim",
    "infer_head_variant",
    "strip_state_dict_prefix",
    "IdentityNormalizer",
    "GreenBorderNormalizer",
    "get_normalizer",
    "srgb_to_linear",
    "linear_to_srgb",
    "make_train_transform",
    "make_eval_transform",
]