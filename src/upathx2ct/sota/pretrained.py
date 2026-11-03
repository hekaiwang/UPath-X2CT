"""Auditable initialization of the custom grayscale ConvNeXt backbone."""

from __future__ import annotations

from typing import Any

import torch


def map_torchvision_convnext_tiny_backbone(
    backbone,
    source_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Map torchvision ConvNeXt-Tiny into the project backbone exactly."""

    target = backbone.state_dict()
    mapped: dict[str, torch.Tensor] = {
        "stem.0.weight": source_state["features.0.0.weight"].mean(
            dim=1,
            keepdim=True,
        ),
        "stem.0.bias": source_state["features.0.0.bias"],
        "stem.1.norm.weight": source_state["features.0.1.weight"],
        "stem.1.norm.bias": source_state["features.0.1.bias"],
    }
    for stage in range(4):
        source_stage = 1 + 2 * stage
        block = 0
        while f"features.{source_stage}.{block}.block.0.weight" in source_state:
            source_prefix = f"features.{source_stage}.{block}"
            target_prefix = f"stages.{stage}.{block}"
            mapped.update(
                {
                    f"{target_prefix}.gamma": source_state[
                        f"{source_prefix}.layer_scale"
                    ].reshape(-1),
                    f"{target_prefix}.dwconv.weight": source_state[
                        f"{source_prefix}.block.0.weight"
                    ],
                    f"{target_prefix}.dwconv.bias": source_state[
                        f"{source_prefix}.block.0.bias"
                    ],
                    f"{target_prefix}.norm.weight": source_state[
                        f"{source_prefix}.block.2.weight"
                    ],
                    f"{target_prefix}.norm.bias": source_state[
                        f"{source_prefix}.block.2.bias"
                    ],
                    f"{target_prefix}.pwconv1.weight": source_state[
                        f"{source_prefix}.block.3.weight"
                    ],
                    f"{target_prefix}.pwconv1.bias": source_state[
                        f"{source_prefix}.block.3.bias"
                    ],
                    f"{target_prefix}.pwconv2.weight": source_state[
                        f"{source_prefix}.block.5.weight"
                    ],
                    f"{target_prefix}.pwconv2.bias": source_state[
                        f"{source_prefix}.block.5.bias"
                    ],
                }
            )
            block += 1
        if stage < 3:
            source_down = 2 + 2 * stage
            target_prefix = f"downsamples.{stage}"
            mapped.update(
                {
                    f"{target_prefix}.0.norm.weight": source_state[
                        f"features.{source_down}.0.weight"
                    ],
                    f"{target_prefix}.0.norm.bias": source_state[
                        f"features.{source_down}.0.bias"
                    ],
                    f"{target_prefix}.1.weight": source_state[
                        f"features.{source_down}.1.weight"
                    ],
                    f"{target_prefix}.1.bias": source_state[
                        f"features.{source_down}.1.bias"
                    ],
                }
            )
    missing = sorted(set(target) - set(mapped))
    unexpected = sorted(set(mapped) - set(target))
    shape_errors = sorted(
        key
        for key in set(target) & set(mapped)
        if target[key].shape != mapped[key].shape
    )
    if missing or unexpected or shape_errors:
        raise ValueError(
            "ConvNeXt mapping is incomplete: "
            f"missing={missing}, unexpected={unexpected}, shape={shape_errors}"
        )
    backbone.load_state_dict(mapped, strict=True)
    return {
        "matched_tensors": len(mapped),
        "target_tensors": len(target),
        "grayscale_stem": "mean of RGB kernels",
    }


def initialize_encoder_from_torchvision_convnext_tiny(encoder) -> dict[str, Any]:
    """Download/cache and apply official ImageNet-1K ConvNeXt-Tiny weights."""

    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        convnext_tiny,
    )

    if getattr(encoder, "variant", None) != "tiny":
        raise ValueError("Official ConvNeXt-Tiny initialization requires variant='tiny'")
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    source = convnext_tiny(weights=weights)
    report = map_torchvision_convnext_tiny_backbone(
        encoder.backbone,
        source.state_dict(),
    )
    report.update(
        {
            "source": "torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "weights_url": weights.url,
            "preprocessing": str(weights.transforms()),
        }
    )
    return report
