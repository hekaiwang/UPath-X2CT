"""Frozen B0 cone-aware reconstruction model for UPath-LIDC-v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from upathx2ct.models.encoder.dual_view_convnext_fpn import (
    DualViewConvNeXtFPNEncoder,
)
from upathx2ct.sota.geometry import (
    ConeBeamGeometry,
    projection_matrix_lps,
)
from upathx2ct.sota.projective_lifting import (
    ConeBeamFeatureLifter,
    PhysicalGrid,
)
from upathx2ct.b0.conditioner import LatentUNet
from upathx2ct.b0.ema import resolve_checkpoint_model_state


@dataclass(frozen=True)
class B0Config:
    encoder_variant: str = "tiny"
    fpn_channels: tuple[int, int, int, int] = (128, 192, 256, 384)
    lifter_levels: tuple[str, ...] = ("p3", "p4", "p5")
    sample_channels: int = 16
    condition_channels: int = 96
    latent_channels: int = 8
    dropout: float = 0.10
    detector_shape: tuple[int, int] = (512, 512)
    detector_pixel_size_uv_mm: tuple[float, float] = (0.9, 0.9)
    source_detector_distance_mm: float = 1800.0
    source_object_distance_mm: float = 1500.0
    principal_point_uv_px: tuple[float, float] = (256.0, 256.0)
    global_shape_zyx: tuple[int, int, int] = (128, 128, 128)
    global_origin_centered_lps_mm: tuple[float, float, float] = (
        -190.5,
        -190.5,
        -190.5,
    )
    global_spacing_xyz_mm: tuple[float, float, float] = (3.0, 3.0, 3.0)
    latent_downsample_factor: int = 4


def b0_projection_matrices(config: B0Config) -> torch.Tensor:
    geometry = ConeBeamGeometry(
        source_detector_distance_mm=config.source_detector_distance_mm,
        source_object_distance_mm=config.source_object_distance_mm,
        detector_shape=config.detector_shape,
        detector_pixel_size_mm=config.detector_pixel_size_uv_mm,
        principal_point_px=config.principal_point_uv_px,
    )
    return torch.from_numpy(
        np.stack(
            (
                projection_matrix_lps(geometry, "frontal_ap"),
                projection_matrix_lps(geometry, "left_lateral"),
            )
        )
    ).float()


class ConeGeometryConditioner(nn.Module):
    """Shared ConvNeXt-FPN followed by exact camera-matrix 2D-to-3D lifting."""

    def __init__(self, config: B0Config = B0Config()) -> None:
        super().__init__()
        self.config = config
        self.encoder = DualViewConvNeXtFPNEncoder(
            in_channels=1,
            variant=config.encoder_variant,
            fpn_channels=config.fpn_channels,
            global_dim=256,
            cross_view_levels=("p4", "p5"),
        )
        self.lifter = ConeBeamFeatureLifter(
            self.encoder.out_channels,
            b0_projection_matrices(config),
            parent_grid=PhysicalGrid(
                config.global_shape_zyx,
                config.global_origin_centered_lps_mm,
                config.global_spacing_xyz_mm,
            ),
            detector_shape=config.detector_shape,
            downsample_factor=config.latent_downsample_factor,
            levels=config.lifter_levels,
            sample_channels=config.sample_channels,
            output_channels=config.condition_channels,
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(self.encoder.global_dim * 4, config.condition_channels * 2),
            nn.SiLU(),
            nn.Linear(config.condition_channels * 2, config.condition_channels),
        )

    def encode_views(
        self,
        ap: torch.Tensor,
        lat: torch.Tensor,
    ) -> dict[str, object]:
        return self.encoder(ap, lat)

    def condition_from_features(
        self,
        features: Mapping[str, Any],
        projection_matrices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local = self.lifter(
            features["ap_feats"],
            features["lat_feats"],
            projection_matrices,
        )
        ap_global = features["ap_global"]
        lat_global = features["lat_global"]
        global_condition = self.global_fusion(
            torch.cat(
                (
                    ap_global,
                    lat_global,
                    ap_global * lat_global,
                    (ap_global - lat_global).abs(),
                ),
                dim=1,
            )
        )
        return local + global_condition[:, :, None, None, None]

    def forward(
        self,
        ap: torch.Tensor,
        lat: torch.Tensor,
        projection_matrices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.condition_from_features(
            self.encode_views(ap, lat),
            projection_matrices,
        )


class B0CoarseReconstructor(nn.Module):
    """Deterministic geometry-aware anchor for the probabilistic B0 system."""

    def __init__(self, config: B0Config = B0Config()) -> None:
        super().__init__()
        self.config = config
        self.conditioner = ConeGeometryConditioner(config)
        self.regressor = LatentUNet(
            config.condition_channels,
            config.latent_channels,
            base_channels=config.condition_channels,
            dropout=config.dropout,
        )
        self.auxiliary = nn.Conv3d(config.condition_channels, 3, kernel_size=1)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return tuple(
            size // self.config.latent_downsample_factor
            for size in self.config.global_shape_zyx
        )

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

    def forward(
        self,
        ap: torch.Tensor,
        lat: torch.Tensor,
        projection_matrices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        condition = self.conditioner(ap, lat, projection_matrices)
        if tuple(condition.shape[-3:]) != self.latent_shape:
            raise RuntimeError(
                f"Condition grid {condition.shape[-3:]} != latent {self.latent_shape}"
            )
        auxiliary = self.auxiliary(condition)
        return {
            "condition": condition,
            "coarse_latent": self.regressor(condition),
            "body_logits": auxiliary[:, 0:1],
            "lung_logits": auxiliary[:, 1:2],
            "logvar": auxiliary[:, 2:3].clamp(-6.0, 4.0),
        }


def b0_config_from_mapping(values: Mapping[str, Any]) -> B0Config:
    defaults = asdict(B0Config())
    merged = {**defaults, **dict(values)}
    tuple_fields = (
        "fpn_channels",
        "lifter_levels",
        "detector_shape",
        "detector_pixel_size_uv_mm",
        "principal_point_uv_px",
        "global_shape_zyx",
        "global_origin_centered_lps_mm",
        "global_spacing_xyz_mm",
    )
    for field in tuple_fields:
        merged[field] = tuple(merged[field])
    return B0Config(**merged)


def load_b0_coarse_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> B0CoarseReconstructor:
    checkpoint = torch.load(path, map_location="cpu")
    config = b0_config_from_mapping(checkpoint["b0_config"])
    model = B0CoarseReconstructor(config)
    model.load_state_dict(
        resolve_checkpoint_model_state(checkpoint, prefer_ema=True),
        strict=True,
    )
    return model.to(device)
