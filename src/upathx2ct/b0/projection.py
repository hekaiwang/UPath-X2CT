from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProjectionConfig:
    """Canonical parallel-beam log-attenuation projection parameters."""

    hu_min: float = -1024.0
    hu_max: float = 1500.0
    spacing_mm: float = 3.0
    attenuation_scale_mm: float = 500.0


def normalized_to_hu(volume: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    return (volume + 1.0) * 0.5 * (hu_max - hu_min) + hu_min


def hu_to_relative_attenuation(hu: torch.Tensor) -> torch.Tensor:
    """Convert HU to non-negative attenuation relative to water."""

    return torch.clamp(hu / 1000.0 + 1.0, min=0.0, max=3.5)


def project_ap_lat(
    volume: torch.Tensor,
    config: ProjectionConfig = ProjectionConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project canonical ``[B, C, Z, Y, X]`` CT volumes to AP and LAT.

    Outputs are calibrated log-attenuation maps in a fixed intensity scale.
    No per-image min/max normalization is used, so body thickness remains
    observable. AP is ``[Z, X]`` and LAT is ``[Z, Y]``.
    """

    if volume.ndim != 5:
        raise ValueError(f"Expected [B,C,Z,Y,X], got {tuple(volume.shape)}")
    hu = normalized_to_hu(volume, config.hu_min, config.hu_max)
    attenuation = hu_to_relative_attenuation(hu)
    ap = attenuation.sum(dim=3) * config.spacing_mm / config.attenuation_scale_mm
    lat = attenuation.sum(dim=4) * config.spacing_mm / config.attenuation_scale_mm
    return torch.clamp(ap, 0.0, 1.5), torch.clamp(lat, 0.0, 1.5)


def backproject_ap_lat(
    ap: torch.Tensor,
    lat: torch.Tensor,
    volume_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Backproject AP/LAT maps into a geometry-aligned 3D evidence volume."""

    if ap.ndim != 4 or lat.ndim != 4:
        raise ValueError("AP and LAT must be [B,C,H,W] tensors")
    depth, height, width = volume_shape
    ap = F.interpolate(ap, size=(depth, width), mode="bilinear", align_corners=True)
    lat = F.interpolate(lat, size=(depth, height), mode="bilinear", align_corners=True)
    ap_volume = ap.unsqueeze(3).expand(-1, -1, depth, height, width)
    lat_volume = lat.unsqueeze(4).expand(-1, -1, depth, height, width)
    return 0.5 * (ap_volume + lat_volume)


def image_gradients_2d(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = image[..., :, 1:] - image[..., :, :-1]
    return dy, dx


def projection_consistency_loss(
    prediction: torch.Tensor,
    target_ap: torch.Tensor,
    target_lat: torch.Tensor,
    config: ProjectionConfig = ProjectionConfig(),
    gradient_weight: float = 0.25,
) -> torch.Tensor:
    pred_ap, pred_lat = project_ap_lat(prediction, config)
    loss = F.l1_loss(pred_ap, target_ap) + F.l1_loss(pred_lat, target_lat)
    if gradient_weight > 0:
        for pred, target in ((pred_ap, target_ap), (pred_lat, target_lat)):
            pred_dy, pred_dx = image_gradients_2d(pred)
            target_dy, target_dx = image_gradients_2d(target)
            loss = loss + gradient_weight * (
                F.l1_loss(pred_dy, target_dy) + F.l1_loss(pred_dx, target_dx)
            )
    return 0.5 * loss


class ProjectionDomainAugmentor(nn.Module):
    """Paired projection augmentation for synthetic-to-real robustness."""

    def __init__(
        self,
        gain_range: tuple[float, float] = (0.85, 1.15),
        bias_range: tuple[float, float] = (-0.04, 0.04),
        noise_std: float = 0.015,
        scatter_strength: float = 0.08,
        view_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.gain_range = gain_range
        self.bias_range = bias_range
        self.noise_std = noise_std
        self.scatter_strength = scatter_strength
        self.view_dropout = view_dropout

    def _augment_one(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        shape = (batch, 1, 1, 1)
        gain = torch.empty(shape, device=image.device, dtype=image.dtype).uniform_(*self.gain_range)
        bias = torch.empty(shape, device=image.device, dtype=image.dtype).uniform_(*self.bias_range)
        scatter = F.avg_pool2d(image, kernel_size=15, stride=1, padding=7)
        strength = torch.rand(shape, device=image.device, dtype=image.dtype) * self.scatter_strength
        out = image * gain + bias + strength * scatter
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        return torch.clamp(out, 0.0, 1.5)

    def forward(self, ap: torch.Tensor, lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return ap, lat
        ap = self._augment_one(ap)
        lat = self._augment_one(lat)
        if self.view_dropout > 0:
            drop_ap = torch.rand(ap.shape[0], 1, 1, 1, device=ap.device) < self.view_dropout * 0.25
            drop_lat = torch.rand(lat.shape[0], 1, 1, 1, device=lat.device) < self.view_dropout
            ap = torch.where(drop_ap, torch.zeros_like(ap), ap)
            lat = torch.where(drop_lat, torch.zeros_like(lat), lat)
        return ap, lat
