"""Cone-beam feature lifting on the frozen UPath-LIDC physical grid."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from upathx2ct.models.lifter.geometry_voxel_lifter import FourierPositionEncoding3D


def _groups(channels: int) -> int:
    for groups in range(min(16, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class PhysicalGrid:
    """Regular voxel-center grid described in DICOM-LPS millimetres."""

    shape_zyx: tuple[int, int, int]
    origin_lps_mm: tuple[float, float, float]
    spacing_xyz_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.shape_zyx) != 3 or min(self.shape_zyx) <= 0:
            raise ValueError(f"Invalid grid shape: {self.shape_zyx}")
        if len(self.origin_lps_mm) != 3:
            raise ValueError("Grid origin must contain three LPS coordinates")
        if len(self.spacing_xyz_mm) != 3 or min(self.spacing_xyz_mm) <= 0:
            raise ValueError(f"Invalid grid spacing: {self.spacing_xyz_mm}")

    def block_centers(self, factor: int) -> "PhysicalGrid":
        """Return exact non-overlapping block centers for an integer reduction."""

        if factor <= 0:
            raise ValueError("Block factor must be positive")
        if any(size % factor for size in self.shape_zyx):
            raise ValueError(
                f"Grid shape {self.shape_zyx} is not divisible by factor {factor}"
            )
        offset = 0.5 * (factor - 1)
        origin = tuple(
            float(value + offset * spacing)
            for value, spacing in zip(self.origin_lps_mm, self.spacing_xyz_mm)
        )
        spacing = tuple(float(factor * value) for value in self.spacing_xyz_mm)
        shape = tuple(int(size // factor) for size in self.shape_zyx)
        return PhysicalGrid(shape, origin, spacing)


def make_lps_grid_zyx(
    grid: PhysicalGrid,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return voxel-center coordinates shaped ``[D, H, W, xyz]``."""

    depth, height, width = grid.shape_zyx
    origin_x, origin_y, origin_z = grid.origin_lps_mm
    spacing_x, spacing_y, spacing_z = grid.spacing_xyz_mm
    x = origin_x + torch.arange(width, device=device, dtype=dtype) * spacing_x
    y = origin_y + torch.arange(height, device=device, dtype=dtype) * spacing_y
    z = origin_z + torch.arange(depth, device=device, dtype=dtype) * spacing_z
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)


def project_lps_grid(
    points_lps_zyx: torch.Tensor,
    projection_matrices: torch.Tensor,
    detector_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project an LPS grid to normalized detector coordinates.

    Projection matrices map homogeneous LPS millimetres to detector ``(u, v)``
    pixels. The returned sampling grid uses ``align_corners=False`` and the
    detector-edge convention, so the even detector principal point ``W/2,H/2``
    maps exactly to normalized zero.
    """

    if points_lps_zyx.ndim not in (4, 5) or points_lps_zyx.shape[-1] != 3:
        raise ValueError(
            "Points must have shape [D,H,W,3] or [B,D,H,W,3], "
            f"got {tuple(points_lps_zyx.shape)}"
        )
    if projection_matrices.ndim not in (3, 4) or projection_matrices.shape[-2:] != (
        3,
        4,
    ):
        raise ValueError(
            "Projection matrices must have shape [V,3,4] or [B,V,3,4], "
            f"got {tuple(projection_matrices.shape)}"
        )
    detector_height, detector_width = detector_shape
    if detector_height <= 0 or detector_width <= 0:
        raise ValueError(f"Invalid detector shape: {detector_shape}")

    points = (
        points_lps_zyx.unsqueeze(0)
        if points_lps_zyx.ndim == 4
        else points_lps_zyx
    )
    matrices = (
        projection_matrices.unsqueeze(0)
        if projection_matrices.ndim == 3
        else projection_matrices
    )
    batch = max(points.shape[0], matrices.shape[0])
    if points.shape[0] not in (1, batch) or matrices.shape[0] not in (1, batch):
        raise ValueError(
            f"Incompatible projection batches: points={points.shape[0]}, "
            f"matrices={matrices.shape[0]}"
        )
    points = points.expand(batch, *points.shape[1:])
    matrices = matrices.expand(batch, *matrices.shape[1:])
    spatial_shape = points.shape[1:4]
    flat = points.reshape(batch, -1, 3)
    homogeneous = torch.cat((flat, torch.ones_like(flat[..., :1])), dim=-1)
    camera = torch.einsum("bvij,bnj->bvni", matrices, homogeneous)
    depth = camera[..., 2]
    safe_depth = torch.where(
        depth.abs() >= torch.finfo(depth.dtype).eps,
        depth,
        torch.ones_like(depth),
    )
    uv = camera[..., :2] / safe_depth.unsqueeze(-1)
    normalized = torch.stack(
        (
            2.0 * uv[..., 0] / float(detector_width) - 1.0,
            2.0 * uv[..., 1] / float(detector_height) - 1.0,
        ),
        dim=-1,
    )
    visible = (
        (depth > 0.0)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= float(detector_width))
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= float(detector_height))
    )
    views = matrices.shape[1]
    return (
        normalized.reshape(batch, views, *spatial_shape, 2),
        depth.reshape(batch, views, *spatial_shape),
        visible.reshape(batch, views, *spatial_shape),
    )


def sample_feature_volume(
    feature_map: torch.Tensor,
    normalized_grid_zyx: torch.Tensor,
    visibility_zyx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bilinearly sample one 2D feature map at every projected 3D voxel."""

    if feature_map.ndim != 4:
        raise ValueError(f"Expected BCHW feature map, got {feature_map.shape}")
    if normalized_grid_zyx.ndim != 5 or normalized_grid_zyx.shape[-1] != 2:
        raise ValueError(
            f"Expected [B,D,H,W,2] sampling grid, got {normalized_grid_zyx.shape}"
        )
    if feature_map.shape[0] != normalized_grid_zyx.shape[0]:
        raise ValueError("Feature and sampling-grid batches differ")
    batch, depth, height, width, _ = normalized_grid_zyx.shape
    flat_grid = normalized_grid_zyx.reshape(batch, depth * height * width, 1, 2)
    sampled = F.grid_sample(
        feature_map,
        flat_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.squeeze(-1).reshape(
        batch,
        feature_map.shape[1],
        depth,
        height,
        width,
    )
    if visibility_zyx is not None:
        if visibility_zyx.shape != (batch, depth, height, width):
            raise ValueError(
                f"Visibility shape {visibility_zyx.shape} does not match the grid"
            )
        sampled = sampled * visibility_zyx.unsqueeze(1).to(sampled.dtype)
    return sampled


class MaskedAdaptiveViewFusion3D(nn.Module):
    """Fuse projected views while excluding voxels outside a detector."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv3d(channels * 3 + 1, channels, 1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, 1, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(
        self,
        views: torch.Tensor,
        visibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if views.ndim != 6:
            raise ValueError(f"Expected [B,V,C,D,H,W] views, got {views.shape}")
        if visibility.shape != (
            views.shape[0],
            views.shape[1],
            1,
            *views.shape[-3:],
        ):
            raise ValueError("View visibility does not match lifted features")
        visible = visibility.to(views.dtype)
        count = visible.sum(dim=1).clamp_min(1.0)
        mean = (views * visible).sum(dim=1) / count
        variance = (
            (views - mean.unsqueeze(1)).square() * visible
        ).sum(dim=1) / count
        logits = torch.stack(
            [
                self.score(
                    torch.cat(
                        (view, mean, variance, view_visible),
                        dim=1,
                    )
                )
                for view, view_visible in zip(
                    views.unbind(dim=1),
                    visible.unbind(dim=1),
                )
            ],
            dim=1,
        )
        logits = logits.masked_fill(visible <= 0.0, -1e4)
        weights = torch.softmax(logits, dim=1) * visible
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused = (views * weights).sum(dim=1)
        return fused, variance, weights


class ConeBeamFeatureLifter(nn.Module):
    """Lift shared AP/LAT FPN maps through exact cone-beam camera matrices."""

    def __init__(
        self,
        feature_channels: Mapping[str, int],
        projection_matrices: torch.Tensor,
        parent_grid: PhysicalGrid,
        detector_shape: tuple[int, int] = (512, 512),
        downsample_factor: int = 4,
        levels: Sequence[str] = ("p3", "p4", "p5"),
        sample_channels: int = 16,
        output_channels: int = 96,
        position_frequencies: int = 4,
        depth_scale_mm: float = 1800.0,
    ) -> None:
        super().__init__()
        self.levels = tuple(levels)
        if len(self.levels) == 0:
            raise ValueError("At least one FPN level is required")
        missing = [level for level in self.levels if level not in feature_channels]
        if missing:
            raise ValueError(f"Missing feature-channel declarations: {missing}")
        if projection_matrices.shape != (2, 3, 4):
            raise ValueError("The formal biplanar lifter requires two 3x4 matrices")
        if depth_scale_mm <= 0:
            raise ValueError("Depth scale must be positive")

        self.detector_shape = tuple(int(value) for value in detector_shape)
        self.parent_grid = parent_grid
        self.latent_grid = parent_grid.block_centers(downsample_factor)
        self.depth_scale_mm = float(depth_scale_mm)
        self.feature_projection = nn.ModuleDict(
            {
                level: nn.Conv2d(
                    feature_channels[level],
                    sample_channels,
                    kernel_size=1,
                )
                for level in self.levels
            }
        )
        self.view_fusion = nn.ModuleDict(
            {
                level: MaskedAdaptiveViewFusion3D(sample_channels)
                for level in self.levels
            }
        )
        self.position = FourierPositionEncoding3D(position_frequencies)
        per_level_channels = sample_channels * 6 + 4
        input_channels = per_level_channels * len(self.levels) + self.position.out_channels
        self.fusion = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(),
            nn.Conv3d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(),
        )

        points = make_lps_grid_zyx(self.latent_grid, dtype=torch.float32)
        sample_grid, depth, visibility = project_lps_grid(
            points,
            projection_matrices.to(dtype=torch.float32),
            self.detector_shape,
        )
        extent_xyz = torch.tensor(
            [
                max(
                    abs(self.latent_grid.origin_lps_mm[axis]),
                    abs(
                        self.latent_grid.origin_lps_mm[axis]
                        + (self.latent_grid.shape_zyx[2 - axis] - 1)
                        * self.latent_grid.spacing_xyz_mm[axis]
                    ),
                    1.0,
                )
                for axis in range(3)
            ],
            dtype=torch.float32,
        )
        coordinates = (points / extent_xyz).permute(3, 0, 1, 2).unsqueeze(0)
        self.register_buffer(
            "default_projection_matrices",
            projection_matrices.to(dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("default_sample_grid", sample_grid, persistent=False)
        self.register_buffer("default_depth", depth, persistent=False)
        self.register_buffer("default_visibility", visibility, persistent=False)
        self.register_buffer("normalized_coordinates", coordinates, persistent=False)

    def _geometry(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        projection_matrices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if projection_matrices is None:
            return (
                self.default_sample_grid.to(device=device, dtype=dtype).expand(
                    batch, -1, -1, -1, -1, -1
                ),
                self.default_depth.to(device=device, dtype=dtype).expand(
                    batch, -1, -1, -1, -1
                ),
                self.default_visibility.to(device=device).expand(
                    batch, -1, -1, -1, -1
                ),
            )
        matrices = projection_matrices.to(device=device, dtype=dtype)
        points = make_lps_grid_zyx(
            self.latent_grid,
            device=device,
            dtype=dtype,
        )
        sample_grid, depth, visibility = project_lps_grid(
            points,
            matrices,
            self.detector_shape,
        )
        if sample_grid.shape[0] not in (1, batch):
            raise ValueError(
                f"Projection batch {sample_grid.shape[0]} does not match features {batch}"
            )
        return (
            sample_grid.expand(batch, -1, -1, -1, -1, -1),
            depth.expand(batch, -1, -1, -1, -1),
            visibility.expand(batch, -1, -1, -1, -1),
        )

    def forward(
        self,
        ap_features: Mapping[str, torch.Tensor],
        lat_features: Mapping[str, torch.Tensor],
        projection_matrices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first = ap_features[self.levels[0]]
        batch = first.shape[0]
        sample_grid, depth, visibility = self._geometry(
            batch,
            first.device,
            first.dtype,
            projection_matrices,
        )
        lifted: list[torch.Tensor] = []
        for level in self.levels:
            ap = self.feature_projection[level](ap_features[level])
            lat = self.feature_projection[level](lat_features[level])
            ap_volume = sample_feature_volume(
                ap,
                sample_grid[:, 0],
                visibility[:, 0],
            )
            lat_volume = sample_feature_volume(
                lat,
                sample_grid[:, 1],
                visibility[:, 1],
            )
            views = torch.stack((ap_volume, lat_volume), dim=1)
            view_visibility = visibility.unsqueeze(2).to(first.dtype)
            fused, variance, _ = self.view_fusion[level](
                views,
                view_visibility,
            )
            product = ap_volume * lat_volume
            difference = (ap_volume - lat_volume).abs()
            depth_features = (
                depth / self.depth_scale_mm * visibility.to(depth.dtype)
            )
            lifted.extend(
                (
                    ap_volume,
                    lat_volume,
                    fused,
                    variance,
                    product,
                    difference,
                    visibility.to(first.dtype),
                    depth_features,
                )
            )
        coordinates = self.normalized_coordinates.to(
            device=first.device,
            dtype=first.dtype,
        ).expand(batch, -1, -1, -1, -1)
        lifted.append(self.position(coordinates))
        return self.fusion(torch.cat(lifted, dim=1))
