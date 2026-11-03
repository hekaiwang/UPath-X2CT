"""Differentiable cone-beam projection on a regular DICOM-LPS CT grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from upathx2ct.sota.geometry import (
    ConeBeamGeometry,
    detector_rays_lps,
)
from upathx2ct.sota.projective_lifting import PhysicalGrid


@dataclass(frozen=True)
class ConeProjectorConfig:
    detector_shape: tuple[int, int] = (128, 128)
    ray_samples: int = 128
    hu_min: float = -1024.0
    hu_max: float = 1500.0


def _ray_box_sampling_grid(
    volume_grid: PhysicalGrid,
    geometry: ConeBeamGeometry,
    view: str,
    ray_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = geometry.detector_shape
    rows, columns = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    pixels = np.stack((columns, rows), axis=-1)
    origins, directions = detector_rays_lps(pixels, geometry, view)
    shape_xyz = np.asarray(volume_grid.shape_zyx[::-1], dtype=np.float64)
    origin_xyz = np.asarray(volume_grid.origin_lps_mm, dtype=np.float64)
    spacing_xyz = np.asarray(volume_grid.spacing_xyz_mm, dtype=np.float64)
    lower = origin_xyz - 0.5 * spacing_xyz
    upper = origin_xyz + (shape_xyz - 0.5) * spacing_xyz
    inverse = np.divide(
        1.0,
        directions,
        out=np.full_like(directions, np.inf),
        where=np.abs(directions) > 1e-12,
    )
    first = (lower - origins) * inverse
    second = (upper - origins) * inverse
    entry = np.max(np.minimum(first, second), axis=-1)
    exit = np.min(np.maximum(first, second), axis=-1)
    valid = (exit > entry) & (exit > 0.0)
    entry = np.maximum(entry, 0.0)
    fractions = (
        np.arange(ray_samples, dtype=np.float64) + 0.5
    ) / float(ray_samples)
    distance = entry[None] + fractions[:, None, None] * (
        exit - entry
    )[None]
    points = origins[None] + distance[..., None] * directions[None]
    index_xyz = (points - origin_xyz) / spacing_xyz
    normalized = 2.0 * (index_xyz + 0.5) / shape_xyz - 1.0
    step_length = np.where(valid, (exit - entry) / ray_samples, 0.0)
    normalized[:, ~valid] = 2.0
    return (
        normalized.astype(np.float32),
        step_length.astype(np.float32),
        valid.astype(np.bool_),
    )


class DifferentiableConeProjector(nn.Module):
    """Ray-sampled AP/LAT projector used consistently in B0 losses."""

    def __init__(
        self,
        volume_grid: PhysicalGrid,
        full_geometry: ConeBeamGeometry,
        config: ConeProjectorConfig = ConeProjectorConfig(),
    ) -> None:
        super().__init__()
        if config.ray_samples <= 0:
            raise ValueError("Cone projector requires positive ray samples")
        full_height, full_width = full_geometry.detector_shape
        height, width = config.detector_shape
        if full_height % height or full_width % width:
            raise ValueError(
                "Projector detector shape must evenly divide the frozen detector"
            )
        scale_u = full_width / width
        scale_v = full_height / height
        geometry = ConeBeamGeometry(
            source_detector_distance_mm=full_geometry.source_detector_distance_mm,
            source_object_distance_mm=full_geometry.source_object_distance_mm,
            detector_shape=config.detector_shape,
            detector_pixel_size_mm=(
                full_geometry.detector_pixel_size_mm[0] * scale_u,
                full_geometry.detector_pixel_size_mm[1] * scale_v,
            ),
            principal_point_px=(
                full_geometry.principal_point_px[0] / scale_u,
                full_geometry.principal_point_px[1] / scale_v,
            ),
        )
        grids = []
        steps = []
        visibility = []
        for view in ("frontal_ap", "left_lateral"):
            grid, step, valid = _ray_box_sampling_grid(
                volume_grid,
                geometry,
                view,
                config.ray_samples,
            )
            grids.append(grid)
            steps.append(step)
            visibility.append(valid)
        self.volume_grid = volume_grid
        self.geometry = geometry
        self.config = config
        self.register_buffer(
            "sampling_grid",
            torch.from_numpy(np.stack(grids)),
            persistent=False,
        )
        self.register_buffer(
            "step_length_mm",
            torch.from_numpy(np.stack(steps)),
            persistent=False,
        )
        self.register_buffer(
            "visibility",
            torch.from_numpy(np.stack(visibility)),
            persistent=False,
        )

    def forward(self, normalized_ct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if normalized_ct.ndim != 5 or normalized_ct.shape[1] != 1:
            raise ValueError(
                f"Expected [B,1,D,H,W] CT, got {tuple(normalized_ct.shape)}"
            )
        if tuple(normalized_ct.shape[-3:]) != self.volume_grid.shape_zyx:
            raise ValueError(
                f"CT shape {normalized_ct.shape[-3:]} != {self.volume_grid.shape_zyx}"
            )
        hu = (
            (normalized_ct + 1.0)
            * 0.5
            * (self.config.hu_max - self.config.hu_min)
            + self.config.hu_min
        )
        attenuation = torch.clamp(hu / 1000.0 + 1.0, 0.0, 3.5)
        projections = []
        for view_index in range(2):
            grid = self.sampling_grid[view_index].to(
                device=normalized_ct.device,
                dtype=normalized_ct.dtype,
            )
            grid = grid.unsqueeze(0).expand(normalized_ct.shape[0], -1, -1, -1, -1)
            samples = F.grid_sample(
                attenuation,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            step = self.step_length_mm[view_index].to(
                device=normalized_ct.device,
                dtype=normalized_ct.dtype,
            )
            projection = samples.sum(dim=2) * step[None, None]
            projections.append(projection)
        return projections[0], projections[1]
