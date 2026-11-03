"""Canonical LPS grids and lesion-independent thoracic centering for D1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ThoracicCenterResult:
    center_xyz: tuple[float, float, float]
    method: str
    body_extent_xyz_voxels: tuple[int, int, int]
    lung_extent_xyz_voxels: tuple[int, int, int]


def stable_shard_index(key: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError("Number of shards must be positive")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def stable_balanced_shards(
    keys: Iterable[str],
    num_shards: int,
) -> list[list[str]]:
    if num_shards <= 0:
        raise ValueError("Number of shards must be positive")
    values = list(keys)
    if len(values) != len(set(values)):
        raise ValueError("Balanced sharding keys must be unique")
    ordered = sorted(
        values,
        key=lambda key: (hashlib.sha256(key.encode("utf-8")).digest(), key),
    )
    return [ordered[index::num_shards] for index in range(num_shards)]


def _largest_components(mask: np.ndarray, count: int = 1) -> np.ndarray:
    labels, number = ndimage.label(np.asarray(mask, dtype=bool))
    if number == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep_count = min(int(count), int(number))
    keep = np.argsort(sizes[1:])[-keep_count:] + 1
    return np.isin(labels, keep)


def body_mask_zyx(
    volume_hu_zyx: np.ndarray,
    threshold_hu: float = -650.0,
) -> np.ndarray:
    """Return a filled patient envelope without using lesion information."""

    volume = np.asarray(volume_hu_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D ZYX volume, got {volume.shape}")
    candidate = np.isfinite(volume) & (volume > threshold_hu)
    output = np.zeros_like(candidate, dtype=bool)
    structure = ndimage.generate_binary_structure(2, 1)
    for index, axial in enumerate(candidate):
        axial = ndimage.binary_closing(axial, structure=structure, iterations=2)
        axial = _largest_components(axial, count=1)
        axial = ndimage.binary_fill_holes(axial)
        output[index] = ndimage.binary_closing(
            axial,
            structure=structure,
            iterations=1,
        )
    return _largest_components(output, count=1)


def lung_mask_zyx(
    volume_hu_zyx: np.ndarray,
    body_zyx: np.ndarray,
    threshold_hu: float = -300.0,
) -> np.ndarray:
    """Return the dominant intrathoracic low-attenuation components."""

    volume = np.asarray(volume_hu_zyx)
    body = np.asarray(body_zyx, dtype=bool)
    if volume.shape != body.shape or volume.ndim != 3:
        raise ValueError("Lung-mask inputs must be same-shape 3D arrays")
    candidate = (volume < threshold_hu) & body
    candidate = ndimage.binary_opening(candidate, iterations=1)
    candidate = _largest_components(candidate, count=3)
    candidate = ndimage.binary_closing(candidate, iterations=1)
    return candidate


def _bbox_zyx(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if not len(coordinates):
        raise ValueError("Cannot compute a bounding box for an empty mask")
    return coordinates.min(axis=0), coordinates.max(axis=0)


def _dominant_profile_center_z(mask_zyx: np.ndarray) -> float | None:
    area = np.asarray(mask_zyx, dtype=bool).sum(axis=(1, 2)).astype(np.float64)
    if not area.size or float(area.max()) < 32.0:
        return None
    smooth = ndimage.gaussian_filter1d(area, sigma=1.5, mode="nearest")
    active = smooth >= max(32.0, 0.15 * float(smooth.max()))
    active = ndimage.binary_closing(active, structure=np.ones(5, dtype=bool))
    labels, number = ndimage.label(active)
    if number == 0:
        return None
    scores = [
        float(smooth[labels == label].sum()) for label in range(1, number + 1)
    ]
    indices = np.flatnonzero(labels == int(np.argmax(scores)) + 1)
    if indices.size < 4:
        return None
    return 0.5 * float(indices[0] + indices[-1])


def thoracic_center_xyz(
    volume_hu_zyx: np.ndarray,
    body_threshold_hu: float = -650.0,
    lung_threshold_hu: float = -300.0,
) -> tuple[ThoracicCenterResult, np.ndarray, np.ndarray]:
    """Estimate a lesion-independent thoracic center in continuous XYZ index."""

    body = body_mask_zyx(volume_hu_zyx, body_threshold_hu)
    if int(body.sum()) < 1000:
        raise ValueError("Body segmentation is too small for thoracic centering")
    lungs = lung_mask_zyx(volume_hu_zyx, body, lung_threshold_hu)
    body_start, body_stop = _bbox_zyx(body)
    body_center_zyx = 0.5 * (body_start + body_stop)
    body_extent_zyx = body_stop - body_start + 1

    method = "body_bbox_fallback"
    center_z = float(body_center_zyx[0])
    lung_extent_zyx = np.zeros(3, dtype=np.int64)
    if int(lungs.sum()) >= 500:
        lung_start, lung_stop = _bbox_zyx(lungs)
        lung_extent_zyx = lung_stop - lung_start + 1
        profile_center = _dominant_profile_center_z(lungs)
        center_z = (
            float(profile_center)
            if profile_center is not None
            else 0.5 * float(lung_start[0] + lung_stop[0])
        )
        method = (
            "dominant_lung_air_profile"
            if profile_center is not None
            else "lung_bbox"
        )

    center_xyz = (
        float(body_center_zyx[2]),
        float(body_center_zyx[1]),
        center_z,
    )
    result = ThoracicCenterResult(
        center_xyz=center_xyz,
        method=method,
        body_extent_xyz_voxels=tuple(int(value) for value in body_extent_zyx[::-1]),
        lung_extent_xyz_voxels=tuple(int(value) for value in lung_extent_zyx[::-1]),
    )
    return result, body.astype(np.uint8), lungs.astype(np.uint8)


def centered_grid_origin_lps(
    center_lps_mm: Iterable[float],
    shape_xyz: Iterable[int],
    spacing_xyz_mm: Iterable[float],
) -> np.ndarray:
    center = np.asarray(tuple(center_lps_mm), dtype=np.float64)
    shape = np.asarray(tuple(shape_xyz), dtype=np.float64)
    spacing = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    if center.shape != (3,) or shape.shape != (3,) or spacing.shape != (3,):
        raise ValueError("Center, shape and spacing must each have three values")
    if np.any(shape <= 0) or np.any(spacing <= 0) or not np.isfinite(center).all():
        raise ValueError("Canonical grid parameters must be finite and positive")
    return center - 0.5 * (shape - 1.0) * spacing


def lps_grid_to_nifti_ras_affine(
    origin_lps_mm: Iterable[float],
    spacing_xyz_mm: Iterable[float],
) -> np.ndarray:
    """Return an XYZ-index NIfTI affine for an axis-aligned DICOM-LPS grid."""

    origin = np.asarray(tuple(origin_lps_mm), dtype=np.float64)
    spacing = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    if origin.shape != (3,) or spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Origin and spacing must each have three valid values")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag((-spacing[0], -spacing[1], spacing[2]))
    affine[:3, 3] = (-origin[0], -origin[1], origin[2])
    return affine


def physical_lps_to_index_xyz(
    point_lps_mm: Iterable[float],
    origin_lps_mm: Iterable[float],
    spacing_xyz_mm: Iterable[float],
) -> np.ndarray:
    point = np.asarray(tuple(point_lps_mm), dtype=np.float64)
    origin = np.asarray(tuple(origin_lps_mm), dtype=np.float64)
    spacing = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    return (point - origin) / spacing


def index_xyz_to_physical_lps(
    index_xyz: Iterable[float],
    origin_lps_mm: Iterable[float],
    spacing_xyz_mm: Iterable[float],
) -> np.ndarray:
    index = np.asarray(tuple(index_xyz), dtype=np.float64)
    origin = np.asarray(tuple(origin_lps_mm), dtype=np.float64)
    spacing = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    return origin + index * spacing


def block_mean_2x_zyx(volume_zyx: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume_zyx)
    if volume.ndim != 3 or any(size % 2 for size in volume.shape):
        raise ValueError(f"Expected an even 3D volume, got {volume.shape}")
    z, y, x = volume.shape
    return volume.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(
        axis=(1, 3, 5),
        dtype=np.float32,
    )


def block_max_2x_zyx(mask_zyx: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_zyx)
    if mask.ndim != 3 or any(size % 2 for size in mask.shape):
        raise ValueError(f"Expected an even 3D mask, got {mask.shape}")
    z, y, x = mask.shape
    return mask.reshape(z // 2, 2, y // 2, 2, x // 2, 2).max(
        axis=(1, 3, 5)
    )


def block_mean_grid(
    full_origin_lps_mm: Iterable[float],
    full_spacing_xyz_mm: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    spacing = np.asarray(tuple(full_spacing_xyz_mm), dtype=np.float64)
    origin = np.asarray(tuple(full_origin_lps_mm), dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Full spacing must contain three positive values")
    return origin + 0.5 * spacing, 2.0 * spacing


def spherical_mask_zyx(
    shape_zyx: Iterable[int],
    center_xyz: Iterable[float],
    diameter_mm: float,
    spacing_xyz_mm: Iterable[float],
    minimum_radius_mm: float = 1.5,
) -> np.ndarray:
    shape = np.asarray(tuple(shape_zyx), dtype=np.int64)
    center = np.asarray(tuple(center_xyz), dtype=np.float64)
    spacing_xyz = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    if shape.shape != (3,) or center.shape != (3,) or spacing_xyz.shape != (3,):
        raise ValueError("Sphere geometry must have three axes")
    if diameter_mm <= 0 or np.any(spacing_xyz <= 0):
        raise ValueError("Sphere diameter and spacing must be positive")
    shape_xyz = shape[::-1]
    if np.any(center < -0.5) or np.any(center > shape_xyz - 0.5):
        raise ValueError(f"Sphere center outside grid: {center.tolist()}")
    radius = max(0.5 * float(diameter_mm), float(minimum_radius_mm))
    radius_voxels_xyz = radius / spacing_xyz
    lower_xyz = np.maximum(
        np.floor(center - radius_voxels_xyz).astype(np.int64),
        0,
    )
    upper_xyz = np.minimum(
        np.ceil(center + radius_voxels_xyz).astype(np.int64) + 1,
        shape_xyz,
    )
    x = np.arange(lower_xyz[0], upper_xyz[0], dtype=np.float64)
    y = np.arange(lower_xyz[1], upper_xyz[1], dtype=np.float64)
    z = np.arange(lower_xyz[2], upper_xyz[2], dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    distance_squared = (
        ((xx - center[0]) * spacing_xyz[0]) ** 2
        + ((yy - center[1]) * spacing_xyz[1]) ** 2
        + ((zz - center[2]) * spacing_xyz[2]) ** 2
    )
    local = distance_squared <= radius**2
    output = np.zeros(tuple(shape), dtype=np.uint8)
    if local.any():
        output[
            lower_xyz[2] : upper_xyz[2],
            lower_xyz[1] : upper_xyz[1],
            lower_xyz[0] : upper_xyz[0],
        ] = local.astype(np.uint8)
    else:
        nearest_xyz = np.clip(np.rint(center).astype(np.int64), 0, shape_xyz - 1)
        output[nearest_xyz[2], nearest_xyz[1], nearest_xyz[0]] = 1
    return output


def sphere_edge_margin_mm(
    shape_xyz: Iterable[int],
    center_xyz: Iterable[float],
    diameter_mm: float,
    spacing_xyz_mm: Iterable[float],
) -> float:
    shape = np.asarray(tuple(shape_xyz), dtype=np.float64)
    center = np.asarray(tuple(center_xyz), dtype=np.float64)
    spacing = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    lower = (center + 0.5) * spacing
    upper = (shape - 0.5 - center) * spacing
    return float(np.min(np.concatenate([lower, upper])) - 0.5 * diameter_mm)


def boundary_face_counts_zyx(mask_zyx: np.ndarray) -> dict[str, int]:
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("Boundary checks require a 3D mask")
    return {
        "inferior_z0": int(mask[0].sum()),
        "superior_z1": int(mask[-1].sum()),
        "posterior_y0": int(mask[:, 0, :].sum()),
        "anterior_y1": int(mask[:, -1, :].sum()),
        "right_x0": int(mask[:, :, 0].sum()),
        "left_x1": int(mask[:, :, -1].sum()),
    }
