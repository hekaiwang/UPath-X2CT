"""Explicit biplanar cone-beam geometry in DICOM LPS coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ViewName = Literal["frontal_ap", "left_lateral"]


@dataclass(frozen=True)
class ConeBeamGeometry:
    """Shared source-detector geometry for AP and left-lateral projections.

    Detector shape is ``(height, width)``. Pixel size and principal point are
    expressed in detector ``(u, v)`` order. The default principal point follows
    DiffDRR's even-detector convention: ``(width / 2, height / 2)``.
    """

    source_detector_distance_mm: float
    source_object_distance_mm: float
    detector_shape: tuple[int, int]
    detector_pixel_size_mm: tuple[float, float]
    principal_point_px: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        sdd = float(self.source_detector_distance_mm)
        sod = float(self.source_object_distance_mm)
        if not np.isfinite(sdd) or not np.isfinite(sod) or sdd <= sod or sod <= 0:
            raise ValueError("Require source_detector_distance > source_object_distance > 0")
        if len(self.detector_shape) != 2 or min(self.detector_shape) <= 0:
            raise ValueError(f"Invalid detector shape: {self.detector_shape}")
        if (
            len(self.detector_pixel_size_mm) != 2
            or not np.isfinite(self.detector_pixel_size_mm).all()
            or min(self.detector_pixel_size_mm) <= 0
        ):
            raise ValueError(
                f"Invalid detector pixel size: {self.detector_pixel_size_mm}"
            )
        if self.principal_point_px is None:
            height, width = self.detector_shape
            object.__setattr__(
                self,
                "principal_point_px",
                (float(width) / 2.0, float(height) / 2.0),
            )
        elif (
            len(self.principal_point_px) != 2
            or not np.isfinite(self.principal_point_px).all()
        ):
            raise ValueError(f"Invalid principal point: {self.principal_point_px}")

    @property
    def magnification_at_isocenter(self) -> float:
        return self.source_detector_distance_mm / self.source_object_distance_mm

    @property
    def isocenter_fov_mm(self) -> tuple[float, float]:
        """Return the detector-covered ``(u, v)`` field of view at isocenter."""

        height, width = self.detector_shape
        du, dv = self.detector_pixel_size_mm
        scale = self.source_object_distance_mm / self.source_detector_distance_mm
        return width * du * scale, height * dv * scale


@dataclass(frozen=True)
class ViewFrame:
    """One X-ray source and detector frame in DICOM LPS coordinates."""

    source_lps_mm: tuple[float, float, float]
    detector_center_lps_mm: tuple[float, float, float]
    detector_u_lps: tuple[float, float, float]
    detector_v_lps: tuple[float, float, float]
    beam_direction_lps: tuple[float, float, float]


def _vector(values: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def view_frame_lps(geometry: ConeBeamGeometry, view: ViewName) -> ViewFrame:
    """Return the frozen AP or left-lateral source-detector frame.

    DICOM LPS axes are +X left, +Y posterior and +Z superior.

    - AP: source anterior, detector posterior, image right is patient left.
    - Left lateral: source patient-right, detector patient-left. Detector
      ``u`` increases toward anterior to match the locked DiffDRR adapter.
    - In both views detector ``v`` increases toward inferior.
    """

    sod = float(geometry.source_object_distance_mm)
    detector_offset = float(
        geometry.source_detector_distance_mm - geometry.source_object_distance_mm
    )
    if view == "frontal_ap":
        return ViewFrame(
            source_lps_mm=(0.0, -sod, 0.0),
            detector_center_lps_mm=(0.0, detector_offset, 0.0),
            detector_u_lps=(1.0, 0.0, 0.0),
            detector_v_lps=(0.0, 0.0, -1.0),
            beam_direction_lps=(0.0, 1.0, 0.0),
        )
    if view == "left_lateral":
        return ViewFrame(
            source_lps_mm=(-sod, 0.0, 0.0),
            detector_center_lps_mm=(detector_offset, 0.0, 0.0),
            detector_u_lps=(0.0, -1.0, 0.0),
            detector_v_lps=(0.0, 0.0, -1.0),
            beam_direction_lps=(1.0, 0.0, 0.0),
        )
    raise ValueError(f"Unknown projection view: {view!r}")


def projection_matrix_lps(
    geometry: ConeBeamGeometry,
    view: ViewName,
) -> np.ndarray:
    """Return a 3x4 world-to-detector projection matrix."""

    frame = view_frame_lps(geometry, view)
    source = _vector(frame.source_lps_mm)
    rotation = np.stack(
        [
            _vector(frame.detector_u_lps),
            _vector(frame.detector_v_lps),
            _vector(frame.beam_direction_lps),
        ],
        axis=0,
    )
    translation = -rotation @ source
    extrinsic = np.concatenate([rotation, translation[:, None]], axis=1)
    du, dv = geometry.detector_pixel_size_mm
    cu, cv = geometry.principal_point_px
    intrinsic = np.asarray(
        [
            [geometry.source_detector_distance_mm / du, 0.0, cu],
            [0.0, geometry.source_detector_distance_mm / dv, cv],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsic @ extrinsic


def project_points_lps(
    points_lps_mm: np.ndarray,
    geometry: ConeBeamGeometry,
    view: ViewName,
) -> tuple[np.ndarray, np.ndarray]:
    """Project LPS points to detector ``(u, v)`` pixels and return ray depth."""

    points = np.asarray(points_lps_mm, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError(f"Expected finite [..., 3] points, got {points.shape}")
    frame = view_frame_lps(geometry, view)
    source = _vector(frame.source_lps_mm)
    relative = points - source
    depth = relative @ _vector(frame.beam_direction_lps)
    if np.any(depth <= 0):
        raise ValueError("All projected points must lie in front of the source")
    scale = geometry.source_detector_distance_mm / depth
    du, dv = geometry.detector_pixel_size_mm
    cu, cv = geometry.principal_point_px
    u = cu + (relative @ _vector(frame.detector_u_lps)) * scale / du
    v = cv + (relative @ _vector(frame.detector_v_lps)) * scale / dv
    return np.stack([u, v], axis=-1), depth


def detector_rays_lps(
    detector_uv_px: np.ndarray,
    geometry: ConeBeamGeometry,
    view: ViewName,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject detector pixels into normalized LPS rays."""

    pixels = np.asarray(detector_uv_px, dtype=np.float64)
    if pixels.shape[-1:] != (2,) or not np.isfinite(pixels).all():
        raise ValueError(f"Expected finite [..., 2] pixels, got {pixels.shape}")
    frame = view_frame_lps(geometry, view)
    source = _vector(frame.source_lps_mm)
    detector_center = _vector(frame.detector_center_lps_mm)
    u_axis = _vector(frame.detector_u_lps)
    v_axis = _vector(frame.detector_v_lps)
    du, dv = geometry.detector_pixel_size_mm
    cu, cv = geometry.principal_point_px
    target = (
        detector_center
        + (pixels[..., 0:1] - cu) * du * u_axis
        + (pixels[..., 1:2] - cv) * dv * v_axis
    )
    directions = target - source
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(source, directions.shape).copy()
    return origins, directions


def triangulate_biplanar_points_lps(
    frontal_uv_px: np.ndarray,
    lateral_uv_px: np.ndarray,
    geometry: ConeBeamGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate matched AP/LAT pixels using closest points between rays."""

    frontal = np.asarray(frontal_uv_px, dtype=np.float64)
    lateral = np.asarray(lateral_uv_px, dtype=np.float64)
    if frontal.shape != lateral.shape or frontal.shape[-1:] != (2,):
        raise ValueError("Frontal and lateral pixels must have matching [..., 2] shape")
    flat_frontal = frontal.reshape(-1, 2)
    flat_lateral = lateral.reshape(-1, 2)
    origin_a, direction_a = detector_rays_lps(
        flat_frontal,
        geometry,
        "frontal_ap",
    )
    origin_b, direction_b = detector_rays_lps(
        flat_lateral,
        geometry,
        "left_lateral",
    )
    points = []
    separations = []
    for oa, da, ob, db in zip(origin_a, direction_a, origin_b, direction_b):
        system = np.stack([da, -db], axis=1)
        parameters, *_ = np.linalg.lstsq(system, ob - oa, rcond=None)
        point_a = oa + parameters[0] * da
        point_b = ob + parameters[1] * db
        points.append(0.5 * (point_a + point_b))
        separations.append(float(np.linalg.norm(point_a - point_b)))
    output_shape = frontal.shape[:-1]
    return (
        np.asarray(points).reshape(*output_shape, 3),
        np.asarray(separations).reshape(output_shape),
    )


def lps_to_ras(points_lps: np.ndarray) -> np.ndarray:
    """Convert point coordinates from DICOM LPS to NIfTI/TorchIO RAS."""

    points = np.asarray(points_lps, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError(f"Expected [..., 3] points, got {points.shape}")
    return points * np.asarray([-1.0, -1.0, 1.0])


def ras_to_lps(points_ras: np.ndarray) -> np.ndarray:
    """Convert point coordinates from NIfTI/TorchIO RAS to DICOM LPS."""

    return lps_to_ras(points_ras)
