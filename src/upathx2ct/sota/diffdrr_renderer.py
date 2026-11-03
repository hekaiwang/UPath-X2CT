"""Reusable DiffDRR adapter for the frozen UPath-LIDC cone geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from upathx2ct.sota.geometry import (
    ConeBeamGeometry,
    lps_to_ras,
    project_points_lps,
    ras_to_lps,
)


@dataclass(frozen=True)
class DiffDRRDensityCalibration:
    """Frozen parameters reproducing DiffDRR's per-volume HU normalization."""

    soft_tissue_min_hu: float
    mapped_max_hu: float
    bone_attenuation_multiplier: float = 1.0

    @property
    def dynamic_range_hu(self) -> float:
        return self.mapped_max_hu - self.soft_tissue_min_hu

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def fit_diffdrr_density_calibration(
    volume_hu: np.ndarray,
    bone_attenuation_multiplier: float = 1.0,
) -> DiffDRRDensityCalibration:
    """Fit the normalization used by DiffDRR without changing the input."""

    volume = np.asarray(volume_hu, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Density calibration requires one finite 3D HU volume")
    multiplier = float(bone_attenuation_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Bone attenuation multiplier must be positive")
    soft_tissue = volume[(volume > -800.0) & (volume <= 350.0)]
    if not soft_tissue.size:
        raise ValueError("HU volume contains no DiffDRR soft-tissue calibration voxels")
    bone = volume[volume > 350.0]
    soft_min = float(soft_tissue.min())
    mapped_max = float(soft_tissue.max())
    if bone.size:
        mapped_max = max(mapped_max, float(bone.max()) * multiplier)
    calibration = DiffDRRDensityCalibration(
        soft_tissue_min_hu=soft_min,
        mapped_max_hu=mapped_max,
        bone_attenuation_multiplier=multiplier,
    )
    if calibration.dynamic_range_hu <= 0:
        raise ValueError("DiffDRR density calibration has zero dynamic range")
    return calibration


def apply_diffdrr_density_calibration(
    volume_hu: Any,
    calibration: DiffDRRDensityCalibration,
) -> Any:
    """Apply one frozen calibration to a torch HU tensor.

    DiffDRR normally refits its minimum and maximum for every volume. That
    behavior creates a detector-wide residual after a local counterfactual
    edit. Freezing the source CT calibration keeps the edit spatially local.
    """

    import torch

    volume = torch.as_tensor(volume_hu).to(torch.float32)
    if volume.ndim < 3 or not bool(torch.isfinite(volume).all()):
        raise ValueError("Expected one finite HU tensor")
    soft = (volume > -800.0) & (volume <= 350.0)
    bone = volume > 350.0
    mapped = torch.empty_like(volume)
    mapped[volume <= -800.0] = calibration.soft_tissue_min_hu
    mapped[soft] = volume[soft]
    mapped[bone] = volume[bone] * calibration.bone_attenuation_multiplier
    density = (
        (mapped - calibration.soft_tissue_min_hu)
        / calibration.dynamic_range_hu
    )
    return density.clamp_(0.0, 1.0)


def geometry_from_protocol(config: dict[str, Any]) -> ConeBeamGeometry:
    projection = config["projection"]
    nominal = projection["nominal_geometry"]
    return ConeBeamGeometry(
        source_detector_distance_mm=float(nominal["source_detector_distance_mm"]),
        source_object_distance_mm=float(nominal["source_object_distance_mm"]),
        detector_shape=tuple(int(value) for value in projection["detector_shape"]),
        detector_pixel_size_mm=tuple(
            float(value) for value in nominal["detector_pixel_size_uv_mm"]
        ),
        principal_point_px=tuple(
            float(value) for value in nominal["principal_point_uv_px"]
        ),
    )


def render_centered_biplanar(
    ct_path: str | Path,
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    renderer: str = "siddon",
    patch_size: int = 256,
    points_patient_lps_mm: np.ndarray | None = None,
    center_patient_lps_mm: np.ndarray | None = None,
    density_calibration: DiffDRRDensityCalibration | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render AP/left-lateral views using the exact D1 adapter."""

    import torch
    from diffdrr.data import read
    from diffdrr.drr import DRR
    from diffdrr.pose import convert

    geometry = geometry_from_protocol(config)
    device = torch.device(device_name)
    points = (
        np.empty((0, 3), dtype=np.float64)
        if points_patient_lps_mm is None
        else np.asarray(points_patient_lps_mm, dtype=np.float64).reshape(-1, 3)
    )
    center = (
        np.zeros(3, dtype=np.float64)
        if center_patient_lps_mm is None
        else np.asarray(center_patient_lps_mm, dtype=np.float64)
    )
    if center.shape != (3,):
        raise ValueError("Patient LPS center must have three coordinates")
    fiducials = None
    if len(points):
        fiducials = torch.as_tensor(
            lps_to_ras(points),
            dtype=torch.float32,
        ).unsqueeze(0)
    adapter = config["projection"]["diffdrr_adapter"]
    subject = read(
        Path(ct_path),
        orientation=adapter["subject_orientation"],
        fiducials=fiducials,
        center_volume=True,
    )
    if density_calibration is not None:
        subject.density.data = apply_diffdrr_density_calibration(
            subject.volume.data,
            density_calibration,
        )
    height, width = geometry.detector_shape
    du, dv = geometry.detector_pixel_size_mm
    drr = DRR(
        subject,
        sdd=geometry.source_detector_distance_mm,
        height=height,
        width=width,
        delx=du,
        dely=dv,
        reverse_x_axis=bool(adapter["reverse_x_axis"]),
        patch_size=patch_size,
        renderer=renderer,
    ).to(device)
    rotations = torch.tensor(
        [
            adapter["frontal_ap_rotation_zxy_rad"],
            adapter["left_lateral_rotation_zxy_rad"],
        ],
        dtype=torch.float32,
        device=device,
    )
    translations = torch.tensor(
        [adapter["translation_xyz_mm"]] * 2,
        dtype=torch.float32,
        device=device,
    )
    pose = convert(
        rotations,
        translations,
        parameterization=adapter["parameterization"],
        convention=adapter["convention"],
    )
    with torch.inference_mode():
        rendered = drr(pose).detach()
        projected = None
        centered_lps = None
        if subject.fiducials is not None:
            repeated = subject.fiducials.to(device).repeat(2, 1, 1)
            projected = (
                drr.perspective_projection(pose, repeated).detach().cpu().numpy()
            )
            centered_lps = ras_to_lps(
                subject.fiducials[0].detach().cpu().numpy()
            )
    images = rendered[:, 0].float().cpu().numpy()
    record: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "centered_fiducial_error_mm": 0.0,
        "max_projection_error_px": 0.0,
        "fiducial_uv_by_view": [[], []],
        "density_calibration": (
            density_calibration.to_dict()
            if density_calibration is not None
            else None
        ),
    }
    if projected is not None and centered_lps is not None:
        expected_centered_lps = points - center[None]
        record["centered_fiducial_error_mm"] = float(
            np.max(np.linalg.norm(centered_lps - expected_centered_lps, axis=1))
        )
        errors = []
        for view_index, view in enumerate(("frontal_ap", "left_lateral")):
            analytic, depth = project_points_lps(
                expected_centered_lps,
                geometry,
                view,
            )
            errors.extend(np.linalg.norm(projected[view_index] - analytic, axis=1))
            record["fiducial_uv_by_view"][view_index] = analytic.tolist()
            record.setdefault(
                "fiducial_depth_mm_by_view",
                [[], []],
            )[view_index] = depth.tolist()
        record["max_projection_error_px"] = float(max(errors, default=0.0))
    del drr, subject, rendered
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return images, record
