#!/usr/bin/env python3
"""Build and audit the deterministic 16-case UPath-LIDC-v1 D1 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import nibabel as nib
import numpy as np
from skimage import exposure

from upathx2ct.sota.canonical import (
    block_max_2x_zyx,
    block_mean_2x_zyx,
    block_mean_grid,
    body_mask_zyx,
    boundary_face_counts_zyx,
    centered_grid_origin_lps,
    index_xyz_to_physical_lps,
    lps_grid_to_nifti_ras_affine,
    lung_mask_zyx,
    physical_lps_to_index_xyz,
    sphere_edge_margin_mm,
    spherical_mask_zyx,
    stable_balanced_shards,
    thoracic_center_xyz,
)
from upathx2ct.sota.dicom import regularize_axial_stack
from upathx2ct.sota.geometry import (
    ConeBeamGeometry,
    lps_to_ras,
    project_points_lps,
    projection_matrix_lps,
    ras_to_lps,
)
from upathx2ct.sota.protocol import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sota/upath_lidc_v1.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/upath_lidc_v1.csv"),
    )
    parser.add_argument(
        "--nodule-table",
        type=Path,
        default=Path(
            "outputs/upath_lidc_v1_observability_preflight_20260724/per_nodule.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/upath_lidc_v1_d1_pilot_20260724"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renderer", choices=("siddon", "trilinear"), default="siddon")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional smoke-test cap; zero means all cases in the selected mode.",
    )
    parser.add_argument("--full-release", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--repair-nonuniform",
        action="store_true",
        help=(
            "When resuming, rebuild only legacy cases whose source slice "
            "spacing residual exceeds 0.1 mm."
        ),
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Rebuild the gate summary from an existing completed case manifest.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def case_id(patient_id: str, series_uid: str) -> str:
    patient = patient_id.lower().replace("lidc-idri-", "")
    suffix = hashlib.sha256(series_uid.encode("utf-8")).hexdigest()[:10]
    return f"lidc_{patient}_{suffix}"


def _median_candidate(
    rows: list[dict[str, str]],
    used_patients: set[str],
) -> dict[str, str]:
    candidates = [row for row in rows if row["patient_id"] not in used_patients]
    if not candidates:
        raise RuntimeError("No unused candidate satisfies the D1 selection stratum")
    diameters = np.asarray([float(row["diameter_mm"]) for row in candidates])
    spacings = np.asarray([float(row["spacing_z_mm"]) for row in candidates])
    center_diameter = float(np.median(diameters))
    center_spacing = float(np.median(spacings))
    scale_diameter = max(float(np.std(diameters)), 1e-6)
    scale_spacing = max(float(np.std(spacings)), 1e-6)
    return min(
        candidates,
        key=lambda row: (
            abs(float(row["diameter_mm"]) - center_diameter) / scale_diameter
            + abs(float(row["spacing_z_mm"]) - center_spacing) / scale_spacing,
            row["patient_id"],
            row["nodule_id"],
        ),
    )


def select_cases(
    manifest_rows: list[dict[str, str]],
    nodule_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    manifest = {row["series_uid"]: row for row in manifest_rows}
    selected: list[dict[str, Any]] = []
    used_patients: set[str] = set()
    used_series: set[str] = set()

    def add_positive(
        reason: str,
        candidates: list[dict[str, str]],
    ) -> None:
        row = _median_candidate(candidates, used_patients)
        series_uid = row["series_uid"]
        if series_uid in used_series:
            raise RuntimeError(f"Duplicate selected series: {series_uid}")
        selected.append(
            {
                "manifest": manifest[series_uid],
                "selection_reason": reason,
                "selected_nodule_id": row["nodule_id"],
                "selection_evidence_tier": row["evidence_tier"],
            }
        )
        used_patients.add(row["patient_id"])
        used_series.add(series_uid)

    tier_to_size = {"low": "<6", "intermediate": "6-10", "high": ">10"}
    for split in ("train", "val", "test"):
        for tier in ("low", "intermediate", "high"):
            exact = [
                row
                for row in nodule_rows
                if row["split"] == split
                and row["evidence_tier"] == tier
                and row["size_group"] == tier_to_size[tier]
            ]
            fallback = [
                row
                for row in nodule_rows
                if row["split"] == split and row["evidence_tier"] == tier
            ]
            add_positive(
                f"{split}_{tier}_{tier_to_size[tier]}",
                exact or fallback,
            )

    unused_train = [
        row
        for row in nodule_rows
        if row["split"] == "train" and row["patient_id"] not in used_patients
    ]
    add_positive(
        "train_min_native_z_spacing",
        sorted(
            unused_train,
            key=lambda row: (
                float(row["spacing_z_mm"]),
                row["patient_id"],
                row["nodule_id"],
            ),
        )[: max(1, min(12, len(unused_train)))],
    )
    unused_train = [
        row
        for row in nodule_rows
        if row["split"] == "train" and row["patient_id"] not in used_patients
    ]
    add_positive(
        "train_max_native_z_spacing",
        sorted(
            unused_train,
            key=lambda row: (
                -float(row["spacing_z_mm"]),
                row["patient_id"],
                row["nodule_id"],
            ),
        )[: max(1, min(12, len(unused_train)))],
    )
    unused_lateral = [
        row
        for row in nodule_rows
        if row["split"] == "train"
        and row["best_view"] == "lateral"
        and row["patient_id"] not in used_patients
    ]
    add_positive(
        "train_lateral_best_view",
        sorted(
            unused_lateral,
            key=lambda row: (
                -float(row["lateral_cnr_mean"]),
                row["patient_id"],
                row["nodule_id"],
            ),
        )[: max(1, min(24, len(unused_lateral)))],
    )

    def add_negative(reason: str, split: str, quantile: float) -> None:
        candidates = [
            row
            for row in manifest_rows
            if row["split"] == split
            and row["has_nodule"] == "0"
            and row["patient_id"] not in used_patients
        ]
        candidates.sort(
            key=lambda row: (float(row["series_size_mb"]), row["patient_id"])
        )
        if not candidates:
            raise RuntimeError(f"No unused negative D1 candidate for {split}")
        index = int(round(quantile * (len(candidates) - 1)))
        row = candidates[index]
        selected.append(
            {
                "manifest": row,
                "selection_reason": reason,
                "selected_nodule_id": "",
                "selection_evidence_tier": "negative",
            }
        )
        used_patients.add(row["patient_id"])
        used_series.add(row["series_uid"])

    add_negative("train_negative_small_series", "train", 0.05)
    add_negative("train_negative_large_series", "train", 0.95)
    add_negative("val_negative_median_series", "val", 0.50)
    add_negative("test_negative_median_series", "test", 0.50)
    if len(selected) != 16:
        raise RuntimeError(f"D1 selection must contain 16 cases, got {len(selected)}")
    return selected


def select_release_shard(
    manifest_rows: list[dict[str, str]],
    nodules_by_series: dict[str, list[dict[str, str]]],
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"Invalid release shard {shard_index} of {num_shards}"
        )
    tier_rank = {"low": 0, "intermediate": 1, "high": 2}
    assigned_series = set(
        stable_balanced_shards(
            [row["series_uid"] for row in manifest_rows],
            num_shards,
        )[shard_index]
    )
    selected = []
    for row in sorted(
        manifest_rows,
        key=lambda item: (item["patient_id"], item["series_uid"]),
    ):
        if row["series_uid"] not in assigned_series:
            continue
        nodules = nodules_by_series.get(row["series_uid"], [])
        representative = max(
            nodules,
            key=lambda item: (
                tier_rank[item["evidence_tier"]],
                float(item["best_view_cnr_mean"]),
                float(item["diameter_mm"]),
            ),
            default=None,
        )
        selected.append(
            {
                "manifest": row,
                "selection_reason": (
                    f"full_release_shard_{shard_index:02d}_of_{num_shards:02d}"
                ),
                "selected_nodule_id": (
                    representative["nodule_id"] if representative else ""
                ),
                "selection_evidence_tier": (
                    representative["evidence_tier"]
                    if representative
                    else "negative"
                ),
            }
        )
    return selected


def read_dicom_series(series_dir: Path, series_uid: str):
    import SimpleITK as sitk

    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir), series_uid))
    if len(files) < 2:
        raise ValueError(f"Only {len(files)} slices for {series_uid}")
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    reader.SetFileNames(files)
    image = reader.Execute()
    positions = []
    orientation = None
    for index in range(len(files)):
        if reader.HasMetaDataKey(index, "0020|0032"):
            positions.append(
                [
                    float(value)
                    for value in reader.GetMetaData(index, "0020|0032").split("\\")
                ]
            )
        if orientation is None and reader.HasMetaDataKey(index, "0020|0037"):
            orientation = [
                float(value)
                for value in reader.GetMetaData(index, "0020|0037").split("\\")
            ]
    residual = 0.0
    median_spacing = float(image.GetSpacing()[2])
    if len(positions) >= 2 and orientation is not None:
        row = np.asarray(orientation[:3], dtype=np.float64)
        column = np.asarray(orientation[3:], dtype=np.float64)
        normal = np.cross(row, column)
        coordinates = np.sort(np.asarray(positions) @ normal)
        differences = np.diff(coordinates)
        median_spacing = float(np.median(np.abs(differences)))
        residual = float(np.max(np.abs(np.abs(differences) - median_spacing)))
    regularization: dict[str, Any] = {
        "applied": False,
        "method": "native GDCM regular stack",
        "source_slices": len(files),
        "output_slices": int(image.GetSize()[2]),
        "nominal_spacing_mm": median_spacing,
        "source_max_spacing_residual_mm": residual,
        "retained_lattice_max_residual_mm": residual,
        "discarded_off_lattice_slices": 0,
        "discarded_disconnected_slices": 0,
        "duplicate_slices_resolved": 0,
        "interpolated_slices": 0,
        "post_regularization_spacing_residual_mm": residual,
    }
    if residual > 0.1:
        if orientation is None or len(positions) != len(files):
            raise ValueError(
                "Non-uniform DICOM stack lacks complete position/orientation "
                "metadata"
            )
        corrected, origin, direction, regularization = regularize_axial_stack(
            sitk.GetArrayFromImage(image),
            np.asarray(positions, dtype=np.float64),
            np.asarray(orientation, dtype=np.float64),
            lattice_tolerance_mm=0.1,
            max_missing_slices=3,
        )
        repaired = sitk.GetImageFromArray(corrected)
        repaired.SetSpacing(
            (
                float(image.GetSpacing()[0]),
                float(image.GetSpacing()[1]),
                float(regularization["output_spacing_mm"]),
            )
        )
        repaired.SetOrigin(tuple(float(value) for value in origin))
        repaired.SetDirection(
            tuple(float(value) for value in direction.reshape(-1))
        )
        image = repaired
        median_spacing = float(regularization["output_spacing_mm"])
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    image = orienter.Execute(image)
    return image, files, median_spacing, regularization


def upgrade_record(record: dict[str, Any]) -> dict[str, Any]:
    source_residual = float(record["slice_spacing_residual_mm"])
    record.setdefault("source_slice_spacing_residual_mm", source_residual)
    record.setdefault("slice_spacing_regularized", 0)
    record.setdefault("slice_spacing_discarded_off_lattice", 0)
    record.setdefault("slice_spacing_off_lattice_resampled", 0)
    record.setdefault("slice_spacing_discarded_disconnected", 0)
    record.setdefault("slice_spacing_duplicates_resolved", 0)
    record.setdefault("slice_spacing_interpolated", 0)
    record.setdefault(
        "slice_spacing_retained_lattice_residual_mm",
        source_residual,
    )
    return record


def image_edge_bounds_lps(image) -> tuple[np.ndarray, np.ndarray]:
    size = np.asarray(image.GetSize(), dtype=np.float64)
    points = []
    for x in (-0.5, size[0] - 0.5):
        for y in (-0.5, size[1] - 0.5):
            for z in (-0.5, size[2] - 0.5):
                points.append(
                    image.TransformContinuousIndexToPhysicalPoint((x, y, z))
                )
    points = np.asarray(points, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def resample_axis_aligned(image, spacing_mm: float):
    import SimpleITK as sitk

    lower, upper = image_edge_bounds_lps(image)
    spacing = np.repeat(float(spacing_mm), 3)
    size = np.ceil((upper - lower) / spacing).astype(np.int64)
    origin = lower + 0.5 * spacing
    return sitk.Resample(
        image,
        [int(value) for value in size],
        sitk.Transform(),
        sitk.sitkLinear,
        [float(value) for value in origin],
        [float(value) for value in spacing],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        -1024.0,
        sitk.sitkFloat32,
    )


def resample_canonical(
    image,
    center_lps_mm: np.ndarray,
    shape_xyz: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
):
    import SimpleITK as sitk

    origin = centered_grid_origin_lps(center_lps_mm, shape_xyz, spacing_xyz_mm)
    output = sitk.Resample(
        image,
        list(shape_xyz),
        sitk.Transform(),
        sitk.sitkLinear,
        [float(value) for value in origin],
        list(spacing_xyz_mm),
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        -1024.0,
        sitk.sitkFloat32,
    )
    return output, origin


def save_nifti_zyx(
    path: Path,
    array_zyx: np.ndarray,
    affine_ras: np.ndarray,
    dtype,
) -> None:
    array_xyz = np.ascontiguousarray(np.transpose(array_zyx, (2, 1, 0)), dtype=dtype)
    image = nib.Nifti1Image(array_xyz, affine_ras)
    image.header.set_xyzt_units("mm")
    image.set_data_dtype(dtype)
    nib.save(image, str(path))


def load_geometry(config: dict[str, Any]) -> ConeBeamGeometry:
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


def render_drrs(
    ct_path: Path,
    points_lps_mm: np.ndarray,
    center_lps_mm: np.ndarray,
    config: dict[str, Any],
    geometry: ConeBeamGeometry,
    device_name: str,
    renderer: str,
    patch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from diffdrr.data import read
    from diffdrr.drr import DRR
    from diffdrr.pose import convert

    device = torch.device(device_name)
    fiducials = None
    if len(points_lps_mm):
        points_ras = lps_to_ras(points_lps_mm)
        fiducials = torch.as_tensor(points_ras, dtype=torch.float32).unsqueeze(0)
    adapter = config["projection"]["diffdrr_adapter"]
    subject = read(
        ct_path,
        orientation=adapter["subject_orientation"],
        fiducials=fiducials,
        center_volume=True,
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
    with torch.no_grad():
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
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "centered_fiducial_error_mm": 0.0,
        "max_projection_error_px": 0.0,
        "fiducial_uv_by_view": [[], []],
    }
    if projected is not None and centered_lps is not None:
        expected_centered_lps = points_lps_mm - center_lps_mm[None]
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
            record.setdefault("fiducial_depth_mm_by_view", [[], []])[view_index] = (
                depth.tolist()
            )
        record["max_projection_error_px"] = float(max(errors, default=0.0))
    del drr, subject, rendered
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return images, record


def display_drr(image: np.ndarray) -> np.ndarray:
    array = np.log1p(np.maximum(np.asarray(image, dtype=np.float32), 0.0))
    positive = array[array > 0]
    if not positive.size:
        return np.zeros_like(array)
    low, high = np.percentile(positive, (0.5, 99.7))
    normalized = np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)
    return exposure.equalize_adapthist(normalized, clip_limit=0.012)


def add_nodule_marker(axis, u: float, v: float, radius_px: float) -> None:
    from matplotlib.patches import Circle

    axis.add_patch(
        Circle(
            (u, v),
            radius=max(2.0, radius_px),
            fill=False,
            edgecolor="#FF1744",
            linewidth=1.5,
        )
    )
    axis.scatter([u], [v], marker="+", color="#FF1744", s=36, linewidths=1.2)


def save_preview(
    path: Path,
    ct_zyx: np.ndarray,
    drrs: np.ndarray,
    nodules: list[dict[str, Any]],
    selected_nodule_id: str,
    case_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    selected = next(
        (item for item in nodules if item["nodule_id"] == selected_nodule_id),
        nodules[0] if nodules else None,
    )
    if selected is None:
        x = y = z = 127.5
    else:
        x, y, z = selected["center_full_xyz"]
    z_index = int(np.clip(round(z), 0, ct_zyx.shape[0] - 1))
    y_index = int(np.clip(round(y), 0, ct_zyx.shape[1] - 1))
    x_index = int(np.clip(round(x), 0, ct_zyx.shape[2] - 1))
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()
    for view_index, title in enumerate(("Frontal AP cone DRR", "Left-lateral cone DRR")):
        axes[view_index].imshow(display_drr(drrs[view_index]), cmap="gray")
        axes[view_index].set_title(title)
        for nodule in nodules:
            u, v = nodule["uv_by_view"][view_index]
            add_nodule_marker(
                axes[view_index],
                u,
                v,
                nodule["projected_radius_px_by_view"][view_index],
            )
    panels = (
        (ct_zyx[z_index], "Target CT axial"),
        (ct_zyx[:, y_index, :], "Target CT coronal"),
        (ct_zyx[:, :, x_index], "Target CT sagittal"),
    )
    for axis, (image, title) in zip(axes[2:5], panels):
        axis.imshow(image, cmap="gray", vmin=-1000, vmax=400, origin="lower")
        axis.set_title(title)
    if selected is not None:
        radius_voxels = 0.5 * float(selected["diameter_mm"]) / 1.5
        axes[2].add_patch(
            Circle(
                (x, y),
                radius=radius_voxels,
                fill=False,
                edgecolor="#FF1744",
                linewidth=1.5,
            )
        )
        axes[3].add_patch(
            Circle(
                (x, z),
                radius=radius_voxels,
                fill=False,
                edgecolor="#FF1744",
                linewidth=1.5,
            )
        )
        axes[4].add_patch(
            Circle(
                (y, z),
                radius=radius_voxels,
                fill=False,
                edgecolor="#FF1744",
                linewidth=1.5,
            )
        )
        zoom_half_voxels = max(14.0, 2.0 * float(selected["diameter_mm"]) / 1.5)
        axes[5].imshow(
            ct_zyx[z_index],
            cmap="gray",
            vmin=-1000,
            vmax=400,
            origin="lower",
        )
        axes[5].add_patch(
            Circle(
                (x, y),
                radius=radius_voxels,
                fill=False,
                edgecolor="#FF1744",
                linewidth=1.5,
            )
        )
        axes[5].set_xlim(x - zoom_half_voxels, x + zoom_half_voxels)
        axes[5].set_ylim(y - zoom_half_voxels, y + zoom_half_voxels)
        axes[5].set_title("Real LUNA16 nodule zoom")
    else:
        axes[5].imshow(ct_zyx[z_index], cmap="gray", vmin=-1000, vmax=400)
        axes[5].set_title("Negative case: no LUNA16 nodule")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"D1 synchronized data QC: {case_label}\n"
        "Red overlays are real LUNA16 annotations; no panel is a model prediction",
        fontsize=14,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_case_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = list(read_csv(path))
    integer_fields = {
        "nodule_count",
        "body_transverse_boundary_voxels",
        "lung_boundary_voxels",
        "slice_spacing_regularized",
        "slice_spacing_discarded_off_lattice",
        "slice_spacing_off_lattice_resampled",
        "slice_spacing_discarded_disconnected",
        "slice_spacing_duplicates_resolved",
        "slice_spacing_interpolated",
    }
    string_fields = {
        "case_id",
        "patient_id",
        "series_uid",
        "split",
        "selection_reason",
        "selection_evidence_tier",
        "preview_path",
        "metadata_path",
    }
    for row in rows:
        for key, value in list(row.items()):
            if key in string_fields:
                continue
            if key in integer_fields:
                row[key] = int(value)
            else:
                row[key] = float(value)
        upgrade_record(row)
    return rows


def make_montage(path: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wanted = []
    for tier in ("negative", "low", "intermediate", "high"):
        candidate = next(
            (
                record
                for record in records
                if record["selection_evidence_tier"] == tier
            ),
            None,
        )
        if candidate is not None:
            wanted.append(candidate)
    for record in records:
        if len(wanted) >= 4:
            break
        if record not in wanted:
            wanted.append(record)
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    for axis, record in zip(axes.ravel(), wanted):
        image = plt.imread(record["preview_path"])
        axis.imshow(image)
        axis.set_title(
            f"{record['patient_id']} | {record['selection_evidence_tier']} | "
            f"{record['split']}"
        )
        axis.axis("off")
    for axis in axes.ravel()[len(wanted) :]:
        axis.axis("off")
    figure.suptitle(
        "UPath-LIDC-v1 D1 real CT / synchronized cone-DRR pilot",
        fontsize=16,
    )
    figure.savefig(path, dpi=120)
    plt.close(figure)


def process_case(
    root: Path,
    output_dir: Path,
    selected: dict[str, Any],
    nodules_by_series: dict[str, list[dict[str, str]]],
    config: dict[str, Any],
    geometry: ConeBeamGeometry,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import SimpleITK as sitk

    row = selected["manifest"]
    series_uid = row["series_uid"]
    identifier = case_id(row["patient_id"], series_uid)
    case_dir = output_dir / "cases" / identifier
    record_path = case_dir / "record.json"
    if case_dir.exists() and args.resume and record_path.is_file():
        existing = upgrade_record(
            json.loads(record_path.read_text(encoding="utf-8"))
        )
        requires_repair = (
            args.repair_nonuniform
            and existing["source_slice_spacing_residual_mm"] > 0.1
            and not bool(existing["slice_spacing_regularized"])
        )
        if not requires_repair:
            return existing
    if case_dir.exists() and args.resume:
        resolved_case = case_dir.resolve()
        resolved_parent = (output_dir / "cases").resolve()
        if resolved_case.parent != resolved_parent:
            raise RuntimeError(f"Refusing to clean unexpected case path: {case_dir}")
        shutil.rmtree(resolved_case)
    case_dir.mkdir(parents=True, exist_ok=False)
    source, files, slice_spacing, spacing_audit = read_dicom_series(
        root / row["dicom_dir"],
        series_uid,
    )
    source_spacing_residual = float(
        spacing_audit["source_max_spacing_residual_mm"]
    )
    spacing_residual = float(
        spacing_audit["post_regularization_spacing_residual_mm"]
    )
    source_spacing = tuple(float(value) for value in source.GetSpacing())
    low_resolution = resample_axis_aligned(source, spacing_mm=3.0)
    low_hu = sitk.GetArrayFromImage(low_resolution).astype(np.float32)
    center_result, _, _ = thoracic_center_xyz(low_hu)
    center_lps = np.asarray(
        low_resolution.TransformContinuousIndexToPhysicalPoint(
            center_result.center_xyz
        ),
        dtype=np.float64,
    )

    full_shape_xyz = tuple(int(value) for value in config["canonical_volume"]["full_shape"])
    full_spacing_xyz = tuple(
        float(value) for value in config["canonical_volume"]["full_spacing_mm"]
    )
    full_image, full_origin_lps = resample_canonical(
        source,
        center_lps,
        full_shape_xyz,
        full_spacing_xyz,
    )
    hu_min, hu_max = (
        float(config["canonical_volume"]["hu_clip"][0]),
        float(config["canonical_volume"]["hu_clip"][1]),
    )
    full_hu = np.clip(
        sitk.GetArrayFromImage(full_image).astype(np.float32),
        hu_min,
        hu_max,
    )
    full_hu = np.rint(full_hu).astype(np.int16)
    full_body = body_mask_zyx(full_hu).astype(np.uint8)
    full_lung = lung_mask_zyx(full_hu, full_body).astype(np.uint8)
    full_affine = lps_grid_to_nifti_ras_affine(
        full_origin_lps,
        full_spacing_xyz,
    )

    full_ct_path = case_dir / "ct_full_256.nii.gz"
    full_body_path = case_dir / "body_mask_full_256.nii.gz"
    full_lung_path = case_dir / "lung_mask_full_256.nii.gz"
    save_nifti_zyx(full_ct_path, full_hu, full_affine, np.int16)
    save_nifti_zyx(full_body_path, full_body, full_affine, np.uint8)
    save_nifti_zyx(full_lung_path, full_lung, full_affine, np.uint8)

    source_nodules = nodules_by_series.get(series_uid, [])
    lesion_full = np.zeros_like(full_body, dtype=np.uint8)
    nodule_records: list[dict[str, Any]] = []
    roundtrip_errors = []
    nifti_errors = []
    edge_margins = []
    points_lps = []
    for source_nodule in source_nodules:
        point_lps = np.asarray(
            [
                float(source_nodule["center_lps_x"]),
                float(source_nodule["center_lps_y"]),
                float(source_nodule["center_lps_z"]),
            ],
            dtype=np.float64,
        )
        diameter = float(source_nodule["diameter_mm"])
        center_xyz = physical_lps_to_index_xyz(
            point_lps,
            full_origin_lps,
            full_spacing_xyz,
        )
        reconstructed = index_xyz_to_physical_lps(
            center_xyz,
            full_origin_lps,
            full_spacing_xyz,
        )
        roundtrip_errors.append(float(np.linalg.norm(reconstructed - point_lps)))
        point_ras = lps_to_ras(point_lps)
        nifti_index = (
            np.linalg.inv(full_affine)
            @ np.asarray([*point_ras, 1.0], dtype=np.float64)
        )[:3]
        nifti_errors.append(float(np.linalg.norm(nifti_index - center_xyz)))
        edge_margin = sphere_edge_margin_mm(
            full_shape_xyz,
            center_xyz,
            diameter,
            full_spacing_xyz,
        )
        edge_margins.append(edge_margin)
        lesion_full |= spherical_mask_zyx(
            full_hu.shape,
            center_xyz,
            diameter,
            full_spacing_xyz,
        )
        points_lps.append(point_lps)
        nodule_records.append(
            {
                "nodule_id": source_nodule["nodule_id"],
                "diameter_mm": diameter,
                "evidence_tier": source_nodule["evidence_tier"],
                "best_view": source_nodule["best_view"],
                "center_lps_mm": point_lps.tolist(),
                "center_full_xyz": center_xyz.tolist(),
                "edge_margin_mm": edge_margin,
            }
        )
    lesion_full_path = case_dir / "lesion_proxy_full_256.nii.gz"
    save_nifti_zyx(lesion_full_path, lesion_full, full_affine, np.uint8)

    global_hu = block_mean_2x_zyx(full_hu)
    global_body = block_max_2x_zyx(full_body).astype(np.uint8)
    global_lung = block_max_2x_zyx(full_lung).astype(np.uint8)
    global_lesion = block_max_2x_zyx(lesion_full).astype(np.uint8)
    global_origin_lps, global_spacing_xyz = block_mean_grid(
        full_origin_lps,
        full_spacing_xyz,
    )
    global_affine = lps_grid_to_nifti_ras_affine(
        global_origin_lps,
        global_spacing_xyz,
    )
    global_ct_path = case_dir / "ct_global_128.nii.gz"
    save_nifti_zyx(global_ct_path, global_hu, global_affine, np.float32)
    save_nifti_zyx(
        case_dir / "body_mask_global_128.nii.gz",
        global_body,
        global_affine,
        np.uint8,
    )
    save_nifti_zyx(
        case_dir / "lung_mask_global_128.nii.gz",
        global_lung,
        global_affine,
        np.uint8,
    )
    save_nifti_zyx(
        case_dir / "lesion_proxy_global_128.nii.gz",
        global_lesion,
        global_affine,
        np.uint8,
    )
    reloaded_global = np.transpose(
        nib.load(str(global_ct_path)).get_fdata(dtype=np.float32),
        (2, 1, 0),
    )
    global_error = float(np.max(np.abs(reloaded_global - global_hu)))

    drrs, render_record = render_drrs(
        full_ct_path,
        np.asarray(points_lps, dtype=np.float64).reshape(-1, 3),
        center_lps,
        config,
        geometry,
        args.device,
        args.renderer,
        args.patch_size,
    )
    if drrs.shape != (2, *geometry.detector_shape):
        raise ValueError(f"Unexpected DRR shape: {drrs.shape}")
    drr_path = case_dir / "drrs_ap_left_lateral.npz"
    np.savez_compressed(
        drr_path,
        frontal_ap=drrs[0].astype(np.float32),
        left_lateral=drrs[1].astype(np.float32),
    )
    matrices_path = case_dir / "projection_matrices_centered_lps.npz"
    np.savez(
        matrices_path,
        frontal_ap=projection_matrix_lps(geometry, "frontal_ap"),
        left_lateral=projection_matrix_lps(geometry, "left_lateral"),
    )
    for nodule_index, nodule in enumerate(nodule_records):
        nodule["uv_by_view"] = [
            render_record["fiducial_uv_by_view"][view][nodule_index]
            for view in range(2)
        ]
        radius = 0.5 * float(nodule["diameter_mm"])
        nodule["projected_radius_px_by_view"] = []
        for view_index in range(2):
            depth = render_record["fiducial_depth_mm_by_view"][view_index][
                nodule_index
            ]
            pixel_size = geometry.detector_pixel_size_mm[view_index]
            nodule["projected_radius_px_by_view"].append(
                radius * geometry.source_detector_distance_mm / depth / pixel_size
            )

    preview_path = case_dir / "qc.png"
    save_preview(
        preview_path,
        full_hu,
        drrs,
        nodule_records,
        selected["selected_nodule_id"],
        f"{row['patient_id']} | {row['split']} | {selected['selection_reason']}",
    )
    body_faces = boundary_face_counts_zyx(full_body)
    lung_faces = boundary_face_counts_zyx(full_lung)
    body_coords = np.argwhere(full_body > 0)
    lung_coords = np.argwhere(full_lung > 0)
    body_extent_mm = (
        (body_coords.max(axis=0) - body_coords.min(axis=0) + 1)
        * np.asarray(full_spacing_xyz[::-1])
    )[::-1]
    lung_extent_mm = (
        (lung_coords.max(axis=0) - lung_coords.min(axis=0) + 1)
        * np.asarray(full_spacing_xyz[::-1])
    )[::-1]
    data_files = [
        full_ct_path,
        global_ct_path,
        full_body_path,
        full_lung_path,
        lesion_full_path,
        drr_path,
        matrices_path,
        preview_path,
    ]
    metadata = {
        "protocol_id": config["protocol_id"],
        "gate": "D1",
        "case_id": identifier,
        "patient_id": row["patient_id"],
        "series_uid": series_uid,
        "split": row["split"],
        "selection_reason": selected["selection_reason"],
        "selection_evidence_tier": selected["selection_evidence_tier"],
        "selected_nodule_id": selected["selected_nodule_id"],
        "crop_uses_lesion_annotation": False,
        "source": {
            "dicom_dir": row["dicom_dir"],
            "dicom_slices": len(files),
            "shape_xyz": list(source.GetSize()),
            "spacing_xyz_mm": list(source_spacing),
            "direction_lps": list(source.GetDirection()),
            "slice_spacing_median_mm": slice_spacing,
            "slice_spacing_source_max_residual_mm": source_spacing_residual,
            "slice_spacing_post_max_residual_mm": spacing_residual,
            "slice_spacing_regularization": spacing_audit,
        },
        "canonical": {
            "thoracic_center_method": center_result.method,
            "center_lps_mm": center_lps.tolist(),
            "full_shape_zyx": list(full_hu.shape),
            "full_spacing_xyz_mm": list(full_spacing_xyz),
            "full_origin_lps_mm": full_origin_lps.tolist(),
            "full_nifti_affine_ras": full_affine.tolist(),
            "global_shape_zyx": list(global_hu.shape),
            "global_spacing_xyz_mm": global_spacing_xyz.tolist(),
            "global_origin_lps_mm": global_origin_lps.tolist(),
            "global_nifti_affine_ras": global_affine.tolist(),
            "global_construction": "exact 2x2x2 block mean of stored full CT",
            "global_mean_max_abs_error_hu": global_error,
            "body_extent_xyz_mm": body_extent_mm.tolist(),
            "lung_extent_xyz_mm": lung_extent_mm.tolist(),
            "body_boundary_face_voxels": body_faces,
            "lung_boundary_face_voxels": lung_faces,
        },
        "annotations": {
            "count": len(nodule_records),
            "semantics": "LUNA16 center/diameter spherical localization proxy, not contour",
            "max_lps_roundtrip_error_mm": max(roundtrip_errors, default=0.0),
            "max_nifti_index_error_voxels": max(nifti_errors, default=0.0),
            "minimum_sphere_edge_margin_mm": min(edge_margins, default=None),
            "nodules": nodule_records,
        },
        "projection": {
            "views": ["frontal_ap", "left_lateral"],
            "shape": list(geometry.detector_shape),
            "intensity_domain": "DiffDRR line-integrated normalized density",
            "nonblank_std": [float(np.std(image)) for image in drrs],
            "minimum": [float(np.min(image)) for image in drrs],
            "maximum": [float(np.max(image)) for image in drrs],
            "centered_fiducial_error_mm": render_record[
                "centered_fiducial_error_mm"
            ],
            "max_analytic_diffdrr_projection_error_px": render_record[
                "max_projection_error_px"
            ],
            "matrices_map": "thorax-centered DICOM-LPS millimetres to detector uv pixels",
            "patient_lps_to_centered_lps": {
                "operation": "subtract",
                "translation_mm": center_lps.tolist(),
            },
        },
        "hashes": {path.name: sha256_file(path) for path in data_files},
    }
    write_json(case_dir / "metadata.json", metadata)
    record = {
        "case_id": identifier,
        "patient_id": row["patient_id"],
        "series_uid": series_uid,
        "split": row["split"],
        "selection_reason": selected["selection_reason"],
        "selection_evidence_tier": selected["selection_evidence_tier"],
        "nodule_count": len(nodule_records),
        "source_spacing_z_mm": source_spacing[2],
        "source_slice_spacing_residual_mm": source_spacing_residual,
        "slice_spacing_residual_mm": spacing_residual,
        "slice_spacing_regularized": int(bool(spacing_audit["applied"])),
        "slice_spacing_discarded_off_lattice": int(
            spacing_audit["discarded_off_lattice_slices"]
        ),
        "slice_spacing_off_lattice_resampled": int(
            spacing_audit.get("off_lattice_slices_resampled", 0)
        ),
        "slice_spacing_discarded_disconnected": int(
            spacing_audit["discarded_disconnected_slices"]
        ),
        "slice_spacing_duplicates_resolved": int(
            spacing_audit["duplicate_slices_resolved"]
        ),
        "slice_spacing_interpolated": int(
            spacing_audit["interpolated_slices"]
        ),
        "slice_spacing_retained_lattice_residual_mm": float(
            spacing_audit["retained_lattice_max_residual_mm"]
        ),
        "body_extent_x_mm": body_extent_mm[0],
        "body_extent_y_mm": body_extent_mm[1],
        "body_extent_z_mm": body_extent_mm[2],
        "lung_extent_x_mm": lung_extent_mm[0],
        "lung_extent_y_mm": lung_extent_mm[1],
        "lung_extent_z_mm": lung_extent_mm[2],
        "body_transverse_boundary_voxels": sum(
            body_faces[key]
            for key in ("posterior_y0", "anterior_y1", "right_x0", "left_x1")
        ),
        "lung_boundary_voxels": sum(lung_faces.values()),
        "minimum_nodule_edge_margin_mm": min(edge_margins, default=math.inf),
        "max_lps_roundtrip_error_mm": max(roundtrip_errors, default=0.0),
        "max_nifti_index_error_voxels": max(nifti_errors, default=0.0),
        "global_mean_error_hu": global_error,
        "max_projection_error_px": render_record["max_projection_error_px"],
        "min_drr_std": min(float(np.std(image)) for image in drrs),
        "preview_path": str(preview_path),
        "metadata_path": str(case_dir / "metadata.json"),
    }
    write_json(record_path, record)
    return record


def main() -> None:
    args = parse_args()
    started = time.time()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    config_path = resolve(args.config)
    manifest_path = resolve(args.manifest)
    nodule_path = resolve(args.nodule_table)
    output_dir = resolve(args.output_dir)
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not (args.finalize_existing or args.resume)
    ):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cases").mkdir(
        exist_ok=(args.finalize_existing or args.resume)
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    geometry = load_geometry(config)
    manifest_rows = read_csv(manifest_path)
    nodule_rows = read_csv(nodule_path)
    nodules_by_series: dict[str, list[dict[str, str]]] = {}
    for row in nodule_rows:
        nodules_by_series.setdefault(row["series_uid"], []).append(row)
    errors = []
    if args.finalize_existing:
        records = read_case_records(output_dir / "case_manifest.csv")
        print(
            f"[D1 finalize] loaded {len(records)} completed case records",
            flush=True,
        )
    else:
        if args.full_release:
            selected = select_release_shard(
                manifest_rows,
                nodules_by_series,
                args.num_shards,
                args.shard_index,
            )
        else:
            selected = select_cases(manifest_rows, nodule_rows)
        if args.max_cases > 0:
            selected = selected[: args.max_cases]
        write_json(
            output_dir / "selection.json",
            [
                {
                    "patient_id": item["manifest"]["patient_id"],
                    "series_uid": item["manifest"]["series_uid"],
                    "split": item["manifest"]["split"],
                    "selection_reason": item["selection_reason"],
                    "selection_evidence_tier": item["selection_evidence_tier"],
                    "selected_nodule_id": item["selected_nodule_id"],
                }
                for item in selected
            ],
        )
        records = []
        progress_label = "R0" if args.full_release else "D1"
        for index, item in enumerate(selected, start=1):
            patient = item["manifest"]["patient_id"]
            print(
                f"[{progress_label} {index:04d}/{len(selected):04d}] {patient} "
                f"{item['selection_reason']}",
                flush=True,
            )
            try:
                record = process_case(
                    root,
                    output_dir,
                    item,
                    nodules_by_series,
                    config,
                    geometry,
                    args,
                )
                records.append(record)
                print(
                    f"  done body={record['body_extent_x_mm']:.1f}x"
                    f"{record['body_extent_y_mm']:.1f} mm "
                    f"drr_std={record['min_drr_std']:.4f}",
                    flush=True,
                )
            except Exception as error:
                errors.append(
                    {
                        "patient_id": patient,
                        "series_uid": item["manifest"]["series_uid"],
                        "selection_reason": item["selection_reason"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"  ERROR {errors[-1]['error']}", flush=True)
    if records:
        if not args.finalize_existing:
            write_rows(output_dir / "case_manifest.csv", records)
        make_montage(output_dir / "d1_real_data_qc.png", records)

    finite_margins = [
        record["minimum_nodule_edge_margin_mm"]
        for record in records
        if math.isfinite(record["minimum_nodule_edge_margin_mm"])
    ]
    body_widths = [
        max(record["body_extent_x_mm"], record["body_extent_y_mm"])
        for record in records
    ]
    expected_case_count = (
        len(
            select_release_shard(
                manifest_rows,
                nodules_by_series,
                args.num_shards,
                args.shard_index,
            )
        )
        if args.full_release
        else 16
    )
    count_check_name = (
        "all_release_shard_cases_completed"
        if args.full_release
        else "exactly_16_cases"
    )
    checks = {
        count_check_name: len(records) == expected_case_count and not errors,
        "all_splits_represented": {record["split"] for record in records}
        == {"train", "val", "test"},
        "all_evidence_strata_represented": {
            record["selection_evidence_tier"] for record in records
        }
        == {"negative", "low", "intermediate", "high"},
        "native_slice_spacing_span_at_least_1mm": (
            len(records) > 1
            and max(record["source_spacing_z_mm"] for record in records)
            - min(record["source_spacing_z_mm"] for record in records)
            >= 1.0
        ),
        "body_size_span_at_least_20mm": (
            len(body_widths) > 1 and max(body_widths) - min(body_widths) >= 20.0
        ),
        "no_transverse_body_truncation": all(
            record["body_transverse_boundary_voxels"] == 0 for record in records
        ),
        "no_lung_truncation": all(
            record["lung_boundary_voxels"] == 0 for record in records
        ),
        "no_nodule_truncation": all(margin >= 0.0 for margin in finite_margins),
        "annotation_lps_roundtrip_below_1e-4mm": all(
            record["max_lps_roundtrip_error_mm"] <= 1e-4 for record in records
        ),
        "nifti_index_roundtrip_below_1e-4vox": all(
            record["max_nifti_index_error_voxels"] <= 1e-4 for record in records
        ),
        "global_is_exact_block_mean": all(
            record["global_mean_error_hu"] <= 1e-6 for record in records
        ),
        "analytic_diffdrr_projection_below_0_05px": all(
            record["max_projection_error_px"] <= 0.05 for record in records
        ),
        "drrs_are_nonblank": all(record["min_drr_std"] > 1e-5 for record in records),
        "slice_spacing_residual_below_0_1mm": all(
            record["slice_spacing_residual_mm"] <= 0.1 for record in records
        ),
        "nonuniform_slice_stacks_were_regularized": all(
            record["source_slice_spacing_residual_mm"] <= 0.1
            or bool(record["slice_spacing_regularized"])
            for record in records
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "pass" if all(checks.values()) else "fail"
    gate_name = (
        f"R0-shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
        if args.full_release
        else "D1"
    )
    summary = {
        "protocol_id": config["protocol_id"],
        "gate": gate_name,
        "status": status,
        "interpretation": (
            "Full-release shard under the passed D1 synchronized CT/DRR "
            "contract."
            if args.full_release
            else "Deterministic real LIDC source DICOM to synchronized "
            "canonical CT, AP/left-lateral cone DRR and LUNA annotation "
            "contract."
        ),
        "expected_case_count": expected_case_count,
        "case_count": len(records),
        "error_count": len(errors),
        "errors": errors,
        "checks": checks,
        "ranges": {
            "source_spacing_z_mm": [
                min((record["source_spacing_z_mm"] for record in records), default=None),
                max((record["source_spacing_z_mm"] for record in records), default=None),
            ],
            "body_transverse_extent_mm": [
                min(body_widths, default=None),
                max(body_widths, default=None),
            ],
            "minimum_nodule_edge_margin_mm": min(finite_margins, default=None),
            "maximum_projection_error_px": max(
                (record["max_projection_error_px"] for record in records),
                default=None,
            ),
            "maximum_slice_spacing_residual_mm": max(
                (record["slice_spacing_residual_mm"] for record in records),
                default=None,
            ),
            "maximum_source_slice_spacing_residual_mm": max(
                (
                    record["source_slice_spacing_residual_mm"]
                    for record in records
                ),
                default=None,
            ),
            "regularized_slice_stacks": sum(
                int(bool(record["slice_spacing_regularized"]))
                for record in records
            ),
        },
        "runtime": {
            "seconds": time.time() - started,
            "device": args.device,
            "renderer": args.renderer,
            "patch_size": args.patch_size,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
            "nodule_table_sha256": sha256_file(nodule_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "canonical_module_sha256": sha256_file(
                root / "src/upathx2ct/sota/canonical.py"
            ),
        },
    }
    if (output_dir / "d1_real_data_qc.png").exists():
        summary["provenance"]["figure_sha256"] = sha256_file(
            output_dir / "d1_real_data_qc.png"
        )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        f"# UPath-LIDC-v1 Gate {gate_name}\n\n"
        f"Status: **{status.upper()}**.\n\n"
        "This release contains real LIDC CT targets, synchronized CT-derived "
        "frontal AP/left-lateral cone DRRs and real LUNA16 center/diameter "
        "localization proxies. No image is a model prediction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if status != "pass":
        raise SystemExit(f"Gate {gate_name} failed; model training remains locked")


if __name__ == "__main__":
    main()
