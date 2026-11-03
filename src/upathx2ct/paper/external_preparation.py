from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import h5py
import matplotlib
import numpy as np
import SimpleITK as sitk

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from upathx2ct.sota.canonical import (
    block_max_2x_zyx,
    block_mean_2x_zyx,
    block_mean_grid,
    centered_grid_origin_lps,
    thoracic_center_xyz,
)
from upathx2ct.sota.diffdrr_renderer import render_centered_biplanar
from upathx2ct.sota.model import B0Config, b0_projection_matrices
from upathx2ct.sota.protocol import sha256_file, write_json


HU_MIN = -1024.0
HU_MAX = 1500.0
FULL_SHAPE_XYZ = (256, 256, 256)
FULL_SPACING_XYZ_MM = (1.5, 1.5, 1.5)
IDENTITY_DIRECTION_3D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ExternalCTCase:
    case_id: str
    patient_id: str
    series_uid: str
    image_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


def resample_to_grid(
    image: sitk.Image,
    origin_lps_mm: np.ndarray,
    *,
    nearest: bool,
    default_value: float,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(FULL_SHAPE_XYZ)
    resampler.SetOutputSpacing(FULL_SPACING_XYZ_MM)
    resampler.SetOutputOrigin(tuple(float(value) for value in origin_lps_mm))
    resampler.SetOutputDirection(IDENTITY_DIRECTION_3D)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(image)


def image_from_zyx(
    volume: np.ndarray,
    origin_lps_mm: np.ndarray,
    spacing_xyz_mm: np.ndarray,
) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(volume))
    image.SetOrigin(tuple(float(value) for value in origin_lps_mm))
    image.SetSpacing(tuple(float(value) for value in spacing_xyz_mm))
    image.SetDirection(IDENTITY_DIRECTION_3D)
    return image


def save_preview(
    path: Path, case_id: str, ct_hu: np.ndarray, drrs: np.ndarray
) -> None:
    indices = tuple(size // 2 for size in ct_hu.shape)
    figure, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
    for axis, index, title, panel in zip(
        range(3),
        indices,
        ("Axial", "Coronal", "Sagittal"),
        axes[:3],
    ):
        panel.imshow(
            np.take(ct_hu, index, axis=axis),
            cmap="gray",
            vmin=-1000,
            vmax=300,
            origin="lower",
        )
        panel.set_title(title)
        panel.axis("off")
    for panel, image, title in zip(
        axes[3:], drrs, ("AP DRR", "Left lateral DRR")
    ):
        low, high = np.percentile(image, (0.5, 99.5))
        panel.imshow(image, cmap="gray", vmin=low, vmax=high, origin="lower")
        panel.set_title(title)
        panel.axis("off")
    figure.suptitle(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def prepare_unlabelled_case(
    case: ExternalCTCase,
    output_root: Path,
    protocol: Mapping[str, Any],
    *,
    device: str,
    renderer: str,
    patch_size: int,
    protocol_id: str,
    make_preview: bool,
    resume: bool,
) -> dict[str, Any]:
    output_h5 = output_root / "cases" / f"{case.case_id}.h5"
    case_dir = output_root / "source_cases" / case.case_id
    metadata_path = case_dir / "metadata.json"
    if resume and output_h5.is_file() and metadata_path.is_file():
        with h5py.File(output_h5, "r") as handle:
            valid = (
                tuple(handle["ct"].shape) == (128, 128, 128)
                and tuple(handle["drr_frontal_ap"].shape) == (512, 512)
                and str(handle.attrs["case_id"]) == case.case_id
                and str(handle.attrs["patient_id"]) == case.patient_id
            )
        if valid:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            return {**record["audit"], "status": "reused"}

    source_ct = sitk.ReadImage(str(case.image_path), sitk.sitkFloat32)
    source_hu = sitk.GetArrayFromImage(source_ct).astype(np.float32, copy=False)
    center_result, _, _ = thoracic_center_xyz(source_hu)
    center_lps = np.asarray(
        source_ct.TransformContinuousIndexToPhysicalPoint(center_result.center_xyz),
        dtype=np.float64,
    )
    full_origin = centered_grid_origin_lps(
        center_lps, FULL_SHAPE_XYZ, FULL_SPACING_XYZ_MM
    )
    canonical_ct = resample_to_grid(
        source_ct, full_origin, nearest=False, default_value=HU_MIN
    )
    full_hu = np.clip(
        sitk.GetArrayFromImage(canonical_ct).astype(np.float32), HU_MIN, HU_MAX
    )
    _, full_body, full_lung = thoracic_center_xyz(full_hu)
    global_hu = block_mean_2x_zyx(full_hu)
    global_body = block_max_2x_zyx(full_body) > 0
    global_lung = block_max_2x_zyx(full_lung) > 0
    global_lesion = np.zeros_like(global_body, dtype=np.uint8)
    global_origin, global_spacing = block_mean_grid(
        full_origin, FULL_SPACING_XYZ_MM
    )

    case_dir.mkdir(parents=True, exist_ok=True)
    render_ct_path = case_dir / "ct_full_256.rendering.nii.gz"
    global_ct_path = case_dir / "ct_global_128.nii.gz"
    sitk.WriteImage(canonical_ct, str(render_ct_path), True)
    sitk.WriteImage(
        image_from_zyx(global_hu, global_origin, global_spacing),
        str(global_ct_path),
        True,
    )
    try:
        drrs, renderer_record = render_centered_biplanar(
            render_ct_path,
            dict(protocol),
            device_name=device,
            renderer=renderer,
            patch_size=patch_size,
        )
    finally:
        render_ct_path.unlink(missing_ok=True)
    if drrs.shape != (2, 512, 512) or not np.isfinite(drrs).all():
        raise ValueError(f"Invalid rendered DRRs for {case.case_id}: {drrs.shape}")
    if min(float(np.std(view)) for view in drrs) <= 1e-5:
        raise ValueError(f"Near-constant rendered DRR for {case.case_id}")
    if float(global_lung.mean()) <= 0.01:
        raise ValueError(f"No usable lung mask for {case.case_id}")

    signed_ct = (
        2.0 * (global_hu - HU_MIN) / (HU_MAX - HU_MIN) - 1.0
    ).astype(np.float16)
    matrices = b0_projection_matrices(B0Config()).numpy()
    metadata = {
        "protocol_id": protocol_id,
        "source": str(case.image_path.resolve()),
        "source_metadata": dict(case.metadata),
        "full_origin_lps_mm": full_origin.tolist(),
        "global_origin_lps_mm": global_origin.tolist(),
        "full_spacing_xyz_mm": list(FULL_SPACING_XYZ_MM),
        "global_spacing_xyz_mm": global_spacing.tolist(),
        "thoracic_center_method": center_result.method,
        "renderer": renderer_record,
    }
    temporary = output_h5.with_suffix(".h5.tmp")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("ct", data=signed_ct, compression="lzf")
        handle.create_dataset(
            "ct_hu", data=np.rint(global_hu).astype(np.int16), compression="lzf"
        )
        handle.create_dataset(
            "body_mask", data=global_body.astype(np.uint8), compression="lzf"
        )
        handle.create_dataset(
            "lung_mask", data=global_lung.astype(np.uint8), compression="lzf"
        )
        handle.create_dataset("lesion_proxy", data=global_lesion, compression="lzf")
        handle.create_dataset(
            "drr_frontal_ap", data=drrs[0].astype(np.float16), compression="lzf"
        )
        handle.create_dataset(
            "drr_left_lateral", data=drrs[1].astype(np.float16), compression="lzf"
        )
        handle.create_dataset("projection_matrices", data=matrices.astype(np.float32))
        handle.attrs["case_id"] = case.case_id
        handle.attrs["patient_id"] = case.patient_id
        handle.attrs["series_uid"] = case.series_uid
        handle.attrs["split"] = "test"
        handle.attrs["nodule_count"] = 0
        handle.attrs["has_lesion_label"] = False
        handle.attrs["source_case_dir"] = str(case_dir.resolve())
        handle.attrs["metadata_json"] = json.dumps(metadata, separators=(",", ":"))
    temporary.replace(output_h5)

    audit = {
        "case_id": case.case_id,
        "patient_id": case.patient_id,
        "series_uid": case.series_uid,
        "split": "test",
        "status": "pass",
        "source_path": str(case.image_path.resolve()),
        "source_shape_xyz": json.dumps(list(source_ct.GetSize())),
        "source_spacing_xyz_mm": json.dumps(list(source_ct.GetSpacing())),
        "body_fraction": float(global_body.mean()),
        "lung_fraction": float(global_lung.mean()),
        "ap_std": float(np.std(drrs[0])),
        "lat_std": float(np.std(drrs[1])),
        "body_boundary_voxels": int(
            global_body[0].sum()
            + global_body[-1].sum()
            + global_body[:, 0].sum()
            + global_body[:, -1].sum()
            + global_body[:, :, 0].sum()
            + global_body[:, :, -1].sum()
        ),
        "packed_path": str(output_h5.resolve()),
        "ct_normalized_key": "ct",
        "ct_hu_key": "ct_hu",
        "ap_key": "drr_frontal_ap",
        "lat_key": "drr_left_lateral",
        "projection_matrices_key": "projection_matrices",
        "h5_sha256": sha256_file(output_h5),
        "h5_bytes": output_h5.stat().st_size,
    }
    write_json(metadata_path, {"metadata": metadata, "audit": audit})
    if make_preview:
        save_preview(
            output_root / "previews" / f"{case.case_id}.png",
            case.case_id,
            global_hu,
            drrs,
        )
    return audit
