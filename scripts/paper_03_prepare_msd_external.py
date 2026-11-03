#!/usr/bin/env python3
"""Prepare MSD Lung as a no-tuning external B0 CT/DRR test cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from upathx2ct.sota.canonical import (
    block_max_2x_zyx,
    block_mean_2x_zyx,
    block_mean_grid,
    centered_grid_origin_lps,
    lung_mask_zyx,
    thoracic_center_xyz,
)
from upathx2ct.sota.diffdrr_renderer import render_centered_biplanar
from upathx2ct.sota.model import B0Config, b0_projection_matrices
from upathx2ct.sota.protocol import evenly_spaced_indices, sha256_file, write_json


HU_MIN = -1024.0
HU_MAX = 1500.0
FULL_SHAPE_XYZ = (256, 256, 256)
FULL_SPACING_XYZ_MM = (1.5, 1.5, 1.5)
IDENTITY_DIRECTION_3D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--internal-pack", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renderer", choices=("siddon", "trilinear"), default="siddon")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--preview-cases", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear)
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
    path: Path,
    case_id: str,
    ct_hu: np.ndarray,
    lesion: np.ndarray,
    drrs: np.ndarray,
) -> None:
    coordinates = np.argwhere(lesion > 0)
    indices = (
        tuple(int(round(value)) for value in coordinates.mean(axis=0))
        if len(coordinates)
        else tuple(size // 2 for size in ct_hu.shape)
    )
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
    for axis, index, title, panel in zip(
        range(3),
        indices,
        ("Axial", "Coronal", "Sagittal"),
        axes[:3],
    ):
        image = np.take(ct_hu, index, axis=axis)
        mask = np.take(lesion, index, axis=axis)
        panel.imshow(image, cmap="gray", vmin=-1000, vmax=300, origin="lower")
        if mask.max() > 0:
            panel.contour(mask, levels=[0.5], colors=["#00E5FF"], linewidths=0.8, origin="lower")
        panel.set_title(title)
        panel.axis("off")
    for panel, image, title in zip(axes[3:], drrs, ("AP DRR", "Left lateral DRR")):
        low, high = np.percentile(image, (0.5, 99.5))
        panel.imshow(image, cmap="gray", vmin=low, vmax=high, origin="lower")
        panel.set_title(title)
        panel.axis("off")
    fig.suptitle(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def process_case(
    image_path: Path,
    label_path: Path,
    output_root: Path,
    protocol: dict[str, Any],
    args: argparse.Namespace,
    make_preview: bool,
) -> dict[str, Any]:
    case_id = image_path.name.removesuffix(".nii.gz")
    output_h5 = output_root / "cases" / f"{case_id}.h5"
    case_dir = output_root / "source_cases" / case_id
    metadata_path = case_dir / "metadata.json"
    if args.resume and output_h5.is_file() and metadata_path.is_file():
        with h5py.File(output_h5, "r") as handle:
            valid = (
                tuple(handle["ct"].shape) == (128, 128, 128)
                and tuple(handle["drr_frontal_ap"].shape) == (512, 512)
                and str(handle.attrs["case_id"]) == case_id
            )
        if valid:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            return {**record["audit"], "case_id": case_id, "status": "reused"}

    source_ct = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    source_label = sitk.ReadImage(str(label_path), sitk.sitkUInt8)
    if (
        source_ct.GetSize() != source_label.GetSize()
        or not np.allclose(source_ct.GetSpacing(), source_label.GetSpacing(), atol=1e-5)
        or not np.allclose(source_ct.GetOrigin(), source_label.GetOrigin(), atol=1e-3)
        or not np.allclose(source_ct.GetDirection(), source_label.GetDirection(), atol=1e-5)
    ):
        raise ValueError(f"CT/label geometry differs for {case_id}")
    source_hu = sitk.GetArrayFromImage(source_ct).astype(np.float32, copy=False)
    center_result, _, _ = thoracic_center_xyz(source_hu)
    center_lps = np.asarray(
        source_ct.TransformContinuousIndexToPhysicalPoint(center_result.center_xyz),
        dtype=np.float64,
    )
    full_origin = centered_grid_origin_lps(
        center_lps,
        FULL_SHAPE_XYZ,
        FULL_SPACING_XYZ_MM,
    )
    canonical_ct = resample_to_grid(
        source_ct,
        full_origin,
        nearest=False,
        default_value=HU_MIN,
    )
    canonical_label = resample_to_grid(
        source_label,
        full_origin,
        nearest=True,
        default_value=0.0,
    )
    full_hu = np.clip(
        sitk.GetArrayFromImage(canonical_ct).astype(np.float32),
        HU_MIN,
        HU_MAX,
    )
    full_lesion = sitk.GetArrayFromImage(canonical_label) > 0
    center_check, full_body, full_lung = thoracic_center_xyz(full_hu)
    global_hu = block_mean_2x_zyx(full_hu)
    global_body = block_max_2x_zyx(full_body) > 0
    global_lung = block_max_2x_zyx(full_lung) > 0
    global_lesion = block_max_2x_zyx(full_lesion.astype(np.uint8)) > 0
    global_origin, global_spacing = block_mean_grid(full_origin, FULL_SPACING_XYZ_MM)
    lesion_components = int(ndimage.label(global_lesion)[1])

    case_dir.mkdir(parents=True, exist_ok=True)
    full_ct_path = case_dir / "ct_full_256.nii.gz"
    global_ct_path = case_dir / "ct_global_128.nii.gz"
    sitk.WriteImage(canonical_ct, str(full_ct_path), True)
    sitk.WriteImage(
        image_from_zyx(global_hu, global_origin, global_spacing),
        str(global_ct_path),
        True,
    )
    sitk.WriteImage(
        image_from_zyx(global_lesion.astype(np.uint8), global_origin, global_spacing),
        str(case_dir / "lesion_global_128.nii.gz"),
        True,
    )
    drrs, renderer_record = render_centered_biplanar(
        full_ct_path,
        protocol,
        device_name=args.device,
        renderer=args.renderer,
        patch_size=args.patch_size,
    )
    if drrs.shape != (2, 512, 512) or not np.isfinite(drrs).all():
        raise ValueError(f"Invalid rendered DRRs for {case_id}: {drrs.shape}")
    if min(float(np.std(view)) for view in drrs) <= 1e-5:
        raise ValueError(f"Near-constant rendered DRR for {case_id}")

    signed_ct = (
        2.0 * (global_hu - HU_MIN) / (HU_MAX - HU_MIN) - 1.0
    ).astype(np.float16)
    matrices = b0_projection_matrices(B0Config()).numpy()
    temporary = output_h5.with_suffix(".h5.tmp")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    metadata = {
        "protocol_id": "UPATH-MSD-LUNG-EXTERNAL-v1.0",
        "source": str(image_path.resolve()),
        "source_label": str(label_path.resolve()),
        "full_origin_lps_mm": full_origin.tolist(),
        "global_origin_lps_mm": global_origin.tolist(),
        "full_spacing_xyz_mm": list(FULL_SPACING_XYZ_MM),
        "global_spacing_xyz_mm": global_spacing.tolist(),
        "thoracic_center_method": center_result.method,
        "renderer": renderer_record,
    }
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("ct", data=signed_ct, compression="lzf")
        handle.create_dataset("ct_hu", data=np.rint(global_hu).astype(np.int16), compression="lzf")
        handle.create_dataset("body_mask", data=global_body.astype(np.uint8), compression="lzf")
        handle.create_dataset("lung_mask", data=global_lung.astype(np.uint8), compression="lzf")
        handle.create_dataset("lesion_proxy", data=global_lesion.astype(np.uint8), compression="lzf")
        handle.create_dataset("drr_frontal_ap", data=drrs[0].astype(np.float16), compression="lzf")
        handle.create_dataset("drr_left_lateral", data=drrs[1].astype(np.float16), compression="lzf")
        handle.create_dataset("projection_matrices", data=matrices.astype(np.float32))
        handle.attrs["case_id"] = case_id
        handle.attrs["patient_id"] = f"MSD-{case_id}"
        handle.attrs["series_uid"] = f"MSD-{case_id}"
        handle.attrs["split"] = "test"
        handle.attrs["nodule_count"] = lesion_components
        handle.attrs["source_case_dir"] = str(case_dir.resolve())
        handle.attrs["metadata_json"] = json.dumps(metadata, separators=(",", ":"))
    temporary.replace(output_h5)

    audit = {
        "case_id": case_id,
        "status": "pass",
        "source_shape_xyz": list(source_ct.GetSize()),
        "source_spacing_xyz_mm": list(source_ct.GetSpacing()),
        "body_fraction": float(global_body.mean()),
        "lung_fraction": float(global_lung.mean()),
        "lesion_voxels": int(global_lesion.sum()),
        "lesion_components": lesion_components,
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
        "h5_sha256": sha256_file(output_h5),
        "h5_bytes": output_h5.stat().st_size,
    }
    write_json(metadata_path, {"metadata": metadata, "audit": audit})
    if make_preview:
        save_preview(
            output_root / "previews" / f"{case_id}.png",
            case_id,
            global_hu,
            global_lesion,
            drrs,
        )
    return audit


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    image_paths = sorted(args.images.glob("*.nii.gz"))
    discovered_cases = len(image_paths)
    if args.max_cases:
        image_paths = image_paths[: args.max_cases]
    if not image_paths:
        raise FileNotFoundError(f"No MSD CTs under {args.images}")
    labels = {path.name: path for path in args.labels.glob("*.nii.gz")}
    missing = [path.name for path in image_paths if path.name not in labels]
    if missing:
        raise FileNotFoundError(f"Missing labels: {missing[:5]}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_stats = args.internal_pack / "projection_stats.json"
    shutil.copy2(source_stats, args.output_root / "projection_stats.json")
    preview_indices = set(
        evenly_spaced_indices(len(image_paths), min(args.preview_cases, len(image_paths)))
        if args.preview_cases
        else []
    )
    rows = []
    failures = []
    for index, image_path in enumerate(image_paths):
        try:
            row = process_case(
                image_path,
                labels[image_path.name],
                args.output_root,
                protocol,
                args,
                index in preview_indices,
            )
            rows.append(row)
            print(
                f"[{index + 1}/{len(image_paths)}] {row['case_id']} "
                f"lesion={row['lesion_voxels']} lung={row['lung_fraction']:.3f}",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {
                    "case_id": image_path.name.removesuffix(".nii.gz"),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"FAIL {image_path.name}: {failures[-1]['error']}", flush=True)
    rows.sort(key=lambda row: row["case_id"])
    case_ids = [row["case_id"] for row in rows]
    (args.output_root / "test.txt").write_text("\n".join(case_ids) + "\n", encoding="utf-8")
    write_csv(args.output_root / "manifest.csv", rows)
    failure_path = args.output_root / "failures.csv"
    if failures:
        write_csv(failure_path, failures)
    else:
        failure_path.unlink(missing_ok=True)
    expected = len(image_paths)
    checks = {
        "all_cases_prepared": len(rows) == expected and not failures,
        "all_lesion_masks_nonempty": all(int(row["lesion_voxels"]) > 0 for row in rows),
        "all_lung_masks_nonempty": all(float(row["lung_fraction"]) > 0.01 for row in rows),
        "all_drrs_nonconstant": all(
            min(float(row["ap_std"]), float(row["lat_std"])) > 1e-5 for row in rows
        ),
        "internal_projection_normalization_reused": sha256_file(
            args.output_root / "projection_stats.json"
        )
        == sha256_file(source_stats),
    }
    summary = {
        "protocol_id": "UPATH-MSD-LUNG-EXTERNAL-v1.0",
        "status": "pass" if all(checks.values()) else "fail",
        "role": "zero-tuning external CT-derived DRR transport test",
        "cases_discovered": discovered_cases,
        "cases_selected": expected,
        "cases": len(rows),
        "failures": len(failures),
        "checks": checks,
        "aggregate": {
            "lesion_voxels_median": (
                float(np.median([row["lesion_voxels"] for row in rows])) if rows else None
            ),
            "lung_fraction_mean": (
                float(np.mean([row["lung_fraction"] for row in rows])) if rows else None
            ),
            "body_boundary_voxels_positive_cases": int(
                sum(int(row["body_boundary_voxels"]) > 0 for row in rows)
            ),
        },
        "provenance": {
            "protocol_config_sha256": sha256_file(args.protocol_config),
            "internal_projection_stats_sha256": sha256_file(source_stats),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "pass":
        raise SystemExit("MSD external preparation failed")


if __name__ == "__main__":
    main()
