#!/usr/bin/env python3
"""Pack the passed R0 release into efficient B0 128-cubed HDF5 cases."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import h5py
import nibabel as nib
import numpy as np

from upathx2ct.sota.canonical import physical_lps_to_index_xyz
from upathx2ct.sota.protocol import sha256_file, write_json


HU_MIN = -1024.0
HU_MAX = 1500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/processed/upath_lidc_v1_release_20260724"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/upath_lidc_v1_training_128"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_zyx(path: Path, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(
        np.transpose(nib.load(str(path)).get_fdata(dtype=np.float32), (2, 1, 0))
    ).astype(dtype, copy=False)


def log_moments(image: np.ndarray) -> dict[str, float | int]:
    values = np.log1p(np.maximum(np.asarray(image, dtype=np.float64), 0.0))
    return {
        "count": int(values.size),
        "sum": float(values.sum(dtype=np.float64)),
        "sum_square": float(np.square(values).sum(dtype=np.float64)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def pack_case(task: dict[str, Any]) -> dict[str, Any]:
    row = task["row"]
    case_dir = Path(row["case_dir"])
    output_path = Path(task["output_path"])
    if output_path.is_file() and task["resume"]:
        with h5py.File(output_path, "r") as handle:
            complete = (
                tuple(handle["ct"].shape) == (128, 128, 128)
                and tuple(handle["drr_frontal_ap"].shape) == (512, 512)
                and tuple(handle["projection_matrices"].shape) == (2, 3, 4)
            )
            ap = np.asarray(handle["drr_frontal_ap"], dtype=np.float32)
            lat = np.asarray(handle["drr_left_lateral"], dtype=np.float32)
        if complete:
            return {
                **row,
                "packed_path": str(output_path.resolve()),
                "packed_sha256": sha256_file(output_path),
                "packed_bytes": output_path.stat().st_size,
                "ap_moments": log_moments(ap),
                "lat_moments": log_moments(lat),
            }

    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    ct_hu = np.clip(
        load_zyx(case_dir / "ct_global_128.nii.gz", np.float32),
        HU_MIN,
        HU_MAX,
    )
    ct = (2.0 * (ct_hu - HU_MIN) / (HU_MAX - HU_MIN) - 1.0).astype(
        np.float16
    )
    body = load_zyx(case_dir / "body_mask_global_128.nii.gz", np.float32) > 0.5
    lung = load_zyx(case_dir / "lung_mask_global_128.nii.gz", np.float32) > 0.5
    lesion = (
        load_zyx(case_dir / "lesion_proxy_global_128.nii.gz", np.float32) > 0.5
    )
    with np.load(case_dir / "drrs_ap_left_lateral.npz") as handle:
        ap = np.asarray(handle["frontal_ap"], dtype=np.float32)
        lat = np.asarray(handle["left_lateral"], dtype=np.float32)
    with np.load(case_dir / "projection_matrices_centered_lps.npz") as handle:
        matrices = np.stack(
            (
                np.asarray(handle["frontal_ap"], dtype=np.float32),
                np.asarray(handle["left_lateral"], dtype=np.float32),
            )
        )
    global_origin = metadata["canonical"]["global_origin_lps_mm"]
    global_spacing = metadata["canonical"]["global_spacing_xyz_mm"]
    nodule_centers_zyx = []
    nodule_diameters = []
    nodule_ids = []
    for nodule in metadata["annotations"]["nodules"]:
        center_xyz = physical_lps_to_index_xyz(
            nodule["center_lps_mm"],
            global_origin,
            global_spacing,
        )
        nodule_centers_zyx.append(center_xyz[::-1])
        nodule_diameters.append(float(nodule["diameter_mm"]))
        nodule_ids.append(str(nodule["nodule_id"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".h5.tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("ct", data=ct, compression="lzf")
        handle.create_dataset(
            "ct_hu",
            data=np.rint(ct_hu).astype(np.int16),
            compression="lzf",
        )
        handle.create_dataset("body_mask", data=body.astype(np.uint8), compression="lzf")
        handle.create_dataset("lung_mask", data=lung.astype(np.uint8), compression="lzf")
        handle.create_dataset(
            "lesion_proxy",
            data=lesion.astype(np.uint8),
            compression="lzf",
        )
        handle.create_dataset("drr_frontal_ap", data=ap.astype(np.float16), compression="lzf")
        handle.create_dataset(
            "drr_left_lateral",
            data=lat.astype(np.float16),
            compression="lzf",
        )
        handle.create_dataset("projection_matrices", data=matrices)
        handle.create_dataset(
            "nodule_centers_zyx",
            data=np.asarray(nodule_centers_zyx, dtype=np.float32).reshape(-1, 3),
        )
        handle.create_dataset(
            "nodule_diameters_mm",
            data=np.asarray(nodule_diameters, dtype=np.float32),
        )
        string_type = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "nodule_ids",
            data=np.asarray(nodule_ids, dtype=object),
            dtype=string_type,
        )
        handle.attrs["case_id"] = row["case_id"]
        handle.attrs["patient_id"] = row["patient_id"]
        handle.attrs["series_uid"] = row["series_uid"]
        handle.attrs["split"] = row["split"]
        handle.attrs["nodule_count"] = len(nodule_ids)
        handle.attrs["source_case_dir"] = str(case_dir.resolve())
        handle.attrs["metadata_json"] = json.dumps(
            {
                "protocol_id": metadata["protocol_id"],
                "full_spacing_xyz_mm": metadata["canonical"][
                    "full_spacing_xyz_mm"
                ],
                "global_spacing_xyz_mm": global_spacing,
                "global_origin_lps_mm": global_origin,
                "nodule_centers_zyx": np.asarray(
                    nodule_centers_zyx,
                    dtype=np.float32,
                ).reshape(-1, 3).tolist(),
                "nodule_diameters_mm": nodule_diameters,
                "nodule_ids": nodule_ids,
            },
            separators=(",", ":"),
        )
    temporary.replace(output_path)
    return {
        **row,
        "packed_path": str(output_path.resolve()),
        "packed_sha256": sha256_file(output_path),
        "packed_bytes": output_path.stat().st_size,
        "ap_moments": log_moments(ap),
        "lat_moments": log_moments(lat),
    }


def combine_moments(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    moments = [row[key] for row in rows if row["split"] == "train"]
    count = sum(int(item["count"]) for item in moments)
    total = sum(float(item["sum"]) for item in moments)
    total_square = sum(float(item["sum_square"]) for item in moments)
    mean = total / count
    variance = max(total_square / count - mean**2, 1e-12)
    return {
        "pixels": count,
        "log1p_mean": mean,
        "log1p_std": float(np.sqrt(variance)),
        "log1p_min": min(float(item["minimum"]) for item in moments),
        "log1p_max": max(float(item["maximum"]) for item in moments),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    release_root = (
        args.release_root if args.release_root.is_absolute() else root / args.release_root
    )
    output_root = (
        args.output_root if args.output_root.is_absolute() else root / args.output_root
    )
    release_summary = json.loads(
        (release_root / "summary.json").read_text(encoding="utf-8")
    )
    if release_summary.get("status") != "pass":
        raise RuntimeError("R0 release is not valid")
    rows = read_csv(release_root / "manifest.csv")
    if args.max_cases:
        rows = rows[: args.max_cases]
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "row": row,
            "output_path": str(output_root / "cases" / f"{row['case_id']}.h5"),
            "resume": args.resume,
        }
        for row in rows
    ]
    packed = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(pack_case, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                packed.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "case_id": task["row"]["case_id"],
                        "series_uid": task["row"]["series_uid"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if completed % 25 == 0 or errors:
                print(
                    f"[{completed}/{len(tasks)}] packed={len(packed)} "
                    f"errors={len(errors)}",
                    flush=True,
                )
    packed.sort(key=lambda row: (row["split"], row["patient_id"], row["series_uid"]))
    if errors:
        write_csv(output_root / "failures.csv", errors)
    if not packed:
        raise RuntimeError("No training cases were packed")
    manifest_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"ap_moments", "lat_moments"}
        }
        for row in packed
    ]
    write_csv(output_root / "manifest.csv", manifest_rows)
    for split in ("train", "val", "test"):
        case_ids = [row["case_id"] for row in packed if row["split"] == split]
        (output_root / f"{split}.txt").write_text(
            "\n".join(case_ids) + ("\n" if case_ids else ""),
            encoding="utf-8",
        )
    stats = {
        "protocol_id": "UPATH-LIDC-v1.0",
        "definition_split": "train",
        "transform": "log1p(max(drr, 0)), then fixed train mean/std and clip [-6,6]",
        "views": {
            "frontal_ap": combine_moments(packed, "ap_moments"),
            "left_lateral": combine_moments(packed, "lat_moments"),
        },
    }
    write_json(output_root / "projection_stats.json", stats)
    checks = {
        "all_selected_cases_packed": len(packed) == len(rows) and not errors,
        "all_splits_present": {row["split"] for row in packed}
        == {"train", "val", "test"},
        "projection_stats_train_only": stats["definition_split"] == "train",
        "all_h5_hashes_recorded": all(
            len(row["packed_sha256"]) == 64 for row in packed
        ),
    }
    summary = {
        "protocol_id": "UPATH-LIDC-v1.0",
        "gate": "P0-pack",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "selected": len(rows),
            "packed": len(packed),
            "errors": len(errors),
            "train": sum(row["split"] == "train" for row in packed),
            "val": sum(row["split"] == "val" for row in packed),
            "test": sum(row["split"] == "test" for row in packed),
        },
        "bytes": sum(int(row["packed_bytes"]) for row in packed),
        "projection_stats": stats,
        "provenance": {
            "release_summary_sha256": sha256_file(release_root / "summary.json"),
            "release_manifest_sha256": sha256_file(release_root / "manifest.csv"),
            "packed_manifest_sha256": sha256_file(output_root / "manifest.csv"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "README.md").write_text(
        "# UPath-LIDC-v1 128-cubed training pack\n\n"
        f"Status: **{summary['status'].upper()}**.\n\n"
        "This derived pack contains normalized global CT, masks, raw cone-beam "
        "DRRs and explicit camera matrices. Projection normalization statistics "
        "use the training split only.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "pass":
        raise SystemExit("Training pack failed")


if __name__ == "__main__":
    main()
